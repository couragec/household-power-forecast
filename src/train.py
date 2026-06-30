#!/usr/bin/env python3
"""Household power forecasting experiments for the course assignment.

The script is intentionally self-contained:
1. Prefer local train.csv/test.csv (or tes.csv) when they are present.
2. Otherwise download the UCI raw minute-level data and aggregate it by day.
3. Train LSTM, Transformer, and a proposed calibrated multi-scale Conv-Transformer.
4. Report MSE/MAE mean and standard deviation across repeated runs.
5. Save plots and table images for the PDF report.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


UCI_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)

POWER_SUM_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]
POWER_MEAN_COLUMNS = ["voltage", "global_intensity"]
WEATHER_COLUMNS = ["rr", "nbjrr1", "nbjrr5", "nbjrr10", "nbjbrou"]
TARGET_COLUMN = "global_active_power"


@dataclass
class PreparedData:
    daily: pd.DataFrame
    split_idx: int
    feature_columns: List[str]
    source: str


@dataclass
class Scalers:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_std: float


class LSTMForecast(nn.Module):
    def __init__(self, n_features: int, horizon: int, hidden: int = 48) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecast(nn.Module):
    def __init__(
        self, n_features: int, horizon: int, d_model: int = 48, nhead: int = 4
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 3,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pos(self.input_proj(x))
        z = self.encoder(z)
        return self.head(z[:, -1])


class MultiScaleConvTransformerForecast(nn.Module):
    """A local-pattern first Transformer for the open-model part.

    Parallel temporal convolutions extract short-cycle patterns at several
    calendar scales before the sequence is passed to a Transformer encoder. A
    learned gate combines the final state and global average state so that
    long-horizon forecasts can use both recent momentum and whole-window
    context.
    """

    def __init__(
        self,
        n_features: int,
        horizon: int,
        d_model: int = 64,
        nhead: int = 4,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        branch_width = d_model // 4
        widths = [
            branch_width,
            branch_width,
            branch_width,
            d_model - 3 * branch_width,
        ]
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(n_features, widths[0], kernel_size=3, padding=1),
                nn.Conv1d(n_features, widths[1], kernel_size=5, padding=2),
                nn.Conv1d(n_features, widths[2], kernel_size=7, padding=3),
                nn.Conv1d(n_features, widths[3], kernel_size=15, padding=7),
            ]
        )
        self.mix = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels_first = x.transpose(1, 2)
        z = torch.cat([conv(channels_first) for conv in self.branches], dim=1)
        z = self.mix(z).transpose(1, 2)
        z = self.encoder(self.pos(z))
        last = z[:, -1]
        pooled = z.mean(dim=1)
        gate = self.gate(torch.cat([last, pooled], dim=-1))
        fused = gate * last + (1.0 - gate) * pooled
        return self.head(fused)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        col: col.strip().lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    }
    df = df.rename(columns=rename)
    aliases = {
        "global_active_power": "global_active_power",
        "global_reactive_power": "global_reactive_power",
        "global_intensity": "global_intensity",
        "sub_metering_1": "sub_metering_1",
        "sub_metering_2": "sub_metering_2",
        "sub_metering_3": "sub_metering_3",
    }
    for col in list(df.columns):
        compact = col.replace("__", "_")
        if compact in aliases and compact != col:
            df = df.rename(columns={col: aliases[compact]})
    return df


def parse_datetime(df: pd.DataFrame) -> pd.Series:
    if "date" in df.columns and "time" in df.columns:
        combined = df["date"].astype(str) + " " + df["time"].astype(str)
        return pd.to_datetime(combined, dayfirst=True, errors="coerce")
    for candidate in ["datetime", "timestamp", "date"]:
        if candidate in df.columns:
            return pd.to_datetime(df[candidate], dayfirst=True, errors="coerce")
    raise ValueError("CSV must include Date/Time, datetime, timestamp, or date columns.")


def read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python", na_values=["?", "NA", ""])
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            encoding="gbk",
            na_values=["?", "NA", ""],
        )
    return normalize_columns(df)


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df).copy()
    df["datetime"] = parse_datetime(df)
    df = df.dropna(subset=["datetime"]).sort_values("datetime")

    numeric_candidates = [
        *POWER_SUM_COLUMNS,
        *POWER_MEAN_COLUMNS,
        *WEATHER_COLUMNS,
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If the data are already daily, this grouping is still safe.
    df["date"] = df["datetime"].dt.floor("D")
    agg: Dict[str, str] = {}
    for col in POWER_SUM_COLUMNS:
        if col in df.columns:
            agg[col] = "sum"
    for col in POWER_MEAN_COLUMNS:
        if col in df.columns:
            agg[col] = "mean"
    for col in WEATHER_COLUMNS:
        if col in df.columns:
            agg[col] = "first"

    if TARGET_COLUMN not in agg:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}")

    daily = df.groupby("date", as_index=False).agg(agg)
    daily = daily.sort_values("date").reset_index(drop=True)
    add_features(daily)
    return daily


def add_features(daily: pd.DataFrame) -> None:
    for col in POWER_SUM_COLUMNS + POWER_MEAN_COLUMNS + WEATHER_COLUMNS:
        if col in daily.columns:
            daily[col] = pd.to_numeric(daily[col], errors="coerce")

    for col in POWER_SUM_COLUMNS + POWER_MEAN_COLUMNS + WEATHER_COLUMNS:
        if col in daily.columns:
            daily[col] = daily[col].interpolate(limit_direction="both")
            daily[col] = daily[col].ffill().bfill()

    needed = {"global_active_power", "sub_metering_1", "sub_metering_2", "sub_metering_3"}
    if needed.issubset(set(daily.columns)):
        daily["sub_metering_remainder"] = (
            daily["global_active_power"] * 1000.0 / 60.0
            - (
                daily["sub_metering_1"]
                + daily["sub_metering_2"]
                + daily["sub_metering_3"]
            )
        )

    date = pd.to_datetime(daily["date"])
    month = date.dt.month.to_numpy()
    dow = date.dt.dayofweek.to_numpy()
    day_of_year = date.dt.dayofyear.to_numpy()
    daily["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    daily["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    daily["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    daily["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    daily["year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    daily["year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    daily["is_weekend"] = (dow >= 5).astype(float)


def download_uci(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "individual_household_power_consumption.zip"
    txt_path = data_dir / "household_power_consumption.txt"
    if txt_path.exists():
        return txt_path

    if not zip_path.exists():
        print(f"Downloading UCI data from {UCI_ZIP_URL}")
        urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("household_power_consumption.txt", data_dir)
    return txt_path


def prepare_from_uci(data_dir: Path, test_days: int) -> PreparedData:
    daily_cache = data_dir / "daily_power.csv"
    if daily_cache.exists():
        daily = pd.read_csv(daily_cache, parse_dates=["date"])
    else:
        txt_path = download_uci(data_dir)
        df = pd.read_csv(
            txt_path,
            sep=";",
            na_values=["?"],
            low_memory=False,
        )
        daily = aggregate_daily(df)
        daily.to_csv(daily_cache, index=False)

    daily = daily.sort_values("date").reset_index(drop=True)
    split_idx = len(daily) - test_days
    if split_idx <= 460:
        raise ValueError("Not enough data after the requested test split.")
    feature_columns = choose_feature_columns(daily)
    return PreparedData(
        daily=daily,
        split_idx=split_idx,
        feature_columns=feature_columns,
        source="UCI raw minute-level data; last 365 days used as test split",
    )


def prepare_from_local(root: Path) -> Optional[PreparedData]:
    train_path = root / "train.csv"
    test_path = root / "test.csv"
    if not test_path.exists() and (root / "tes.csv").exists():
        test_path = root / "tes.csv"
    if not train_path.exists() or not test_path.exists():
        return None

    train_daily = aggregate_daily(read_csv_flexible(train_path))
    test_daily = aggregate_daily(read_csv_flexible(test_path))
    split_idx = len(train_daily)
    daily = pd.concat([train_daily, test_daily], ignore_index=True)
    daily = daily.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    # Recompute split after date sorting in case files are continuous but unsorted.
    last_train_date = pd.to_datetime(train_daily["date"]).max()
    split_idx = int((pd.to_datetime(daily["date"]) <= last_train_date).sum())
    feature_columns = choose_feature_columns(daily)
    return PreparedData(
        daily=daily,
        split_idx=split_idx,
        feature_columns=feature_columns,
        source="local train.csv/test.csv",
    )


def choose_feature_columns(daily: pd.DataFrame) -> List[str]:
    candidates = [
        *POWER_SUM_COLUMNS,
        *POWER_MEAN_COLUMNS,
        "sub_metering_remainder",
        *WEATHER_COLUMNS,
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "year_sin",
        "year_cos",
        "is_weekend",
    ]
    return [col for col in candidates if col in daily.columns and col != "date"]


def load_or_prepare(root: Path, data_dir: Path, test_days: int) -> PreparedData:
    local = prepare_from_local(root)
    if local is not None:
        return local
    return prepare_from_uci(data_dir, test_days)


def make_scalers(
    features: np.ndarray, target: np.ndarray, split_idx: int
) -> Scalers:
    feature_mean = features[:split_idx].mean(axis=0)
    feature_std = features[:split_idx].std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    target_mean = float(target[:split_idx].mean())
    target_std = float(target[:split_idx].std())
    if target_std < 1e-6:
        target_std = 1.0
    return Scalers(feature_mean, feature_std, target_mean, target_std)


def build_windows(
    features: np.ndarray,
    target: np.ndarray,
    input_len: int,
    horizon: int,
    split_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    train_x, train_y = [], []
    train_stop = split_idx - input_len - horizon + 1
    for start in range(max(0, train_stop)):
        train_x.append(features[start : start + input_len])
        train_y.append(target[start + input_len : start + input_len + horizon])

    test_x, test_y, test_starts = [], [], []
    first_test_start = split_idx - input_len
    last_start = len(features) - input_len - horizon
    for start in range(max(0, first_test_start), last_start + 1):
        output_start = start + input_len
        output_end = output_start + horizon
        if output_start >= split_idx and output_end <= len(features):
            test_x.append(features[start : start + input_len])
            test_y.append(target[output_start:output_end])
            test_starts.append(start)

    if not train_x or not test_x:
        raise ValueError(
            f"Unable to build windows for input={input_len}, horizon={horizon}."
        )

    return (
        np.asarray(train_x, dtype=np.float32),
        np.asarray(train_y, dtype=np.float32),
        np.asarray(test_x, dtype=np.float32),
        np.asarray(test_y, dtype=np.float32),
        test_starts,
    )


def make_model(name: str, n_features: int, horizon: int) -> nn.Module:
    if name == "lstm":
        return LSTMForecast(n_features, horizon)
    if name == "transformer":
        return TransformerForecast(n_features, horizon)
    if name == "conv_transformer":
        return MultiScaleConvTransformerForecast(n_features, horizon)
    raise ValueError(f"Unknown model: {name}")


def train_one(
    model_name: str,
    horizon: int,
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y_raw: np.ndarray,
    scalers: Scalers,
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], np.ndarray]:
    set_seed(seed)
    device = torch.device(args.device)
    model = make_model(model_name, train_x.shape[-1], horizon).to(device)
    n_train = len(train_x)
    n_fit = max(1, int(n_train * 0.8))
    fit_ds = TensorDataset(
        torch.from_numpy(train_x[:n_fit]), torch.from_numpy(train_y[:n_fit])
    )
    val_x = torch.from_numpy(train_x[n_fit:]).to(device)
    val_y = torch.from_numpy(train_y[n_fit:]).to(device)
    if len(val_x) == 0:
        val_x = torch.from_numpy(train_x[:n_fit]).to(device)
        val_y = torch.from_numpy(train_y[:n_fit]).to(device)

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        fit_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = float("inf")
    stale_epochs = 0

    for _epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).cpu().item())
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    def predict_scaled(array: np.ndarray) -> np.ndarray:
        preds = []
        with torch.no_grad():
            for start in range(0, len(array), args.batch_size):
                xb = torch.from_numpy(array[start : start + args.batch_size]).to(device)
                preds.append(model(xb).cpu().numpy())
        return np.concatenate(preds, axis=0)

    pred_scaled = predict_scaled(test_x)
    should_calibrate = (
        model_name == "conv_transformer"
        and args.conv_calibration != "none"
        and horizon >= args.conv_calibration_horizon
    )
    if should_calibrate:
        val_pred = predict_scaled(train_x[n_fit:])
        val_true = train_y[n_fit:]
        if len(val_pred) == 0:
            val_pred = predict_scaled(train_x[:n_fit])
            val_true = train_y[:n_fit]
        if args.conv_calibration == "global":
            pred_mean = float(val_pred.mean())
            true_mean = float(val_true.mean())
            denom = float(((val_pred - pred_mean) ** 2).mean())
            slope = 1.0 if denom < 1e-8 else float(
                ((val_pred - pred_mean) * (val_true - true_mean)).mean() / denom
            )
            intercept = true_mean - slope * pred_mean
            pred_scaled = slope * pred_scaled + intercept
        else:
            raise ValueError(f"Unknown calibration mode: {args.conv_calibration}")

    pred = pred_scaled * scalers.target_std + scalers.target_mean
    err = pred - test_y_raw
    metrics = {
        "mse": float(np.mean(err**2)),
        "mae": float(np.mean(np.abs(err))),
        "best_val_loss": best_val,
    }
    return metrics, pred


def summarize_metrics(rows: List[Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["horizon", "model"], as_index=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            best_val_mean=("best_val_loss", "mean"),
        )
        .sort_values(["horizon", "mse_mean"])
        .reset_index(drop=True)
    )
    return summary


def save_metrics_table(summary: pd.DataFrame, output_dir: Path) -> None:
    display = summary.copy()
    display["task"] = display["horizon"].map({90: "90-day", 365: "365-day"}).fillna(
        display["horizon"].astype(str) + "-day"
    )
    display["model"] = display["model"].map(
        {
            "lstm": "LSTM",
            "transformer": "Transformer",
            "conv_transformer": "Multi-scale Conv-Transformer",
        }
    )
    table_df = display[
        ["task", "model", "mse_mean", "mse_std", "mae_mean", "mae_std"]
    ].copy()
    for col in ["mse_mean", "mse_std", "mae_mean", "mae_std"]:
        table_df[col] = table_df[col].map(lambda x: f"{x:.3f}")

    fig_h = 0.72 + 0.42 * len(table_df)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=["Task", "Model", "MSE mean", "MSE std", "MAE mean", "MAE std"],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#263238")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#eef3f5")
        else:
            cell.set_facecolor("#ffffff")
        cell.set_edgecolor("#b0bec5")
    ax.set_title("Experiment metrics across 5 runs", fontsize=14, weight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_table.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_prediction_plot(
    horizon: int,
    dates: Sequence[pd.Timestamp],
    ground_truth: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    labels = {
        "lstm": "LSTM",
        "transformer": "Transformer",
        "conv_transformer": "Multi-scale Conv-Transformer",
    }
    colors = {
        "ground_truth": "#111827",
        "lstm": "#1f77b4",
        "transformer": "#ff7f0e",
        "conv_transformer": "#2ca02c",
    }
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(dates, ground_truth, label="Ground Truth", color=colors["ground_truth"], lw=2.2)
    for name, pred in predictions.items():
        ax.plot(dates, pred, label=labels[name], lw=1.6, color=colors[name], alpha=0.92)
    ax.set_title(f"{horizon}-day forecast: power prediction vs. ground truth")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily global active power")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / f"forecast_{horizon}d.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_run_config(args: argparse.Namespace, prepared: PreparedData, output_dir: Path) -> None:
    config = {
        "source": prepared.source,
        "rows": int(len(prepared.daily)),
        "train_rows": int(prepared.split_idx),
        "test_rows": int(len(prepared.daily) - prepared.split_idx),
        "features": prepared.feature_columns,
        "input_len": args.input_len,
        "horizons": args.horizons,
        "runs": args.runs,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": args.patience,
        "conv_calibration": args.conv_calibration,
        "conv_calibration_horizon": args.conv_calibration_horizon,
        "device": args.device,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--horizons", type=int, nargs="+", default=[90, 365])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-days", type=int, default=365)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--conv-calibration",
        choices=["none", "global"],
        default="global",
        help="Validation-set output calibration for the proposed Conv-Transformer.",
    )
    parser.add_argument(
        "--conv-calibration-horizon",
        type=int,
        default=365,
        help="Apply proposed-model calibration only at or above this horizon.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    args.root = args.root.resolve()
    args.data_dir = (args.root / args.data_dir).resolve()
    args.output_dir = (args.root / args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    prepared = load_or_prepare(args.root, args.data_dir, args.test_days)
    prepared.daily.to_csv(args.output_dir / "daily_dataset_used.csv", index=False)
    save_run_config(args, prepared, args.output_dir)

    features_raw = prepared.daily[prepared.feature_columns].to_numpy(dtype=np.float32)
    target_raw = prepared.daily[TARGET_COLUMN].to_numpy(dtype=np.float32)
    scalers = make_scalers(features_raw, target_raw, prepared.split_idx)
    features = (features_raw - scalers.feature_mean) / scalers.feature_std
    target_scaled = (target_raw - scalers.target_mean) / scalers.target_std

    model_names = ["lstm", "transformer", "conv_transformer"]
    seeds = list(range(2026, 2026 + args.runs))
    all_rows: List[Dict[str, float]] = []
    forecast_store: Dict[int, Dict[str, np.ndarray]] = {}
    forecast_truth: Dict[int, np.ndarray] = {}
    forecast_dates: Dict[int, Sequence[pd.Timestamp]] = {}

    print(f"Data source: {prepared.source}")
    print(
        f"Daily rows: {len(prepared.daily)}, train: {prepared.split_idx}, "
        f"test: {len(prepared.daily) - prepared.split_idx}"
    )
    print(f"Features ({len(prepared.feature_columns)}): {prepared.feature_columns}")

    for horizon in args.horizons:
        train_x, train_y, test_x, test_y_raw, test_starts = build_windows(
            features,
            target_scaled,
            args.input_len,
            horizon,
            prepared.split_idx,
        )
        _, _, _, test_y_unscaled, _ = build_windows(
            features_raw,
            target_raw,
            args.input_len,
            horizon,
            prepared.split_idx,
        )
        print(
            f"\nHorizon {horizon}: train windows={len(train_x)}, "
            f"test windows={len(test_x)}"
        )
        forecast_store[horizon] = {}
        first_output_start = test_starts[0] + args.input_len
        forecast_dates[horizon] = pd.to_datetime(
            prepared.daily["date"].iloc[first_output_start : first_output_start + horizon]
        )
        forecast_truth[horizon] = test_y_unscaled[0]

        for model_name in model_names:
            first_seed_pred: Optional[np.ndarray] = None
            for seed in seeds:
                metrics, pred = train_one(
                    model_name,
                    horizon,
                    seed,
                    train_x,
                    train_y,
                    test_x,
                    test_y_unscaled,
                    scalers,
                    args,
                )
                row = {
                    "horizon": horizon,
                    "model": model_name,
                    "seed": seed,
                    **metrics,
                }
                all_rows.append(row)
                if seed == seeds[0]:
                    first_seed_pred = pred[0]
                print(
                    f"{horizon:>3}d {model_name:<16} seed={seed} "
                    f"MSE={metrics['mse']:.3f} MAE={metrics['mae']:.3f}"
                )
            assert first_seed_pred is not None
            forecast_store[horizon][model_name] = first_seed_pred

    metrics = pd.DataFrame(all_rows)
    summary = summarize_metrics(all_rows)
    metrics.to_csv(args.output_dir / "metrics_runs.csv", index=False)
    summary.to_csv(args.output_dir / "metrics_summary.csv", index=False)
    save_metrics_table(summary, args.output_dir)
    for horizon in args.horizons:
        save_prediction_plot(
            horizon,
            forecast_dates[horizon],
            forecast_truth[horizon],
            forecast_store[horizon],
            args.output_dir,
        )

    elapsed = time.time() - start_time
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nFinished in {elapsed:.1f} seconds. Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
