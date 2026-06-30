#!/usr/bin/env python3
"""Search stronger open-model variants on the household forecasting task.

This script is intentionally separate from ``src/train.py``. It lets us test
candidate proposed models on the GPU before promoting one into the official
training entry point.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "src" / "train.py"


def load_train_module():
    spec = importlib.util.spec_from_file_location("zuoye_train", TRAIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {TRAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["zuoye_train"] = module
    spec.loader.exec_module(module)
    return module


train_mod = load_train_module()


class LinearSkipConvTransformer(nn.Module):
    """Multi-scale Conv-Transformer with a direct temporal linear skip.

    The linear skip maps the past target curve directly to the future horizon.
    The Conv-Transformer branch then learns nonlinear residual corrections from
    the full multivariate window.
    """

    def __init__(
        self,
        n_features: int,
        horizon: int,
        input_len: int,
        d_model: int = 64,
        nhead: int = 4,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        branch_width = d_model // 4
        widths = [branch_width, branch_width, branch_width, d_model - 3 * branch_width]
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
        self.pos = train_mod.PositionalEncoding(d_model)
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
        self.residual_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )
        self.linear_skip = nn.Linear(input_len, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_history = x[:, :, 0]
        linear = self.linear_skip(target_history)
        z = torch.cat([conv(x.transpose(1, 2)) for conv in self.branches], dim=1)
        z = self.mix(z).transpose(1, 2)
        z = self.encoder(self.pos(z))
        last = z[:, -1]
        pooled = z.mean(dim=1)
        gate = self.gate(torch.cat([last, pooled], dim=-1))
        fused = gate * last + (1.0 - gate) * pooled
        return linear + self.residual_head(fused)


class FlexibleConvTransformer(nn.Module):
    """Capacity/scales variant of the proposed Conv-Transformer."""

    def __init__(
        self,
        n_features: int,
        horizon: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        kernels: tuple[int, ...] = (3, 5, 7, 15),
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        base = d_model // len(kernels)
        widths = [base] * (len(kernels) - 1) + [d_model - base * (len(kernels) - 1)]
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(n_features, width, kernel_size=kernel, padding=kernel // 2)
                for width, kernel in zip(widths, kernels)
            ]
        )
        self.mix = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos = train_mod.PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
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


class LstmSkipConvTransformer(nn.Module):
    """Linear skip plus Conv-Transformer and LSTM sequence summaries."""

    def __init__(
        self,
        n_features: int,
        horizon: int,
        input_len: int,
        d_model: int = 64,
        nhead: int = 4,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        branch_width = d_model // 4
        widths = [branch_width, branch_width, branch_width, d_model - 3 * branch_width]
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
        self.pos = train_mod.PositionalEncoding(d_model)
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
        self.lstm = nn.LSTM(n_features, d_model, num_layers=1, batch_first=True)
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )
        self.linear_skip = nn.Linear(input_len, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_history = x[:, :, 0]
        linear = self.linear_skip(target_history)
        z = torch.cat([conv(x.transpose(1, 2)) for conv in self.branches], dim=1)
        z = self.mix(z).transpose(1, 2)
        z = self.encoder(self.pos(z))
        lstm_out, _ = self.lstm(x)
        fused = self.fuse(torch.cat([z[:, -1], z.mean(dim=1), lstm_out[:, -1]], dim=-1))
        return linear + self.residual_head(fused)


class ResidualLastConvTransformer(nn.Module):
    """Conv-Transformer residual around the last observed target value."""

    def __init__(
        self,
        n_features: int,
        horizon: int,
        input_len: int,
        d_model: int = 64,
        nhead: int = 4,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        self.core = LinearSkipConvTransformer(
            n_features=n_features,
            horizon=horizon,
            input_len=input_len,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )
        self.core.linear_skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1, 0].unsqueeze(-1)
        residual = self.core.residual_head(
            self._encode(x)
        )
        return last + residual

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        core = self.core
        z = torch.cat([conv(x.transpose(1, 2)) for conv in core.branches], dim=1)
        z = core.mix(z).transpose(1, 2)
        z = core.encoder(core.pos(z))
        last = z[:, -1]
        pooled = z.mean(dim=1)
        gate = core.gate(torch.cat([last, pooled], dim=-1))
        return gate * last + (1.0 - gate) * pooled


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(variant: str, n_features: int, horizon: int, input_len: int) -> nn.Module:
    if variant == "current":
        return train_mod.MultiScaleConvTransformerForecast(n_features, horizon)
    if variant == "conv64":
        return FlexibleConvTransformer(n_features, horizon, d_model=64, num_layers=1)
    if variant == "conv64_deep":
        return FlexibleConvTransformer(n_features, horizon, d_model=64, num_layers=2)
    if variant == "conv96":
        return FlexibleConvTransformer(n_features, horizon, d_model=96, nhead=4, num_layers=1)
    if variant == "linear_skip":
        return LinearSkipConvTransformer(n_features, horizon, input_len)
    if variant == "lstm_skip":
        return LstmSkipConvTransformer(n_features, horizon, input_len)
    if variant == "last_residual":
        return ResidualLastConvTransformer(n_features, horizon, input_len)
    raise ValueError(f"Unknown variant: {variant}")


def train_one(
    variant: str,
    horizon: int,
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y_raw: np.ndarray,
    scalers,
    args: argparse.Namespace,
) -> Tuple[Dict[str, float], np.ndarray]:
    set_seed(seed)
    device = torch.device(args.device)
    model = make_model(variant, train_x.shape[-1], horizon, args.input_len).to(device)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss() if args.loss == "mse" else nn.SmoothL1Loss(beta=args.huber_beta)
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
    if args.calibration != "none":
        val_pred = predict_scaled(train_x[n_fit:])
        val_true = train_y[n_fit:]
        if len(val_pred) == 0:
            val_pred = predict_scaled(train_x[:n_fit])
            val_true = train_y[:n_fit]
        if args.calibration == "global":
            x_mean = float(val_pred.mean())
            y_mean = float(val_true.mean())
            denom = float(((val_pred - x_mean) ** 2).mean())
            slope = 1.0 if denom < 1e-8 else float(((val_pred - x_mean) * (val_true - y_mean)).mean() / denom)
            intercept = y_mean - slope * x_mean
            pred_scaled = slope * pred_scaled + intercept
        elif args.calibration == "per_step":
            x_mean = val_pred.mean(axis=0, keepdims=True)
            y_mean = val_true.mean(axis=0, keepdims=True)
            denom = ((val_pred - x_mean) ** 2).mean(axis=0, keepdims=True)
            slope = np.where(
                denom < 1e-8,
                1.0,
                ((val_pred - x_mean) * (val_true - y_mean)).mean(axis=0, keepdims=True) / denom,
            )
            intercept = y_mean - slope * x_mean
            pred_scaled = slope * pred_scaled + intercept
        else:
            raise ValueError(f"Unknown calibration: {args.calibration}")
    pred = pred_scaled * scalers.target_std + scalers.target_mean
    err = pred - test_y_raw
    return {
        "mse": float(np.mean(err**2)),
        "mae": float(np.mean(np.abs(err))),
        "best_val_loss": best_val,
    }, pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model_search"))
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--horizons", type=int, nargs="+", default=[90, 365])
    parser.add_argument("--variants", nargs="+", default=["current", "linear_skip", "lstm_skip"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--huber-beta", type=float, default=0.5)
    parser.add_argument("--calibration", choices=["none", "global", "per_step"], default="none")
    parser.add_argument("--test-days", type=int, default=365)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    args.root = args.root.resolve()
    args.data_dir = (args.root / args.data_dir).resolve()
    args.output_dir = (args.root / args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prepared = train_mod.load_or_prepare(args.root, args.data_dir, args.test_days)
    features_raw = prepared.daily[prepared.feature_columns].to_numpy(dtype=np.float32)
    target_raw = prepared.daily[train_mod.TARGET_COLUMN].to_numpy(dtype=np.float32)
    scalers = train_mod.make_scalers(features_raw, target_raw, prepared.split_idx)
    features = (features_raw - scalers.feature_mean) / scalers.feature_std
    target_scaled = (target_raw - scalers.target_mean) / scalers.target_std

    seeds = list(range(2026, 2026 + args.runs))
    rows: List[Dict[str, float]] = []
    start_time = time.time()
    print(f"device={args.device} variants={args.variants} seeds={seeds}")

    for horizon in args.horizons:
        train_x, train_y, test_x, _, _ = train_mod.build_windows(
            features, target_scaled, args.input_len, horizon, prepared.split_idx
        )
        _, _, _, test_y_unscaled, _ = train_mod.build_windows(
            features_raw, target_raw, args.input_len, horizon, prepared.split_idx
        )
        print(f"\nHorizon {horizon}: train={len(train_x)} test={len(test_x)}")
        for variant in args.variants:
            for seed in seeds:
                metrics, _ = train_one(
                    variant, horizon, seed, train_x, train_y, test_x, test_y_unscaled, scalers, args
                )
                row = {"horizon": horizon, "variant": variant, "seed": seed, **metrics}
                rows.append(row)
                print(
                    f"{horizon:>3}d {variant:<14} seed={seed} "
                    f"MSE={metrics['mse']:.3f} MAE={metrics['mae']:.3f} "
                    f"val={metrics['best_val_loss']:.5f}",
                    flush=True,
                )

    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby(["horizon", "variant"], as_index=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            best_val_mean=("best_val_loss", "mean"),
        )
        .sort_values(["horizon", "mse_mean"])
    )
    metrics.to_csv(args.output_dir / "model_search_runs.csv", index=False)
    summary.to_csv(args.output_dir / "model_search_summary.csv", index=False)
    (args.output_dir / "model_search_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nFinished in {time.time() - start_time:.1f}s -> {args.output_dir}")


if __name__ == "__main__":
    main()
