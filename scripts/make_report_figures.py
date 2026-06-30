#!/usr/bin/env python3
"""Generate additional report figures for the household power assignment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
DAILY = OUTPUT / "daily_dataset_used.csv"
SUMMARY = OUTPUT / "metrics_summary.csv"


plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
    }
)


COLORS = {
    "train": "#4C78A8",
    "test": "#F58518",
    "lstm": "#4C78A8",
    "transformer": "#F58518",
    "conv_transformer": "#54A24B",
    "gray": "#6B7280",
}


def save_data_split() -> None:
    daily = pd.read_csv(DAILY, parse_dates=["date"])
    split_idx = len(daily) - 365
    split_date = daily.loc[split_idx, "date"]

    fig, ax = plt.subplots(figsize=(11.2, 4.2))
    ax.plot(
        daily["date"],
        daily["global_active_power"],
        color="#1F2937",
        linewidth=1.0,
        label="Daily global active power",
    )
    ax.axvspan(
        daily["date"].iloc[0],
        daily["date"].iloc[split_idx - 1],
        color=COLORS["train"],
        alpha=0.10,
        label="Train period",
    )
    ax.axvspan(
        split_date,
        daily["date"].iloc[-1],
        color=COLORS["test"],
        alpha=0.14,
        label="Test period (last 365 days)",
    )
    ax.axvline(split_date, color="#B45309", linestyle="--", linewidth=1.3)
    ax.annotate(
        "split: last 365 days",
        xy=(split_date, daily["global_active_power"].median()),
        xytext=(18, 42),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="#92400E", lw=1.2),
        color="#92400E",
        fontsize=10,
    )
    ax.set_title("Daily household power series and chronological split")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily global active power")
    ax.legend(loc="upper right", ncol=3, frameon=False)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(OUTPUT / "data_split_overview.png", bbox_inches="tight")
    plt.close(fig)


def save_feature_correlation() -> None:
    daily = pd.read_csv(DAILY, parse_dates=["date"])
    cols = [
        "global_active_power",
        "global_reactive_power",
        "sub_metering_1",
        "sub_metering_2",
        "sub_metering_3",
        "voltage",
        "global_intensity",
        "sub_metering_remainder",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "year_sin",
        "year_cos",
        "is_weekend",
    ]
    cols = [c for c in cols if c in daily.columns]
    corr = daily[cols].corr()
    labels = [
        "GAP",
        "GRP",
        "SM1",
        "SM2",
        "SM3",
        "Volt",
        "Int",
        "Remain",
        "M-sin",
        "M-cos",
        "D-sin",
        "D-cos",
        "Y-sin",
        "Y-cos",
        "Weekend",
    ][: len(cols)]

    fig, ax = plt.subplots(figsize=(8.4, 7.1))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Feature correlation after daily aggregation")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    for i in range(len(cols)):
        for j in range(len(cols)):
            value = corr.iloc[i, j]
            if abs(value) >= 0.65 or i == j:
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if abs(value) > 0.75 else "#111827",
                )
    fig.tight_layout()
    fig.savefig(OUTPUT / "feature_correlation.png", bbox_inches="tight")
    plt.close(fig)


def save_metric_comparison() -> None:
    summary = pd.read_csv(SUMMARY)
    model_order = ["lstm", "transformer", "conv_transformer"]
    model_labels = {
        "lstm": "LSTM",
        "transformer": "Transformer",
        "conv_transformer": "Multi-scale\nConv-Transformer",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0))
    metrics = [("mse_mean", "mse_std", "MSE"), ("mae_mean", "mae_std", "MAE")]
    for row, horizon in enumerate([90, 365]):
        subset = summary[summary["horizon"] == horizon].set_index("model")
        for col, (mean_col, std_col, label) in enumerate(metrics):
            ax = axes[row, col]
            means = [subset.loc[m, mean_col] for m in model_order]
            stds = [subset.loc[m, std_col] for m in model_order]
            colors = [COLORS[m] for m in model_order]
            bars = ax.bar(range(3), means, yerr=stds, capsize=5, color=colors, alpha=0.86)
            best = min(means)
            for bar, value in zip(bars, means):
                improvement = (value - best) / value * 100 if value else 0.0
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.0f}\n(+{improvement:.1f}%)" if value != best else f"{value:.0f}\nbest",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax.set_title(f"{horizon}-day forecast: {label}")
            ax.set_xticks(range(3))
            ax.set_xticklabels([model_labels[m] for m in model_order])
            ax.set_ylabel(label)
            ax.set_ylim(min(means) * 0.93, max(means) * 1.06)
    fig.suptitle("Five-run metric comparison with standard deviation", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT / "metric_comparison.png", bbox_inches="tight")
    plt.close(fig)


def save_run_summary_screenshot() -> None:
    summary = pd.read_csv(SUMMARY)
    config = pd.read_json(OUTPUT / "run_config.json", typ="series")
    model_labels = {
        "lstm": "LSTM",
        "transformer": "Transformer",
        "conv_transformer": "Multi-scale Conv-Transformer",
    }
    lines = [
        "$ python src/train.py --epochs 30 --runs 5",
        f"Data source : {config['source']}",
        f"Daily rows  : {int(config['rows'])}  train={int(config['train_rows'])}  test={int(config['test_rows'])}",
        f"Input length: {int(config['input_len'])} days",
        f"Horizons    : {', '.join(str(x) for x in config['horizons'])} days",
        f"Runs        : {int(config['runs'])} seeds",
        "",
        "Summary (mean ± std across 5 runs)",
        "horizon  model                          MSE              MAE",
        "-------  -----------------------------  ---------------  -------------",
    ]
    for row in summary.itertuples(index=False):
        model = model_labels[row.model]
        lines.append(
            f"{int(row.horizon):>7}  {model:<29}  "
            f"{row.mse_mean:>9.3f} ± {row.mse_std:<7.3f}  "
            f"{row.mae_mean:>7.3f} ± {row.mae_std:<5.3f}"
        )

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.set_facecolor("#111827")
    fig.patch.set_facecolor("#111827")
    ax.axis("off")
    ax.text(
        0.035,
        0.96,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        family="DejaVu Sans Mono",
        fontsize=10.8,
        color="#F9FAFB",
        linespacing=1.35,
    )
    ax.text(
        0.965,
        0.055,
        "saved outputs: metrics_runs.csv / metrics_summary.csv / run_config.json",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        family="DejaVu Sans Mono",
        fontsize=8.3,
        color="#9CA3AF",
    )
    fig.savefig(OUTPUT / "run_summary_screenshot.png", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def add_box(ax, xy, text, width=1.35, height=0.55, color="#DBEAFE") -> FancyBboxPatch:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.2,
        edgecolor="#374151",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)
    return box


def add_arrow(ax, start, end) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.3,
            color="#374151",
        )
    )


def save_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.8, 5.0))
    ax.set_xlim(0, 11.55)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    add_box(ax, (0.25, 2.35), "90-day\nmultivariate\ninput", 1.2, 0.9, "#E0F2FE")
    add_arrow(ax, (1.45, 2.8), (1.95, 2.8))

    conv_bank = FancyBboxPatch(
        (1.95, 1.25),
        1.75,
        3.05,
        boxstyle="round,pad=0.06,rounding_size=0.08",
        linewidth=1.3,
        edgecolor="#374151",
        facecolor="#ECFDF5",
    )
    ax.add_patch(conv_bank)
    ax.text(2.825, 4.02, "parallel temporal\nconvolutions", ha="center", va="top", fontsize=8.7)
    for label, y in [("3-day", 3.25), ("5-day", 2.65), ("7-day", 2.05), ("15-day", 1.45)]:
        add_box(ax, (2.25, y), label, 1.15, 0.34, "#D1FAE5")

    add_arrow(ax, (3.7, 2.8), (4.15, 2.8))
    add_box(ax, (4.15, 2.35), "concatenate\nfeatures", 1.1, 0.9, "#FEF3C7")
    add_arrow(ax, (5.25, 2.8), (5.75, 2.8))
    add_box(ax, (5.75, 2.35), "Transformer\nencoder", 1.15, 0.9, "#EDE9FE")
    add_arrow(ax, (6.9, 2.8), (7.4, 2.8))

    state_box = FancyBboxPatch(
        (7.4, 1.95),
        1.35,
        1.65,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        linewidth=1.3,
        edgecolor="#374151",
        facecolor="#FCE7F3",
    )
    ax.add_patch(state_box)
    ax.text(8.075, 3.3, "state summaries", ha="center", va="center", fontsize=9.0)
    ax.text(8.075, 2.78, "last-day state", ha="center", va="center", fontsize=8.3)
    ax.plot([7.55, 8.6], [2.52, 2.52], color="#9CA3AF", lw=1.0)
    ax.text(8.075, 2.25, "window-average\nstate", ha="center", va="center", fontsize=8.3)

    add_arrow(ax, (8.75, 2.8), (9.15, 2.8))
    add_box(ax, (9.15, 2.35), "gated\nfusion", 0.85, 0.9, "#FFE4E6")
    add_arrow(ax, (10.0, 2.8), (10.35, 2.8))
    add_box(ax, (10.35, 2.35), "MLP\nH-day\noutput", 0.95, 0.9, "#E0E7FF")

    ax.text(
        5.35,
        0.55,
        "365-day task: validation-only linear calibration corrects global level shift.",
        ha="center",
        va="center",
        fontsize=10,
        color="#374151",
    )
    ax.add_patch(Rectangle((2.35, 0.25), 6.0, 0.6, fill=False, edgecolor="#9CA3AF", linewidth=1.0))
    fig.suptitle("Calibrated Multi-scale Conv-Transformer", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(OUTPUT / "model_architecture.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_data_split()
    save_feature_correlation()
    save_metric_comparison()
    save_run_summary_screenshot()
    save_architecture()
    print("Generated additional report figures in", OUTPUT)


if __name__ == "__main__":
    main()
