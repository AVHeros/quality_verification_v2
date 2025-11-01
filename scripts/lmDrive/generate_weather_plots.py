#!/usr/bin/env python3
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ANALYSIS_DIR = Path(__file__).resolve().parent / "analysis_results"
VIS_DIR = ANALYSIS_DIR / "visualizations"
EVENTS_OUT = VIS_DIR / "events"
FRAMES_OUT = VIS_DIR / "frames"


# Static thresholds for common metrics (Good/Excellent)
STATIC_THRESHOLDS = {
    "frames": {
        "psnr_mean": {"good": 30.0, "excellent": 40.0, "higher_is_better": True},
        "ssim_mean": {"good": 0.92, "excellent": 0.95, "higher_is_better": True},
        "lpips_mean": {"good": 0.10, "excellent": 0.05, "higher_is_better": False},
        "mse_mean": {"good": 0.01, "excellent": 0.005, "higher_is_better": False},
    },
    "events": {
        "polarity_accuracy_mean": {"good": 0.80, "excellent": 0.90, "higher_is_better": True},
    },
}


def ensure_dirs() -> None:
    EVENTS_OUT.mkdir(parents=True, exist_ok=True)
    FRAMES_OUT.mkdir(parents=True, exist_ok=True)


def get_metric_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.endswith("_mean")]
    # Coerce numeric columns to numeric (ignore non-numeric quietly)
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return sorted(cols)


def plot_by_weather(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    # Ensure weather column exists and numeric
    if "weather" not in df.columns:
        return
    df = df.copy()
    df["weather"] = pd.to_numeric(df["weather"], errors="coerce").astype(pd.Int64Dtype())

    # Aggregate mean of metric per weather
    grouped = (
        df.dropna(subset=["weather"])
        .groupby("weather")[metric]
        .mean()
        .rename("value")
    )

    # Create full weather index 0..20 and align
    weather_index = pd.Index(range(0, 21), name="weather")
    aligned = grouped.reindex(weather_index)

    # Plot (scatter points only)
    fig, ax = plt.subplots(figsize=(10, 5))
    x_vals = aligned.index.astype(int)
    y_vals = aligned.values

    base_name = metric[:-5] if metric.endswith("_mean") else metric
    title = f"{base_name.replace('_', ' ').title()} by Weather"
    ax.set_title(title)
    ax.set_xlabel("Weather (0–20)")
    ax.set_ylabel(base_name)
    ax.set_xticks(list(range(0, 21, 1)))
    ax.grid(True, axis="y", alpha=0.3)

    # Threshold lines (Good/Excellent) gated by allowlist
    from quality_verification.visualization.plot_config import plot_config
    scope = out_path.parent.name  # "events" or "frames"
    base_name = metric[:-5] if metric.endswith("_mean") else metric
    # Only compute thresholds if metric is allowed to be shaded
    if scope in plot_config.should_shade and base_name in plot_config.should_shade[scope]:
        series = pd.to_numeric(df[metric], errors="coerce")
        thresholds = plot_config.compute_thresholds(base_name, series.to_numpy(), scope)
    else:
        thresholds = None

    # Determine y-limits to include thresholds and data, then shade regions
    valid = y_vals[~np.isnan(y_vals)]
    data_min = float(valid.min()) if valid.size else 0.0
    data_max = float(valid.max()) if valid.size else 1.0
    y_candidates_min = [data_min]
    y_candidates_max = [data_max]
    if thresholds:
        if thresholds.get("good") is not None:
            y_candidates_min.append(float(thresholds["good"]))
            y_candidates_max.append(float(thresholds["good"]))
        if thresholds.get("excellent") is not None:
            y_candidates_min.append(float(thresholds["excellent"]))
            y_candidates_max.append(float(thresholds["excellent"]))
    y_min = min(y_candidates_min)
    y_max = max(y_candidates_max)
    # add small margin
    margin = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    y_min -= margin
    y_max += margin
    ax.set_ylim(y_min, y_max)

    # Region shading: bad/good/excellent with light colors, gated by thresholds presence
    if thresholds and thresholds.get("good") is not None and thresholds.get("excellent") is not None:
        if thresholds.get("higher_is_better", True):
            ax.axhspan(y_min, thresholds["good"], facecolor="#f8d7da", alpha=0.35, zorder=1)
            ax.axhspan(thresholds["good"], thresholds["excellent"], facecolor="#cfe2ff", alpha=0.35, zorder=1)
            ax.axhspan(thresholds["excellent"], y_max, facecolor="#d1e7dd", alpha=0.35, zorder=1)
        else:
            ax.axhspan(y_min, thresholds["excellent"], facecolor="#d1e7dd", alpha=0.35, zorder=1)
            ax.axhspan(thresholds["excellent"], thresholds["good"], facecolor="#cfe2ff", alpha=0.35, zorder=1)
            ax.axhspan(thresholds["good"], y_max, facecolor="#f8d7da", alpha=0.35, zorder=1)

    # Scatter points
    mask = ~np.isnan(y_vals)
    ax.scatter(x_vals[mask], y_vals[mask], s=28, color="#1B2838", zorder=3)

    if thresholds and thresholds.get("good") is not None:
        if thresholds.get("good") is not None:
            ax.axhline(thresholds["good"], color="orange", linestyle="--", linewidth=1.5, label="Good", zorder=4)
        if thresholds.get("excellent") is not None:
            ax.axhline(thresholds["excellent"], color="green", linestyle=":", linewidth=1.8, label="Excellent", zorder=4)
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_dirs()

    events_csv = ANALYSIS_DIR / "events_metrics_summary.csv"
    frames_csv = ANALYSIS_DIR / "frames_metrics_summary.csv"

    if events_csv.exists():
        events_df = pd.read_csv(events_csv)
        event_metrics = get_metric_columns(events_df)
        for m in event_metrics:
            out_file = EVENTS_OUT / f"{m[:-5] if m.endswith('_mean') else m}.png"
            plot_by_weather(events_df, m, out_file)

    if frames_csv.exists():
        frames_df = pd.read_csv(frames_csv)
        frame_metrics = get_metric_columns(frames_df)
        for m in frame_metrics:
            out_file = FRAMES_OUT / f"{m[:-5] if m.endswith('_mean') else m}.png"
            plot_by_weather(frames_df, m, out_file)


if __name__ == "__main__":
    main()