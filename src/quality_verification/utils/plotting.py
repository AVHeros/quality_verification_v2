from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from quality_verification.utils.path_utils import ensure_directory
import numpy as np
from quality_verification.visualization.plot_config import plot_config


def _require_pyplot():  # pragma: no cover - optional dependency helper
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except ImportError:
        return None


def plot_metric_series(
    values: Sequence[Optional[float]],
    output_path: Path | str,
    title: str,
    ylabel: str,
    xlabel: str = "Index",
) -> Optional[Path]:
    plt = _require_pyplot()
    if plt is None:
        return None

    points = [(idx, val) for idx, val in enumerate(values) if val is not None]
    if not points:
        return None

    ensure_directory(Path(output_path).parent)
    indices, metric_values = zip(*points)

    fig, ax = plt.subplots()
    # Use scatter instead of line plot and add quality shading
    ax.scatter(indices, metric_values, s=28, color="#1B2838")
    # Try to infer metric name and scope from output filename
    metric_name = Path(output_path).stem
    scope = "frames"
    if metric_name in plot_config.static_thresholds.get("events", {}):
        scope = "events"
    elif metric_name in plot_config.static_thresholds.get("frames", {}):
        scope = "frames"
    else:
        # Heuristic scope inference
        lower = metric_name.lower()
        if any(tok in lower for tok in ["event", "polarity", "temporal"]):
            scope = "events"
        else:
            scope = "frames"

    # Respect explicit gating for shaded regions to avoid misleading visuals
    try:
        plot_config.apply_quality_shading(ax, np.asarray(metric_values, dtype=float), metric_name, scope)
    except Exception:
        # Fallback gracefully if shading fails
        pass
    # Derive publication-ready labels with units
    display_name, unit = plot_config.get_metric_info(metric_name)
    final_title = f"{display_name} ({unit}) across samples"
    ax.set_title(final_title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{display_name} ({unit})")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = Path(output_path).expanduser().resolve()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_histogram(
    values: Iterable[float],
    output_path: Path | str,
    title: str,
    xlabel: str,
    bins: int = 30,
) -> Optional[Path]:
    plt = _require_pyplot()
    if plt is None:
        return None

    values_list = [float(v) for v in values]
    if not values_list:
        return None

    ensure_directory(Path(output_path).parent)

    fig, ax = plt.subplots()
    ax.hist(values_list, bins=bins, alpha=0.75, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = Path(output_path).expanduser().resolve()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
