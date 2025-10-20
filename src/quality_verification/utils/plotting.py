from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

from quality_verification.utils.path_utils import ensure_directory


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
    ax.plot(indices, metric_values, marker="o", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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
