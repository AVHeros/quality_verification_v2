from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from quality_verification.io.frame_io import list_frame_files, load_image
from quality_verification.metrics.frame_metrics import compute_frame_metrics
from quality_verification.utils.path_utils import ensure_directory, find_modality_dirs, pair_frames
from quality_verification.utils.progress import progress_iter
from quality_verification.utils.plotting import plot_metric_series
from quality_verification.utils.reporting import aggregate_metric_series, aggregate_to_dict


@dataclass
class FrameMetricRecord:
    stem: str
    rgb_path: Path
    dvs_path: Path
    metrics: Dict[str, float | None]

    def to_dict(self) -> Dict[str, object]:
        return {
            "stem": self.stem,
            "rgb_path": str(self.rgb_path),
            "dvs_path": str(self.dvs_path),
            "metrics": self.metrics,
        }


def evaluate_rgb_vs_dvs_frames(
    root: Path | str | None = None,
    rgb_dir: Path | str | None = None,
    dvs_dir: Path | str | None = None,
    metrics: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    device: str = "cpu",
) -> Dict[str, object]:
    root_path = Path(root).expanduser().resolve() if root is not None else None
    rgb_path = Path(rgb_dir).expanduser().resolve() if rgb_dir is not None else None
    dvs_path = Path(dvs_dir).expanduser().resolve() if dvs_dir is not None else None

    if rgb_path is None or dvs_path is None:
        if root_path is None:
            raise ValueError("Provide a root directory or explicit RGB and DVS directories.")
        modalities = find_modality_dirs(root_path)
        if rgb_path is None:
            if "rgb" not in modalities:
                raise FileNotFoundError("Could not locate an RGB directory under the root.")
            rgb_path = modalities["rgb"]
        if dvs_path is None:
            if "dvs" not in modalities:
                raise FileNotFoundError("Could not locate a DVS directory under the root.")
            dvs_path = modalities["dvs"]

    if rgb_path is None or dvs_path is None:  # defensive; should not happen
        raise RuntimeError("Failed to resolve RGB and DVS directories.")
    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB input path not found: {rgb_path}")
    if not rgb_path.is_dir():
        raise NotADirectoryError(f"RGB input path must be a directory: {rgb_path}")
    if not dvs_path.exists():
        raise FileNotFoundError(f"DVS input path not found: {dvs_path}")
    if not dvs_path.is_dir():
        raise NotADirectoryError(f"DVS input path must be a directory: {dvs_path}")

    rgb_files = list_frame_files(rgb_path)
    dvs_files = list_frame_files(dvs_path)

    frame_pairs = pair_frames(rgb_files, dvs_files, limit=limit)
    if not frame_pairs:
        raise RuntimeError("No matching frame pairs were found between RGB and DVS data.")

    records: List[FrameMetricRecord] = []
    metric_values: List[Dict[str, float]] = []
    selected_metrics = [m.lower() for m in metrics] if metrics is not None else ["mse", "psnr", "ssim", "lpips"]
    diagnostic_metrics = [
        "mean_intensity_diff",
        "rgb_mean",
        "dvs_mean",
        "contrast_ratio",
        "rgb_std",
        "dvs_std",
    ]
    metric_series: Dict[str, List[Optional[float]]] = {
        name: [] for name in selected_metrics + diagnostic_metrics
    }

    for rgb_path, dvs_path in progress_iter(frame_pairs, desc="Comparing frame pairs", total=len(frame_pairs)):
        rgb_img = load_image(rgb_path)
        dvs_img = load_image(dvs_path)
        metric_result = compute_frame_metrics(rgb_img, dvs_img, metrics=selected_metrics, device=device)

        rgb_mean = float(np.mean(rgb_img))
        dvs_mean = float(np.mean(dvs_img))
        rgb_std = float(np.std(rgb_img))
        dvs_std = float(np.std(dvs_img))
        mean_diff = rgb_mean - dvs_mean
        contrast_ratio = float(rgb_std / dvs_std) if dvs_std > 1e-6 else None

        metric_result.update(
            {
                "mean_intensity_diff": mean_diff,
                "rgb_mean": rgb_mean,
                "dvs_mean": dvs_mean,
                "contrast_ratio": contrast_ratio,
                "rgb_std": rgb_std,
                "dvs_std": dvs_std,
            }
        )
        records.append(
            FrameMetricRecord(
                stem=rgb_path.stem,
                rgb_path=rgb_path,
                dvs_path=dvs_path,
                metrics=metric_result,
            )
        )
        metric_values.append({k: v for k, v in metric_result.items() if v is not None})
        for name in selected_metrics + diagnostic_metrics:
            metric_series[name].append(metric_result.get(name))

    aggregates = aggregate_metric_series(metric_values)

    plot_paths: Dict[str, str] = {}
    if output_dir is not None:
        base_dir = ensure_directory(Path(output_dir))
        plot_root = ensure_directory(base_dir / "plots")
        for name, series in metric_series.items():
            plot_path = plot_metric_series(
                series,
                plot_root / f"{name}.png",
                title=f"{name.upper()} across frame pairs",
                ylabel=name.upper(),
            )
            if plot_path is not None:
                plot_paths[name] = str(plot_path)

    return {
    "root": str(root_path) if root_path is not None else None,
    "rgb_dir": str(rgb_path),
    "dvs_dir": str(dvs_path),
        "pair_count": len(frame_pairs),
        "metrics_summary": aggregate_to_dict(aggregates),
        "per_pair": [record.to_dict() for record in records],
        "plots": plot_paths,
        "device": device,
    }
