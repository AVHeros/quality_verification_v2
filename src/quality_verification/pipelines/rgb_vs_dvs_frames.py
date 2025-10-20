from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
    root: Path | str,
    metrics: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    device: str = "cpu",
) -> Dict[str, object]:
    root_path = Path(root).expanduser().resolve()
    modalities = find_modality_dirs(root_path)
    if "rgb" not in modalities:
        raise FileNotFoundError("Could not locate an RGB directory under the root.")
    if "dvs" not in modalities:
        raise FileNotFoundError("Could not locate a DVS directory under the root.")

    rgb_files = list_frame_files(modalities["rgb"])
    dvs_files = list_frame_files(modalities["dvs"])

    frame_pairs = pair_frames(rgb_files, dvs_files, limit=limit)
    if not frame_pairs:
        raise RuntimeError("No matching frame pairs were found between RGB and DVS data.")

    records: List[FrameMetricRecord] = []
    metric_values: List[Dict[str, float]] = []
    selected_metrics = [m.lower() for m in metrics] if metrics is not None else ["mse", "psnr", "ssim", "lpips"]
    metric_series: Dict[str, List[Optional[float]]] = {name: [] for name in selected_metrics}

    for rgb_path, dvs_path in progress_iter(frame_pairs, desc="Comparing frame pairs", total=len(frame_pairs)):
        rgb_img = load_image(rgb_path)
        dvs_img = load_image(dvs_path)
        metric_result = compute_frame_metrics(rgb_img, dvs_img, metrics=selected_metrics, device=device)
        records.append(
            FrameMetricRecord(
                stem=rgb_path.stem,
                rgb_path=rgb_path,
                dvs_path=dvs_path,
                metrics=metric_result,
            )
        )
        metric_values.append({k: v for k, v in metric_result.items() if v is not None})
        for name in selected_metrics:
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
        "root": str(root_path),
        "pair_count": len(frame_pairs),
        "metrics_summary": aggregate_to_dict(aggregates),
        "per_pair": [record.to_dict() for record in records],
        "plots": plot_paths,
        "device": device,
    }
