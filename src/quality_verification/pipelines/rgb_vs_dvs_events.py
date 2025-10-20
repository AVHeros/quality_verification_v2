from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from quality_verification.io.event_io import AEDAT4Loader, EventArray
from quality_verification.io.frame_io import list_frame_files, load_grayscale
from quality_verification.metrics.event_metrics import (
    accumulate_event_image,
    compute_event_density,
    compute_event_edge_correlation,
    compute_event_rate,
    compute_polarity_balance,
    compute_temporal_precision,
    polarity_accuracy_against_brightness,
)
from quality_verification.utils.path_utils import collect_files, ensure_directory, find_modality_dirs
from quality_verification.utils.progress import progress_iter
from quality_verification.utils.plotting import plot_histogram, plot_metric_series
from quality_verification.utils.reporting import aggregate_metric_series, aggregate_to_dict


@dataclass
class EventMetricRecord:
    index: int
    frame_path: Path
    next_frame_path: Path
    metrics: Dict[str, float | None]

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "frame_path": str(self.frame_path),
            "next_frame_path": str(self.next_frame_path),
            "metrics": self.metrics,
        }


def _discover_aedat_file(dvs_root: Path) -> Path:
    files = collect_files(dvs_root, suffixes={".aedat", ".aedat4"})
    if not files:
        raise FileNotFoundError("No AEDAT or AEDAT4 files found under the DVS directory.")
    if len(files) > 1:
        # Prefer AEDAT4 by suffix ordering
        files.sort(key=lambda p: (0 if p.suffix.lower() == ".aedat4" else 1, p.name))
    return files[0]


def evaluate_rgb_vs_dvs_events(
    root: Path | str,
    frame_rate: float,
    sync_offset_us: int = 0,
    dvs_frames_resolution_fallback: Optional[tuple[int, int]] = None,
    limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    device: str = "cpu",
) -> Dict[str, object]:
    if frame_rate <= 0:
        raise ValueError("Frame rate must be positive.")
    frame_period_us = int(1_000_000 / frame_rate)

    root_path = Path(root).expanduser().resolve()
    modalities = find_modality_dirs(root_path)
    if "rgb" not in modalities or "dvs" not in modalities:
        raise FileNotFoundError("Root must contain both RGB and DVS directories.")

    rgb_files = list_frame_files(modalities["rgb"])
    if limit is not None:
        rgb_files = rgb_files[: limit + 1]
    if len(rgb_files) < 2:
        raise RuntimeError("At least two RGB frames are required to compare against events.")

    aedat_path = _discover_aedat_file(modalities["dvs"])

    with AEDAT4Loader(aedat_path) as loader:
        width, height = loader.geometry
        if width is None or height is None:
            if dvs_frames_resolution_fallback is None:
                # attempt to infer from RGB frame dimensions
                sample_img = load_grayscale(rgb_files[0], normalize=False)
                height, width = sample_img.shape
            else:
                width, height = dvs_frames_resolution_fallback
        width = int(width)
        height = int(height)

        events_all = loader.load_all()

    if events_all.timestamps.size == 0:
        raise RuntimeError("The AEDAT file does not contain any events.")

    base_timestamp = int(events_all.timestamps[0]) + int(sync_offset_us)

    records: List[EventMetricRecord] = []
    metric_values: List[Dict[str, float]] = []
    plot_metric_names = [
        "event_density",
        "event_rate",
        "temporal_precision_us_std",
        "on_ratio",
        "off_ratio",
        "polarity_accuracy",
        "event_edge_correlation",
        "brightness_delta",
    ]
    metric_series: Dict[str, List[Optional[float]]] = {name: [] for name in plot_metric_names}

    window_count = max(0, len(rgb_files) - 1)
    window_indices = range(window_count)
    for idx in progress_iter(window_indices, desc="Evaluating frame windows", total=window_count):
        frame_path = rgb_files[idx]
        next_frame_path = rgb_files[idx + 1]

        frame_start = base_timestamp + idx * frame_period_us
        frame_end = frame_start + frame_period_us
        mask = (events_all.timestamps >= frame_start) & (events_all.timestamps < frame_end)
        window_events = EventArray(
            timestamps=events_all.timestamps[mask],
            x=events_all.x[mask],
            y=events_all.y[mask],
            polarity=events_all.polarity[mask],
        )

        frame_gray = load_grayscale(frame_path)
        next_gray = load_grayscale(next_frame_path)
        brightness_delta = float(np.mean(np.log(next_gray + 1e-6) - np.log(frame_gray + 1e-6)))

        metrics: Dict[str, float | None] = {}
        metrics["event_density"] = (
            compute_event_density(window_events, width, height) if window_events.timestamps.size else 0.0
        )
        metrics["event_rate"] = compute_event_rate(window_events)
        metrics["temporal_precision_us_std"] = compute_temporal_precision(window_events)

        polarity_stats = compute_polarity_balance(window_events)
        metrics.update(polarity_stats)
        metrics["polarity_accuracy"] = polarity_accuracy_against_brightness(window_events, brightness_delta)
        metrics["brightness_delta"] = brightness_delta

        # Edge correlation
        try:
            import cv2  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency path
            edge_correlation = None
        else:
            event_image = accumulate_event_image(window_events, width, height, polarity_weighted=True)
            frame_uint8 = (next_gray * 255.0).astype(np.uint8)
            if frame_uint8.shape != event_image.shape:
                frame_uint8 = cv2.resize(
                    frame_uint8,
                    (event_image.shape[1], event_image.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
            edge_image = cv2.Canny(frame_uint8, 100, 200)
            edge_correlation = compute_event_edge_correlation(event_image, edge_image)
        metrics["event_edge_correlation"] = edge_correlation

        records.append(
            EventMetricRecord(
                index=idx,
                frame_path=frame_path,
                next_frame_path=next_frame_path,
                metrics=metrics,
            )
        )
        metric_values.append({k: v for k, v in metrics.items() if isinstance(v, (float, int))})
        for name in plot_metric_names:
            metric_series[name].append(metrics.get(name))

    aggregates = aggregate_metric_series(metric_values)

    plot_paths: Dict[str, str] = {}
    if output_dir is not None:
        base_dir = ensure_directory(Path(output_dir))
        plot_root = ensure_directory(base_dir / "plots")
        for name, series in metric_series.items():
            plot_path = plot_metric_series(
                series,
                plot_root / f"{name}.png",
                title=f"{name.replace('_', ' ').title()} across windows",
                ylabel=name.replace("_", " ").title(),
            )
            if plot_path is not None:
                plot_paths[name] = str(plot_path)
        # Histogram focusing on event rate distribution
        histogram_path = plot_histogram(
            [v for v in metric_series["event_rate"] if v is not None],
            plot_root / "event_rate_hist.png",
            title="Event Rate Distribution",
            xlabel="Events per second",
        )
        if histogram_path is not None:
            plot_paths["event_rate_histogram"] = str(histogram_path)

    return {
        "root": str(root_path),
        "frame_rate": frame_rate,
        "frame_count": len(rgb_files),
        "sensor_geometry": {"width": width, "height": height},
        "metrics_summary": aggregate_to_dict(aggregates),
        "per_window": [record.to_dict() for record in records],
        "plots": plot_paths,
        "device": device,
    }
