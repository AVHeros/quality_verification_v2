from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from quality_verification.io.event_io import AEDAT4Loader, EventArray
from quality_verification.io.frame_io import list_frame_files, load_image
from quality_verification.metrics.event_metrics import (
    accumulate_event_image,
    compute_event_density,
    compute_event_rate,
    compute_polarity_balance,
    compute_temporal_precision,
)
from quality_verification.metrics.no_reference import (
    compute_brisque,
    compute_iwe_contrast,
    compute_niqe,
)
from quality_verification.utils.path_utils import collect_files, ensure_directory, find_modality_dirs
from quality_verification.utils.progress import progress_iter
from quality_verification.utils.plotting import plot_histogram, plot_metric_series
from quality_verification.utils.reporting import aggregate_metric_series, aggregate_to_dict


@dataclass
class EventWindowRecord:
    index: int
    start_us: int
    end_us: int
    metrics: Dict[str, float | None]

    def to_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "start_us": self.start_us,
            "end_us": self.end_us,
            "metrics": self.metrics,
        }


def evaluate_dvs_self_quality(
    root: Path | str,
    window_ms: float = 50.0,
    limit: Optional[int] = None,
    output_dir: Optional[Path] = None,
    device: str = "cpu",
) -> Dict[str, object]:
    if window_ms <= 0:
        raise ValueError("Window size must be positive.")

    root_path = Path(root).expanduser().resolve()
    modalities = find_modality_dirs(root_path)
    if "dvs" not in modalities:
        raise FileNotFoundError("DVS directory not found under the root path.")

    dvs_root = modalities["dvs"]
    aedat_files = collect_files(dvs_root, suffixes={".aedat", ".aedat4"})
    events_summary: Dict[str, object] = {}
    event_series: Dict[str, List[Optional[float]]] = {
        "event_density": [],
        "event_rate": [],
        "temporal_precision_us_std": [],
        "on_ratio": [],
        "off_ratio": [],
        "iwe_contrast": [],
    }

    if aedat_files:
        aedat_path = aedat_files[0]
        window_us = int(window_ms * 1_000.0)
        with AEDAT4Loader(aedat_path) as loader:
            width, height = loader.geometry
            events_all = loader.load_all()

        total_duration = events_all.duration_seconds()
        metrics_per_window: List[Dict[str, float]] = []
        window_records: List[EventWindowRecord] = []

        if events_all.timestamps.size:
            start_ts = int(events_all.timestamps[0])
            end_ts = int(events_all.timestamps[-1])
            duration_us = max(0, end_ts - start_ts)
            total_windows = (duration_us + window_us - 1) // window_us
            if limit is not None:
                total_windows = min(total_windows, limit)
            window_range = range(total_windows)
            for idx in progress_iter(window_range, desc="Analyzing event windows", total=total_windows):
                current = start_ts + idx * window_us
                if current >= end_ts:
                    break
                window_end = min(current + window_us, end_ts)
                mask = (events_all.timestamps >= current) & (events_all.timestamps < window_end)
                window_events = EventArray(
                    timestamps=events_all.timestamps[mask],
                    x=events_all.x[mask],
                    y=events_all.y[mask],
                    polarity=events_all.polarity[mask],
                )
                metrics: Dict[str, float | None] = {}
                if width is not None and height is not None:
                    metrics["event_density"] = compute_event_density(window_events, width, height)
                    event_image = accumulate_event_image(window_events, width, height, polarity_weighted=True)
                    metrics["iwe_contrast"] = compute_iwe_contrast(event_image)
                metrics["event_rate"] = compute_event_rate(window_events)
                metrics["temporal_precision_us_std"] = compute_temporal_precision(window_events)
                polarity_stats = compute_polarity_balance(window_events)
                metrics.update(polarity_stats)

                metrics_per_window.append({k: float(v) for k, v in metrics.items() if isinstance(v, (float, int))})
                window_records.append(
                    EventWindowRecord(index=idx, start_us=current, end_us=window_end, metrics=metrics)
                )
                for name in event_series:
                    event_series[name].append(metrics.get(name))

        aggregates = aggregate_metric_series(metrics_per_window)
        events_summary = {
            "aedat_path": str(aedat_path),
            "total_duration_seconds": total_duration,
            "metrics_summary": aggregate_to_dict(aggregates),
            "per_window": [record.to_dict() for record in window_records],
        }

    # Optional: assess DVS frames with no-reference metrics if available
    dvs_frame_files = list_frame_files(dvs_root) if dvs_root.exists() else []
    frame_metrics: List[Dict[str, float]] = []
    frame_records: List[Dict[str, object]] = []
    frame_series: Dict[str, List[Optional[float]]] = {"brisque": [], "niqe": []}

    if limit is not None:
        dvs_frame_files = dvs_frame_files[:limit]

    for frame_path in progress_iter(dvs_frame_files, desc="Assessing DVS frames", total=len(dvs_frame_files)):
        frame = load_image(frame_path)
        record: Dict[str, float | None] = {
            "brisque": compute_brisque(frame, device=device),
            "niqe": compute_niqe(frame, device=device),
        }
        frame_records.append({"path": str(frame_path), "metrics": record})
        frame_metrics.append({k: v for k, v in record.items() if isinstance(v, (float, int))})
        for name in frame_series:
            frame_series[name].append(record.get(name))

    frame_summary = aggregate_to_dict(aggregate_metric_series(frame_metrics)) if frame_metrics else {}

    plot_paths: Dict[str, Dict[str, str]] = {}
    if output_dir is not None:
        base_dir = ensure_directory(Path(output_dir))
        if events_summary:
            event_plot_dir = ensure_directory(base_dir / "plots" / "events")
            event_plots: Dict[str, str] = {}
            for name, series in event_series.items():
                plot_path = plot_metric_series(
                    series,
                    event_plot_dir / f"{name}.png",
                    title=f"{name.replace('_', ' ').title()} across windows",
                    ylabel=name.replace("_", " ").title(),
                )
                if plot_path is not None:
                    event_plots[name] = str(plot_path)
            hist_path = plot_histogram(
                [v for v in event_series.get("event_rate", []) if v is not None],
                event_plot_dir / "event_rate_hist.png",
                title="Event Rate Distribution",
                xlabel="Events per second",
            )
            if hist_path is not None:
                event_plots["event_rate_histogram"] = str(hist_path)
            if event_plots:
                plot_paths["event"] = event_plots
        if frame_records:
            frame_plot_dir = ensure_directory(base_dir / "plots" / "frames")
            frame_plots: Dict[str, str] = {}
            for name, series in frame_series.items():
                plot_path = plot_metric_series(
                    series,
                    frame_plot_dir / f"{name}.png",
                    title=f"{name.upper()} across frames",
                    ylabel=name.upper(),
                )
                if plot_path is not None:
                    frame_plots[name] = str(plot_path)
            if frame_plots:
                plot_paths["frame"] = frame_plots

    return {
        "root": str(root_path),
        "window_ms": window_ms,
        "event_quality": events_summary,
        "frame_quality": {
            "metrics_summary": frame_summary,
            "per_frame": frame_records,
        },
        "plots": plot_paths,
        "device": device,
    }
