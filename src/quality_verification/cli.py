from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from quality_verification.pipelines.dvs_self_quality import evaluate_dvs_self_quality
from quality_verification.pipelines.rgb_vs_dvs_events import evaluate_rgb_vs_dvs_events
from quality_verification.pipelines.rgb_vs_dvs_frames import evaluate_rgb_vs_dvs_frames
from quality_verification.utils.device import resolve_device
from quality_verification.utils.path_utils import ensure_directory
from quality_verification.utils.reporting import write_json_report


def _print_summary(result: Dict[str, Any]) -> None:
    print(json.dumps(result.get("metrics_summary", {}), indent=2))


def _handle_outputs(result: Dict[str, Any], output_folder: Optional[Path]) -> None:
    if output_folder is None:
        return
    folder = ensure_directory(output_folder)
    report_path = folder / "report.json"
    write_json_report(result, report_path)
    print(f"Report written to {report_path}")


def _print_plot_locations(result: Dict[str, Any]) -> None:
    plots = result.get("plots")
    if not plots:
        return

    lines: List[str] = []

    def _walk(prefix: str, obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}{key}/" if prefix else f"{key}/"
                _walk(new_prefix, value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                _walk(prefix, value)
        else:
            lines.append(f"- {prefix}{obj}")

    _walk("", plots)
    if lines:
        print("Generated plot files:")
        for line in lines:
            print(line)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Synthetic DVS data quality evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    frames_parser = subparsers.add_parser(
        "rgb-vs-dvs-frames",
        help="Compare RGB frames against synthesized DVS frames.",
    )
    frames_parser.add_argument("--root", required=True, type=Path)
    frames_parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Subset of frame metrics to compute (default: all).",
    )
    frames_parser.add_argument("--limit", type=int, default=None, help="Limit number of frame pairs.")
    frames_parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help="Directory where reports and plots will be written.",
    )
    frames_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device for learned metrics (e.g., cpu, cuda, mps).",
    )

    events_parser = subparsers.add_parser(
        "rgb-vs-dvs-events",
        help="Compare RGB frames against synthesized DVS events (AEDAT4).",
    )
    events_parser.add_argument("--root", required=True, type=Path)
    events_parser.add_argument(
        "--frame-rate",
        required=True,
        type=float,
        help="Frame rate (FPS) of the original RGB sequence.",
    )
    events_parser.add_argument(
        "--sync-offset-us",
        type=int,
        default=0,
        help="Temporal offset in microseconds to align events with frames.",
    )
    events_parser.add_argument("--limit", type=int, default=None, help="Limit number of frame windows.")
    events_parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help="Directory where reports and plots will be written.",
    )
    events_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device for learned metrics (e.g., cpu, cuda, mps).",
    )

    dvs_parser = subparsers.add_parser(
        "dvs-self-quality",
        help="Evaluate DVS outputs without ground-truth references.",
    )
    dvs_parser.add_argument("--root", required=True, type=Path)
    dvs_parser.add_argument(
        "--window-ms",
        type=float,
        default=50.0,
        help="Sliding window size in milliseconds for event statistics.",
    )
    dvs_parser.add_argument("--limit", type=int, default=None, help="Limit number of windows/frames.")
    dvs_parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help="Directory where reports and plots will be written.",
    )
    dvs_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device for learned metrics (e.g., cpu, cuda, mps).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "rgb-vs-dvs-frames":
        output_dir = args.output_folder
        device, device_msg = resolve_device(getattr(args, "device", None))
        if device_msg:
            print(device_msg)
        result = evaluate_rgb_vs_dvs_frames(
            root=args.root,
            metrics=args.metrics,
            limit=args.limit,
            output_dir=output_dir,
            device=device,
        )
        _print_summary(result)
        _print_plot_locations(result)
        _handle_outputs(result, output_dir)
    elif args.command == "rgb-vs-dvs-events":
        output_dir = args.output_folder
        device, device_msg = resolve_device(getattr(args, "device", None))
        if device_msg:
            print(device_msg)
        result = evaluate_rgb_vs_dvs_events(
            root=args.root,
            frame_rate=args.frame_rate,
            sync_offset_us=args.sync_offset_us,
            limit=args.limit,
            output_dir=output_dir,
            device=device,
        )
        _print_summary(result)
        _print_plot_locations(result)
        _handle_outputs(result, output_dir)
    elif args.command == "dvs-self-quality":
        output_dir = args.output_folder
        device, device_msg = resolve_device(getattr(args, "device", None))
        if device_msg:
            print(device_msg)
        result = evaluate_dvs_self_quality(
            root=args.root,
            window_ms=args.window_ms,
            limit=args.limit,
            output_dir=output_dir,
            device=device,
        )
        print(json.dumps(result.get("event_quality", {}).get("metrics_summary", {}), indent=2))
        _print_plot_locations(result)
        _handle_outputs(result, output_dir)
    else:  # pragma: no cover - defensive path
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
