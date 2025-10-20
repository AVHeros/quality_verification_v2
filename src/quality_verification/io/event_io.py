from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


@dataclass
class EventArray:
    timestamps: np.ndarray  # microseconds
    x: np.ndarray
    y: np.ndarray
    polarity: np.ndarray  # 1 for ON, 0/ -1 for OFF depending on source

    def duration_seconds(self) -> float:
        if self.timestamps.size < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0]) / 1_000_000.0


class AEDAT4Loader:
    """Lightweight wrapper around dv-processing to expose event data as numpy arrays."""

    def __init__(self, file_path: Path | str):
        self.path = Path(file_path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"AEDAT4 file not found: {self.path}")

        try:
            import dv_processing as dv  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "dv-processing is required to read AEDAT4 files. Install via 'pip install dv-processing'."
            ) from exc

        self._dv = dv
        self._reader, self._reader_kind = self._create_reader()
        self._geometry = self._infer_geometry()
        self._cached_events: Optional[EventArray] = None

    def _create_reader(self):
        io_module = getattr(self._dv, "io", None)
        if io_module is not None:
            if hasattr(io_module, "AedatFile"):
                return io_module.AedatFile(str(self.path)), "aedatfile"
            if hasattr(io_module, "MonoCameraRecording"):
                return io_module.MonoCameraRecording(str(self.path)), "monorecording"
        if hasattr(self._dv, "AedatFile"):
            return self._dv.AedatFile(str(self.path)), "aedatfile"
        if hasattr(self._dv, "MonoCameraRecording"):
            return self._dv.MonoCameraRecording(str(self.path)), "monorecording"
        raise AttributeError(
            "dv-processing does not expose an AedatFile reader. Ensure that your dv-processing installation satisfies this API."
        )

    def _infer_geometry(self) -> tuple[Optional[int], Optional[int]]:
        width = None
        height = None
        if hasattr(self._reader, "getEventResolution"):
            resolution = self._reader.getEventResolution()
            if isinstance(resolution, tuple) and len(resolution) == 2:
                width, height = resolution
        else:  # pragma: no cover - defensive fallback
            width = getattr(self._reader, "getEventResolutionWidth", lambda: None)()
            height = getattr(self._reader, "getEventResolutionHeight", lambda: None)()
        return width, height

    @property
    def geometry(self) -> tuple[Optional[int], Optional[int]]:
        return self._geometry

    def load_all(self) -> EventArray:
        if self._cached_events is None:
            events = self._read_events()
            self._cached_events = events
        return self._cached_events

    def _read_events(self) -> EventArray:
        if self._reader_kind == "aedatfile":
            return self._read_events_aedatfile()
        if self._reader_kind == "monorecording":
            return self._read_events_monorecording()
        raise RuntimeError("Unsupported reader kind encountered while parsing events.")

    def _read_events_aedatfile(self) -> EventArray:
        packets = []
        stream = self._reader["events"]
        for packet in stream:
            data = packet.numpy()
            if data.size == 0:
                continue
            packets.append(data)
        if not packets:
            empty = np.empty((0,), dtype=[("t", np.uint64), ("x", np.uint16), ("y", np.uint16), ("p", np.int8)])
            return EventArray(
                timestamps=empty["t"],
                x=empty["x"],
                y=empty["y"],
                polarity=empty["p"],
            )

        raw = np.concatenate(packets)
        if raw.dtype.names:  # structured array
            timestamps = raw["t"].astype(np.int64, copy=False)
            x = raw["x"].astype(np.int16, copy=False)
            y = raw["y"].astype(np.int16, copy=False)
            polarity = raw["p"].astype(np.int8, copy=False)
        else:  # fallback to standard Nx4 float/int representation
            timestamps = raw[:, 0].astype(np.int64, copy=False)
            x = raw[:, 1].astype(np.int16, copy=False)
            y = raw[:, 2].astype(np.int16, copy=False)
            polarity = raw[:, 3].astype(np.int8, copy=False)
        return EventArray(timestamps=timestamps, x=x, y=y, polarity=polarity)

    def _read_events_monorecording(self) -> EventArray:
        reader = self._reader
        if hasattr(reader, "resetSequentialRead"):
            reader.resetSequentialRead()

        packets: list[np.ndarray] = []
        while reader.isRunning():
            batch = reader.getNextEventBatch()
            if batch is None:
                break
            if not hasattr(batch, "numpy"):
                continue
            array = batch.numpy()
            if array.size == 0:
                continue
            packets.append(np.array(array, copy=True))

        if not packets:
            return EventArray(
                timestamps=np.empty((0,), dtype=np.int64),
                x=np.empty((0,), dtype=np.int16),
                y=np.empty((0,), dtype=np.int16),
                polarity=np.empty((0,), dtype=np.int8),
            )

        raw = np.concatenate(packets)
        if raw.dtype.names:
            timestamp_key = "timestamp" if "timestamp" in raw.dtype.names else raw.dtype.names[0]
            x_key = "x"
            y_key = "y"
            polarity_key = "polarity" if "polarity" in raw.dtype.names else raw.dtype.names[-1]
            timestamps = raw[timestamp_key].astype(np.int64, copy=False)
            x_vals = raw[x_key].astype(np.int16, copy=False)
            y_vals = raw[y_key].astype(np.int16, copy=False)
            polarity_vals = raw[polarity_key].astype(np.int8, copy=False)
        else:
            timestamps = raw[:, 0].astype(np.int64, copy=False)
            x_vals = raw[:, 1].astype(np.int16, copy=False)
            y_vals = raw[:, 2].astype(np.int16, copy=False)
            polarity_vals = raw[:, 3].astype(np.int8, copy=False)

        return EventArray(timestamps=timestamps, x=x_vals, y=y_vals, polarity=polarity_vals)

    def iter_time_windows(self, window_us: int) -> Iterator[EventArray]:
        events = self.load_all()
        if events.timestamps.size == 0:
            return
        start_ts = int(events.timestamps[0])
        end_ts = int(events.timestamps[-1])
        current_start = start_ts
        window_us = max(window_us, 1)

        while current_start < end_ts:
            current_end = min(current_start + window_us, end_ts)
            mask = (events.timestamps >= current_start) & (events.timestamps < current_end)
            if np.any(mask):
                yield EventArray(
                    timestamps=events.timestamps[mask],
                    x=events.x[mask],
                    y=events.y[mask],
                    polarity=events.polarity[mask],
                )
            current_start = current_end

    def slice_time_range(self, start_us: int, end_us: int) -> EventArray:
        events = self.load_all()
        mask = (events.timestamps >= start_us) & (events.timestamps < end_us)
        return EventArray(
            timestamps=events.timestamps[mask],
            x=events.x[mask],
            y=events.y[mask],
            polarity=events.polarity[mask],
        )

    def close(self) -> None:
        if hasattr(self._reader, "close"):
            self._reader.close()

    def __enter__(self) -> "AEDAT4Loader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
