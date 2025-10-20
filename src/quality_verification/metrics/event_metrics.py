from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from quality_verification.io.event_io import EventArray


def compute_event_density(events: EventArray, width: int, height: int) -> float:
    """Events per pixel per second."""
    if width <= 0 or height <= 0:
        raise ValueError("Invalid sensor geometry for event density computation.")
    duration = events.duration_seconds()
    if duration <= 0.0:
        return 0.0
    return float(events.timestamps.size) / (width * height * duration)


def compute_event_rate(events: EventArray) -> float:
    """Events per second."""
    duration = events.duration_seconds()
    if duration <= 0.0:
        return 0.0
    return float(events.timestamps.size) / duration


def compute_polarity_balance(events: EventArray) -> Dict[str, float]:
    """Return the fraction of ON/OFF events."""
    if events.timestamps.size == 0:
        return {"on_ratio": 0.0, "off_ratio": 0.0}
    # Polarity is expected to be 1 for ON, 0 or -1 for OFF.
    on_mask = events.polarity > 0
    on_ratio = float(np.count_nonzero(on_mask)) / float(events.timestamps.size)
    off_ratio = 1.0 - on_ratio
    return {"on_ratio": on_ratio, "off_ratio": off_ratio}


def compute_temporal_precision(events: EventArray) -> float:
    """Standard deviation of inter-event intervals in microseconds."""
    if events.timestamps.size < 2:
        return 0.0
    intervals = np.diff(events.timestamps.astype(np.int64))
    return float(np.std(intervals))


def accumulate_event_image(
    events: EventArray,
    width: int,
    height: int,
    polarity_weighted: bool = True,
) -> np.ndarray:
    """Accumulate events into a 2D histogram image."""
    if width <= 0 or height <= 0:
        raise ValueError("Invalid sensor geometry for accumulation.")
    image = np.zeros((height, width), dtype=np.float32)
    if events.timestamps.size == 0:
        return image

    x = np.clip(events.x.astype(np.int32), 0, width - 1)
    y = np.clip(events.y.astype(np.int32), 0, height - 1)
    if polarity_weighted:
        weights = np.where(events.polarity > 0, 1.0, -1.0)
    else:
        weights = np.ones_like(x, dtype=np.float32)

    np.add.at(image, (y, x), weights)
    return image


def compute_event_edge_correlation(event_image: np.ndarray, edge_image: np.ndarray) -> Optional[float]:
    """Compute Pearson correlation between event accumulation and frame edges."""
    if event_image.size == 0 or edge_image.size == 0:
        return None
    if event_image.shape != edge_image.shape:
        raise ValueError("Event and edge images must share the same spatial shape.")
    event_flat = event_image.reshape(-1)
    edge_flat = edge_image.reshape(-1)
    if np.all(event_flat == 0) or np.all(edge_flat == 0):
        return None
    event_norm = (event_flat - np.mean(event_flat))
    edge_norm = (edge_flat - np.mean(edge_flat))
    denom = np.linalg.norm(event_norm) * np.linalg.norm(edge_norm)
    if denom == 0.0:
        return None
    return float(np.dot(event_norm, edge_norm) / denom)


def polarity_accuracy_against_brightness(
    events: EventArray,
    brightness_delta: float,
) -> Optional[float]:
    """Compare event polarity with expected sign from frame brightness change."""
    if events.timestamps.size == 0:
        return None
    on_mask = events.polarity > 0
    if brightness_delta > 0:
        match = np.count_nonzero(on_mask)
    elif brightness_delta < 0:
        match = np.count_nonzero(~on_mask)
    else:
        # For negligible change, balanced polarities are desired.
        match = np.count_nonzero(on_mask) if events.timestamps.size == 0 else min(
            np.count_nonzero(on_mask),
            events.timestamps.size - np.count_nonzero(on_mask),
        )
    return float(match) / float(events.timestamps.size)
