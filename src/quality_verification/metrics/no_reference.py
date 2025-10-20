from __future__ import annotations

from typing import Optional

import numpy as np


def compute_brisque(image: np.ndarray) -> Optional[float]:
    try:
        import pyiqa  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency path
        return None

    metric = pyiqa.create_metric("brisque", device="cpu")
    value = metric(image, data_range=1.0)
    return float(value)


def compute_niqe(image: np.ndarray) -> Optional[float]:
    try:
        import pyiqa  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency path
        return None

    metric = pyiqa.create_metric("niqe", device="cpu")
    value = metric(image, data_range=1.0)
    return float(value)


def compute_iwe_contrast(event_image: np.ndarray) -> float:
    """Contrast metric based on image variance."""
    if event_image.size == 0:
        return 0.0
    return float(np.var(event_image))
