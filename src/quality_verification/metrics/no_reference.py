from __future__ import annotations

from typing import Optional

import numpy as np


def compute_brisque(image: np.ndarray, device: str = "cpu") -> Optional[float]:
    try:
        import pyiqa  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency path
        return None

    metric = pyiqa.create_metric("brisque", device=device)
    tensor = _to_pyiqa_tensor(image, device)
    value = metric(tensor, data_range=1.0)
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def compute_niqe(image: np.ndarray, device: str = "cpu") -> Optional[float]:
    try:
        import pyiqa  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency path
        return None

    metric = pyiqa.create_metric("niqe", device=device)
    tensor = _to_pyiqa_tensor(image, device)
    value = metric(tensor, data_range=1.0)
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def compute_iwe_contrast(event_image: np.ndarray) -> float:
    """Contrast metric based on image variance."""
    if event_image.size == 0:
        return 0.0
    return float(np.var(event_image))


def _to_pyiqa_tensor(image: np.ndarray, device: str):  # pragma: no cover - helper
    import torch

    if image.ndim == 2:
        tensor = torch.from_numpy(image).unsqueeze(0)
    else:
        tensor = torch.from_numpy(image).permute(2, 0, 1)
    tensor = tensor.float().unsqueeze(0)
    return tensor.to(device)
