from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np

from quality_verification.utils.path_utils import collect_files


def list_frame_files(directory: Path | str) -> List[Path]:
    """Return a sorted list of JPG frame files contained in the directory."""
    return collect_files(directory, suffixes={".jpg", ".jpeg", ".png"})


def _require_cv2():  # pragma: no cover - import helper
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - defensive path
        raise ImportError(
            "opencv-python is required for frame-based metrics. Install via 'pip install opencv-python'."
        ) from exc
    return cv2


def load_image(path: Path | str, normalize: bool = True) -> np.ndarray:
    """Load an image file as an RGB array with optional normalization to [0, 1]."""
    cv2 = _require_cv2()
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if normalize:
        return image_rgb.astype(np.float32) / 255.0
    return image_rgb


def load_grayscale(path: Path | str, normalize: bool = True) -> np.ndarray:
    """Load a grayscale image with optional normalization to [0, 1]."""
    cv2 = _require_cv2()
    image_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    if normalize:
        return image_gray.astype(np.float32) / 255.0
    return image_gray
