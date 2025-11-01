from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

from functools import lru_cache

import numpy as np


def compute_mse(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Mean squared error between two images."""
    diff = reference.astype(np.float32) - candidate.astype(np.float32)
    return float(np.mean(np.square(diff)))


def compute_psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB."""
    from skimage.metrics import peak_signal_noise_ratio  # type: ignore

    return float(peak_signal_noise_ratio(reference, candidate, data_range=1.0))


def compute_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Structural similarity index."""
    from skimage.metrics import structural_similarity  # type: ignore

    return float(
        structural_similarity(reference, candidate, data_range=1.0, channel_axis=-1)
    )


def compute_lpips(
    reference: np.ndarray,
    candidate: np.ndarray,
    net: str = "alex",
    device: str = "cpu",
) -> Optional[float]:
    """Learned perceptual similarity metric using the lpips library.

    Returns None if required dependencies are unavailable. Falls back to CPU
    when the requested device is not available. As a secondary fallback,
    attempts to use pyiqa's LPIPS implementation when lpips cannot be
    imported.
    """
    # Attempt torch import
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency path
        # Try pyiqa fallback without torch (pyiqa still requires torch typically)
        try:
            import pyiqa  # type: ignore
        except Exception:
            return None
        try:
            model = pyiqa.create_metric('lpips', device='cpu')  # type: ignore
            value = float(model(reference, candidate).item())  # type: ignore
            return value
        except Exception:
            return None

    # Resolve torch device with graceful CPU fallback
    try:
        torch_device = torch.device(device)
        # If CUDA requested but unavailable, fallback to CPU
        if torch_device.type == 'cuda' and not torch.cuda.is_available():
            torch_device = torch.device('cpu')
    except Exception:
        torch_device = torch.device('cpu')

    # Primary: lpips package
    try:
        loss_fn = _get_lpips_model(net, str(torch_device))
        ref_tensor = _to_lpips_tensor(reference, torch_device)
        cand_tensor = _to_lpips_tensor(candidate, torch_device)
        with torch.no_grad():
            value = loss_fn(ref_tensor, cand_tensor).item()
        return float(value)
    except ImportError:  # pragma: no cover - optional dependency path
        # Secondary: pyiqa LPIPS
        try:
            import pyiqa  # type: ignore
            model = pyiqa.create_metric('lpips', device=str(torch_device))  # type: ignore
            value = float(model(reference, candidate).item())  # type: ignore
            return value
        except Exception:
            return None


@lru_cache(maxsize=None)
def _get_lpips_model(net: str, device: str):  # pragma: no cover - cache helper
    import lpips  # type: ignore
    import torch

    torch_device = torch.device(device)
    model = lpips.LPIPS(net=net)
    model = model.to(torch_device)
    model.eval()
    return model


def _to_lpips_tensor(image: np.ndarray, device):  # pragma: no cover - helper
    import torch

    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() * 2.0 - 1.0
    return tensor.unsqueeze(0).to(device)


MetricFn = Callable[[np.ndarray, np.ndarray], Optional[float]]


def compute_frame_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    metrics: Optional[Iterable[str]] = None,
    device: str = "cpu",
) -> Dict[str, Optional[float]]:
    """Compute the selected metrics between two RGB images in [0, 1] range."""
    metric_map: Dict[str, MetricFn] = {
        "mse": compute_mse,
        "psnr": compute_psnr,
        "ssim": compute_ssim,
    }
    names = list(metrics) if metrics is not None else list(metric_map.keys())
    results: Dict[str, Optional[float]] = {}
    for name in names:
        key = name.lower()
        if key == "lpips":
            results[key] = compute_lpips(reference, candidate, device=device)
            continue
        func = metric_map.get(key)
        if func is None:
            continue
        results[key] = func(reference, candidate)
    return results
