from __future__ import annotations

from typing import Optional, Tuple


def resolve_device(requested: Optional[str]) -> Tuple[str, Optional[str]]:
    """Return a torch device string based on the requested value with graceful fallback."""
    if not requested:
        return "cpu", None

    req = requested.strip().lower()
    target = req
    if req in {"gpu", "cuda"}:
        target = "cuda"
    elif req in {"metal"}:
        target = "mps"

    if target == "cpu":
        return "cpu", None

    try:
        import torch
    except ImportError:
        return "cpu", f"Requested device '{requested}' requires torch; falling back to CPU."

    if target.startswith("cuda") or target == "gpu":
        if torch.cuda.is_available():
            return target if target.startswith("cuda") else "cuda", None
        return "cpu", "CUDA is not available on this system; using CPU instead."

    if target == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and torch.backends.mps.is_available():
            return "mps", None
        return "cpu", "MPS backend is not available; using CPU instead."

    try:
        torch.device(target)
    except Exception:
        return "cpu", f"Unrecognized device '{requested}'; using CPU instead."
    return target, None
