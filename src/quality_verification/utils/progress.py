from __future__ import annotations

from typing import Iterable, Optional, TypeVar

T = TypeVar("T")


def progress_iter(
    iterable: Iterable[T],
    desc: Optional[str] = None,
    total: Optional[int] = None,
) -> Iterable[T]:
    """Wrap an iterable with tqdm progress feedback when available."""
    try:
        from tqdm.auto import tqdm  # type: ignore
    except ImportError:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=True)
