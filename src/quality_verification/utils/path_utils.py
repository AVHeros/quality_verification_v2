from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def find_modality_dirs(root: Path | str) -> Dict[str, Path]:
    """Automatically detect modality directories under the given root."""
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Root path not found: {root_path}")

    modality_dirs: Dict[str, Path] = {}
    for child in root_path.iterdir():
        if not child.is_dir():
            continue
        name_lower = child.name.lower()
        if "rgb" in name_lower and "rgb" not in modality_dirs:
            modality_dirs["rgb"] = child
        elif "dvs" in name_lower and "dvs" not in modality_dirs:
            modality_dirs["dvs"] = child
    return modality_dirs


def collect_files(directory: Path | str, suffixes: Iterable[str]) -> List[Path]:
    """Collect files recursively that match one of the provided suffixes."""
    dir_path = Path(directory).expanduser().resolve()
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    suffix_set = {s.lower() for s in suffixes}
    results: List[Path] = []
    for path in dir_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffix_set:
            results.append(path)
    return sorted(results)


def pair_frames(
    rgb_files: Iterable[Path],
    dvs_files: Iterable[Path],
    limit: Optional[int] = None,
    allow_mismatch: bool = False,
    max_count_difference: int = 5,
) -> List[Tuple[Path, Path]]:
    """Pair RGB and DVS frame files by matching stem names.

    When stems do not overlap and ``allow_mismatch`` is True, falls back to
    pairing files in order provided the directory counts are within
    ``max_count_difference`` of one another. This supports datasets where one
    modality has a small number of extra frames.
    """

    rgb_list = list(rgb_files)
    dvs_list = list(dvs_files)

    rgb_map = {p.stem: p for p in rgb_list}
    dvs_map = {p.stem: p for p in dvs_list}

    common_stems = sorted(set(rgb_map) & set(dvs_map))
    if common_stems:
        if limit is not None:
            common_stems = common_stems[:limit]
        return [(rgb_map[stem], dvs_map[stem]) for stem in common_stems]

    if not allow_mismatch:
        return []

    if not rgb_list or not dvs_list:
        return []

    if abs(len(rgb_list) - len(dvs_list)) > max_count_difference:
        return []

    rgb_sorted = sorted(rgb_list)
    dvs_sorted = sorted(dvs_list)
    pair_count = min(len(rgb_sorted), len(dvs_sorted))
    if limit is not None:
        pair_count = min(pair_count, limit)
    return list(zip(rgb_sorted[:pair_count], dvs_sorted[:pair_count]))


def ensure_directory(path: Path | str) -> Path:
    """Create the directory if it does not yet exist."""
    dir_path = Path(path).expanduser().resolve()
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
