from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional


@dataclass
class MetricAggregate:
    count: int
    mean: float
    std: float
    minimum: float
    maximum: float


def aggregate_metric_series(records: Iterable[Mapping[str, float]]) -> Dict[str, MetricAggregate]:
    """Aggregate scalar metrics across multiple records."""
    value_lists: MutableMapping[str, List[float]] = {}
    for record in records:
        for key, value in record.items():
            if value is None:
                continue
            value_lists.setdefault(key, []).append(float(value))

    aggregates: Dict[str, MetricAggregate] = {}
    for key, values in value_lists.items():
        if not values:
            continue
        aggregates[key] = MetricAggregate(
            count=len(values),
            mean=mean(values),
            std=pstdev(values) if len(values) > 1 else 0.0,
            minimum=min(values),
            maximum=max(values),
        )
    return aggregates


def aggregate_to_dict(aggregates: Mapping[str, MetricAggregate]) -> Dict[str, Dict[str, float]]:
    """Convert aggregates to plain dictionaries suitable for serialization."""
    return {
        key: {
            "count": agg.count,
            "mean": agg.mean,
            "std": agg.std,
            "min": agg.minimum,
            "max": agg.maximum,
        }
        for key, agg in aggregates.items()
    }


def write_json_report(payload: Mapping, output_path: Path | str) -> Path:
    """Write the payload to JSON at the given path."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path
