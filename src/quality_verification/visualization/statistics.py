#!/usr/bin/env python3
"""Statistical utilities backing the visualization layer.

This module provides bootstrap confidence intervals, group-wise summaries,
multiple-comparison corrections, and correlation helpers tailored for the
metrics tracked in the quality verification pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


@dataclass
class MetricSummary:
    """Container describing central tendency and dispersion for a metric."""

    mean: float
    std: float
    count: int
    ci_low: float
    ci_high: float

    @property
    def half_width(self) -> float:
        return 0.5 * (self.ci_high - self.ci_low)


def _as_clean_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return arr[~np.isnan(arr)]


def bootstrap_mean_ci(values: Iterable[float], *, n_boot: int = 2000,
                      ci: float = 0.95, random_state: Optional[int] = None) -> Tuple[float, float]:
    """Return lower/upper bootstrap CI bounds for the mean."""

    clean = _as_clean_array(values)
    if clean.size == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    samples = rng.choice(clean, (n_boot, clean.size), replace=True).mean(axis=1)
    alpha = (1 - ci) / 2
    low, high = np.quantile(samples, [alpha, 1 - alpha])
    return float(low), float(high)


def summarise_metric(values: Iterable[float], *, ci: float = 0.95) -> MetricSummary:
    """Compute descriptive statistics for a 1-D collection."""

    clean = _as_clean_array(values)
    if clean.size == 0:
        return MetricSummary(np.nan, np.nan, 0, np.nan, np.nan)

    ci_low, ci_high = bootstrap_mean_ci(clean, ci=ci)
    return MetricSummary(
        mean=float(np.mean(clean)),
        std=float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0,
        count=int(clean.size),
        ci_low=ci_low,
        ci_high=ci_high,
    )


def group_metric_summary(df: pd.DataFrame, metric: str, group_col: str,
                         *, ci: float = 0.95) -> Dict[str, MetricSummary]:
    """Summarise ``metric`` within each level of ``group_col``."""

    if metric not in df.columns or group_col not in df.columns:
        return {}

    summaries: Dict[str, MetricSummary] = {}
    for level, subset in df.groupby(group_col):
        summaries[str(level)] = summarise_metric(subset[metric], ci=ci)
    return summaries


def cohen_d(a: Iterable[float], b: Iterable[float]) -> float:
    """Compute Cohen's d effect size (Welch's pooled standard deviation)."""

    a_clean = _as_clean_array(a)
    b_clean = _as_clean_array(b)
    if a_clean.size < 2 or b_clean.size < 2:
        return np.nan

    mean_diff = a_clean.mean() - b_clean.mean()
    var_a = a_clean.var(ddof=1)
    var_b = b_clean.var(ddof=1)
    pooled = np.sqrt(((var_a + var_b) / 2.0))
    if pooled == 0:
        return np.nan
    return float(mean_diff / pooled)


def pairwise_tests(df: pd.DataFrame, metric: str, group_col: str) -> pd.DataFrame:
    """Run pairwise Welch t-tests between all group levels with Holm correction."""

    if metric not in df.columns or group_col not in df.columns:
        return pd.DataFrame()

    groups = {level: _as_clean_array(subset[metric]) for level, subset in df.groupby(group_col)}
    levels = list(groups.keys())
    records = []
    p_values = []

    for i in range(len(levels)):
        for j in range(i + 1, len(levels)):
            g_i, g_j = levels[i], levels[j]
            vals_i, vals_j = groups[g_i], groups[g_j]
            if vals_i.size < 2 or vals_j.size < 2:
                p_val = np.nan
                t_stat = np.nan
            else:
                t_stat, p_val = stats.ttest_ind(vals_i, vals_j, equal_var=False)
            effect = cohen_d(vals_i, vals_j)
            records.append({
                'group_a': g_i,
                'group_b': g_j,
                't_stat': t_stat,
                'p_value': p_val,
                'effect_size': effect,
            })
            p_values.append(p_val)

    adjusted = holm_correction(p_values)
    for record, adj in zip(records, adjusted):
        record['p_value_holm'] = adj
        record['significant'] = adj < 0.05 if not np.isnan(adj) else False

    return pd.DataFrame.from_records(records)


def holm_correction(p_values: Sequence[Optional[float]]) -> List[float]:
    """Apply Holm-Bonferroni correction to a list of p-values."""

    indexed = [(idx, p) for idx, p in enumerate(p_values) if p is not None and not np.isnan(p)]
    if not indexed:
        return [np.nan for _ in p_values]

    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    adjusted = [np.nan for _ in p_values]

    for rank, (idx, p) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p)
        adjusted[idx] = adj

    # Ensure monotonicity
    for i in range(m - 2, -1, -1):
        idx_i, _ = indexed[i]
        idx_j, _ = indexed[i + 1]
        adjusted[idx_i] = min(adjusted[idx_i], adjusted[idx_j])

    return [adj if adj is not None else np.nan for adj in adjusted]


def anova_f_oneway(df: pd.DataFrame, metric: str, group_col: str) -> Dict[str, float]:
    """Run one-way ANOVA returning F-statistic and p-value."""

    if metric not in df.columns or group_col not in df.columns:
        return {'f_stat': np.nan, 'p_value': np.nan}

    groups = [vals for _, vals in df.groupby(group_col)[metric]]
    cleaned = [ _as_clean_array(vals) for vals in groups if _as_clean_array(vals).size > 1]
    if len(cleaned) < 2:
        return {'f_stat': np.nan, 'p_value': np.nan}

    f_stat, p_val = stats.f_oneway(*cleaned)
    return {'f_stat': float(f_stat), 'p_value': float(p_val)}


def reorder_correlation(corr: pd.DataFrame) -> pd.DataFrame:
    """Cluster rows/columns of a correlation matrix for visual clarity."""

    if corr.empty:
        return corr

    mat = 1 - np.abs(corr.to_numpy())
    np.fill_diagonal(mat, 0.0)
    if mat.shape[0] < 2:
        return corr

    condensed = squareform(mat, checks=False)
    linkage = hierarchy.linkage(condensed, method='average')
    order = hierarchy.leaves_list(linkage)
    ordered = corr.iloc[order, order]
    ordered.columns = corr.columns[order]
    return ordered


def partial_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the partial correlation matrix via precision matrix inversion."""

    if df.empty:
        return df

    arr = df.to_numpy(dtype=float)
    arr = arr[~np.isnan(arr).any(axis=1)]
    if arr.size == 0:
        return pd.DataFrame(np.nan, index=df.columns, columns=df.columns)

    cov = np.cov(arr, rowvar=False)
    try:
        prec = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return pd.DataFrame(np.nan, index=df.columns, columns=df.columns)

    d = np.sqrt(np.diag(prec))
    denom = np.outer(d, d)
    pcorr = -prec / denom
    np.fill_diagonal(pcorr, 1.0)
    return pd.DataFrame(pcorr, index=df.columns, columns=df.columns)


def summary_table_from_groups(group_stats: Mapping[str, MetricSummary]) -> pd.DataFrame:
    """Convert grouped ``MetricSummary`` instances into a tidy table."""

    if not group_stats:
        return pd.DataFrame(columns=['group', 'mean', 'ci_low', 'ci_high', 'std', 'count'])

    records = []
    for group, summary in group_stats.items():
        records.append({
            'group': group,
            'mean': summary.mean,
            'ci_low': summary.ci_low,
            'ci_high': summary.ci_high,
            'std': summary.std,
            'count': summary.count,
        })
    return pd.DataFrame.from_records(records)


__all__ = [
    'MetricSummary',
    'anova_f_oneway',
    'bootstrap_mean_ci',
    'cohen_d',
    'group_metric_summary',
    'holm_correction',
    'pairwise_tests',
    'partial_correlation',
    'reorder_correlation',
    'summarise_metric',
    'summary_table_from_groups',
]
