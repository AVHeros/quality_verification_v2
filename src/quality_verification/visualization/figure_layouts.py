#!/usr/bin/env python3
"""Reusable figure layout helpers for complex multi-panel visuals."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt


def label_subplots(axes: Sequence[plt.Axes], *, labels: Iterable[str] | None = None,
                   x_offset: float = -0.05, y_offset: float = 1.05,
                   fontdict: dict | None = None) -> None:
    """Annotate each axis with subplot lettering (A, B, ...).

    Parameters
    ----------
    axes:
        Iterable of axes in desired order.
    labels:
        Custom label sequence. Defaults to uppercase alphabet.
    x_offset, y_offset:
        Relative offsets applied in axes coordinates.
    fontdict:
        Optional Matplotlib text properties.
    """

    if labels is None:
        labels = (chr(65 + idx) for idx in range(len(axes)))
    for ax, label in zip(axes, labels):
        ax.text(x_offset, y_offset, label, transform=ax.transAxes,
                fontweight='bold', fontsize=12, fontdict=fontdict)


def build_shared_colorbar(fig: plt.Figure, im, *, label: str = '',
                          location: str = 'right', pad: float = 0.02,
                          fraction: float = 0.05) -> plt.colorbar:
    """Attach a shared colorbar with consistent sizing."""

    cbar = fig.colorbar(im, ax=fig.axes, location=location, pad=pad, fraction=fraction)
    if label:
        cbar.set_label(label, rotation=270 if location in {'right', 'left'} else 0,
                       labelpad=15 if location in {'right', 'left'} else 10)
    return cbar


def align_ylabels(axes: Sequence[plt.Axes]) -> None:
    """Align y-axis labels for a row of subplots."""

    fig = axes[0].figure
    fig.align_ylabels(axes)


__all__ = ['align_ylabels', 'build_shared_colorbar', 'label_subplots']
