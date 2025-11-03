#!/usr/bin/env python3
"""Plot configuration and styling helpers for visualization modules.

This module centralises Matplotlib/NumPy configuration so that all figures
adhere to the same publication-quality style guidelines. It also exposes
utility methods for exporting figures in multiple formats and working with
color-blind–safe palettes.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

# -- Styling presets -----------------------------------------------------

_SERIF_FONT = 'DejaVu Serif'

_BASE_STYLE = {
    'figure.dpi': 110,
    'savefig.dpi': 300,
    'font.family': _SERIF_FONT,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.axisbelow': True,
    'axes.titleweight': 'semibold',
    'pdf.fonttype': 42,   # Preserve editable text in vector exports
    'ps.fonttype': 42,
}

_PRINT_STYLE = {
    'figure.figsize': (6.5, 4.0),  # Two-column journal width
    'axes.prop_cycle': plt.cycler(color=[
        '#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666'
    ]),
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'lines.linewidth': 2.0,
    'legend.frameon': False,
}

_PRESENTATION_STYLE = {
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'legend.fontsize': 13,
    'figure.titlesize': 22,
    'figure.figsize': (9.0, 5.5),
}


def _apply_style(style: Dict[str, float | str]) -> None:
    """Apply a style dictionary to Matplotlib without clearing rc defaults."""

    plt.rcParams.update(style)


# Apply style presets at import time (base + print default).
plt.style.use('default')
_apply_style(_BASE_STYLE)
_apply_style(_PRINT_STYLE)


# -- Color palettes ------------------------------------------------------

COLORS = {
    'primary': '#1b9e77',
    'secondary': '#d95f02',
    'accent': '#7570b3',
    'success': '#66a61e',
    'warning': '#e6ab02',
    'danger': '#d73027',
    'info': '#1f78b4',
    'light': '#f0f0f0',
    'dark': '#252525',
}

_COLORBLIND_SET = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666']
_GREYSCALE_COMPANION = ['#252525', '#525252', '#737373', '#969696', '#bdbdbd', '#d9d9d9']

CATEGORY_COLORS = _COLORBLIND_SET


PathLike = Union[str, Path]


@dataclass
class ExportSettings:
    """Configuration for exporting figures."""

    formats: Sequence[str] = ("png", "pdf")
    bbox_inches: str = "tight"
    transparent: bool = False

    def iter_paths(self, base_path: PathLike) -> Iterable[tuple[str, str]]:
        base = str(base_path)
        for fmt in self.formats:
            suffix = fmt.lower().lstrip('.')
            yield suffix, f"{base}.{suffix}"


class PlotConfig:
    """Configuration and convenience utilities for all plotting modules."""

    def __init__(self):
        self.colors = COLORS
        self.category_colors = CATEGORY_COLORS
        self.greyscale_palette = _GREYSCALE_COMPANION
        self.export_settings = ExportSettings()
        self.active_style = 'print'
        
        # Event metrics configuration
        self.event_metrics_config = {
            'event_density_mean': ('Event Density', 'Events/pixel'),
            'event_rate_mean': ('Event Rate', 'Events/sec'),
            'polarity_accuracy_mean': ('Polarity Accuracy', 'Ratio'),
            'temporal_precision_us_std': ('Temporal Precision', 'μs std'),
            'on_ratio': ('ON Ratio', 'Ratio'),
            'off_ratio': ('OFF Ratio', 'Ratio'),
            'event_edge_correlation': ('Edge Correlation', 'Correlation'),
            'brightness_delta': ('Brightness Delta', 'Delta')
        }
        
        # Frame metrics configuration
        self.frame_metrics_config = {
            'ssim_mean': ('SSIM', 'Similarity index'),
            'psnr_mean': ('PSNR', 'dB'),
            'mse_mean': ('MSE', 'Mean squared error'),
            'lpips_mean': ('LPIPS', 'Perceptual distance'),
            'mean_intensity_diff': ('Mean Intensity Diff', 'Intensity'),
            'rgb_mean': ('RGB Mean', 'Intensity'),
            'dvs_mean': ('DVS Mean', 'Intensity'),
            'contrast_ratio': ('Contrast Ratio', 'Ratio'),
            'rgb_std': ('RGB Std', 'Std deviation'),
            'dvs_std': ('DVS Std', 'Std deviation')
        }
        
        # Default figure sizes
        self.figure_sizes = {
            'overview': (18, 12),
            'comparison': (16, 12),
            'correlation': (12, 8),
            'single_metric': (10, 6),
            'large': (20, 15)
        }

        # Static thresholds for common metrics (Good/Excellent)
        # Mirrors the logic used in scripts/lmDrive/generate_weather_plots.py
        self.static_thresholds = {
            'frames': {
                'psnr_mean': {
                    'bad': 20.0,
                    'good': 25.0,
                    'excellent': 30.0,
                    'higher_is_better': True,
                },
                'ssim_mean': {
                    'bad': 0.50,
                    'good': 0.70,
                    'excellent': 0.90,
                    'higher_is_better': True,
                },
                'lpips_mean': {
                    'bad': 0.50,
                    'good': 0.30,
                    'excellent': 0.10,
                    'higher_is_better': False,
                },
                'mse_mean': {
                    'bad': 0.10,
                    'good': 0.05,
                    'excellent': 0.01,
                    'higher_is_better': False,
                },
                # Base metric names used in per-pair/per-frame series plots
                'psnr': {
                    'bad': 20.0,
                    'good': 25.0,
                    'excellent': 30.0,
                    'higher_is_better': True,
                },
                'ssim': {
                    'bad': 0.50,
                    'good': 0.70,
                    'excellent': 0.90,
                    'higher_is_better': True,
                },
                'lpips': {
                    'bad': 0.50,
                    'good': 0.30,
                    'excellent': 0.10,
                    'higher_is_better': False,
                },
                'mse': {
                    'bad': 0.10,
                    'good': 0.05,
                    'excellent': 0.01,
                    'higher_is_better': False,
                },
            },
            'events': {
                'polarity_accuracy_mean': {'good': 0.80, 'excellent': 0.90, 'higher_is_better': True},
                # Base metric name used in per-window series plots
                'polarity_accuracy': {'good': 0.80, 'excellent': 0.90, 'higher_is_better': True},
            },
        }

        # Metrics where shaded regions meaningfully communicate quality bands.
        # Only these will show good/bad/excellent shading to avoid misleading visuals.
        self.should_shade: Dict[str, set[str]] = {
            'frames': {
                'psnr_mean', 'ssim_mean', 'lpips_mean', 'mse_mean',
                'psnr', 'ssim', 'lpips', 'mse'
            },
            'events': {
                # Polarity accuracy is interpretable as a quality ratio.
                'polarity_accuracy_mean', 'polarity_accuracy'
            },
        }

    # ------------------------------------------------------------------
    # Styling helpers

    def apply_style(self, mode: str = 'print') -> None:
        """Switch global Matplotlib style presets.

        Parameters
        ----------
        mode:
            Either ``'print'`` (default) or ``'presentation'``. Passing any
            other value reverts to the base style only.
        """

        plt.style.use('default')
        _apply_style(_BASE_STYLE)
        if mode == 'presentation':
            _apply_style(_PRESENTATION_STYLE)
            self.active_style = 'presentation'
        elif mode == 'print':
            _apply_style(_PRINT_STYLE)
            self.active_style = 'print'
        else:
            self.active_style = 'custom'

    def iter_color_cycle(self, greyscale: bool = False) -> Iterable[str]:
        """Return an infinite color iterator for consistent categorical hues."""

        palette = self.greyscale_palette if greyscale else self.category_colors
        return itertools.cycle(palette)

    # ------------------------------------------------------------------
    # Metric helpers
    
    def get_metric_info(self, metric_name: str) -> tuple[str, str]:
        """Get display name and unit for a metric."""
        if metric_name in self.event_metrics_config:
            return self.event_metrics_config[metric_name]
        elif metric_name in self.frame_metrics_config:
            return self.frame_metrics_config[metric_name]
        else:
            # Fallback for unknown metrics
            clean_name = metric_name.replace('_mean', '').replace('_', ' ').title()
            return (clean_name, 'Value')
    
    def get_color_for_index(self, index: int) -> str:
        """Get color for a given index using the category colors."""
        return self.category_colors[index % len(self.category_colors)]
    
    def get_colors_for_count(self, count: int, *, greyscale: bool = False) -> List[str]:
        """Get a list of colors for a given count.

        When ``greyscale`` is True the palette is suitable for print in
        monochrome journals.
        """

        palette = self.greyscale_palette if greyscale else self.category_colors
        return [palette[i % len(palette)] for i in range(count)]
    
    def setup_clean_bar_plot(self, ax, x_data, y_data, colors=None, title="", xlabel="", ylabel="", 
                           add_value_labels=True, rotation=45):
        """Setup a clean bar plot with consistent styling."""
        if colors is None:
            colors = self.get_colors_for_count(len(x_data))
        
        bars = ax.bar(x_data, y_data, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on top of bars
        if add_value_labels:
            for bar, value in zip(bars, y_data):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold',
                       fontsize=9)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=rotation)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(y_data) * 1.15 if len(y_data) > 0 else 1)
        
        return bars

    def _infer_higher_is_better(self, metric_name: str) -> bool:
        """Infer if higher values mean better quality for a metric."""
        lower_better_tokens = ['lpips', 'mse', 'std', 'temporal_precision']
        metric_lower = metric_name.lower()
        return not any(tok in metric_lower for tok in lower_better_tokens)

    def compute_thresholds(self, metric_name: str, values: np.ndarray, scope: str) -> Dict[str, Optional[float]]:
        """Compute quality thresholds for a metric.

        Returns dict with keys: 'good', 'excellent', 'higher_is_better'.
        Uses static thresholds when available; otherwise falls back to data-driven quantiles.
        """
        # Guard: only compute thresholds for metrics that are designated for shading
        if scope in self.should_shade and metric_name not in self.should_shade[scope]:
            hib = self._infer_higher_is_better(metric_name)
            return {'good': None, 'excellent': None, 'higher_is_better': hib}

        # Use static thresholds if available
        if scope in self.static_thresholds and metric_name in self.static_thresholds[scope]:
            return self.static_thresholds[scope][metric_name]

        # Fallback: dataset-driven quantiles
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        higher_is_better = self._infer_higher_is_better(metric_name)
        if arr.size == 0:
            return {'good': None, 'excellent': None, 'higher_is_better': higher_is_better}

        if higher_is_better:
            good = float(np.quantile(arr, 0.75))
            excellent = float(np.quantile(arr, 0.90))
        else:
            good = float(np.quantile(arr, 0.25))
            excellent = float(np.quantile(arr, 0.10))

        return {'good': good, 'excellent': excellent, 'higher_is_better': higher_is_better}

    def apply_quality_shading(self, ax, y_values: np.ndarray, metric_name: str, scope: str):
        """Shade bad/good/excellent regions on the y-axis and draw threshold lines.

        Adjusts y-limits to fit data and thresholds, and adds a legend.
        """
        # Respect should_shade gating: if not eligible, do not add shading.
        if scope in self.should_shade and metric_name not in self.should_shade[scope]:
            # Still set reasonable y-limits from data without shading.
            valid = np.asarray(y_values, dtype=float)
            valid = valid[~np.isnan(valid)]
            if valid.size:
                data_min = float(np.min(valid))
                data_max = float(np.max(valid))
                margin = 0.05 * (data_max - data_min if data_max > data_min else 1.0)
                ax.set_ylim(data_min - margin, data_max + margin)
            return {'good': None, 'excellent': None, 'higher_is_better': self._infer_higher_is_better(metric_name)}

        thresholds = self.compute_thresholds(metric_name, np.asarray(y_values, dtype=float), scope)

        valid = np.asarray(y_values, dtype=float)
        valid = valid[~np.isnan(valid)]
        data_min = float(np.min(valid)) if valid.size else 0.0
        data_max = float(np.max(valid)) if valid.size else 1.0

        good_thr = thresholds.get('good')
        exc_thr = thresholds.get('excellent')
        bad_thr = thresholds.get('bad')
        hib = thresholds.get('higher_is_better', True)

        y_candidates_min = [data_min]
        y_candidates_max = [data_max]
        for candidate in (bad_thr, good_thr, exc_thr):
            if candidate is not None:
                y_candidates_min.append(float(candidate))
                y_candidates_max.append(float(candidate))

        y_min = min(y_candidates_min)
        y_max = max(y_candidates_max)
        margin = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
        y_min -= margin
        y_max += margin
        ax.set_ylim(y_min, y_max)

        # Region shading: bad/good/excellent bands
        if good_thr is not None and exc_thr is not None:
            if hib:
                if bad_thr is not None:
                    ax.axhspan(y_min, bad_thr, facecolor='#f8d7da', alpha=0.42, zorder=1)
                    ax.axhspan(bad_thr, good_thr, facecolor='#fff3cd', alpha=0.34, zorder=1)
                else:
                    ax.axhspan(y_min, good_thr, facecolor='#f8d7da', alpha=0.35, zorder=1)
                ax.axhspan(good_thr, exc_thr, facecolor='#cfe2ff', alpha=0.34, zorder=1)
                ax.axhspan(exc_thr, y_max, facecolor='#d1e7dd', alpha=0.35, zorder=1)
            else:
                ax.axhspan(y_min, exc_thr, facecolor='#d1e7dd', alpha=0.35, zorder=1)
                ax.axhspan(exc_thr, good_thr, facecolor='#cfe2ff', alpha=0.34, zorder=1)
                if bad_thr is not None:
                    ax.axhspan(good_thr, bad_thr, facecolor='#fff3cd', alpha=0.34, zorder=1)
                    ax.axhspan(bad_thr, y_max, facecolor='#f8d7da', alpha=0.42, zorder=1)
                else:
                    ax.axhspan(good_thr, y_max, facecolor='#f8d7da', alpha=0.35, zorder=1)

        # Threshold lines with ordered legend entries
        def _format_threshold_value(value: float | None) -> str:
            if value is None:
                return ''
            magnitude = abs(float(value))
            if magnitude >= 100 or float(value).is_integer():
                formatted = f"{float(value):.0f}"
            elif magnitude >= 10:
                formatted = f"{float(value):.1f}"
            elif magnitude >= 1:
                formatted = f"{float(value):.2f}"
            else:
                formatted = f"{float(value):.3f}"
            return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted

        threshold_lines: list[tuple] = []
        comparator = '≥' if hib else '≤'

        if good_thr is not None:
            good_label = f"Good {comparator} {_format_threshold_value(good_thr)}"
            good_line = ax.axhline(
                good_thr,
                color=self.colors['warning'],
                linestyle='--',
                linewidth=1.6,
                label=good_label,
                zorder=5,
            )
            threshold_lines.append((good_line, good_label, float(good_thr)))

        if exc_thr is not None:
            exc_label = f"Excellent {comparator} {_format_threshold_value(exc_thr)}"
            exc_line = ax.axhline(
                exc_thr,
                color=self.colors['success'],
                linestyle=':',
                linewidth=1.8,
                label=exc_label,
                zorder=6,
            )
            threshold_lines.append((exc_line, exc_label, float(exc_thr)))

        existing_handles, existing_labels = ax.get_legend_handles_labels()

        if threshold_lines or existing_handles:
            threshold_label_map = {label: (handle, pos) for handle, label, pos in threshold_lines}
            other_entries: list[tuple] = []
            for handle, label in zip(existing_handles, existing_labels):
                if label in threshold_label_map:
                    continue
                other_entries.append((handle, label))

            ordered_thresholds = sorted(threshold_lines, key=lambda item: item[2], reverse=True)
            final_handles = [h for h, _ in other_entries] + [item[0] for item in ordered_thresholds]
            final_labels = [lbl for _, lbl in other_entries] + [item[1] for item in ordered_thresholds]

            if final_handles:
                ax.legend(final_handles, final_labels, loc='upper right', frameon=False)

        return thresholds

    def scatter_with_quality_shading(self, ax, x_values, y_values, metric_name: str, scope: str,
                                     color: Optional[str] = None, s: int = 28):
        """Draw scatter points and overlay quality shading and threshold lines."""
        x_arr = np.asarray(x_values)
        y_arr = np.asarray(y_values, dtype=float)
        mask = ~np.isnan(y_arr)
        scatter_color = color if color is not None else self.colors['primary']
        ax.scatter(x_arr[mask], y_arr[mask], s=s, color=scatter_color, zorder=3)
        return self.apply_quality_shading(ax, y_arr[mask], metric_name, scope)
    
    def setup_clean_bar_plot_with_error(self, ax, x_data, y_data, error_data, colors=None, 
                                      title="", xlabel="", ylabel="", add_value_labels=True, rotation=90):
        """Setup a clean bar plot with error bars and consistent styling."""
        if colors is None:
            colors = self.get_colors_for_count(len(x_data))
        
        bars = ax.bar(x_data, y_data, yerr=error_data, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1)
        
        # Add value labels on top of bars (vertical text to avoid overlap)
        if add_value_labels:
            for bar, value in zip(bars, y_data):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(error_data)*0.15,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold',
                       rotation=90, fontsize=10)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=rotation)
        ax.grid(True, alpha=0.3)
        
        return bars
    
    def create_summary_text_plot(self, ax, summary_data: Dict, title: str = "Summary"):
        """Create a text-based summary plot."""
        summary_text = f"{title}\n\n"
        
        for key, value in summary_data.items():
            if isinstance(value, dict) and 'mean' in value:
                summary_text += f"{key}:\n"
                summary_text += f"  Mean: {value['mean']:.3f}\n"
                summary_text += f"  Count: {value.get('count', 'N/A')}\n\n"
            elif isinstance(value, (int, float)):
                summary_text += f"{key}: {value:.3f}\n"
            else:
                summary_text += f"{key}: {value}\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    def save_plot(self, fig, output_path, *, dpi: int = 300,
                  export_settings: Optional[ExportSettings] = None) -> List[Path]:
        """Save plot with consistent settings and multi-format export.

        Parameters
        ----------
        fig:
            Matplotlib figure to save.
        output_path:
            Target path. If a suffix is included it will be stripped so that
            every requested format receives its own file.
        dpi:
            Dots-per-inch for raster formats.
        export_settings:
            Optional override for :class:`ExportSettings`.
        """

        settings = export_settings or self.export_settings
        plt.tight_layout()

        output_path = Path(output_path)
        base_root = output_path.with_suffix('')
        saved_paths: List[Path] = []
        for fmt, target in settings.iter_paths(base_root):
            path = Path(target)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                path,
                dpi=dpi if fmt in {'png', 'jpg', 'jpeg', 'tiff'} else None,
                bbox_inches=settings.bbox_inches,
                transparent=settings.transparent,
            )
            saved_paths.append(path)

        plt.close(fig)
        return saved_paths
    
    def get_available_metrics(self, df, metric_configs: Dict) -> Dict:
        """Get available metrics from dataframe that match the configuration."""
        if df.empty:
            return {}
        
        return {k: v for k, v in metric_configs.items() if k in df.columns}


# Global configuration instance
plot_config = PlotConfig()