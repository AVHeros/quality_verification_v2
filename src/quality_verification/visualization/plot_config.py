#!/usr/bin/env python3
"""
Plot configuration module for DVS Quality Verification visualizations.

This module defines plotting styles, colors, and common configuration
settings used across all visualization modules.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional

# Set matplotlib style for better, cleaner plots
plt.style.use('default')
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True
})

# Define clear color palette for consistent plotting
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'info': '#5D737E',
    'light': '#F5F5F5',
    'dark': '#2C3E50'
}

# Color palette for multiple categories
CATEGORY_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#5D737E', '#8E44AD', '#27AE60', '#E67E22']


class PlotConfig:
    """Configuration class for DVS visualization plots."""
    
    def __init__(self):
        self.colors = COLORS
        self.category_colors = CATEGORY_COLORS
        
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
                'psnr_mean': {'good': 30.0, 'excellent': 40.0, 'higher_is_better': True},
                'ssim_mean': {'good': 0.92, 'excellent': 0.95, 'higher_is_better': True},
                'lpips_mean': {'good': 0.10, 'excellent': 0.05, 'higher_is_better': False},
                'mse_mean': {'good': 0.01, 'excellent': 0.005, 'higher_is_better': False},
                # Base metric names used in per-pair/per-frame series plots
                'psnr': {'good': 30.0, 'excellent': 40.0, 'higher_is_better': True},
                'ssim': {'good': 0.92, 'excellent': 0.95, 'higher_is_better': True},
                'lpips': {'good': 0.10, 'excellent': 0.05, 'higher_is_better': False},
                'mse': {'good': 0.01, 'excellent': 0.005, 'higher_is_better': False},
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
    
    def get_colors_for_count(self, count: int) -> List[str]:
        """Get a list of colors for a given count."""
        return [self.get_color_for_index(i) for i in range(count)]
    
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

        y_candidates_min = [data_min]
        y_candidates_max = [data_max]
        if thresholds.get('good') is not None:
            y_candidates_min.append(float(thresholds['good']))
            y_candidates_max.append(float(thresholds['good']))
        if thresholds.get('excellent') is not None:
            y_candidates_min.append(float(thresholds['excellent']))
            y_candidates_max.append(float(thresholds['excellent']))

        y_min = min(y_candidates_min)
        y_max = max(y_candidates_max)
        margin = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
        y_min -= margin
        y_max += margin
        ax.set_ylim(y_min, y_max)

        # Region shading: bad/good/excellent
        good_thr = thresholds.get('good')
        exc_thr = thresholds.get('excellent')
        hib = thresholds.get('higher_is_better', True)
        if good_thr is not None and exc_thr is not None:
            if hib:
                ax.axhspan(y_min, good_thr, facecolor='#f8d7da', alpha=0.35, zorder=1)  # bad
                ax.axhspan(good_thr, exc_thr, facecolor='#cfe2ff', alpha=0.35, zorder=1)  # good
                ax.axhspan(exc_thr, y_max, facecolor='#d1e7dd', alpha=0.35, zorder=1)  # excellent
            else:
                ax.axhspan(y_min, exc_thr, facecolor='#d1e7dd', alpha=0.35, zorder=1)  # excellent
                ax.axhspan(exc_thr, good_thr, facecolor='#cfe2ff', alpha=0.35, zorder=1)  # good
                ax.axhspan(good_thr, y_max, facecolor='#f8d7da', alpha=0.35, zorder=1)  # bad

        # Threshold lines
        if good_thr is not None:
            ax.axhline(good_thr, color='orange', linestyle='--', linewidth=1.5, label='Good', zorder=4)
        if exc_thr is not None:
            ax.axhline(exc_thr, color='green', linestyle=':', linewidth=1.8, label='Excellent', zorder=4)
        ax.legend(loc='upper right')

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
    
    def save_plot(self, fig, output_path, dpi=300):
        """Save plot with consistent settings."""
        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    def get_available_metrics(self, df, metric_configs: Dict) -> Dict:
        """Get available metrics from dataframe that match the configuration."""
        if df.empty:
            return {}
        
        return {k: v for k, v in metric_configs.items() if k in df.columns}


# Global configuration instance
plot_config = PlotConfig()