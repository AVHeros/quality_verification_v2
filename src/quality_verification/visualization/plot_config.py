#!/usr/bin/env python3
"""
Plot configuration module for DVS Quality Verification visualizations.

This module defines plotting styles, colors, and common configuration
settings used across all visualization modules.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

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