#!/usr/bin/env python3
"""
Frame-specific plotting module for DVS Quality Verification visualizations.

This module handles all frame-related plotting functions including
overview plots, distributions, and frame-specific comparisons.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .plot_config import plot_config

logger = logging.getLogger(__name__)


class FramePlotter:
    """Handles frame-specific plotting functions."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = plot_config
    
    def plot_frames_overview(self, frames_df: pd.DataFrame):
        """Create comprehensive frames metrics overview with clean design."""
        logger.info("Creating comprehensive frames overview")
        
        available_metrics = self.config.get_available_metrics(frames_df, self.config.frame_metrics_config)
        
        if not available_metrics:
            logger.warning("No key frame metrics available for overview")
            return
        
        # Create a larger layout to accommodate more metrics
        fig, axes = plt.subplots(2, 3, figsize=self.config.figure_sizes['overview'])
        fig.suptitle('Comprehensive Frame Metrics Overview', fontsize=16, fontweight='bold')
        
        # Plot 1: Distribution of key metrics
        ax = axes[0, 0]
        metric_data = []
        metric_names = []
        for metric, (name, _) in available_metrics.items():
            data = frames_df[metric].dropna()
            if len(data) > 0:
                metric_data.append(data)
                metric_names.append(name)
        
        if metric_data and len(metric_data) > 0:
            means = [np.mean(data) for data in metric_data]
            colors = self.config.get_colors_for_count(len(means))
            
            self.config.setup_clean_bar_plot(
                ax, metric_names, means, colors,
                title='Frame Metrics Distribution (Mean Values)',
                xlabel='', ylabel='Value'
            )
        
        # Plot 2: Weather comparison (SSIM)
        ax = axes[0, 1]
        if 'ssim_mean' in available_metrics and 'weather' in frames_df.columns:
            weather_data = frames_df.groupby('weather')['ssim_mean'].agg(['mean']).reset_index()
            colors = self.config.get_colors_for_count(len(weather_data))
            
            self.config.setup_clean_bar_plot(
                ax, weather_data['weather'], weather_data['mean'], colors,
                title='SSIM by Weather',
                xlabel='Weather Condition', ylabel='SSIM'
            )
        
        # Plot 3: Route type comparison (PSNR)
        ax = axes[0, 2]
        if 'psnr_mean' in available_metrics and 'route_type' in frames_df.columns:
            route_data = frames_df.groupby('route_type')['psnr_mean'].agg(['mean']).reset_index()
            
            self.config.setup_clean_bar_plot(
                ax, route_data['route_type'], route_data['mean'],
                [self.config.colors['primary']] * len(route_data),
                title='PSNR by Route Type',
                xlabel='Route Type', ylabel='PSNR (dB)'
            )
        
        # Plot 4: LPIPS by Weather
        ax = axes[1, 0]
        if 'lpips_mean' in available_metrics and 'weather' in frames_df.columns:
            weather_data = frames_df.groupby('weather')['lpips_mean'].agg(['mean']).reset_index()
            colors = self.config.get_colors_for_count(len(weather_data))
            
            self.config.setup_clean_bar_plot(
                ax, weather_data['weather'], weather_data['mean'], colors,
                title='LPIPS by Weather',
                xlabel='Weather Condition', ylabel='LPIPS'
            )
        
        # Plot 5: MSE by Route Type
        ax = axes[1, 1]
        if 'mse_mean' in available_metrics and 'route_type' in frames_df.columns:
            route_data = frames_df.groupby('route_type')['mse_mean'].agg(['mean']).reset_index()
            
            self.config.setup_clean_bar_plot(
                ax, route_data['route_type'], route_data['mean'],
                [self.config.colors['secondary']] * len(route_data),
                title='MSE by Route Type',
                xlabel='Route Type', ylabel='MSE'
            )
        
        # Plot 6: Summary statistics
        ax = axes[1, 2]
        summary_data = {}
        for metric, (name, unit) in available_metrics.items():
            data = frames_df[metric].dropna()
            if len(data) > 0:
                summary_data[name] = {
                    'mean': data.mean(),
                    'count': len(data)
                }
        
        self.config.create_summary_text_plot(ax, summary_data, "Frame Metrics Summary")
        
        # Save plot
        frames_overview_path = self.output_dir / 'frames_metrics_overview.png'
        self.config.save_plot(fig, frames_overview_path)
        
        logger.info(f"Frames overview saved to {frames_overview_path}")
    
    def plot_frames_route_comparison(self, frames_df: pd.DataFrame):
        """Create focused frames route type comparison."""
        logger.info("Creating frames route type comparison")
        
        # Focus on key frame metrics
        key_metrics = {
            'ssim_mean': ('SSIM', 'Similarity index'),
            'psnr_mean': ('PSNR', 'dB'),
            'lpips_mean': ('LPIPS', 'Perceptual distance')
        }
        
        available_metrics = {k: v for k, v in key_metrics.items() if k in frames_df.columns}
        
        if not available_metrics:
            logger.warning("No key frame metrics available for route comparison")
            return
        
        # Create clean subplot layout
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle('Frame Metrics by Route Type', fontsize=16, fontweight='bold')
        
        for i, (metric, (name, unit)) in enumerate(available_metrics.items()):
            ax = axes[i]
            
            # Create bar chart with error bars
            route_data = frames_df.groupby('route_type')[metric].agg(['mean', 'std']).reset_index()
            colors = self.config.get_colors_for_count(len(route_data))
            
            self.config.setup_clean_bar_plot_with_error(
                ax, route_data['route_type'], route_data['mean'], route_data['std'],
                colors, title=f'{name} by Route Type',
                xlabel='Route Type', ylabel=f'{name} ({unit})'
            )
            
            # Add sample count annotations
            route_counts = frames_df['route_type'].value_counts()
            for j, route in enumerate(frames_df['route_type'].unique()):
                count = route_counts[route]
                ax.text(j, ax.get_ylim()[1] * 0.95, f'n={count}', 
                       ha='center', va='top', fontsize=10, fontweight='bold')
        
        # Save plot
        frames_route_path = self.output_dir / 'frames_route_type_comparison.png'
        self.config.save_plot(fig, frames_route_path)
        
        logger.info(f"Frames route comparison saved to {frames_route_path}")
    
    def plot_frame_metric_distribution(self, frames_df: pd.DataFrame, metric_name: str):
        """Plot distribution of a specific frame metric."""
        if metric_name not in frames_df.columns:
            logger.warning(f"Metric {metric_name} not found in frames data")
            return
        
        data = frames_df[metric_name].dropna()
        if len(data) == 0:
            logger.warning(f"No valid data for metric {metric_name}")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histogram
        ax1.hist(data, bins=20, alpha=0.7, color=self.config.colors['primary'], edgecolor='black')
        ax1.set_title(f'{metric_name} Distribution')
        ax1.set_xlabel(metric_name)
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(data, patch_artist=True, 
                   boxprops=dict(facecolor=self.config.colors['primary'], alpha=0.7))
        ax2.set_title(f'{metric_name} Box Plot')
        ax2.set_ylabel(metric_name)
        ax2.grid(True, alpha=0.3)
        
        # Save plot
        metric_dist_path = self.output_dir / f'frame_{metric_name}_distribution.png'
        self.config.save_plot(fig, metric_dist_path)
        
        logger.info(f"Frame metric distribution saved to {metric_dist_path}")
    
    def plot_frames_weather_detailed(self, frames_df: pd.DataFrame):
        """Create detailed weather comparison for frames."""
        if 'weather' not in frames_df.columns:
            logger.warning("No weather column found for detailed weather analysis")
            return
        
        available_metrics = self.config.get_available_metrics(frames_df, self.config.frame_metrics_config)
        
        if not available_metrics:
            logger.warning("No frame metrics available for weather analysis")
            return
        
        # Select top 4 metrics for detailed analysis
        key_metrics = ['ssim_mean', 'psnr_mean', 'lpips_mean', 'mse_mean']
        selected_metrics = {k: v for k, v in available_metrics.items() if k in key_metrics}
        
        if not selected_metrics:
            # Fallback to first 4 available metrics
            selected_metrics = dict(list(available_metrics.items())[:4])
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_sizes['comparison'])
        fig.suptitle('Frame Metrics by Weather Conditions (Detailed)', fontsize=16, fontweight='bold')
        
        axes_flat = axes.flatten()
        
        for i, (metric, (name, unit)) in enumerate(selected_metrics.items()):
            if i >= 4:  # Only plot first 4 metrics
                break
                
            ax = axes_flat[i]
            weather_data = frames_df.groupby('weather')[metric].agg(['mean', 'std']).reset_index()
            colors = self.config.get_colors_for_count(len(weather_data))
            
            self.config.setup_clean_bar_plot_with_error(
                ax, weather_data['weather'], weather_data['mean'], weather_data['std'],
                colors, title=f'{name} by Weather',
                xlabel='Weather Condition', ylabel=f'{name} ({unit})'
            )
        
        # Hide unused subplots
        for i in range(len(selected_metrics), 4):
            axes_flat[i].axis('off')
        
        # Save plot
        weather_detailed_path = self.output_dir / 'frames_weather_detailed.png'
        self.config.save_plot(fig, weather_detailed_path)
        
        logger.info(f"Detailed frames weather analysis saved to {weather_detailed_path}")
    
    def plot_frame_quality_heatmap(self, frames_df: pd.DataFrame):
        """Create a heatmap showing frame quality metrics correlation."""
        quality_metrics = ['ssim_mean', 'psnr_mean', 'lpips_mean', 'mse_mean']
        available_metrics = [m for m in quality_metrics if m in frames_df.columns]
        
        if len(available_metrics) < 2:
            logger.warning("Not enough frame quality metrics for correlation heatmap")
            return
        
        # Calculate correlation matrix
        corr_data = frames_df[available_metrics].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        im = ax.imshow(corr_data.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(available_metrics)))
        ax.set_yticks(range(len(available_metrics)))
        ax.set_xticklabels([self.config.frame_metrics_config.get(m, (m, ''))[0] 
                           for m in available_metrics], rotation=45, ha='right')
        ax.set_yticklabels([self.config.frame_metrics_config.get(m, (m, ''))[0] 
                           for m in available_metrics])
        
        # Add correlation values as text
        for i in range(len(available_metrics)):
            for j in range(len(available_metrics)):
                text = ax.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
        
        ax.set_title('Frame Quality Metrics Correlation', fontsize=14, fontweight='bold')
        
        # Save plot
        heatmap_path = self.output_dir / 'frame_quality_correlation_heatmap.png'
        self.config.save_plot(fig, heatmap_path)
        
        logger.info(f"Frame quality correlation heatmap saved to {heatmap_path}")
    
    def plot_frame_quality_scatter(self, frames_df: pd.DataFrame, metric_x: str, metric_y: str):
        """Create scatter plot between two frame quality metrics."""
        if metric_x not in frames_df.columns or metric_y not in frames_df.columns:
            logger.warning(f"Metrics {metric_x} or {metric_y} not found in frames data")
            return
        
        # Remove NaN values
        clean_data = frames_df[[metric_x, metric_y]].dropna()
        if len(clean_data) == 0:
            logger.warning(f"No valid data for scatter plot between {metric_x} and {metric_y}")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create scatter plot
        scatter = ax.scatter(clean_data[metric_x], clean_data[metric_y], 
                           alpha=0.6, color=self.config.colors['primary'])
        
        # Trend line removed to keep plot purely scatter-based
        
        # Calculate correlation
        correlation = clean_data[metric_x].corr(clean_data[metric_y])
        
        # Labels and title
        x_name = self.config.frame_metrics_config.get(metric_x, (metric_x, ''))[0]
        y_name = self.config.frame_metrics_config.get(metric_y, (metric_y, ''))[0]
        
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(f'{x_name} vs {y_name}\n(Correlation: {correlation:.3f})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Save plot
        scatter_path = self.output_dir / f'frame_{metric_x}_vs_{metric_y}_scatter.png'
        self.config.save_plot(fig, scatter_path)
        
        logger.info(f"Frame quality scatter plot saved to {scatter_path}")