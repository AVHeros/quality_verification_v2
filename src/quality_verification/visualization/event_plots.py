#!/usr/bin/env python3
"""
Event-specific plotting module for DVS Quality Verification visualizations.

This module handles all event-related plotting functions including
overview plots, distributions, and event-specific comparisons.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .plot_config import plot_config

logger = logging.getLogger(__name__)


class EventPlotter:
    """Handles event-specific plotting functions."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = plot_config
    
    def plot_events_overview(self, events_df: pd.DataFrame):
        """Create comprehensive events metrics overview with clean design."""
        logger.info("Creating comprehensive events overview")
        
        logger.debug(f"Events DataFrame shape: {events_df.shape}")
        logger.debug(f"Events DataFrame columns: {list(events_df.columns)}")
        
        available_metrics = self.config.get_available_metrics(events_df, self.config.event_metrics_config)
        
        logger.debug(f"Available metrics: {available_metrics}")
        
        if not available_metrics:
            logger.warning("No key event metrics available for overview")
            return
        
        # Create a larger layout to accommodate more metrics
        fig, axes = plt.subplots(2, 3, figsize=self.config.figure_sizes['overview'])
        fig.suptitle('Comprehensive Event Metrics Overview', fontsize=16, fontweight='bold')
        
        # Plot 1: Distribution of key metrics
        ax = axes[0, 0]
        metric_data = []
        metric_names = []
        try:
            for metric, (name, _) in available_metrics.items():
                logger.debug(f"Processing metric: {metric}")
                data = events_df[metric].dropna()
                logger.debug(f"Data for {metric}: {data}")
                if len(data) > 0:
                    metric_data.append(data)
                    metric_names.append(name)
            
            logger.debug(f"Metric data length: {len(metric_data)}")
            if len(metric_data) > 0:
                means = [np.mean(data) for data in metric_data]
                logger.debug(f"Means: {means}")
                colors = self.config.get_colors_for_count(len(means))
                
                self.config.setup_clean_bar_plot(
                    ax, metric_names, means, colors,
                    title='Event Metrics Distribution (Mean Values)',
                    xlabel='', ylabel='Value'
                )
        except Exception as e:
            logger.error(f"Error in Plot 1: {e}")
            raise
        
        # Plot 2: Weather comparison (Event Density)
        ax = axes[0, 1]
        try:
            logger.debug("Starting Plot 2")
            logger.debug(f"'event_density_mean' in available_metrics: {'event_density_mean' in available_metrics}")
            logger.debug(f"'weather' in events_df.columns: {'weather' in events_df.columns}")
            
            if ('event_density_mean' in available_metrics) and ('weather' in events_df.columns):
                logger.debug("Conditions met for Plot 2")
                weather_data = events_df.groupby('weather')['event_density_mean'].agg(['mean']).reset_index()
                logger.debug(f"Weather data: {weather_data}")
                colors = self.config.get_colors_for_count(len(weather_data))
                
                self.config.setup_clean_bar_plot(
                    ax, weather_data['weather'], weather_data['mean'], colors,
                    title='Event Density by Weather',
                    xlabel='Weather Condition', ylabel='Event Density'
                )
            else:
                logger.debug("Conditions not met for Plot 2")
        except Exception as e:
            logger.error(f"Error in Plot 2: {e}")
            raise
        
        # Plot 3: Route type comparison (Event Rate)
        ax = axes[1, 0]
        if ('event_rate_mean' in available_metrics) and ('route_type' in events_df.columns):
            route_data = events_df.groupby('route_type')['event_rate_mean'].agg(['mean']).reset_index()
            
            self.config.setup_clean_bar_plot(
                ax, route_data['route_type'], route_data['mean'], 
                [self.config.colors['primary']] * len(route_data),
                title='Event Rate by Route Type',
                xlabel='Route Type', ylabel='Event Rate'
            )
        
        # Plot 4: Polarity Accuracy by Weather
        ax = axes[0, 2]
        if ('polarity_accuracy_mean' in available_metrics) and ('weather' in events_df.columns):
            weather_data = events_df.groupby('weather')['polarity_accuracy_mean'].agg(['mean']).reset_index()
            colors = self.config.get_colors_for_count(len(weather_data))
            
            self.config.setup_clean_bar_plot(
                ax, weather_data['weather'], weather_data['mean'], colors,
                title='Polarity Accuracy by Weather',
                xlabel='Weather Condition', ylabel='Polarity Accuracy'
            )
        
        # Plot 5: Temporal Precision by Route Type
        ax = axes[1, 1]
        if ('temporal_precision_us_std' in available_metrics) and ('route_type' in events_df.columns):
            route_data = events_df.groupby('route_type')['temporal_precision_us_std'].agg(['mean']).reset_index()
            
            self.config.setup_clean_bar_plot(
                ax, route_data['route_type'], route_data['mean'],
                [self.config.colors['secondary']] * len(route_data),
                title='Temporal Precision by Route Type',
                xlabel='Route Type', ylabel='Temporal Precision (μs std)'
            )
        
        # Plot 6: Summary statistics
        ax = axes[1, 2]
        summary_data = {}
        for metric, (name, unit) in available_metrics.items():
            data = events_df[metric].dropna()
            if len(data) > 0:
                summary_data[name] = {
                    'mean': data.mean(),
                    'count': len(data)
                }
        
        self.config.create_summary_text_plot(ax, summary_data, "Event Metrics Summary")
        
        # Save plot
        events_overview_path = self.output_dir / 'events_metrics_overview.png'
        self.config.save_plot(fig, events_overview_path)
        
        logger.info(f"Events overview saved to {events_overview_path}")
    
    def plot_events_route_comparison(self, events_df: pd.DataFrame):
        """Create focused events route type comparison."""
        logger.info("Creating events route type comparison")
        
        # Focus on key event metrics
        key_metrics = {
            'event_density_mean': ('Event Density', 'Events per pixel'),
            'polarity_accuracy_mean': ('Polarity Accuracy', 'Accuracy ratio')
        }
        
        available_metrics = {k: v for k, v in key_metrics.items() if k in events_df.columns}
        
        if not available_metrics:
            logger.warning("No key event metrics available for route comparison")
            return
        
        # Create clean subplot layout
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle('Event Metrics by Route Type', fontsize=16, fontweight='bold')
        
        for i, (metric, (name, unit)) in enumerate(available_metrics.items()):
            ax = axes[i]
            
            # Create bar chart with error bars
            route_data = events_df.groupby('route_type')[metric].agg(['mean', 'std']).reset_index()
            colors = self.config.get_colors_for_count(len(route_data))
            
            self.config.setup_clean_bar_plot_with_error(
                ax, route_data['route_type'], route_data['mean'], route_data['std'],
                colors, title=f'{name} by Route Type',
                xlabel='Route Type', ylabel=f'{name} ({unit})'
            )
            
            # Add sample count annotations
            route_counts = events_df['route_type'].value_counts()
            for j, route in enumerate(events_df['route_type'].unique()):
                count = route_counts[route]
                ax.text(j, ax.get_ylim()[1] * 0.95, f'n={count}', 
                       ha='center', va='top', fontsize=10, fontweight='bold')
        
        # Save plot
        events_route_path = self.output_dir / 'events_route_type_comparison.png'
        self.config.save_plot(fig, events_route_path)
        
        logger.info(f"Events route comparison saved to {events_route_path}")
    
    def plot_event_metric_distribution(self, events_df: pd.DataFrame, metric_name: str):
        """Plot distribution of a specific event metric."""
        if metric_name not in events_df.columns:
            logger.warning(f"Metric {metric_name} not found in events data")
            return
        
        data = events_df[metric_name].dropna()
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
        metric_dist_path = self.output_dir / f'event_{metric_name}_distribution.png'
        self.config.save_plot(fig, metric_dist_path)
        
        logger.info(f"Event metric distribution saved to {metric_dist_path}")
    
    def plot_events_weather_detailed(self, events_df: pd.DataFrame):
        """Create detailed weather comparison for events."""
        if 'weather' not in events_df.columns:
            logger.warning("No weather column found for detailed weather analysis")
            return
        
        available_metrics = self.config.get_available_metrics(events_df, self.config.event_metrics_config)
        
        if not available_metrics:
            logger.warning("No event metrics available for weather analysis")
            return
        
        # Select top 4 metrics for detailed analysis
        key_metrics = ['event_density_mean', 'event_rate_mean', 'polarity_accuracy_mean', 'temporal_precision_us_std']
        selected_metrics = {k: v for k, v in available_metrics.items() if k in key_metrics}
        
        if not selected_metrics:
            # Fallback to first 4 available metrics
            selected_metrics = dict(list(available_metrics.items())[:4])
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_sizes['comparison'])
        fig.suptitle('Event Metrics by Weather Conditions (Detailed)', fontsize=16, fontweight='bold')
        
        axes_flat = axes.flatten()
        
        for i, (metric, (name, unit)) in enumerate(selected_metrics.items()):
            if i >= 4:  # Only plot first 4 metrics
                break
                
            ax = axes_flat[i]
            weather_data = events_df.groupby('weather')[metric].agg(['mean', 'std']).reset_index()
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
        weather_detailed_path = self.output_dir / 'events_weather_detailed.png'
        self.config.save_plot(fig, weather_detailed_path)
        
        logger.info(f"Detailed events weather analysis saved to {weather_detailed_path}")