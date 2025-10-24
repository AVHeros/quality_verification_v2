#!/usr/bin/env python3
"""
Overview plotting module for DVS Quality Verification visualizations.

This module handles high-level overview plots that combine multiple
metrics and provide comprehensive analysis summaries.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .plot_config import plot_config

logger = logging.getLogger(__name__)


class OverviewPlotter:
    """Handles overview and summary plotting functions."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = plot_config
    
    def plot_comprehensive_overview(self, events_df: pd.DataFrame, frames_df: pd.DataFrame):
        """Create a comprehensive overview combining event and frame metrics."""
        logger.info("Creating comprehensive overview plot")
        
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=self.config.figure_sizes['large'])
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('DVS Quality Verification - Comprehensive Analysis Overview', 
                    fontsize=18, fontweight='bold')
        
        # Top row: Data overview
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])
        
        # Middle row: Event metrics
        ax5 = fig.add_subplot(gs[1, :2])
        ax6 = fig.add_subplot(gs[1, 2:])
        
        # Bottom row: Frame metrics
        ax7 = fig.add_subplot(gs[2, :2])
        ax8 = fig.add_subplot(gs[2, 2:])
        
        # Plot 1: Data availability
        self._plot_data_availability(ax1, events_df, frames_df)
        
        # Plot 2: Weather distribution
        self._plot_weather_distribution(ax2, events_df)
        
        # Plot 3: Route type distribution
        self._plot_route_distribution(ax3, events_df)
        
        # Plot 4: Quality score summary
        self._plot_quality_scores(ax4, events_df, frames_df)
        
        # Plot 5: Event metrics overview
        self._plot_event_metrics_overview(ax5, events_df)
        
        # Plot 6: Event performance by conditions
        self._plot_event_performance_conditions(ax6, events_df)
        
        # Plot 7: Frame metrics overview
        self._plot_frame_metrics_overview(ax7, frames_df)
        
        # Plot 8: Frame performance by conditions
        self._plot_frame_performance_conditions(ax8, frames_df)
        
        # Save plot
        overview_path = self.output_dir / 'comprehensive_overview.png'
        self.config.save_plot(fig, overview_path)
        
        logger.info(f"Comprehensive overview saved to {overview_path}")
    
    def _plot_data_availability(self, ax, events_df: pd.DataFrame, frames_df: pd.DataFrame):
        """Plot data availability summary."""
        ax.set_title('Data Availability', fontsize=12, fontweight='bold')
        
        categories = ['Events', 'Frames']
        counts = [len(events_df), len(frames_df)]
        colors = [self.config.colors['primary'], self.config.colors['secondary']]
        
        bars = ax.bar(categories, counts, color=colors, alpha=0.8)
        ax.set_ylabel('Sequences')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_weather_distribution(self, ax, events_df: pd.DataFrame):
        """Plot weather conditions distribution."""
        ax.set_title('Weather Conditions', fontsize=12, fontweight='bold')
        
        if 'weather' in events_df.columns and not events_df.empty:
            weather_counts = events_df['weather'].value_counts()
            colors = self.config.get_colors_for_count(len(weather_counts))
            
            wedges, texts, autotexts = ax.pie(weather_counts.values, 
                                            labels=weather_counts.index,
                                            colors=colors, autopct='%1.0f%%', 
                                            startangle=90)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(10)
        else:
            ax.text(0.5, 0.5, 'Weather data\nnot available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=11)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
    
    def _plot_route_distribution(self, ax, events_df: pd.DataFrame):
        """Plot route type distribution."""
        ax.set_title('Route Types', fontsize=12, fontweight='bold')
        
        if 'route_type' in events_df.columns and not events_df.empty:
            route_counts = events_df['route_type'].value_counts()
            colors = self.config.get_colors_for_count(len(route_counts))
            
            bars = ax.bar(range(len(route_counts)), route_counts.values, 
                         color=colors, alpha=0.8)
            ax.set_xticks(range(len(route_counts)))
            ax.set_xticklabels(route_counts.index, rotation=45, ha='right')
            ax.set_ylabel('Sequences')
            
            # Add value labels
            for bar, count in zip(bars, route_counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'Route type data\nnot available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=11)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_quality_scores(self, ax, events_df: pd.DataFrame, frames_df: pd.DataFrame):
        """Plot overall quality scores."""
        ax.set_title('Quality Scores', fontsize=12, fontweight='bold')
        
        scores = {}
        
        # Calculate event quality score
        if not events_df.empty and 'polarity_accuracy_mean' in events_df.columns:
            event_score = events_df['polarity_accuracy_mean'].mean() * 100
            scores['Event\nQuality'] = event_score
        
        # Calculate frame quality score (using SSIM)
        if not frames_df.empty and 'ssim_mean' in frames_df.columns:
            frame_score = frames_df['ssim_mean'].mean() * 100
            scores['Frame\nQuality'] = frame_score
        
        # Calculate overall score
        if scores:
            overall_score = np.mean(list(scores.values()))
            scores['Overall\nQuality'] = overall_score
        
        if scores:
            categories = list(scores.keys())
            values = list(scores.values())
            colors = self.config.get_colors_for_count(len(categories))
            
            bars = ax.bar(categories, values, color=colors, alpha=0.8)
            ax.set_ylabel('Quality Score (%)')
            ax.set_ylim(0, 100)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'Quality scores\nnot available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=11)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_event_metrics_overview(self, ax, events_df: pd.DataFrame):
        """Plot event metrics overview."""
        ax.set_title('Event Metrics Overview', fontsize=12, fontweight='bold')
        
        if events_df.empty:
            ax.text(0.5, 0.5, 'No event data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=11)
            return
        
        # Select key event metrics
        key_metrics = {
            'event_density_mean': 'Event Density',
            'event_rate_mean': 'Event Rate',
            'polarity_accuracy_mean': 'Polarity Accuracy'
        }
        
        available_metrics = {k: v for k, v in key_metrics.items() if k in events_df.columns}
        
        if available_metrics:
            metrics = list(available_metrics.keys())
            names = list(available_metrics.values())
            means = [events_df[metric].mean() for metric in metrics]
            stds = [events_df[metric].std() for metric in metrics]
            
            colors = self.config.get_colors_for_count(len(metrics))
            
            bars = ax.bar(names, means, yerr=stds, color=colors, alpha=0.8, 
                         capsize=5, error_kw={'linewidth': 2})
            
            ax.set_ylabel('Metric Value')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{mean:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No key event metrics\navailable', ha='center', va='center',
                   transform=ax.transAxes, fontsize=11)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_event_performance_conditions(self, ax, events_df: pd.DataFrame):
        """Plot event performance by conditions."""
        ax.set_title('Event Performance by Conditions', fontsize=12, fontweight='bold')
        
        if events_df.empty or 'weather' not in events_df.columns or 'polarity_accuracy_mean' not in events_df.columns:
            ax.text(0.5, 0.5, 'Insufficient data for\ncondition analysis', ha='center', va='center',
                   transform=ax.transAxes, fontsize=11)
            return
        
        # Group by weather and calculate mean polarity accuracy
        weather_performance = events_df.groupby('weather')['polarity_accuracy_mean'].mean()
        
        colors = self.config.get_colors_for_count(len(weather_performance))
        bars = ax.bar(weather_performance.index, weather_performance.values, 
                     color=colors, alpha=0.8)
        
        ax.set_ylabel('Polarity Accuracy')
        ax.set_xlabel('Weather Condition')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, weather_performance.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_frame_metrics_overview(self, ax, frames_df: pd.DataFrame):
        """Plot frame metrics overview."""
        ax.set_title('Frame Metrics Overview', fontsize=12, fontweight='bold')
        
        if frames_df.empty:
            ax.text(0.5, 0.5, 'No frame data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=11)
            return
        
        # Select key frame metrics
        key_metrics = {
            'ssim_mean': 'SSIM',
            'psnr_mean': 'PSNR',
            'lpips_mean': 'LPIPS'
        }
        
        available_metrics = {k: v for k, v in key_metrics.items() if k in frames_df.columns}
        
        if available_metrics:
            metrics = list(available_metrics.keys())
            names = list(available_metrics.values())
            means = [frames_df[metric].mean() for metric in metrics]
            stds = [frames_df[metric].std() for metric in metrics]
            
            colors = self.config.get_colors_for_count(len(metrics))
            
            bars = ax.bar(names, means, yerr=stds, color=colors, alpha=0.8, 
                         capsize=5, error_kw={'linewidth': 2})
            
            ax.set_ylabel('Metric Value')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{mean:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No key frame metrics\navailable', ha='center', va='center',
                   transform=ax.transAxes, fontsize=11)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_frame_performance_conditions(self, ax, frames_df: pd.DataFrame):
        """Plot frame performance by conditions."""
        ax.set_title('Frame Performance by Conditions', fontsize=12, fontweight='bold')
        
        if frames_df.empty or 'weather' not in frames_df.columns or 'ssim_mean' not in frames_df.columns:
            ax.text(0.5, 0.5, 'Insufficient data for\ncondition analysis', ha='center', va='center',
                   transform=ax.transAxes, fontsize=11)
            return
        
        # Group by weather and calculate mean SSIM
        weather_performance = frames_df.groupby('weather')['ssim_mean'].mean()
        
        colors = self.config.get_colors_for_count(len(weather_performance))
        bars = ax.bar(weather_performance.index, weather_performance.values, 
                     color=colors, alpha=0.8)
        
        ax.set_ylabel('SSIM')
        ax.set_xlabel('Weather Condition')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, weather_performance.values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='y')
    
    def plot_metrics_dashboard(self, events_df: pd.DataFrame, frames_df: pd.DataFrame):
        """Create a dashboard-style metrics overview."""
        logger.info("Creating metrics dashboard")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DVS Quality Verification - Metrics Dashboard', 
                    fontsize=20, fontweight='bold')
        
        # Dashboard panels
        self._create_dashboard_panel_1(axes[0, 0], events_df, frames_df)
        self._create_dashboard_panel_2(axes[0, 1], events_df)
        self._create_dashboard_panel_3(axes[0, 2], frames_df)
        self._create_dashboard_panel_4(axes[1, 0], events_df)
        self._create_dashboard_panel_5(axes[1, 1], frames_df)
        self._create_dashboard_panel_6(axes[1, 2], events_df, frames_df)
        
        # Save plot
        dashboard_path = self.output_dir / 'metrics_dashboard.png'
        self.config.save_plot(fig, dashboard_path)
        
        logger.info(f"Metrics dashboard saved to {dashboard_path}")
    
    def _create_dashboard_panel_1(self, ax, events_df: pd.DataFrame, frames_df: pd.DataFrame):
        """Dashboard panel 1: Data summary."""
        ax.set_title('Data Summary', fontsize=14, fontweight='bold')
        
        summary_text = []
        summary_text.append(f"Event Sequences: {len(events_df)}")
        summary_text.append(f"Frame Sequences: {len(frames_df)}")
        
        if 'weather' in events_df.columns:
            unique_weather = events_df['weather'].nunique()
            summary_text.append(f"Weather Conditions: {unique_weather}")
        
        if 'route_type' in events_df.columns:
            unique_routes = events_df['route_type'].nunique()
            summary_text.append(f"Route Types: {unique_routes}")
        
        ax.text(0.05, 0.95, '\n'.join(summary_text), transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontfamily='monospace')
        ax.axis('off')
    
    def _create_dashboard_panel_2(self, ax, events_df: pd.DataFrame):
        """Dashboard panel 2: Event metrics summary."""
        ax.set_title('Event Metrics', fontsize=14, fontweight='bold')
        
        if events_df.empty:
            ax.text(0.5, 0.5, 'No event data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            return
        
        metrics_text = []
        if 'event_density_mean' in events_df.columns:
            mean_val = events_df['event_density_mean'].mean()
            metrics_text.append(f"Avg Event Density: {mean_val:.3f}")
        
        if 'polarity_accuracy_mean' in events_df.columns:
            mean_val = events_df['polarity_accuracy_mean'].mean()
            metrics_text.append(f"Avg Polarity Accuracy: {mean_val:.3f}")
        
        if 'event_rate_mean' in events_df.columns:
            mean_val = events_df['event_rate_mean'].mean()
            metrics_text.append(f"Avg Event Rate: {mean_val:.1f}")
        
        if metrics_text:
            ax.text(0.05, 0.95, '\n'.join(metrics_text), transform=ax.transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace')
        else:
            ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
        
        ax.axis('off')
    
    def _create_dashboard_panel_3(self, ax, frames_df: pd.DataFrame):
        """Dashboard panel 3: Frame metrics summary."""
        ax.set_title('Frame Metrics', fontsize=14, fontweight='bold')
        
        if frames_df.empty:
            ax.text(0.5, 0.5, 'No frame data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            return
        
        metrics_text = []
        if 'ssim_mean' in frames_df.columns:
            mean_val = frames_df['ssim_mean'].mean()
            metrics_text.append(f"Avg SSIM: {mean_val:.3f}")
        
        if 'psnr_mean' in frames_df.columns:
            mean_val = frames_df['psnr_mean'].mean()
            metrics_text.append(f"Avg PSNR: {mean_val:.1f} dB")
        
        if 'lpips_mean' in frames_df.columns:
            mean_val = frames_df['lpips_mean'].mean()
            metrics_text.append(f"Avg LPIPS: {mean_val:.3f}")
        
        if metrics_text:
            ax.text(0.05, 0.95, '\n'.join(metrics_text), transform=ax.transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace')
        else:
            ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
        
        ax.axis('off')
    
    def _create_dashboard_panel_4(self, ax, events_df: pd.DataFrame):
        """Dashboard panel 4: Event performance trends."""
        ax.set_title('Event Performance Trends', fontsize=14, fontweight='bold')
        
        if events_df.empty or 'weather' not in events_df.columns:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Simple trend visualization
        if 'polarity_accuracy_mean' in events_df.columns:
            weather_perf = events_df.groupby('weather')['polarity_accuracy_mean'].mean()
            ax.plot(range(len(weather_perf)), weather_perf.values, 'o-', 
                   color=self.config.colors['primary'], linewidth=2, markersize=8)
            ax.set_xticks(range(len(weather_perf)))
            ax.set_xticklabels(weather_perf.index, rotation=45)
            ax.set_ylabel('Polarity Accuracy')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No performance data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
    
    def _create_dashboard_panel_5(self, ax, frames_df: pd.DataFrame):
        """Dashboard panel 5: Frame performance trends."""
        ax.set_title('Frame Performance Trends', fontsize=14, fontweight='bold')
        
        if frames_df.empty or 'weather' not in frames_df.columns:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            return
        
        # Simple trend visualization
        if 'ssim_mean' in frames_df.columns:
            weather_perf = frames_df.groupby('weather')['ssim_mean'].mean()
            ax.plot(range(len(weather_perf)), weather_perf.values, 's-', 
                   color=self.config.colors['secondary'], linewidth=2, markersize=8)
            ax.set_xticks(range(len(weather_perf)))
            ax.set_xticklabels(weather_perf.index, rotation=45)
            ax.set_ylabel('SSIM')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No performance data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
    
    def _create_dashboard_panel_6(self, ax, events_df: pd.DataFrame, frames_df: pd.DataFrame):
        """Dashboard panel 6: Quality indicators."""
        ax.set_title('Quality Indicators', fontsize=14, fontweight='bold')
        
        # Create quality indicators
        indicators = []
        
        if not events_df.empty and 'polarity_accuracy_mean' in events_df.columns:
            event_quality = events_df['polarity_accuracy_mean'].mean()
            if event_quality > 0.9:
                indicators.append("✓ Event Quality: Excellent")
            elif event_quality > 0.8:
                indicators.append("⚠ Event Quality: Good")
            else:
                indicators.append("✗ Event Quality: Needs Improvement")
        
        if not frames_df.empty and 'ssim_mean' in frames_df.columns:
            frame_quality = frames_df['ssim_mean'].mean()
            if frame_quality > 0.9:
                indicators.append("✓ Frame Quality: Excellent")
            elif frame_quality > 0.8:
                indicators.append("⚠ Frame Quality: Good")
            else:
                indicators.append("✗ Frame Quality: Needs Improvement")
        
        if indicators:
            ax.text(0.05, 0.95, '\n'.join(indicators), transform=ax.transAxes,
                   fontsize=12, verticalalignment='top')
        else:
            ax.text(0.5, 0.5, 'No quality data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
        
        ax.axis('off')