#!/usr/bin/env python3
"""
Comparison plotting module for DVS Quality Verification visualizations.

This module handles weather and route type comparison plots for both
event and frame metrics.
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .plot_config import plot_config

logger = logging.getLogger(__name__)


class ComparisonPlotter:
    """Handles weather and route type comparison plotting functions."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = plot_config
    
    def plot_weather_comparison(self, events_df: pd.DataFrame, frames_df: pd.DataFrame):
        """Create comprehensive weather comparison plots."""
        logger.info("Creating weather comparison plots")
        
        # Check if weather column exists
        has_events_weather = 'weather' in events_df.columns if not events_df.empty else False
        has_frames_weather = 'weather' in frames_df.columns if not frames_df.empty else False
        
        if not has_events_weather and not has_frames_weather:
            logger.warning("No weather data available for comparison")
            return
        
        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_sizes['comparison'])
        fig.suptitle('Performance Comparison Across Weather Conditions', fontsize=16, fontweight='bold')
        
        # Plot 1: Event Density by Weather (Events)
        ax = axes[0, 0]
        if has_events_weather and 'event_density_mean' in events_df.columns:
            weather_data = events_df.groupby('weather')['event_density_mean'].agg(['mean', 'std']).reset_index()
            colors = self.config.get_colors_for_count(len(weather_data))
            
            self.config.setup_clean_bar_plot_with_error(
                ax, weather_data['weather'], weather_data['mean'], weather_data['std'],
                colors, title='Event Density by Weather (Events)',
                xlabel='Weather Condition', ylabel='Event Density'
            )
        else:
            ax.text(0.5, 0.5, 'Event Density\nData Not Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Event Density by Weather (Events)')
        
        # Plot 2: Event Rate by Weather (Events)
        ax = axes[0, 1]
        if has_events_weather and 'event_rate_mean' in events_df.columns:
            weather_data = events_df.groupby('weather')['event_rate_mean'].agg(['mean', 'std']).reset_index()
            colors = self.config.get_colors_for_count(len(weather_data))
            
            self.config.setup_clean_bar_plot_with_error(
                ax, weather_data['weather'], weather_data['mean'], weather_data['std'],
                colors, title='Event Rate by Weather (Events)',
                xlabel='Weather Condition', ylabel='Event Rate (events/sec)'
            )
        else:
            ax.text(0.5, 0.5, 'Event Rate\nData Not Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Event Rate by Weather (Events)')
        
        # Plot 3: SSIM by Weather (Frames)
        ax = axes[1, 0]
        if has_frames_weather and 'ssim_mean' in frames_df.columns:
            weather_data = frames_df.groupby('weather')['ssim_mean'].agg(['mean', 'std']).reset_index()
            colors = self.config.get_colors_for_count(len(weather_data))
            
            self.config.setup_clean_bar_plot_with_error(
                ax, weather_data['weather'], weather_data['mean'], weather_data['std'],
                colors, title='SSIM by Weather (Frames)',
                xlabel='Weather Condition', ylabel='SSIM'
            )
        else:
            ax.text(0.5, 0.5, 'SSIM\nData Not Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('SSIM by Weather (Frames)')
        
        # Plot 4: PSNR by Weather (Frames)
        ax = axes[1, 1]
        if has_frames_weather and 'psnr_mean' in frames_df.columns:
            weather_data = frames_df.groupby('weather')['psnr_mean'].agg(['mean', 'std']).reset_index()
            colors = self.config.get_colors_for_count(len(weather_data))
            
            self.config.setup_clean_bar_plot_with_error(
                ax, weather_data['weather'], weather_data['mean'], weather_data['std'],
                colors, title='PSNR by Weather (Frames)',
                xlabel='Weather Condition', ylabel='PSNR (dB)'
            )
        else:
            ax.text(0.5, 0.5, 'PSNR\nData Not Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('PSNR by Weather (Frames)')
        
        # Save plot
        weather_comparison_path = self.output_dir / 'weather_comparison.png'
        self.config.save_plot(fig, weather_comparison_path)
        
        logger.info(f"Weather comparison saved to {weather_comparison_path}")
    
    def plot_route_type_comparison(self, events_df: pd.DataFrame, frames_df: pd.DataFrame):
        """Create comprehensive route type comparison plots."""
        logger.info("Creating route type comparison plots")
        
        # Check if route_type column exists
        has_events_route = 'route_type' in events_df.columns if not events_df.empty else False
        has_frames_route = 'route_type' in frames_df.columns if not frames_df.empty else False
        
        if not has_events_route and not has_frames_route:
            logger.warning("No route type data available for comparison")
            return
        
        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_sizes['comparison'])
        fig.suptitle('Performance Comparison Across Route Types', fontsize=16, fontweight='bold')
        
        # Plot 1: Event Density by Route Type (Events)
        ax = axes[0, 0]
        if has_events_route and 'event_density_mean' in events_df.columns:
            route_data = events_df.groupby('route_type')['event_density_mean'].agg(['mean', 'std']).reset_index()
            colors = self.config.get_colors_for_count(len(route_data))
            
            self.config.setup_clean_bar_plot_with_error(
                ax, route_data['route_type'], route_data['mean'], route_data['std'],
                colors, title='Event Density by Route Type',
                xlabel='Route Type', ylabel='Event Density'
            )
            
            # Add sample counts
            route_counts = events_df['route_type'].value_counts()
            for i, route in enumerate(route_data['route_type']):
                count = route_counts[route]
                ax.text(i, ax.get_ylim()[1] * 0.95, f'n={count}', 
                       ha='center', va='top', fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Event Density\nData Not Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Event Density by Route Type')
        
        # Plot 2: Polarity Accuracy by Route Type (Events)
        ax = axes[0, 1]
        if has_events_route and 'polarity_accuracy_mean' in events_df.columns:
            route_data = events_df.groupby('route_type')['polarity_accuracy_mean'].agg(['mean', 'std']).reset_index()
            colors = self.config.get_colors_for_count(len(route_data))
            
            self.config.setup_clean_bar_plot_with_error(
                ax, route_data['route_type'], route_data['mean'], route_data['std'],
                colors, title='Polarity Accuracy by Route Type',
                xlabel='Route Type', ylabel='Polarity Accuracy'
            )
            
            # Add sample counts
            route_counts = events_df['route_type'].value_counts()
            for i, route in enumerate(route_data['route_type']):
                count = route_counts[route]
                ax.text(i, ax.get_ylim()[1] * 0.95, f'n={count}', 
                       ha='center', va='top', fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Polarity Accuracy\nData Not Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Polarity Accuracy by Route Type')
        
        # Plot 3: SSIM by Route Type (Frames)
        ax = axes[1, 0]
        if has_frames_route and 'ssim_mean' in frames_df.columns:
            route_data = frames_df.groupby('route_type')['ssim_mean'].agg(['mean', 'std']).reset_index()
            colors = self.config.get_colors_for_count(len(route_data))
            
            self.config.setup_clean_bar_plot_with_error(
                ax, route_data['route_type'], route_data['mean'], route_data['std'],
                colors, title='SSIM by Route Type',
                xlabel='Route Type', ylabel='SSIM'
            )
            
            # Add sample counts
            route_counts = frames_df['route_type'].value_counts()
            for i, route in enumerate(route_data['route_type']):
                count = route_counts[route]
                ax.text(i, ax.get_ylim()[1] * 0.95, f'n={count}', 
                       ha='center', va='top', fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'SSIM\nData Not Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('SSIM by Route Type')
        
        # Plot 4: PSNR by Route Type (Frames)
        ax = axes[1, 1]
        if has_frames_route and 'psnr_mean' in frames_df.columns:
            route_data = frames_df.groupby('route_type')['psnr_mean'].agg(['mean', 'std']).reset_index()
            colors = self.config.get_colors_for_count(len(route_data))
            
            self.config.setup_clean_bar_plot_with_error(
                ax, route_data['route_type'], route_data['mean'], route_data['std'],
                colors, title='PSNR by Route Type',
                xlabel='Route Type', ylabel='PSNR (dB)'
            )
            
            # Add sample counts
            route_counts = frames_df['route_type'].value_counts()
            for i, route in enumerate(route_data['route_type']):
                count = route_counts[route]
                ax.text(i, ax.get_ylim()[1] * 0.95, f'n={count}', 
                       ha='center', va='top', fontsize=10, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'PSNR\nData Not Available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('PSNR by Route Type')
        
        # Save plot
        route_comparison_path = self.output_dir / 'route_type_comparison.png'
        self.config.save_plot(fig, route_comparison_path)
        
        logger.info(f"Route type comparison saved to {route_comparison_path}")
    
    def plot_cross_correlation_heatmap(self, events_df: pd.DataFrame, frames_df: pd.DataFrame):
        """Create cross-correlation heatmap between event and frame metrics."""
        logger.info("Creating cross-correlation heatmap")
        
        # Get available metrics
        event_metrics = self.config.get_available_metrics(events_df, self.config.event_metrics_config)
        frame_metrics = self.config.get_available_metrics(frames_df, self.config.frame_metrics_config)
        
        if not event_metrics or not frame_metrics:
            logger.warning("Insufficient metrics for cross-correlation analysis")
            return
        
        # Select key metrics for correlation
        key_event_metrics = ['event_density_mean', 'event_rate_mean', 'polarity_accuracy_mean']
        key_frame_metrics = ['ssim_mean', 'psnr_mean', 'lpips_mean', 'mse_mean']
        
        selected_event_metrics = [m for m in key_event_metrics if m in event_metrics]
        selected_frame_metrics = [m for m in key_frame_metrics if m in frame_metrics]
        
        if not selected_event_metrics or not selected_frame_metrics:
            logger.warning("No key metrics available for cross-correlation")
            return
        
        # Merge dataframes on common columns (assuming they have sequence or similar identifier)
        common_cols = ['sequence', 'weather', 'route_type']
        merge_cols = [col for col in common_cols if col in events_df.columns and col in frames_df.columns]
        
        if not merge_cols:
            logger.warning("No common columns found for merging event and frame data")
            return
        
        # Merge data
        merged_df = pd.merge(events_df[merge_cols + selected_event_metrics], 
                           frames_df[merge_cols + selected_frame_metrics], 
                           on=merge_cols, how='inner')
        
        if merged_df.empty:
            logger.warning("No matching data found after merging")
            return
        
        # Calculate cross-correlation matrix
        all_metrics = selected_event_metrics + selected_frame_metrics
        corr_matrix = merged_df[all_metrics].corr()
        
        # Extract cross-correlations (events vs frames)
        cross_corr = corr_matrix.loc[selected_event_metrics, selected_frame_metrics]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(cross_corr.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Set labels
        event_labels = [self.config.event_metrics_config.get(m, (m, ''))[0] for m in selected_event_metrics]
        frame_labels = [self.config.frame_metrics_config.get(m, (m, ''))[0] for m in selected_frame_metrics]
        
        ax.set_xticks(range(len(selected_frame_metrics)))
        ax.set_yticks(range(len(selected_event_metrics)))
        ax.set_xticklabels(frame_labels, rotation=45, ha='right')
        ax.set_yticklabels(event_labels)
        
        # Add correlation values as text
        for i in range(len(selected_event_metrics)):
            for j in range(len(selected_frame_metrics)):
                text = ax.text(j, i, f'{cross_corr.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', rotation=270, labelpad=20)
        
        ax.set_title('Cross-Correlation: Event vs Frame Metrics', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frame Metrics')
        ax.set_ylabel('Event Metrics')
        
        # Save plot
        cross_corr_path = self.output_dir / 'cross_correlation_heatmap.png'
        self.config.save_plot(fig, cross_corr_path)
        
        logger.info(f"Cross-correlation heatmap saved to {cross_corr_path}")
    
    def plot_performance_summary(self, events_df: pd.DataFrame, frames_df: pd.DataFrame):
        """Create a comprehensive performance summary plot."""
        logger.info("Creating performance summary plot")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config.figure_sizes['large'])
        fig.suptitle('DVS Quality Verification - Performance Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Data availability summary
        ax1.set_title('Data Availability Summary')
        categories = ['Events', 'Frames']
        counts = [len(events_df), len(frames_df)]
        colors = [self.config.colors['primary'], self.config.colors['secondary']]
        
        bars = ax1.bar(categories, counts, color=colors, alpha=0.7)
        ax1.set_ylabel('Number of Sequences')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Weather distribution
        ax2.set_title('Weather Conditions Distribution')
        if 'weather' in events_df.columns:
            weather_counts = events_df['weather'].value_counts()
            colors_weather = self.config.get_colors_for_count(len(weather_counts))
            
            wedges, texts, autotexts = ax2.pie(weather_counts.values, labels=weather_counts.index, 
                                              colors=colors_weather, autopct='%1.1f%%', startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax2.text(0.5, 0.5, 'Weather data\nnot available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
        
        # Plot 3: Route type distribution
        ax3.set_title('Route Types Distribution')
        if 'route_type' in events_df.columns:
            route_counts = events_df['route_type'].value_counts()
            colors_route = self.config.get_colors_for_count(len(route_counts))
            
            bars = ax3.bar(route_counts.index, route_counts.values, color=colors_route, alpha=0.7)
            ax3.set_ylabel('Number of Sequences')
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, count in zip(bars, route_counts.values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{count}', ha='center', va='bottom', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Route type data\nnot available', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
        
        # Plot 4: Key metrics summary
        ax4.set_title('Key Metrics Summary')
        summary_text = []
        
        # Event metrics summary
        if not events_df.empty:
            summary_text.append("EVENT METRICS:")
            if 'event_density_mean' in events_df.columns:
                mean_density = events_df['event_density_mean'].mean()
                summary_text.append(f"  Avg Event Density: {mean_density:.2f}")
            if 'polarity_accuracy_mean' in events_df.columns:
                mean_accuracy = events_df['polarity_accuracy_mean'].mean()
                summary_text.append(f"  Avg Polarity Accuracy: {mean_accuracy:.3f}")
        
        # Frame metrics summary
        if not frames_df.empty:
            summary_text.append("\nFRAME METRICS:")
            if 'ssim_mean' in frames_df.columns:
                mean_ssim = frames_df['ssim_mean'].mean()
                summary_text.append(f"  Avg SSIM: {mean_ssim:.3f}")
            if 'psnr_mean' in frames_df.columns:
                mean_psnr = frames_df['psnr_mean'].mean()
                summary_text.append(f"  Avg PSNR: {mean_psnr:.1f} dB")
        
        if summary_text:
            ax4.text(0.05, 0.95, '\n'.join(summary_text), transform=ax4.transAxes, 
                    fontsize=11, verticalalignment='top', fontfamily='monospace')
        else:
            ax4.text(0.5, 0.5, 'No metrics\navailable', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
        
        ax4.axis('off')
        
        # Save plot
        summary_path = self.output_dir / 'performance_summary.png'
        self.config.save_plot(fig, summary_path)
        
        logger.info(f"Performance summary saved to {summary_path}")