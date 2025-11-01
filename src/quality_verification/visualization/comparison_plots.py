#!/usr/bin/env python3
"""
Comparison plotting module for DVS Quality Verification visualizations.

This module handles weather and route type comparison plots for both
event and frame metrics.
"""

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figure_layouts import align_ylabels, build_shared_colorbar, label_subplots
from .plot_config import plot_config
from .statistics import (
    MetricSummary,
    anova_f_oneway,
    group_metric_summary,
    pairwise_tests,
    reorder_correlation,
    summarise_metric,
    summary_table_from_groups,
)

logger = logging.getLogger(__name__)


class ComparisonPlotter:
    """Handles weather and route type comparison plotting functions."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = plot_config

    # ------------------------------------------------------------------
    # Internal helpers

    def _export_table(self, df: pd.DataFrame, stem: str, caption: str | None = None) -> None:
        if df.empty:
            return

        markdown_path = self.output_dir / f"{stem}.md"
        latex_path = self.output_dir / f"{stem}.tex"

        with markdown_path.open('w', encoding='utf-8') as handle:
            if caption:
                handle.write(f"# {caption}\n\n")
            handle.write(df.to_markdown(index=False, floatfmt=".4f"))
            handle.write("\n")

        df.to_latex(latex_path, index=False, float_format="%.4f", caption=caption or '')

    def _plot_group_metric(self, ax, df: pd.DataFrame, metric: str, group_col: str,
                           title: str, ylabel: str, scope: str) -> None:
        summaries = group_metric_summary(df, metric, group_col)
        if not summaries:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_axis_off()
            return

        table = summary_table_from_groups(summaries).sort_values('group')
        self._export_table(table, f'{metric}_{group_col}_comparison', caption=f'{metric} by {group_col}')

        x_positions = np.arange(len(table))
        colors = self.config.get_colors_for_count(len(table))
        means = table['mean'].to_numpy()
        err_low = means - table['ci_low'].to_numpy()
        err_high = table['ci_high'].to_numpy() - means

        for idx, (x_pos, mean, low, high, color) in enumerate(zip(x_positions, means, err_low, err_high, colors)):
            ax.errorbar(
                x_pos,
                mean,
                yerr=[[low], [high]],
                fmt='o',
                color=color,
                ecolor=color,
                capsize=6,
                elinewidth=1.5,
                markersize=6,
            )

        ax.plot(x_positions, means, color=self.config.colors['primary'], linewidth=1.1, alpha=0.6)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(table['group'], rotation=30, ha='right')
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        self.config.apply_quality_shading(ax, means, metric, scope)

        anova_stats = anova_f_oneway(df, metric, group_col)
        pairwise_df = pairwise_tests(df, metric, group_col)
        annotations = []
        if not np.isnan(anova_stats.get('p_value', np.nan)):
            annotations.append(f"ANOVA p={anova_stats['p_value']:.3f}")
        if not pairwise_df.empty:
            sig_pairs = pairwise_df[pairwise_df['significant']]
            for _, row in sig_pairs.iterrows():
                annotations.append(
                    f"{row['group_a']} vs {row['group_b']}: p={row['p_value_holm']:.3f}, d={row['effect_size']:.2f}"
                )
        if annotations:
            ax.text(0.02, -0.35, '\n'.join(annotations), transform=ax.transAxes,
                    fontsize=9, va='top', ha='left', fontfamily='monospace')
    
    def plot_weather_comparison(self, events_df: pd.DataFrame, frames_df: pd.DataFrame):
        """Create comprehensive weather comparison plots."""
        logger.info("Creating weather comparison plots")
        
        has_events_weather = 'weather' in events_df.columns if not events_df.empty else False
        has_frames_weather = 'weather' in frames_df.columns if not frames_df.empty else False

        if not has_events_weather and not has_frames_weather:
            logger.warning("No weather data available for comparison")
            return

        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_sizes['comparison'])
        fig.suptitle('Performance Comparison Across Weather Conditions', fontsize=18, fontweight='bold')
        label_subplots(fig.axes)

        if has_events_weather and 'event_density_mean' in events_df.columns:
            self._plot_group_metric(
                axes[0, 0], events_df, 'event_density_mean', 'weather',
                title='Event Density (Events)', ylabel='Events/pixel', scope='events'
            )
        else:
            axes[0, 0].text(0.5, 0.5, 'Event density data not available',
                            ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12)
            axes[0, 0].set_axis_off()

        if has_events_weather and 'event_rate_mean' in events_df.columns:
            self._plot_group_metric(
                axes[0, 1], events_df, 'event_rate_mean', 'weather',
                title='Event Rate (Events)', ylabel='Events/sec', scope='events'
            )
        else:
            axes[0, 1].text(0.5, 0.5, 'Event rate data not available',
                            ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_axis_off()

        if has_frames_weather and 'ssim_mean' in frames_df.columns:
            self._plot_group_metric(
                axes[1, 0], frames_df, 'ssim_mean', 'weather',
                title='SSIM (Frames)', ylabel='SSIM', scope='frames'
            )
        else:
            axes[1, 0].text(0.5, 0.5, 'SSIM data not available',
                            ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_axis_off()

        if has_frames_weather and 'psnr_mean' in frames_df.columns:
            self._plot_group_metric(
                axes[1, 1], frames_df, 'psnr_mean', 'weather',
                title='PSNR (Frames)', ylabel='dB', scope='frames'
            )
        else:
            axes[1, 1].text(0.5, 0.5, 'PSNR data not available',
                            ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_axis_off()

        align_ylabels((axes[0, 0], axes[0, 1]))
        align_ylabels((axes[1, 0], axes[1, 1]))

        weather_comparison_path = self.output_dir / 'weather_comparison'
        self.config.save_plot(fig, weather_comparison_path)
        logger.info("Weather comparison saved to %s.[formats]", weather_comparison_path)
    
    def plot_route_type_comparison(self, events_df: pd.DataFrame, frames_df: pd.DataFrame):
        """Create comprehensive route type comparison plots."""
        logger.info("Creating route type comparison plots")
        
        # Check if route_type column exists
        has_events_route = 'route_type' in events_df.columns if not events_df.empty else False
        has_frames_route = 'route_type' in frames_df.columns if not frames_df.empty else False
        
        if not has_events_route and not has_frames_route:
            logger.warning("No route type data available for comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_sizes['comparison'])
        fig.suptitle('Performance Comparison Across Route Types', fontsize=18, fontweight='bold')
        label_subplots(fig.axes)

        if has_events_route and 'event_density_mean' in events_df.columns:
            self._plot_group_metric(
                axes[0, 0], events_df, 'event_density_mean', 'route_type',
                title='Event Density (Events)', ylabel='Events/pixel', scope='events'
            )
        else:
            axes[0, 0].text(0.5, 0.5, 'Event density data not available',
                            ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12)
            axes[0, 0].set_axis_off()

        if has_events_route and 'polarity_accuracy_mean' in events_df.columns:
            self._plot_group_metric(
                axes[0, 1], events_df, 'polarity_accuracy_mean', 'route_type',
                title='Polarity Accuracy (Events)', ylabel='Ratio', scope='events'
            )
        else:
            axes[0, 1].text(0.5, 0.5, 'Polarity data not available',
                            ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_axis_off()

        if has_frames_route and 'ssim_mean' in frames_df.columns:
            self._plot_group_metric(
                axes[1, 0], frames_df, 'ssim_mean', 'route_type',
                title='SSIM (Frames)', ylabel='SSIM', scope='frames'
            )
        else:
            axes[1, 0].text(0.5, 0.5, 'SSIM data not available',
                            ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_axis_off()

        if has_frames_route and 'psnr_mean' in frames_df.columns:
            self._plot_group_metric(
                axes[1, 1], frames_df, 'psnr_mean', 'route_type',
                title='PSNR (Frames)', ylabel='dB', scope='frames'
            )
        else:
            axes[1, 1].text(0.5, 0.5, 'PSNR data not available',
                            ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_axis_off()

        align_ylabels((axes[0, 0], axes[0, 1]))
        align_ylabels((axes[1, 0], axes[1, 1]))

        route_comparison_path = self.output_dir / 'route_type_comparison'
        self.config.save_plot(fig, route_comparison_path)
        logger.info("Route type comparison saved to %s.[formats]", route_comparison_path)
    
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

        cross_corr = corr_matrix.loc[selected_event_metrics, selected_frame_metrics]

        # Cluster rows/columns for readability
        event_order = reorder_correlation(merged_df[selected_event_metrics].corr()).index.tolist()
        frame_order = reorder_correlation(merged_df[selected_frame_metrics].corr()).index.tolist()
        cross_corr = cross_corr.loc[event_order, frame_order]

        # Export table with correlation coefficients
        tidy_records = []
        for e_metric in cross_corr.index:
            for f_metric in cross_corr.columns:
                coeff = cross_corr.loc[e_metric, f_metric]
                try:
                    from scipy import stats as scipy_stats

                    r, p_value = scipy_stats.pearsonr(merged_df[e_metric], merged_df[f_metric])
                except Exception:
                    r, p_value = np.nan, np.nan
                tidy_records.append({
                    'event_metric': e_metric,
                    'frame_metric': f_metric,
                    'correlation': coeff,
                    'p_value': p_value,
                })

        corr_table = pd.DataFrame.from_records(tidy_records)
        self._export_table(corr_table, 'event_frame_cross_correlation', caption='Event vs Frame correlations')

        event_labels = [self.config.event_metrics_config.get(m, (m, ''))[0] for m in cross_corr.index]
        frame_labels = [self.config.frame_metrics_config.get(m, (m, ''))[0] for m in cross_corr.columns]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle('Cross-Correlation: Event vs Frame Metrics', fontsize=18, fontweight='bold')
        im = ax.imshow(cross_corr.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

        ax.set_xticks(range(len(frame_labels)))
        ax.set_yticks(range(len(event_labels)))
        ax.set_xticklabels(frame_labels, rotation=45, ha='right')
        ax.set_yticklabels(event_labels)

        for i in range(cross_corr.shape[0]):
            for j in range(cross_corr.shape[1]):
                ax.text(j, i, f"{cross_corr.iloc[i, j]:.2f}",
                        ha='center', va='center', color='black', fontsize=9, fontweight='bold')

        build_shared_colorbar(fig, im, label='Correlation coefficient')
        ax.set_xlabel('Frame Metrics')
        ax.set_ylabel('Event Metrics')

        cross_corr_path = self.output_dir / 'cross_correlation_heatmap'
        self.config.save_plot(fig, cross_corr_path)
        logger.info("Cross-correlation heatmap saved to %s.[formats]", cross_corr_path)
    
    def plot_performance_summary(self, events_df: pd.DataFrame, frames_df: pd.DataFrame):
        """Create a comprehensive performance summary plot."""
        logger.info("Creating performance summary plot")
        self.config.apply_style()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config.figure_sizes['large'])
        fig.suptitle('DVS Quality Verification – Performance Overview', fontsize=18, fontweight='bold')

        # Plot 1: acquisition coverage
        ax1.set_title('Sequence Availability')
        categories = ['Events', 'Frames']
        counts = np.array([len(events_df), len(frames_df)])
        colors = [self.config.colors['primary'], self.config.colors['secondary']]
        bars = ax1.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_ylabel('Number of sequences')
        ax1.set_ylim(0, counts.max() * 1.2 if counts.size and counts.max() > 0 else 1)
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(1, bar.get_height() * 0.05),
                     f'{int(count)}', ha='center', va='bottom', fontweight='bold')

        # Plot 2: weather distribution (events fallback to frames)
        ax2.set_title('Weather Coverage')
        weather_source = events_df if 'weather' in events_df.columns else frames_df
        if not weather_source.empty and 'weather' in weather_source.columns:
            weather_counts = weather_source['weather'].value_counts(normalize=True).sort_index()
            colors_weather = self.config.get_colors_for_count(len(weather_counts))
            ax2.barh(weather_counts.index, weather_counts.values * 100, color=colors_weather, alpha=0.85)
            ax2.set_xlabel('Share of sequences (%)')
            for y, val in enumerate(weather_counts.values * 100):
                ax2.text(val + 1, y, f'{val:.1f}%', va='center', ha='left', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Weather data not available', ha='center', va='center',
                     transform=ax2.transAxes, fontsize=12)

        # Plot 3: route distribution (events fallback to frames)
        ax3.set_title('Route Type Coverage')
        route_source = events_df if 'route_type' in events_df.columns else frames_df
        if not route_source.empty and 'route_type' in route_source.columns:
            route_counts = route_source['route_type'].value_counts(normalize=True).sort_index()
            colors_route = self.config.get_colors_for_count(len(route_counts))
            ax3.barh(route_counts.index, route_counts.values * 100, color=colors_route, alpha=0.85)
            ax3.set_xlabel('Share of sequences (%)')
            for y, val in enumerate(route_counts.values * 100):
                ax3.text(val + 1, y, f'{val:.1f}%', va='center', ha='left', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Route type data not available', ha='center', va='center',
                     transform=ax3.transAxes, fontsize=12)

        # Plot 4: statistical summary with bootstrap CIs
        ax4.set_title('Key Metric Estimates (95% CI)')
        metric_candidates = {
            'event_density_mean': ('Event Density', events_df, 'events'),
            'event_rate_mean': ('Event Rate', events_df, 'events'),
            'polarity_accuracy_mean': ('Polarity Accuracy', events_df, 'events'),
            'ssim_mean': ('SSIM', frames_df, 'frames'),
            'psnr_mean': ('PSNR', frames_df, 'frames'),
        }

        metric_summaries: Dict[str, MetricSummary] = {}
        for metric_name, (label, source_df, scope) in metric_candidates.items():
            if source_df.empty or metric_name not in source_df.columns:
                continue
            summary = summarise_metric(source_df[metric_name])
            if np.isnan(summary.mean):
                continue
            metric_summaries[label] = summary

        if metric_summaries:
            ordered_labels = list(metric_summaries.keys())
            means = np.array([metric_summaries[label].mean for label in ordered_labels])
            ci_low = np.array([metric_summaries[label].ci_low for label in ordered_labels])
            ci_high = np.array([metric_summaries[label].ci_high for label in ordered_labels])
            lower_err = means - ci_low
            upper_err = ci_high - means

            y_positions = np.arange(len(ordered_labels))
            color_cycle = self.config.iter_color_cycle()
            for idx, label in enumerate(ordered_labels):
                color = next(color_cycle)
                ax4.errorbar(
                    means[idx],
                    y_positions[idx],
                    xerr=[[lower_err[idx]], [upper_err[idx]]],
                    fmt='o',
                    color=color,
                    ecolor=color,
                    capsize=5,
                    markersize=7,
                    elinewidth=1.6,
                )
            ax4.set_yticks(y_positions)
            ax4.set_yticklabels(ordered_labels)
            ax4.set_xlabel('Metric value')
            ax4.grid(True, axis='x', alpha=0.3)

            summary_lines = [
                f"{label}: {metric_summaries[label].mean:.3f} "
                f"(CI {metric_summaries[label].ci_low:.3f}–{metric_summaries[label].ci_high:.3f}, n={metric_summaries[label].count})"
                for label in ordered_labels
            ]
            fig.text(0.51, 0.05, '\n'.join(summary_lines), fontsize=10, fontfamily='monospace',
                     va='bottom', ha='left')

            stats_table = summary_table_from_groups({label: metric_summaries[label] for label in ordered_labels})
            self._export_table(stats_table, 'performance_summary_metrics', caption='Key metric estimates')
        else:
            ax4.text(0.5, 0.5, 'No metric columns available', ha='center', va='center',
                     transform=ax4.transAxes, fontsize=12)
            ax4.set_axis_off()

        fig.tight_layout(rect=(0, 0.08, 1, 0.98))
        summary_path = self.output_dir / 'performance_summary'
        saved_paths = self.config.save_plot(fig, summary_path)
        logger.info("Performance summary saved to: %s", ", ".join(map(str, saved_paths)))