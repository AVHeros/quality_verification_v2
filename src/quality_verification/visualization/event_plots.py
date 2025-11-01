#!/usr/bin/env python3
"""
Event-specific plotting module for DVS Quality Verification visualizations.

This module handles all event-related plotting functions including
overview plots, distributions, and event-specific comparisons.
"""

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figure_layouts import align_ylabels, label_subplots
from .plot_config import plot_config
from .statistics import (
    anova_f_oneway,
    group_metric_summary,
    pairwise_tests,
    summarise_metric,
    summary_table_from_groups,
)

logger = logging.getLogger(__name__)


class EventPlotter:
    """Handles event-specific plotting functions."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = plot_config

    # ------------------------------------------------------------------
    # Internal helpers

    def _export_table(self, df: pd.DataFrame, stem: str, caption: str | None = None) -> None:
        """Persist a tidy dataframe as Markdown and LaTeX for publication."""

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

    def _plot_grouped_metric(self, ax, df: pd.DataFrame, metric: str, group_col: str,
                             title: str, ylabel: str, scope: str) -> None:
        """Render group-wise means with 95% CIs and annotate statistics."""

        summaries = group_metric_summary(df, metric, group_col)
        if not summaries:
            ax.text(0.5, 0.5, f'{metric}\nData Not Available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
            return

        table = summary_table_from_groups(summaries).sort_values('group')
        self._export_table(table, f'{metric}_{group_col}_summary',
                           caption=f'{metric} by {group_col}')

        x_positions = np.arange(len(table))
        colors = self.config.get_colors_for_count(len(table))
        means = table['mean'].to_numpy()
        err_low = means - table['ci_low'].to_numpy()
        err_high = table['ci_high'].to_numpy() - means

        for idx, (x, mean, low, high, color) in enumerate(zip(x_positions, means, err_low, err_high, colors)):
            ax.errorbar(
                x,
                mean,
                yerr=[[low], [high]],
                fmt='o',
                color=color,
                ecolor=color,
                capsize=6,
                markersize=6,
                elinewidth=1.5,
                label=str(table.iloc[idx]['group'])
            )

        ax.plot(x_positions, means, color=self.config.colors['primary'], linewidth=1.2, alpha=0.6)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(table['group'], rotation=30, ha='right')
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        # Apply quality shading if available
        self.config.apply_quality_shading(ax, means, metric, scope)

        # Statistical tests summary
        anova_stats = anova_f_oneway(df, metric, group_col)
        pairwise_df = pairwise_tests(df, metric, group_col)
        stats_lines = []
        if not np.isnan(anova_stats.get('p_value', np.nan)):
            stats_lines.append(f"ANOVA p={anova_stats['p_value']:.3f}")
        if not pairwise_df.empty:
            sig = pairwise_df[pairwise_df['significant']]
            for _, row in sig.iterrows():
                stats_lines.append(
                    f"{row['group_a']} vs {row['group_b']}: p={row['p_value_holm']:.3f}, d={row['effect_size']:.2f}"
                )

        if stats_lines:
            ax.text(0.02, -0.35, '\n'.join(stats_lines), transform=ax.transAxes,
                    fontsize=9, va='top', ha='left', fontfamily='monospace')

    def _summarise_metrics_panel(self, ax, df: pd.DataFrame, metrics: Dict[str, tuple[str, str]]) -> None:
        """Render textual summary with mean ± CI for selected metrics."""

        summary_lines = []
        for metric, (name, unit) in metrics.items():
            if metric not in df.columns:
                continue
            summary = summarise_metric(df[metric])
            if summary.count == 0:
                continue
            summary_lines.append(
                f"{name}: {summary.mean:.3f} ± {summary.half_width:.3f} ({unit}), n={summary.count}"
            )

        if summary_lines:
            ax.text(0.02, 0.98, '\n'.join(summary_lines), transform=ax.transAxes,
                    va='top', ha='left', fontsize=10, fontfamily='monospace')
        else:
            ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)

        ax.set_axis_off()

    def _plot_violin_panel(self, ax, df: pd.DataFrame, metrics: Dict[str, tuple[str, str]]) -> None:
        """Plot violin distributions for a subset of metrics."""

        data_series = []
        labels = []
        for metric, (name, _) in metrics.items():
            if metric not in df.columns:
                continue
            series = df[metric].dropna()
            if series.size < 2:
                continue
            data_series.append(series)
            labels.append(name)

        if not data_series:
            ax.text(0.5, 0.5, 'Insufficient data for distribution plots',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_axis_off()
            return

        violins = ax.violinplot(data_series, showmeans=True, showextrema=False, showmedians=True)
        colors = self.config.get_colors_for_count(len(data_series))
        for body, color in zip(violins['bodies'], colors):
            body.set_facecolor(color)
            body.set_alpha(0.65)

        # Custom mean markers
        means = [np.mean(series) for series in data_series]
        ax.scatter(np.arange(1, len(means) + 1), means, color=self.config.colors['dark'], s=18, zorder=3)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_title('Metric Distributions', fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
    
    def plot_events_overview(self, events_df: pd.DataFrame):
        """Create comprehensive events metrics overview with clean design."""
        logger.info("Creating comprehensive events overview")
        
        logger.debug("Events DataFrame shape: %s", events_df.shape)
        logger.debug("Events DataFrame columns: %s", list(events_df.columns))

        available_metrics = self.config.get_available_metrics(events_df, self.config.event_metrics_config)
        if not available_metrics:
            logger.warning("No key event metrics available for overview")
            return

        top_metrics = dict(list(available_metrics.items())[:5])

        fig, axes = plt.subplots(2, 3, figsize=self.config.figure_sizes['overview'])
        fig.suptitle('Comprehensive Event Metrics Overview', fontsize=18, fontweight='bold')
        label_subplots(fig.axes)

        # Panel A: Distribution overview
        self._plot_violin_panel(axes[0, 0], events_df, top_metrics)

        # Panel B: Event density by weather
        if ('event_density_mean' in events_df.columns) and ('weather' in events_df.columns):
            self._plot_grouped_metric(
                axes[0, 1], events_df, 'event_density_mean', 'weather',
                title='Event Density by Weather',
                ylabel='Events/pixel',
                scope='events'
            )
        else:
            axes[0, 1].text(0.5, 0.5, 'Weather data not available',
                            ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_axis_off()

        # Panel C: Polarity accuracy by weather
        if ('polarity_accuracy_mean' in events_df.columns) and ('weather' in events_df.columns):
            self._plot_grouped_metric(
                axes[0, 2], events_df, 'polarity_accuracy_mean', 'weather',
                title='Polarity Accuracy by Weather',
                ylabel='Ratio',
                scope='events'
            )
        else:
            axes[0, 2].text(0.5, 0.5, 'Polarity data not available',
                            ha='center', va='center', transform=axes[0, 2].transAxes, fontsize=12)
            axes[0, 2].set_axis_off()

        # Panel D: Event rate by route
        if ('event_rate_mean' in events_df.columns) and ('route_type' in events_df.columns):
            self._plot_grouped_metric(
                axes[1, 0], events_df, 'event_rate_mean', 'route_type',
                title='Event Rate by Route Type',
                ylabel='Events/sec',
                scope='events'
            )
        else:
            axes[1, 0].text(0.5, 0.5, 'Route data not available',
                            ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_axis_off()

        # Panel E: Temporal precision by route
        if ('temporal_precision_us_std' in events_df.columns) and ('route_type' in events_df.columns):
            self._plot_grouped_metric(
                axes[1, 1], events_df, 'temporal_precision_us_std', 'route_type',
                title='Temporal Precision by Route Type',
                ylabel='μs std',
                scope='events'
            )
        else:
            axes[1, 1].text(0.5, 0.5, 'Temporal precision data not available',
                            ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_axis_off()

        # Panel F: Summary text
        self._summarise_metrics_panel(axes[1, 2], events_df, top_metrics)

        align_ylabels((axes[0, 1], axes[0, 2]))
        align_ylabels((axes[1, 0], axes[1, 1]))

        events_overview_path = self.output_dir / 'events_metrics_overview'
        self.config.save_plot(fig, events_overview_path)
        logger.info("Events overview saved to %s.[formats]", events_overview_path)
    
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
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5.5))
        axes = np.asarray(axes).reshape(-1)  # Ensures iterable even for single subplot

        fig.suptitle('Event Metrics by Route Type', fontsize=18, fontweight='bold')
        label_subplots(axes)

        for ax, (metric, (name, unit)) in zip(axes, available_metrics.items()):
            self._plot_grouped_metric(
                ax, events_df, metric, 'route_type',
                title=f'{name} by Route Type',
                ylabel=f'{unit}',
                scope='events'
            )

        align_ylabels(axes)

        events_route_path = self.output_dir / 'events_route_type_comparison'
        self.config.save_plot(fig, events_route_path)
        logger.info("Events route comparison saved to %s.[formats]", events_route_path)
    
    def plot_event_metric_distribution(self, events_df: pd.DataFrame, metric_name: str):
        """Plot distribution of a specific event metric."""

        if metric_name not in events_df.columns:
            logger.warning("Metric %s not found in events data", metric_name)
            return

        data = events_df[metric_name].dropna().to_numpy(dtype=float)
        if data.size == 0:
            logger.warning("No valid data for metric %s", metric_name)
            return

        summary = summarise_metric(data)
        display_name, unit = self.config.get_metric_info(metric_name)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'{display_name} Distribution', fontsize=16, fontweight='bold')
        label_subplots(axes)

        # Histogram + KDE
        ax_hist = axes[0]
        ax_hist.hist(data, bins=20, density=True, alpha=0.6,
                     color=self.config.colors['primary'], edgecolor='black')
        try:
            from scipy import stats as scipy_stats  # Local import to keep module load light

            kde = scipy_stats.gaussian_kde(data)
            x_support = np.linspace(data.min(), data.max(), 200)
            ax_hist.plot(x_support, kde(x_support), color=self.config.colors['secondary'], linewidth=2)
        except Exception as exc:  # pragma: no cover - KDE optional
            logger.debug("Skipping KDE overlay for %s due to %s", metric_name, exc)

        ax_hist.axvline(summary.mean, color=self.config.colors['dark'], linestyle='--', linewidth=1.5,
                        label='Mean')
        ax_hist.fill_betweenx(
            [0, ax_hist.get_ylim()[1]],
            summary.ci_low,
            summary.ci_high,
            color=self.config.colors['accent'],
            alpha=0.2,
            label='95% CI'
        )
        ax_hist.set_xlabel(f'{display_name} ({unit})')
        ax_hist.set_ylabel('Density')
        ax_hist.legend(frameon=False)
        ax_hist.grid(True, alpha=0.3)

        # Empirical CDF
        ax_cdf = axes[1]
        sorted_data = np.sort(data)
        cdf = np.linspace(0, 1, sorted_data.size, endpoint=False)
        ax_cdf.step(sorted_data, cdf, where='post', color=self.config.colors['primary'])
        ax_cdf.set_xlabel(f'{display_name} ({unit})')
        ax_cdf.set_ylabel('Empirical CDF')
        ax_cdf.grid(True, alpha=0.3)
        ax_cdf.axvline(summary.mean, color=self.config.colors['dark'], linestyle='--', linewidth=1.0)

        # Summary panel
        ax_text = axes[2]
        lines = [
            f"Mean: {summary.mean:.3f}",
            f"Std: {summary.std:.3f}",
            f"95% CI: [{summary.ci_low:.3f}, {summary.ci_high:.3f}]",
            f"n: {summary.count}",
        ]
        ax_text.text(0.05, 0.95, '\n'.join(lines), transform=ax_text.transAxes,
                     fontsize=11, va='top', ha='left', fontfamily='monospace')
        ax_text.set_axis_off()

        metric_dist_path = self.output_dir / f'event_{metric_name}_distribution'
        self.config.save_plot(fig, metric_dist_path)
        logger.info("Event metric distribution saved to %s.[formats]", metric_dist_path)
    
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
        fig.suptitle('Event Metrics by Weather Conditions (Detailed)', fontsize=18, fontweight='bold')
        label_subplots(fig.axes)

        axes_flat = axes.flatten()

        for i, (metric, (name, unit)) in enumerate(selected_metrics.items()):
            if i >= len(axes_flat):
                break
            self._plot_grouped_metric(
                axes_flat[i], events_df, metric, 'weather',
                title=f'{name} by Weather',
                ylabel=f'{unit}',
                scope='events'
            )

        for j in range(len(selected_metrics), len(axes_flat)):
            axes_flat[j].set_axis_off()

        weather_detailed_path = self.output_dir / 'events_weather_detailed'
        self.config.save_plot(fig, weather_detailed_path)
        logger.info("Detailed events weather analysis saved to %s.[formats]", weather_detailed_path)