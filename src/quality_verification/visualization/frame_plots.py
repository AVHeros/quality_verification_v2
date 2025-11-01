#!/usr/bin/env python3
"""
Frame-specific plotting module for DVS Quality Verification visualizations.

This module handles all frame-related plotting functions including
overview plots, distributions, and frame-specific comparisons.
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
    anova_f_oneway,
    group_metric_summary,
    pairwise_tests,
    partial_correlation,
    reorder_correlation,
    summarise_metric,
    summary_table_from_groups,
)

logger = logging.getLogger(__name__)


class FramePlotter:
    """Handles frame-specific plotting functions."""
    
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

    def _plot_grouped_metric(self, ax, df: pd.DataFrame, metric: str, group_col: str,
                             title: str, ylabel: str, scope: str) -> None:
        summaries = group_metric_summary(df, metric, group_col)
        if not summaries:
            ax.text(0.5, 0.5, f'{metric}\nData Not Available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
            return

        table = summary_table_from_groups(summaries).sort_values('group')
        self._export_table(table, f'{metric}_{group_col}_summary', caption=f'{metric} by {group_col}')

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

        self.config.apply_quality_shading(ax, means, metric, scope)

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
        lines = []
        for metric, (name, unit) in metrics.items():
            if metric not in df.columns:
                continue
            summary = summarise_metric(df[metric])
            if summary.count == 0:
                continue
            lines.append(f"{name}: {summary.mean:.3f} ± {summary.half_width:.3f} ({unit}), n={summary.count}")

        if lines:
            ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes,
                    va='top', ha='left', fontsize=10, fontfamily='monospace')
        else:
            ax.text(0.5, 0.5, 'No metrics available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()

    def _plot_violin_panel(self, ax, df: pd.DataFrame, metrics: Dict[str, tuple[str, str]]) -> None:
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

        means = [np.mean(series) for series in data_series]
        ax.scatter(np.arange(1, len(means) + 1), means, color=self.config.colors['dark'], s=18, zorder=3)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_title('Metric Distributions', fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
    
    def plot_frames_overview(self, frames_df: pd.DataFrame):
        """Create comprehensive frames metrics overview with clean design."""
        logger.info("Creating comprehensive frames overview")
        
        available_metrics = self.config.get_available_metrics(frames_df, self.config.frame_metrics_config)
        if not available_metrics:
            logger.warning("No key frame metrics available for overview")
            return

        top_metrics = dict(list(available_metrics.items())[:5])

        fig, axes = plt.subplots(2, 3, figsize=self.config.figure_sizes['overview'])
        fig.suptitle('Comprehensive Frame Metrics Overview', fontsize=18, fontweight='bold')
        label_subplots(fig.axes)

        self._plot_violin_panel(axes[0, 0], frames_df, top_metrics)

        if ('ssim_mean' in frames_df.columns) and ('weather' in frames_df.columns):
            self._plot_grouped_metric(
                axes[0, 1], frames_df, 'ssim_mean', 'weather',
                title='SSIM by Weather',
                ylabel='SSIM',
                scope='frames'
            )
        else:
            axes[0, 1].text(0.5, 0.5, 'Weather data not available',
                            ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)
            axes[0, 1].set_axis_off()

        if ('psnr_mean' in frames_df.columns) and ('route_type' in frames_df.columns):
            self._plot_grouped_metric(
                axes[0, 2], frames_df, 'psnr_mean', 'route_type',
                title='PSNR by Route Type',
                ylabel='dB',
                scope='frames'
            )
        else:
            axes[0, 2].text(0.5, 0.5, 'Route data not available',
                            ha='center', va='center', transform=axes[0, 2].transAxes, fontsize=12)
            axes[0, 2].set_axis_off()

        if ('lpips_mean' in frames_df.columns) and ('weather' in frames_df.columns):
            self._plot_grouped_metric(
                axes[1, 0], frames_df, 'lpips_mean', 'weather',
                title='LPIPS by Weather',
                ylabel='Distance',
                scope='frames'
            )
        else:
            axes[1, 0].text(0.5, 0.5, 'LPIPS data not available',
                            ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
            axes[1, 0].set_axis_off()

        if ('mse_mean' in frames_df.columns) and ('route_type' in frames_df.columns):
            self._plot_grouped_metric(
                axes[1, 1], frames_df, 'mse_mean', 'route_type',
                title='MSE by Route Type',
                ylabel='MSE',
                scope='frames'
            )
        else:
            axes[1, 1].text(0.5, 0.5, 'MSE data not available',
                            ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].set_axis_off()

        self._summarise_metrics_panel(axes[1, 2], frames_df, top_metrics)

        align_ylabels((axes[0, 1], axes[0, 2]))
        align_ylabels((axes[1, 0], axes[1, 1]))

        frames_overview_path = self.output_dir / 'frames_metrics_overview'
        self.config.save_plot(fig, frames_overview_path)
        logger.info("Frames overview saved to %s.[formats]", frames_overview_path)
    
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
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5.5))
        axes = np.asarray(axes).reshape(-1)

        fig.suptitle('Frame Metrics by Route Type', fontsize=18, fontweight='bold')
        label_subplots(axes)

        for ax, (metric, (name, unit)) in zip(axes, available_metrics.items()):
            self._plot_grouped_metric(
                ax, frames_df, metric, 'route_type',
                title=f'{name} by Route Type',
                ylabel=unit,
                scope='frames'
            )

        align_ylabels(axes)

        frames_route_path = self.output_dir / 'frames_route_type_comparison'
        self.config.save_plot(fig, frames_route_path)
        logger.info("Frames route comparison saved to %s.[formats]", frames_route_path)
    
    def plot_frame_metric_distribution(self, frames_df: pd.DataFrame, metric_name: str):
        """Plot distribution of a specific frame metric."""
        if metric_name not in frames_df.columns:
            logger.warning(f"Metric {metric_name} not found in frames data")
            return
        
        data = frames_df[metric_name].dropna().to_numpy(dtype=float)
        if data.size == 0:
            logger.warning("No valid data for metric %s", metric_name)
            return

        summary = summarise_metric(data)
        display_name, unit = self.config.get_metric_info(metric_name)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'{display_name} Distribution', fontsize=16, fontweight='bold')
        label_subplots(axes)

        ax_hist = axes[0]
        ax_hist.hist(data, bins=20, density=True, alpha=0.6,
                     color=self.config.colors['primary'], edgecolor='black')
        try:
            from scipy import stats as scipy_stats  # Local import to reduce module load costs

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

        ax_cdf = axes[1]
        sorted_data = np.sort(data)
        cdf = np.linspace(0, 1, sorted_data.size, endpoint=False)
        ax_cdf.step(sorted_data, cdf, where='post', color=self.config.colors['primary'])
        ax_cdf.set_xlabel(f'{display_name} ({unit})')
        ax_cdf.set_ylabel('Empirical CDF')
        ax_cdf.grid(True, alpha=0.3)
        ax_cdf.axvline(summary.mean, color=self.config.colors['dark'], linestyle='--', linewidth=1.0)

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

        metric_dist_path = self.output_dir / f'frame_{metric_name}_distribution'
        self.config.save_plot(fig, metric_dist_path)
        logger.info("Frame metric distribution saved to %s.[formats]", metric_dist_path)
    
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
        fig.suptitle('Frame Metrics by Weather Conditions (Detailed)', fontsize=18, fontweight='bold')
        label_subplots(fig.axes)

        axes_flat = axes.flatten()

        for i, (metric, (name, unit)) in enumerate(selected_metrics.items()):
            if i >= len(axes_flat):
                break
            self._plot_grouped_metric(
                axes_flat[i], frames_df, metric, 'weather',
                title=f'{name} by Weather',
                ylabel=unit,
                scope='frames'
            )

        for j in range(len(selected_metrics), len(axes_flat)):
            axes_flat[j].set_axis_off()

        weather_detailed_path = self.output_dir / 'frames_weather_detailed'
        self.config.save_plot(fig, weather_detailed_path)
        logger.info("Detailed frames weather analysis saved to %s.[formats]", weather_detailed_path)
    
    def plot_frame_quality_heatmap(self, frames_df: pd.DataFrame):
        """Create a heatmap showing frame quality metrics correlation."""
        quality_metrics = ['ssim_mean', 'psnr_mean', 'lpips_mean', 'mse_mean']
        available_metrics = [m for m in quality_metrics if m in frames_df.columns]
        
        if len(available_metrics) < 2:
            logger.warning("Not enough frame quality metrics for correlation heatmap")
            return

        corr_data = frames_df[available_metrics].corr()
        clustered = reorder_correlation(corr_data)
        partial = partial_correlation(frames_df[available_metrics]).loc[clustered.index, clustered.columns]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Frame Quality Metric Associations', fontsize=18, fontweight='bold')
        label_subplots(axes)

        cmap = 'RdBu_r'
        vmin, vmax = -1, 1

        # Full correlation heatmap
        im_corr = axes[0].imshow(clustered.values, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        axes[0].set_title('Pearson Correlation (clustered)', fontweight='bold')
        axes[0].set_xticks(range(len(clustered.columns)))
        axes[0].set_yticks(range(len(clustered.index)))
        display_cols = [self.config.frame_metrics_config.get(col, (col, ''))[0] for col in clustered.columns]
        display_rows = [self.config.frame_metrics_config.get(idx, (idx, ''))[0] for idx in clustered.index]
        axes[0].set_xticklabels(display_cols, rotation=45, ha='right')
        axes[0].set_yticklabels(display_rows)

        for i in range(clustered.shape[0]):
            for j in range(clustered.shape[1]):
                axes[0].text(j, i, f"{clustered.iloc[i, j]:.2f}",
                             ha='center', va='center', color='black', fontsize=9, fontweight='bold')

        # Partial correlation heatmap
        im_partial = axes[1].imshow(partial.values, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        axes[1].set_title('Partial Correlation', fontweight='bold')
        axes[1].set_xticks(range(len(partial.columns)))
        axes[1].set_yticks(range(len(partial.index)))
        axes[1].set_xticklabels(display_cols, rotation=45, ha='right')
        axes[1].set_yticklabels(display_rows)

        for i in range(partial.shape[0]):
            for j in range(partial.shape[1]):
                axes[1].text(j, i, f"{partial.iloc[i, j]:.2f}",
                             ha='center', va='center', color='black', fontsize=9, fontweight='bold')

        build_shared_colorbar(fig, im_corr, label='Correlation coefficient')

        heatmap_path = self.output_dir / 'frame_quality_correlation_heatmap'
        self.config.save_plot(fig, heatmap_path)
        logger.info("Frame quality correlation heatmap saved to %s.[formats]", heatmap_path)
    
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
        
        x = clean_data[metric_x].to_numpy(dtype=float)
        y = clean_data[metric_y].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        ax.scatter(x, y, alpha=0.6, color=self.config.colors['primary'], edgecolor='black', linewidth=0.4)

        try:
            from scipy import stats as scipy_stats

            lr = scipy_stats.linregress(x, y)
            x_grid = np.linspace(x.min(), x.max(), 200)
            y_fit = lr.intercept + lr.slope * x_grid

            # 95% confidence band for the regression line
            dof = x.size - 2
            if dof > 0:
                y_hat = lr.intercept + lr.slope * x
                residuals = y - y_hat
                s_err = np.sqrt(np.sum(residuals ** 2) / dof)
                x_mean = np.mean(x)
                t_val = scipy_stats.t.ppf(0.975, dof)
                denom = np.sum((x - x_mean) ** 2)
                conf_band = t_val * s_err * np.sqrt(1 / x.size + (x_grid - x_mean) ** 2 / denom)
                ax.fill_between(x_grid, y_fit - conf_band, y_fit + conf_band,
                                color=self.config.colors['secondary'], alpha=0.2, label='95% CI')

            ax.plot(x_grid, y_fit, color=self.config.colors['secondary'], linewidth=2, label='Linear fit')
            summary_text = f"r={lr.rvalue:.3f}, p={lr.pvalue:.3e}, slope={lr.slope:.3f}"
        except Exception as exc:  # pragma: no cover - regression optional
            logger.debug("Skipping regression overlay due to %s", exc)
            summary_text = f"r={clean_data[metric_x].corr(clean_data[metric_y]):.3f}"

        x_name = self.config.frame_metrics_config.get(metric_x, (metric_x, ''))[0]
        y_name = self.config.frame_metrics_config.get(metric_y, (metric_y, ''))[0]

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(f'{x_name} vs {y_name}', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.02, summary_text, transform=ax.transAxes,
                fontsize=10, fontfamily='monospace', ha='left', va='bottom')
        ax.legend(frameon=False)

        self.config.apply_quality_shading(ax, y, metric_y, scope='frames')

        scatter_path = self.output_dir / f'frame_{metric_x}_vs_{metric_y}_scatter'
        self.config.save_plot(fig, scatter_path)
        logger.info("Frame quality scatter plot saved to %s.[formats]", scatter_path)