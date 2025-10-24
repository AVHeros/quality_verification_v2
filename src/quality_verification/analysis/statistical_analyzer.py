#!/usr/bin/env python3
"""
Statistical analysis module for DVS Quality Verification results.

This module handles statistical computations, summary generation,
and data aggregation for analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DVSStatisticalAnalyzer:
    """Handles statistical analysis of DVS quality verification data."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_dataframes(self, events_data: List[Dict], frames_data: List[Dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert collected data to pandas DataFrames."""
        events_df = pd.DataFrame(events_data) if events_data else pd.DataFrame()
        frames_df = pd.DataFrame(frames_data) if frames_data else pd.DataFrame()
        
        logger.info(f"Created DataFrames - Events: {len(events_df)}, Frames: {len(frames_df)}")
        
        return events_df, frames_df
    
    def generate_summary_statistics(self, events_df: pd.DataFrame, frames_df: pd.DataFrame, 
                                  skipped_dirs: List[str]) -> Dict[str, Any]:
        """Generate and save comprehensive summary statistics."""
        logger.info("Generating summary statistics")
        
        summary = {
            'total_routes_events': len(events_df),
            'total_routes_frames': len(frames_df),
            'unique_weather_conditions': len(set(events_df['weather'].unique()) | set(frames_df['weather'].unique())) if not events_df.empty or not frames_df.empty else 0,
            'route_types': list(set(events_df['route_type'].unique()) | set(frames_df['route_type'].unique())) if not events_df.empty or not frames_df.empty else [],
            'skipped_directories': len(skipped_dirs),
            'skipped_list': skipped_dirs
        }
        
        # Weather distribution
        if not events_df.empty:
            summary['weather_distribution_events'] = events_df['weather'].value_counts().to_dict()
        if not frames_df.empty:
            summary['weather_distribution_frames'] = frames_df['weather'].value_counts().to_dict()
        
        # Route type distribution
        if not events_df.empty:
            summary['route_type_distribution_events'] = events_df['route_type'].value_counts().to_dict()
        if not frames_df.empty:
            summary['route_type_distribution_frames'] = frames_df['route_type'].value_counts().to_dict()
        
        # Add detailed metrics statistics
        if not events_df.empty:
            summary['events_metrics_stats'] = self._compute_metrics_statistics(events_df, 'events')
        if not frames_df.empty:
            summary['frames_metrics_stats'] = self._compute_metrics_statistics(frames_df, 'frames')
        
        # Save summary
        summary_path = self.output_dir / 'summary_statistics.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Summary statistics saved to {summary_path}")
        return summary
    
    def _compute_metrics_statistics(self, df: pd.DataFrame, data_type: str) -> Dict[str, Dict]:
        """Compute detailed statistics for metrics columns."""
        metrics_stats = {}
        
        # Find all metric columns (ending with _mean)
        metric_cols = [col for col in df.columns if col.endswith('_mean')]
        
        for col in metric_cols:
            data = df[col].dropna()
            if len(data) > 0:
                metrics_stats[col] = {
                    'count': len(data),
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'median': float(data.median()),
                    'q25': float(data.quantile(0.25)),
                    'q75': float(data.quantile(0.75)),
                    'null_count': int(df[col].isnull().sum()),
                    'null_percentage': float(df[col].isnull().sum() / len(df) * 100)
                }
        
        return metrics_stats
    
    def compute_weather_statistics(self, df: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Dict]:
        """Compute statistics grouped by weather conditions."""
        if df.empty or 'weather' not in df.columns:
            return {}
        
        weather_stats = {}
        
        for weather in sorted(df['weather'].unique()):
            weather_data = df[df['weather'] == weather]
            weather_stats[f'weather_{weather}'] = {}
            
            for col in metric_cols:
                if col in weather_data.columns:
                    data = weather_data[col].dropna()
                    if len(data) > 0:
                        weather_stats[f'weather_{weather}'][col] = {
                            'count': len(data),
                            'mean': float(data.mean()),
                            'std': float(data.std()),
                            'min': float(data.min()),
                            'max': float(data.max())
                        }
        
        return weather_stats
    
    def compute_route_statistics(self, df: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Dict]:
        """Compute statistics grouped by route types."""
        if df.empty or 'route_type' not in df.columns:
            return {}
        
        route_stats = {}
        
        for route_type in sorted(df['route_type'].unique()):
            route_data = df[df['route_type'] == route_type]
            route_stats[route_type] = {}
            
            for col in metric_cols:
                if col in route_data.columns:
                    data = route_data[col].dropna()
                    if len(data) > 0:
                        route_stats[route_type][col] = {
                            'count': len(data),
                            'mean': float(data.mean()),
                            'std': float(data.std()),
                            'min': float(data.min()),
                            'max': float(data.max())
                        }
        
        return route_stats
    
    def compute_correlation_matrix(self, events_df: pd.DataFrame, frames_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute correlation matrices for events and frames data."""
        correlation_data = {}
        
        # Events correlation
        if not events_df.empty:
            event_metric_cols = [col for col in events_df.columns if col.endswith('_mean')]
            if len(event_metric_cols) > 1:
                events_numeric = events_df[event_metric_cols].select_dtypes(include=[np.number])
                if not events_numeric.empty:
                    correlation_data['events_correlation'] = events_numeric.corr().to_dict()
        
        # Frames correlation
        if not frames_df.empty:
            frame_metric_cols = [col for col in frames_df.columns if col.endswith('_mean')]
            if len(frame_metric_cols) > 1:
                frames_numeric = frames_df[frame_metric_cols].select_dtypes(include=[np.number])
                if not frames_numeric.empty:
                    correlation_data['frames_correlation'] = frames_numeric.corr().to_dict()
        
        # Cross-correlation between events and frames
        if not events_df.empty and not frames_df.empty:
            # Merge on common columns for cross-correlation
            common_cols = ['route_name', 'weather', 'route_type']
            available_common = [col for col in common_cols if col in events_df.columns and col in frames_df.columns]
            
            if available_common:
                merged_df = pd.merge(events_df, frames_df, on=available_common, suffixes=('_events', '_frames'))
                
                event_cols = [col for col in merged_df.columns if col.endswith('_events')]
                frame_cols = [col for col in merged_df.columns if col.endswith('_frames')]
                
                if event_cols and frame_cols:
                    cross_corr_data = merged_df[event_cols + frame_cols].select_dtypes(include=[np.number])
                    if not cross_corr_data.empty:
                        correlation_data['cross_correlation'] = cross_corr_data.corr().to_dict()
        
        return correlation_data
    
    def save_processed_data(self, events_df: pd.DataFrame, frames_df: pd.DataFrame, skipped_dirs: List[str]):
        """Save processed data to CSV files."""
        logger.info("Saving processed data")
        
        if not events_df.empty:
            events_path = self.output_dir / 'events_metrics_summary.csv'
            events_df.to_csv(events_path, index=False)
            logger.info(f"Events data saved to {events_path}")
        
        if not frames_df.empty:
            frames_path = self.output_dir / 'frames_metrics_summary.csv'
            frames_df.to_csv(frames_path, index=False)
            logger.info(f"Frames data saved to {frames_path}")
        
        # Save skipped directories list
        skipped_path = self.output_dir / 'skipped_directories.txt'
        with open(skipped_path, 'w') as f:
            f.write("Skipped Directories:\n")
            f.write("===================\n\n")
            for skip_dir in skipped_dirs:
                f.write(f"{skip_dir}\n")
        
        logger.info(f"Skipped directories list saved to {skipped_path}")
    
    def generate_detailed_report(self, events_df: pd.DataFrame, frames_df: pd.DataFrame, 
                               skipped_dirs: List[str]) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        logger.info("Generating detailed analysis report")
        
        report = {
            'analysis_summary': self.generate_summary_statistics(events_df, frames_df, skipped_dirs),
            'correlation_analysis': self.compute_correlation_matrix(events_df, frames_df)
        }
        
        # Add weather-based analysis
        if not events_df.empty:
            event_metric_cols = [col for col in events_df.columns if col.endswith('_mean')]
            report['events_weather_analysis'] = self.compute_weather_statistics(events_df, event_metric_cols)
            report['events_route_analysis'] = self.compute_route_statistics(events_df, event_metric_cols)
        
        if not frames_df.empty:
            frame_metric_cols = [col for col in frames_df.columns if col.endswith('_mean')]
            report['frames_weather_analysis'] = self.compute_weather_statistics(frames_df, frame_metric_cols)
            report['frames_route_analysis'] = self.compute_route_statistics(frames_df, frame_metric_cols)
        
        # Save detailed report
        report_path = self.output_dir / 'detailed_analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Detailed analysis report saved to {report_path}")
        return report