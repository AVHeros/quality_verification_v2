#!/usr/bin/env python3
"""
Data validation module for DVS Quality Verification results.

This module handles validation of DataFrames, metrics availability,
and data quality checks.
"""

import logging
from typing import List, Dict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DVSDataValidator:
    """Handles validation of DVS quality verification data."""
    
    def __init__(self):
        self.event_metrics = {
            'event_density', 'event_rate', 'temporal_precision_us_std',
            'on_ratio', 'off_ratio', 'polarity_accuracy', 
            'event_edge_correlation', 'brightness_delta'
        }
        
        self.frame_metrics = {
            'mse', 'psnr', 'ssim', 'lpips', 'mean_intensity_diff',
            'rgb_mean', 'dvs_mean', 'contrast_ratio', 'rgb_std', 'dvs_std'
        }
    
    def validate_dataframe(self, df: pd.DataFrame, df_name: str) -> bool:
        """Comprehensive dataframe validation."""
        if df is None:
            logger.error(f"{df_name} is None")
            return False
        
        if df.empty:
            logger.warning(f"{df_name} is empty")
            return False
        
        logger.info(f"{df_name} validation: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check for required columns
        required_cols = ['route_name']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"{df_name} missing required columns: {missing_cols}")
        
        # Check for data quality issues
        null_counts = df.isnull().sum()
        high_null_cols = null_counts[null_counts > len(df) * 0.5].index.tolist()
        if high_null_cols:
            logger.warning(f"{df_name} has >50% null values in columns: {high_null_cols}")
        
        # Check for duplicate routes
        if 'route_name' in df.columns:
            duplicates = df['route_name'].duplicated().sum()
            if duplicates > 0:
                logger.warning(f"{df_name} has {duplicates} duplicate route names")
        
        return True
    
    def validate_metric_columns(self, df: pd.DataFrame, expected_metrics: List[str], df_name: str) -> List[str]:
        """Validate and return available metric columns."""
        if df.empty:
            return []
        
        metric_cols = [col for col in df.columns if col.endswith('_mean')]
        available_metrics = [col for col in metric_cols if any(metric in col for metric in expected_metrics)]
        
        logger.info(f"{df_name} available metrics: {len(available_metrics)}/{len(expected_metrics)} expected")
        
        if not available_metrics:
            logger.warning(f"No expected metrics found in {df_name}")
        
        return available_metrics
    
    def check_data_sufficiency(self, df: pd.DataFrame, min_samples: int = 5) -> bool:
        """Check if dataframe has sufficient data for meaningful analysis."""
        if df.empty:
            return False
        
        if len(df) < min_samples:
            logger.warning(f"Insufficient data: {len(df)} samples (minimum: {min_samples})")
            return False
        
        return True
    
    def validate_events_dataframe(self, events_df: pd.DataFrame) -> bool:
        """Validate events dataframe specifically."""
        if not self.validate_dataframe(events_df, "Events DataFrame"):
            return False
        
        if not self.check_data_sufficiency(events_df):
            return False
        
        # Validate event-specific metrics
        available_event_metrics = self.validate_metric_columns(
            events_df, list(self.event_metrics), "Events"
        )
        logger.info(f"Available event metrics: {available_event_metrics}")
        
        return len(available_event_metrics) > 0
    
    def validate_frames_dataframe(self, frames_df: pd.DataFrame) -> bool:
        """Validate frames dataframe specifically."""
        if not self.validate_dataframe(frames_df, "Frames DataFrame"):
            return False
        
        if not self.check_data_sufficiency(frames_df):
            return False
        
        # Validate frame-specific metrics
        available_frame_metrics = self.validate_metric_columns(
            frames_df, list(self.frame_metrics), "Frames"
        )
        logger.info(f"Available frame metrics: {available_frame_metrics}")
        
        return len(available_frame_metrics) > 0
    
    def validate_for_weather_analysis(self, df: pd.DataFrame, df_name: str) -> bool:
        """Validate dataframe for weather-based analysis."""
        if not self.validate_dataframe(df, f"{df_name} for weather analysis"):
            return False
        
        if 'weather' not in df.columns:
            logger.warning(f"{df_name} missing 'weather' column for weather analysis")
            return False
        
        weather_counts = df['weather'].value_counts()
        if len(weather_counts) < 2:
            logger.warning(f"{df_name} has insufficient weather conditions for comparison")
            return False
        
        return True
    
    def validate_for_route_analysis(self, df: pd.DataFrame, df_name: str) -> bool:
        """Validate dataframe for route type analysis."""
        if not self.validate_dataframe(df, f"{df_name} for route analysis"):
            return False
        
        if 'route_type' not in df.columns:
            logger.warning(f"{df_name} missing 'route_type' column for route analysis")
            return False
        
        route_counts = df['route_type'].value_counts()
        if len(route_counts) < 2:
            logger.warning(f"{df_name} has insufficient route types for comparison")
            return False
        
        return True
    
    def get_available_metrics(self, df: pd.DataFrame, metric_type: str) -> List[str]:
        """Get available metrics for a specific type (events or frames)."""
        if df.empty:
            return []
        
        if metric_type == 'events':
            expected_metrics = self.event_metrics
        elif metric_type == 'frames':
            expected_metrics = self.frame_metrics
        else:
            logger.error(f"Unknown metric type: {metric_type}")
            return []
        
        return self.validate_metric_columns(df, list(expected_metrics), f"{metric_type.title()} DataFrame")
    
    def check_numeric_data_quality(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict]:
        """Check quality of numeric data in specified columns."""
        quality_report = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            data = df[col].dropna()
            if len(data) == 0:
                quality_report[col] = {'status': 'no_data', 'issues': ['All values are NaN']}
                continue
            
            issues = []
            
            # Check for infinite values
            if np.isinf(data).any():
                issues.append('Contains infinite values')
            
            # Check for extreme outliers (beyond 3 standard deviations)
            if len(data) > 3:
                z_scores = np.abs((data - data.mean()) / data.std())
                outliers = (z_scores > 3).sum()
                if outliers > 0:
                    issues.append(f'{outliers} extreme outliers detected')
            
            # Check for constant values
            if data.nunique() == 1:
                issues.append('All values are identical')
            
            quality_report[col] = {
                'status': 'good' if not issues else 'issues',
                'issues': issues,
                'count': len(data),
                'null_count': df[col].isnull().sum(),
                'mean': data.mean(),
                'std': data.std()
            }
        
        return quality_report