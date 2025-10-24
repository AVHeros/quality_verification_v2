#!/usr/bin/env python3
"""
Core analyzer module for DVS Quality Verification.

This module orchestrates all analysis and visualization components,
providing a unified interface for the complete analysis pipeline.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

from .data_collector import DVSDataCollector
from .data_validator import DVSDataValidator
from .statistical_analyzer import DVSStatisticalAnalyzer
from ..visualization.plot_config import plot_config
from ..visualization.event_plots import EventPlotter
from ..visualization.frame_plots import FramePlotter
from ..visualization.comparison_plots import ComparisonPlotter
from ..visualization.overview_plots import OverviewPlotter

logger = logging.getLogger(__name__)


class DVSCoreAnalyzer:
    """
    Core analyzer that orchestrates the complete DVS quality verification analysis pipeline.
    
    This class integrates data collection, validation, statistical analysis, and visualization
    components to provide a comprehensive analysis of DVS quality verification results.
    """
    
    def __init__(self, root_dir: Path, output_dir: Path):
        """
        Initialize the core analyzer.
        
        Args:
            root_dir: Root directory containing DVS quality verification results
            output_dir: Directory where analysis outputs will be saved
        """
        self.root_dir = Path(root_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_collector = DVSDataCollector(self.root_dir)
        self.data_validator = DVSDataValidator()
        self.statistical_analyzer = DVSStatisticalAnalyzer(self.output_dir)
        
        # Initialize plotters
        self.event_plotter = EventPlotter(self.output_dir)
        self.frame_plotter = FramePlotter(self.output_dir)
        self.comparison_plotter = ComparisonPlotter(self.output_dir)
        self.overview_plotter = OverviewPlotter(self.output_dir)
        
        # Data storage
        self.events_data: Dict[str, Any] = {}
        self.frames_data: Dict[str, Any] = {}
        self.skipped_dirs: List[str] = []
        self.events_df: Optional[pd.DataFrame] = None
        self.frames_df: Optional[pd.DataFrame] = None
        self.summary_stats: Dict[str, Any] = {}
        
        logger.info(f"DVS Core Analyzer initialized with root: {self.root_dir}, output: {self.output_dir}")
    
    def collect_data(self) -> bool:
        """
        Collect all DVS quality verification data from the root directory.
        
        Returns:
            bool: True if data collection was successful, False otherwise
        """
        logger.info("Starting data collection phase")
        
        try:
            # Collect events and frames data
            self.data_collector.collect_all_data()
            self.events_data, self.frames_data, self.skipped_dirs = self.data_collector.get_collected_data()
            
            if not self.events_data and not self.frames_data:
                logger.error("No data collected from the specified directory")
                return False
            
            logger.info(f"Data collection completed: {len(self.events_data)} event sequences, "
                       f"{len(self.frames_data)} frame sequences")
            return True
            
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
            return False
    
    def validate_data(self) -> bool:
        """
        Validate the collected data for quality and completeness.
        
        Returns:
            bool: True if validation passed, False otherwise
        """
        logger.info("Starting data validation phase")
        
        try:
            # Basic validation - check if we have any data
            if not self.events_data and not self.frames_data:
                logger.error("No valid data available after validation")
                return False
            
            logger.info(f"Data validation completed: {len(self.events_data)} events, {len(self.frames_data)} frames")
            return True
            
        except Exception as e:
            logger.error(f"Error during data validation: {e}")
            return False
    
    def create_dataframes(self) -> bool:
        """
        Create pandas DataFrames from the collected data.
        
        Returns:
            bool: True if DataFrame creation was successful, False otherwise
        """
        logger.info("Creating DataFrames from collected data")
        
        try:
            # Create DataFrames using the statistical analyzer
            self.events_df, self.frames_df = self.statistical_analyzer.create_dataframes(
                self.events_data, self.frames_data
            )
            
            # Validate DataFrames
            if self.events_df is not None and not self.events_df.empty:
                events_df_valid = self.data_validator.validate_events_dataframe(self.events_df)
                if not events_df_valid:
                    logger.warning("Events DataFrame validation failed")
            
            if self.frames_df is not None and not self.frames_df.empty:
                frames_df_valid = self.data_validator.validate_frames_dataframe(self.frames_df)
                if not frames_df_valid:
                    logger.warning("Frames DataFrame validation failed")
            
            logger.info(f"DataFrames created successfully - Events: {len(self.events_df) if self.events_df is not None else 0}, "
                       f"Frames: {len(self.frames_df) if self.frames_df is not None else 0}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating DataFrames: {e}")
            return False
    
    def generate_statistics(self) -> bool:
        """
        Generate comprehensive statistical analysis.
        
        Returns:
            bool: True if statistics generation was successful, False otherwise
        """
        logger.info("Generating statistical analysis")
        
        try:
            # Generate summary statistics
            self.summary_stats = self.statistical_analyzer.generate_summary_statistics(
                self.events_df, self.frames_df, self.skipped_dirs
            )
            
            logger.info("Statistical analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            return False
    
    def create_visualizations(self) -> bool:
        """
        Create all visualization plots.
        
        Returns:
            bool: True if visualization creation was successful, False otherwise
        """
        logger.info("Creating visualizations")
        
        try:
            # Create overview plots
            if self.events_df is not None or self.frames_df is not None:
                self.overview_plotter.plot_comprehensive_overview(
                    self.events_df if self.events_df is not None else pd.DataFrame(),
                    self.frames_df if self.frames_df is not None else pd.DataFrame()
                )
                
                self.overview_plotter.plot_metrics_dashboard(
                    self.events_df if self.events_df is not None else pd.DataFrame(),
                    self.frames_df if self.frames_df is not None else pd.DataFrame()
                )
            
            # Create event-specific plots
            if self.events_df is not None and not self.events_df.empty:
                self.event_plotter.plot_events_overview(self.events_df)
                
                if 'route_type' in self.events_df.columns:
                    self.event_plotter.plot_events_route_comparison(self.events_df)
                
                if 'weather' in self.events_df.columns:
                    self.event_plotter.plot_events_weather_detailed(self.events_df)
            
            # Create frame-specific plots
            if self.frames_df is not None and not self.frames_df.empty:
                self.frame_plotter.plot_frames_overview(self.frames_df)
                
                if 'route_type' in self.frames_df.columns:
                    self.frame_plotter.plot_frames_route_comparison(self.frames_df)
                
                if 'weather' in self.frames_df.columns:
                    self.frame_plotter.plot_frames_weather_detailed(self.frames_df)
                
                # Create frame quality correlation heatmap
                self.frame_plotter.plot_frame_quality_heatmap(self.frames_df)
            
            # Create comparison plots
            self.comparison_plotter.plot_weather_comparison(
                self.events_df if self.events_df is not None else pd.DataFrame(),
                self.frames_df if self.frames_df is not None else pd.DataFrame()
            )
            
            self.comparison_plotter.plot_route_type_comparison(
                self.events_df if self.events_df is not None else pd.DataFrame(),
                self.frames_df if self.frames_df is not None else pd.DataFrame()
            )
            
            self.comparison_plotter.plot_cross_correlation_heatmap(
                self.events_df if self.events_df is not None else pd.DataFrame(),
                self.frames_df if self.frames_df is not None else pd.DataFrame()
            )
            
            self.comparison_plotter.plot_performance_summary(
                self.events_df if self.events_df is not None else pd.DataFrame(),
                self.frames_df if self.frames_df is not None else pd.DataFrame()
            )
            
            logger.info("All visualizations created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return False
    
    def save_results(self) -> bool:
        """
        Save all analysis results to files.
        
        Returns:
            bool: True if saving was successful, False otherwise
        """
        logger.info("Saving analysis results")
        
        try:
            # Save processed data
            self.statistical_analyzer.save_processed_data(
                self.events_df, self.frames_df, self.skipped_dirs
            )
            
            # Summary statistics are already saved in generate_statistics method
            
            # Save skipped directories log
            if self.skipped_dirs:
                skipped_file = self.output_dir / 'skipped_directories.txt'
                with open(skipped_file, 'w') as f:
                    f.write("Directories skipped during analysis:\n")
                    for directory in self.skipped_dirs:
                        f.write(f"{directory}\n")
                logger.info(f"Skipped directories log saved to {skipped_file}")
            
            logger.info("All results saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def run_complete_analysis(self) -> bool:
        """
        Run the complete analysis pipeline.
        
        Returns:
            bool: True if the complete analysis was successful, False otherwise
        """
        logger.info("Starting complete DVS quality verification analysis")
        
        # Step 1: Collect data
        if not self.collect_data():
            logger.error("Data collection failed - aborting analysis")
            return False
        
        # Step 2: Validate data
        if not self.validate_data():
            logger.error("Data validation failed - aborting analysis")
            return False
        
        # Step 3: Create DataFrames
        if not self.create_dataframes():
            logger.error("DataFrame creation failed - aborting analysis")
            return False
        
        # Step 4: Generate statistics
        if not self.generate_statistics():
            logger.error("Statistics generation failed - aborting analysis")
            return False
        
        # Step 5: Create visualizations
        if not self.create_visualizations():
            logger.error("Visualization creation failed - aborting analysis")
            return False
        
        # Step 6: Save results
        if not self.save_results():
            logger.error("Results saving failed - aborting analysis")
            return False
        
        logger.info("Complete DVS quality verification analysis finished successfully")
        return True
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the analysis results.
        
        Returns:
            Dict containing analysis summary information
        """
        summary = {
            'data_collection': {
                'events_sequences': len(self.events_data) if self.events_data else 0,
                'frames_sequences': len(self.frames_data) if self.frames_data else 0,
                'skipped_directories': len(self.skipped_dirs)
            },
            'dataframes': {
                'events_rows': len(self.events_df) if self.events_df is not None else 0,
                'frames_rows': len(self.frames_df) if self.frames_df is not None else 0,
                'events_columns': len(self.events_df.columns) if self.events_df is not None else 0,
                'frames_columns': len(self.frames_df.columns) if self.frames_df is not None else 0
            },
            'output_directory': str(self.output_dir),
            'summary_statistics': self.summary_stats
        }
        
        return summary
    
    def create_individual_metric_plots(self, metric_name: str, data_type: str = 'both'):
        """
        Create individual plots for specific metrics.
        
        Args:
            metric_name: Name of the metric to plot
            data_type: Type of data ('events', 'frames', or 'both')
        """
        logger.info(f"Creating individual plots for metric: {metric_name}")
        
        try:
            if data_type in ['events', 'both'] and self.events_df is not None:
                self.event_plotter.plot_event_metric_distribution(self.events_df, metric_name)
            
            if data_type in ['frames', 'both'] and self.frames_df is not None:
                self.frame_plotter.plot_frame_metric_distribution(self.frames_df, metric_name)
                
        except Exception as e:
            logger.error(f"Error creating individual metric plots: {e}")
    
    def create_scatter_plots(self, metric_x: str, metric_y: str, data_type: str = 'frames'):
        """
        Create scatter plots between two metrics.
        
        Args:
            metric_x: X-axis metric name
            metric_y: Y-axis metric name
            data_type: Type of data ('events' or 'frames')
        """
        logger.info(f"Creating scatter plot: {metric_x} vs {metric_y}")
        
        try:
            if data_type == 'frames' and self.frames_df is not None:
                self.frame_plotter.plot_frame_quality_scatter(self.frames_df, metric_x, metric_y)
            elif data_type == 'events' and self.events_df is not None:
                # Could implement event scatter plots if needed
                logger.warning("Event scatter plots not yet implemented")
                
        except Exception as e:
            logger.error(f"Error creating scatter plots: {e}")