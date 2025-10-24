#!/usr/bin/env python3
"""
Data collection module for DVS Quality Verification results.

This module handles traversing directory structures, loading report data,
and collecting metrics from various sources.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class DVSDataCollector:
    """Handles data collection from DVS quality verification results."""
    
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.events_data = []
        self.frames_data = []
        self.skipped_dirs = []
        
        # Metrics categorization
        self.event_metrics = {
            'event_density', 'event_rate', 'temporal_precision_us_std',
            'on_ratio', 'off_ratio', 'polarity_accuracy', 
            'event_edge_correlation', 'brightness_delta'
        }
        
        self.frame_metrics = {
            'mse', 'psnr', 'ssim', 'lpips', 'mean_intensity_diff',
            'rgb_mean', 'dvs_mean', 'contrast_ratio', 'rgb_std', 'dvs_std'
        }
    
    def parse_route_info(self, route_name: str) -> Dict[str, str]:
        """Extract route information from directory name."""
        # Pattern: routes_town05_tiny_w0_10_07_16_14_12
        pattern = r'routes_(\w+)_(\w+)_w(\d+)_(.+)'
        match = re.match(pattern, route_name)
        
        if match:
            return {
                'town': match.group(1),
                'route_type': match.group(2),  # tiny, short, long
                'weather': int(match.group(3)),
                'timestamp': match.group(4)
            }
        else:
            logger.warning(f"Could not parse route name: {route_name}")
            return {
                'town': 'unknown',
                'route_type': 'unknown',
                'weather': -1,
                'timestamp': 'unknown'
            }
    
    def load_report_data(self, report_path: Path) -> Optional[Dict]:
        """Load and validate report JSON data."""
        try:
            with open(report_path, 'r') as f:
                data = json.load(f)
            
            # Validate required fields
            if 'metrics_summary' not in data:
                logger.warning(f"No metrics_summary in {report_path}")
                return None
                
            return data
        except Exception as e:
            logger.error(f"Error loading {report_path}: {e}")
            return None
    
    def collect_all_data(self):
        """Traverse directory structure and collect all metrics data."""
        logger.info(f"Starting data collection from {self.root_dir}")
        
        weather_dirs = [d for d in self.root_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('weather-')]
        
        logger.info(f"Found {len(weather_dirs)} weather directories")
        
        for weather_dir in sorted(weather_dirs):
            weather_num = int(weather_dir.name.split('-')[1])
            logger.info(f"Processing {weather_dir.name}")
            
            route_dirs = [d for d in weather_dir.iterdir() 
                         if d.is_dir() and d.name.startswith('routes_')]
            
            for route_dir in route_dirs:
                route_info = self.parse_route_info(route_dir.name)
                route_info['weather'] = weather_num
                
                # Process events data
                events_report = route_dir / 'rgb_vs_dvs_events' / 'report.json'
                if events_report.exists():
                    events_data = self.load_report_data(events_report)
                    if events_data:
                        self._process_events_data(events_data, route_info, route_dir)
                    else:
                        self.skipped_dirs.append(str(events_report.parent))
                else:
                    self.skipped_dirs.append(str(route_dir / 'rgb_vs_dvs_events'))
                
                # Process frames data
                frames_report = route_dir / 'rgb_vs_dvs_frames' / 'report.json'
                if frames_report.exists():
                    frames_data = self.load_report_data(frames_report)
                    if frames_data:
                        self._process_frames_data(frames_data, route_info, route_dir)
                    else:
                        self.skipped_dirs.append(str(frames_report.parent))
                else:
                    self.skipped_dirs.append(str(route_dir / 'rgb_vs_dvs_frames'))
        
        logger.info(f"Data collection complete. Events: {len(self.events_data)}, "
                   f"Frames: {len(self.frames_data)}, Skipped: {len(self.skipped_dirs)}")
    
    def _process_events_data(self, data: Dict, route_info: Dict, route_dir: Path):
        """Process events metrics data."""
        metrics_summary = data.get('metrics_summary', {})
        
        record = {
            'route_dir': str(route_dir),
            'route_name': route_dir.name,
            'weather': route_info['weather'],
            'route_type': route_info['route_type'],
            'town': route_info['town'],
            'timestamp': route_info['timestamp'],
            'frame_count': data.get('frame_count', 0),
            'frame_rate': data.get('frame_rate', 0),
            'device': data.get('device', 'unknown')
        }
        
        # Add all metrics
        for metric_name, metric_data in metrics_summary.items():
            if isinstance(metric_data, dict):
                record[f'{metric_name}_mean'] = metric_data.get('mean', float('nan'))
                record[f'{metric_name}_std'] = metric_data.get('std', float('nan'))
                record[f'{metric_name}_min'] = metric_data.get('min', float('nan'))
                record[f'{metric_name}_max'] = metric_data.get('max', float('nan'))
                record[f'{metric_name}_count'] = metric_data.get('count', 0)
        
        self.events_data.append(record)
    
    def _process_frames_data(self, data: Dict, route_info: Dict, route_dir: Path):
        """Process frames metrics data."""
        metrics_summary = data.get('metrics_summary', {})
        
        record = {
            'route_dir': str(route_dir),
            'route_name': route_dir.name,
            'weather': route_info['weather'],
            'route_type': route_info['route_type'],
            'town': route_info['town'],
            'timestamp': route_info['timestamp'],
            'pair_count': data.get('pair_count', 0),
            'device': data.get('device', 'unknown')
        }
        
        # Add all metrics
        for metric_name, metric_data in metrics_summary.items():
            if isinstance(metric_data, dict):
                record[f'{metric_name}_mean'] = metric_data.get('mean', float('nan'))
                record[f'{metric_name}_std'] = metric_data.get('std', float('nan'))
                record[f'{metric_name}_min'] = metric_data.get('min', float('nan'))
                record[f'{metric_name}_max'] = metric_data.get('max', float('nan'))
                record[f'{metric_name}_count'] = metric_data.get('count', 0)
        
        self.frames_data.append(record)
    
    def get_collected_data(self) -> tuple[List[Dict], List[Dict], List[str]]:
        """Return collected data."""
        return self.events_data, self.frames_data, self.skipped_dirs
    
    def clear_data(self):
        """Clear collected data."""
        self.events_data.clear()
        self.frames_data.clear()
        self.skipped_dirs.clear()