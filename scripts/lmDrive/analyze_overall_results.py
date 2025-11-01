#!/usr/bin/env python3
"""
DVS Quality Verification Analysis Script

This script provides a clean interface to the comprehensive DVS quality verification
analysis pipeline, utilizing the modular architecture for better maintainability
and reusability.

Usage:
    python analyze_overall_results.py <root_dir> [--output-dir <output_dir>]

Example:
    python analyze_overall_results.py /path/to/outputs/Town05 --output-dir ./analysis_results
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from quality_verification.analysis.core_analyzer import DVSCoreAnalyzer
except ImportError as e:
    print(f"Error importing DVSCoreAnalyzer: {e}")
    print("Please ensure the quality_verification package is properly installed.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dvs_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def validate_input_directory(root_dir: Path) -> bool:
    """
    Validate that the input directory exists and contains expected structure.
    
    Args:
        root_dir: Path to the root directory containing DVS results
        
    Returns:
        bool: True if directory is valid, False otherwise
    """
    if not root_dir.exists():
        logger.error(f"Root directory does not exist: {root_dir}")
        return False
    
    if not root_dir.is_dir():
        logger.error(f"Root path is not a directory: {root_dir}")
        return False
    
    # Check for expected weather directories
    weather_dirs = [d for d in root_dir.iterdir() 
                   if d.is_dir() and d.name.startswith('weather-')]
    
    if not weather_dirs:
        logger.warning(f"No weather directories found in {root_dir}")
        logger.warning("Expected directories with pattern 'weather-*'")
        return False
    
    logger.info(f"Found {len(weather_dirs)} weather directories in {root_dir}")
    return True


def run_analysis(root_dir: Path, output_dir: Path) -> bool:
    """
    Run the complete DVS quality verification analysis.
    
    Args:
        root_dir: Root directory containing DVS results
        output_dir: Directory where analysis outputs will be saved
        
    Returns:
        bool: True if analysis completed successfully, False otherwise
    """
    logger.info("=" * 60)
    logger.info("DVS QUALITY VERIFICATION ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Input directory: {root_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)
    
    try:
        # Create the core analyzer
        analyzer = DVSCoreAnalyzer(root_dir, output_dir)
        
        # Run the complete analysis pipeline
        success = analyzer.run_complete_analysis()
        
        if success:
            # Get and display analysis summary
            summary = analyzer.get_analysis_summary()
            
            logger.info("=" * 60)
            logger.info("ANALYSIS COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info("SUMMARY:")
            logger.info(f"  Events sequences processed: {summary.get('events_count', 0)}")
            logger.info(f"  Frames sequences processed: {summary.get('frames_count', 0)}")
            logger.info(f"  Total metrics computed: {summary.get('total_metrics', 0)}")
            logger.info(f"  Visualizations created: {summary.get('plots_created', 0)}")
            logger.info(f"  Results saved to: {output_dir}")
            logger.info("=" * 60)
            
            return True
        else:
            logger.error("Analysis pipeline failed. Check logs for details.")
            return False
            
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}")
        logger.exception("Full traceback:")
        return False


def main():
    """Main function to run the DVS quality verification analysis."""
    parser = argparse.ArgumentParser(
        description="DVS Quality Verification Results Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/outputs/Town05
  %(prog)s /path/to/outputs/Town05 --output-dir ./my_analysis
  %(prog)s /path/to/outputs/Town05 --output-dir /tmp/analysis --verbose
        """
    )
    
    parser.add_argument(
        'root_dir',
        type=Path,
        help='Root directory containing DVS quality verification results '
             '(should contain weather-* subdirectories)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./analysis_results'),
        help='Output directory for analysis results (default: ./analysis_results)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    parser.add_argument(
        '--log-file',
        type=Path,
        default=Path('dvs_analysis.log'),
        help='Log file path (default: dvs_analysis.log)'
    )
    
    args = parser.parse_args()
    
    # Update logging level if verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    
    # Update log file if specified
    if args.log_file != Path('dvs_analysis.log'):
        # Remove existing file handler and add new one
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                root_logger.removeHandler(handler)
        
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {args.log_file}")
    
    # Validate input directory
    if not validate_input_directory(args.root_dir):
        logger.error("Input directory validation failed")
        sys.exit(1)
    
    # Create output directory
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory created/verified: {args.output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory {args.output_dir}: {e}")
        sys.exit(1)
    
    # Run the analysis
    success = run_analysis(args.root_dir, args.output_dir)
    
    if success:
        logger.info("Analysis completed successfully!")
        sys.exit(0)
    else:
        logger.error("Analysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()