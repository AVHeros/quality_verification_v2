#!/usr/bin/env python3
"""
Script to automatically generate SLURM job scripts for all weather/route combinations
in the Town05 dataset.

This script:
1. Scans the input dataset directory for weather directories
2. Finds all route directories within each weather directory
3. Generates two SLURM scripts per route (events and frames)
4. Organizes scripts and outputs in a structured hierarchy
5. Creates a main submission script to run all jobs
"""

import os
import glob
import argparse
from pathlib import Path


def create_slurm_script(template_type, weather_dir, route_name, route_path, output_base_dir, script_output_dir):
    """
    Create a SLURM script based on the template type (events or frames).
    
    Args:
        template_type: 'events' or 'frames'
        weather_dir: weather directory name (e.g., 'weather-0')
        route_name: route directory name
        route_path: full path to the route directory
        output_base_dir: base directory for outputs
        script_output_dir: directory where the script will be saved
    """
    
    # Define job names and CLI commands
    if template_type == 'events':
        job_name = f"rgb_vs_dvs_events_{weather_dir}_{route_name}"
        cli_command = "rgb-vs-dvs-events"
        frame_rate_arg = " --frame-rate 10"
        script_name = "run_rgb_vs_dvs_events.sh"
    else:  # frames
        job_name = f"rgb_vs_dvs_frames_{weather_dir}_{route_name}"
        cli_command = "rgb-vs-dvs-frames"
        frame_rate_arg = ""
        script_name = "run_rgb_vs_dvs_frames.sh"
    
    # Create output directory path
    output_folder = f"{output_base_dir}/Town05/{weather_dir}/{route_name}/{cli_command.replace('-', '_')}"
    
    # Create the script content
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH -A plgdyplomanci6-gpu-a100
#SBATCH --partition=plgrid-gpu-a100

# Load CUDA module
module load CUDA/12.8.0

# Find and source the main conda.sh script to initialize conda
__conda_setup="$('/net/tscratch/people/plgaidankst/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/net/tscratch/people/plgaidankst/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/net/tscratch/people/plgaidankst/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/net/tscratch/people/plgaidankst/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

# Now, activate your specific environment
conda activate quality_verification

# Add the src directory to PYTHONPATH so the quality_verification module can be found
export PYTHONPATH="/net/tscratch/people/plgaidankst/quality_verification_v2/src:$PYTHONPATH"

python /net/tscratch/people/plgaidankst/quality_verification_v2/src/quality_verification/cli.py {cli_command} --root {route_path}{frame_rate_arg} --output-folder {output_folder} --device cuda

conda deactivate
"""
    
    # Write the script file
    script_path = os.path.join(script_output_dir, script_name)
    os.makedirs(script_output_dir, exist_ok=True)
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    return script_path


def scan_dataset_directory(dataset_root):
    """
    Scan the dataset directory to find all weather directories and their routes.
    
    Args:
        dataset_root: Root directory of the dataset (e.g., .../Town05/)
    
    Returns:
        Dictionary mapping weather directories to lists of route paths
    """
    weather_routes = {}
    
    # Find all weather directories
    weather_pattern = os.path.join(dataset_root, "weather-*")
    weather_dirs = glob.glob(weather_pattern)
    
    for weather_dir in sorted(weather_dirs):
        weather_name = os.path.basename(weather_dir)
        data_dir = os.path.join(weather_dir, "data")
        
        if not os.path.exists(data_dir):
            print(f"Warning: No data directory found in {weather_dir}")
            continue
        
        # Find all route directories
        route_pattern = os.path.join(data_dir, "routes_town05_*")
        route_dirs = glob.glob(route_pattern)
        
        if route_dirs:
            weather_routes[weather_name] = sorted(route_dirs)
            print(f"Found {len(route_dirs)} routes in {weather_name}")
        else:
            print(f"Warning: No route directories found in {data_dir}")
    
    return weather_routes


def generate_all_scripts(dataset_root, scripts_base_dir, outputs_base_dir):
    """
    Generate all SLURM scripts for the dataset.
    
    Args:
        dataset_root: Root directory of the dataset
        scripts_base_dir: Base directory for generated scripts
        outputs_base_dir: Base directory for outputs
    
    Returns:
        List of all generated script paths
    """
    weather_routes = scan_dataset_directory(dataset_root)
    all_scripts = []
    
    for weather_name, route_paths in weather_routes.items():
        print(f"\nProcessing {weather_name}...")
        
        for route_path in route_paths:
            route_name = os.path.basename(route_path)
            print(f"  Creating scripts for {route_name}")
            
            # Create directory for this route's scripts
            script_dir = os.path.join(scripts_base_dir, "Town05", weather_name, route_name)
            
            # Generate both event and frame scripts
            for script_type in ['events', 'frames']:
                script_path = create_slurm_script(
                    script_type, weather_name, route_name, route_path,
                    outputs_base_dir, script_dir
                )
                all_scripts.append(script_path)
                print(f"    Created: {script_path}")
    
    return all_scripts


def create_main_submission_script(all_scripts, scripts_base_dir):
    """
    Create a main bash script that submits all generated SLURM jobs.
    
    Args:
        all_scripts: List of all generated script paths
        scripts_base_dir: Base directory for scripts
    """
    main_script_path = os.path.join(scripts_base_dir, "submit_all_jobs.sh")
    
    script_content = """#!/bin/bash
# Main script to submit all generated SLURM jobs
# Generated automatically by generate_all_scripts.py

echo "Submitting all SLURM jobs for Town05 dataset processing..."
echo "Total scripts to submit: """ + str(len(all_scripts)) + """"

"""
    
    # Add submission commands for each script
    for i, script_path in enumerate(all_scripts, 1):
        # Make path relative to the main script location
        rel_path = os.path.relpath(script_path, scripts_base_dir)
        script_content += f"""
echo "Submitting job {i}/{len(all_scripts)}: {rel_path}"
sbatch "{rel_path}"
sleep 1  # Small delay to avoid overwhelming the scheduler
"""
    
    script_content += """
echo "All jobs submitted successfully!"
echo "Use 'squeue -u $USER' to check job status"
"""
    
    # Write the main script
    with open(main_script_path, 'w') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(main_script_path, 0o755)
    
    print(f"\nMain submission script created: {main_script_path}")
    return main_script_path


def create_logs_directory(scripts_base_dir):
    """Create logs directory for SLURM output files."""
    logs_dir = os.path.join(scripts_base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    print(f"Created logs directory: {logs_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate SLURM scripts for Town05 dataset processing")
    parser.add_argument(
        "--dataset-root",
        default="/net/pr2/projects/plgrid/plggwie/plgminkhant/Interfuser_DVS/InterFuser/dataset/Town05/",
        help="Root directory of the Town05 dataset"
    )
    parser.add_argument(
        "--scripts-dir",
        default="/net/tscratch/people/plgaidankst/quality_verification_v2/scripts",
        help="Base directory for generated scripts"
    )
    parser.add_argument(
        "--outputs-dir",
        default="/net/tscratch/people/plgaidankst/quality_verification_v2/scripts/outputs",
        help="Base directory for outputs"
    )
    
    args = parser.parse_args()
    
    print("Town05 Dataset SLURM Script Generator")
    print("=" * 50)
    print(f"Dataset root: {args.dataset_root}")
    print(f"Scripts directory: {args.scripts_dir}")
    print(f"Outputs directory: {args.outputs_dir}")
    print()
    
    # Check if dataset root exists
    if not os.path.exists(args.dataset_root):
        print(f"Error: Dataset root directory does not exist: {args.dataset_root}")
        return 1
    
    # Create logs directory
    create_logs_directory(args.scripts_dir)
    
    # Generate all scripts
    all_scripts = generate_all_scripts(args.dataset_root, args.scripts_dir, args.outputs_dir)
    
    if not all_scripts:
        print("No scripts were generated. Please check the dataset directory structure.")
        return 1
    
    # Create main submission script
    main_script = create_main_submission_script(all_scripts, args.scripts_dir)
    
    print(f"\nGeneration complete!")
    print(f"Generated {len(all_scripts)} SLURM scripts")
    print(f"Main submission script: {main_script}")
    print(f"\nTo submit all jobs, run:")
    print(f"cd {args.scripts_dir} && ./submit_all_jobs.sh")
    
    return 0


if __name__ == "__main__":
    exit(main())