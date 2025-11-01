#!/bin/bash
#SBATCH --job-name=analyze_overall_results
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH -A plgdyplomanci6-gpu-a100
#SBATCH --partition=plgrid-gpu-a100


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

python /net/tscratch/people/plgaidankst/quality_verification_v2/scripts/analyze_overall_results.py /net/tscratch/people/plgaidankst/quality_verification_v2/scripts/outputs/Town05

conda deactivate