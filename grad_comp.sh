#!/bin/bash
#SBATCH --job-name="grad_compare"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/grad_compare_%j.out  # CHANGE: Specify log directory
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/grad_compare_%j.err   # CHANGE: Specify log directory
#SBATCH --cpus-per-task=2                  # Request CPUs
# SBATCH --gres=gpu:1                       # Request GPU (if needed)
#SBATCH --time=05:00:00                      # Time limit (e.g., 2 hours, may need more)
#SBATCH --mem=32G                          # Memory requirement (may need more for variance runs)
#SBATCH --partition=general                  # CHANGE: Specify your cluster partition

# --- Environment Setup ---
echo "Setting up environment..."
# Activate your conda environment (adjust path if needed)
source /home/$USER/miniconda/etc/profile.d/conda.sh # CHANGE: Specify conda path
conda activate venv      
# Or load modules if not using conda
# module load python/3.9 cuda/11.x etc...         # CHANGE: Adapt to your cluster modules

# --- Script Execution ---
SCRIPT_DIR="/data/user_data/mswaroop/Subset-Selection-Code" # CHANGE: Set path to your script directory
SCRIPT_NAME="gradient_comparison.py"
SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"

# --- Parameters for the Gradient Comparison ---
# Adjust sample sizes, alpha values, MC samples, variance runs etc.
PARAMS="--m 10 \
--A-scale 1.5 \
--y-noise-scale 0.1 \
--n-base-eyx 20000 \
--sample-sizes 1000 5000 \
--alpha-values 0.01 0.1 1.0 5.0 \
--n-grad-samples-list 10 25 50 \
--base-model-type xgb \
--n-variance-runs 15 \
--penalty-type Reciprocal_L1 \
--penalty-lambda 0.01 \
--seeds 42 123 \
--save-dir $SCRIPT_DIR/results_compare_gradient/linear_run_$(date +%Y%m%d_%H%M%S)/"
# Add --no-use-baseline to disable baseline for REINFORCE

echo "----------------------------------------"
echo "Running $SCRIPT_NAME with parameters:"
echo "$PARAMS" | sed 's/\\//g' # Print params without backslashes
echo "----------------------------------------"

# Execute the script
python3 "$SCRIPT_PATH" $PARAMS

echo "----------------------------------------"
echo "Gradient comparison finished."
echo "----------------------------------------"

