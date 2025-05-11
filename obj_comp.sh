#!/bin/bash
#SBATCH --job-name="obj_compare"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/obj_compare_%j.out  # CHANGE: Specify log directory
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/obj_compare_%j.err   # CHANGE: Specify log directory
#SBATCH --cpus-per-task=1                # Request CPUs
#SBATCH --time=05:00:00                      # Time limit (e.g., 1 hour)
#SBATCH --mem=64G                          # Memory requirement
#SBATCH --partition=general                  # CHANGE: Specify your cluster partition

# --- Environment Setup ---
echo "Setting up environment..."
# Activate your conda environment (adjust path if needed)
source /home/$USER/miniconda/etc/profile.d/conda.sh # CHANGE: Specify conda path
conda activate venv                       # CHANGE: Specify your environment name

# Or load modules if not using conda
# module load python/3.9 cuda/11.x etc...         # CHANGE: Adapt to your cluster modules

# --- Script Execution ---
SCRIPT_DIR="/data/user_data/mswaroop/Subset-Selection-Code" # CHANGE: Set path to your script directory
SCRIPT_NAME="objective_comparison.py"
SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"

# --- Parameters for the Objective Comparison ---
# Adjust sample sizes, alpha values, MC samples, etc.
PARAMS="--m 10 \
--A-scale 1.5 \
--y-noise-scale 0.1 \
--n-base-t1 50000 \
--sample-sizes 10000 5000 1000 \
--alpha-values 0.01 0.05 0.1 0.5 1.0 2.0 5.0 10.0 \
--n-mc-samples-obj 30 \
--base-model-type xgb \
--penalty-lambda 0.01 \
--penalty-types Reciprocal_L1 Quadratic_Barrier Exponential Max_Dev \
--seeds 42 123 456 \
--save-dir $SCRIPT_DIR/results_compare_objective/linear_run_$(date +%Y%m%d_%H%M%S)/"

echo "----------------------------------------"
echo "Running $SCRIPT_NAME with parameters:"
echo "$PARAMS" | sed 's/\\//g' # Print params without backslashes
echo "----------------------------------------"

# Execute the script
python3 "$SCRIPT_PATH" $PARAMS

echo "----------------------------------------"
echo "Objective comparison finished."
echo "----------------------------------------"
