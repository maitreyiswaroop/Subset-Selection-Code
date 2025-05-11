#!/bin/bash
#SBATCH --job-name="optuna_v6"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/v6_tune_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/v6_tune_%j.err
#SBATCH --gres=gpu:1                         # Request 1 GPU per trial (adjust if using n_jobs > 1)
#SBATCH --cpus-per-task=4                  # More CPUs might help if base models use them (RF/XGB)
#SBATCH --time=10:00:00                     # Adjust time limit for total tuning duration
#SBATCH --mem=32G                          # Adjust memory based on single trial needs
#SBATCH --partition=general                  # CHANGE: Specify your cluster partition

# --- Environment Setup ---
echo "Setting up environment..."
# Activate your conda environment
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv 

# --- Script Execution ---
SCRIPT_DIR="/data/user_data/mswaroop/Subset-Selection-Code" # Root directory
SCRIPT_NAME="gd_pops_tune_v6.py" # The Optuna tuning script
SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"

# --- Parameters for the Optuna Study ---
STUDY_NAME="gd_pops_v6_theta_autograd_study_$(date +%Y%m%d)"
# Use a persistent database file for resuming/analysis
STORAGE_DB="sqlite:///$SCRIPT_DIR/optuna_studies/tuning_v6.db"
# Ensure the directory for the database exists
mkdir -p "$SCRIPT_DIR/optuna_studies"

# --- Fixed Parameters for the underlying experiment ---
# These are passed to the tuning script, which passes them to gd_pops_v6.py
FIXED_EXP_PARAMS="--m1 4 \
--m 20 \
--dataset-size 5000 \
--noise-scale 0.1 \
--corr-strength 0.0 \
--populations linear_regression linear_regression linear_regression \
--num-epochs 100 \
--patience 15 \
--seed 123 \
--base-save-path $SCRIPT_DIR/results_v6_tuning/$STUDY_NAME/ \
--objective-value-estimator mc"
# Add other fixed params like --param-freezing if needed

# --- Optuna Parameters ---
OPTUNA_PARAMS="--n-trials 100 \
--study-name $STUDY_NAME \
--storage $STORAGE_DB"

echo "----------------------------------------"
echo "Starting Optuna Study: $STUDY_NAME"
echo "Running $SCRIPT_NAME"
echo "Optuna Args: $OPTUNA_PARAMS"
echo "Fixed Experiment Args: $FIXED_EXP_PARAMS"
echo "----------------------------------------"

# Execute the Optuna script
python3 "$SCRIPT_PATH" \
    $OPTUNA_PARAMS \
    $FIXED_EXP_PARAMS

echo "----------------------------------------"
echo "Optuna tuning finished for study: $STUDY_NAME"
echo "----------------------------------------"
