#!/bin/bash
#SBATCH --job-name="test_acs_regression_data"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/test_acs_regression_data_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/test_acs_regression_data_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00 # Increased time slightly for potential first download
#SBATCH --mem=32G
#SBATCH --partition=general

# --- Environment Setup ---
echo "Setting up environment..."
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv      

echo "Ensuring folktables is installed..."
pip install folktables

# --- Script Execution ---
SCRIPT_DIR="/data/user_data/mswaroop/Subset-Selection-Code"
SCRIPT_NAME="data_regression_loader.py"
SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"

# Define a specific, writable cache directory for folktables data
# This path will be absolute or resolved to absolute by the Python script
FOLKTABLES_CACHE_PATH="$SCRIPT_DIR/folktables_data_storage" # Changed name for clarity

# --- Parameters for ACS Income Regression ---
DATASET_CHOICE="acs_income_reg"
ACS_STATES_CHOICE="CA,NY,FL"
ACS_YEAR_CHOICE="2018"        
DOMAIN_COLUMN_CHOICE="STATE_DOMAIN" 
DOMAIN_METHOD_CHOICE="categorical"  
PREPROCESSING_CHOICE="onehot"       
NUM_FEATURES_TO_PLOT=10             
MIN_SAMPLES_PER_DOMAIN_CHOICE=500   
NUM_BINS_CHOICE=3 # Default, not used by categorical domain method

# Add echo statements for debugging paths in SLURM
echo "SLURM: SCRIPT_DIR is $SCRIPT_DIR"
echo "SLURM: Python script to run is $SCRIPT_PATH"
echo "SLURM: Folktables cache path set to $FOLKTABLES_CACHE_PATH"

# Ensure the cache directory is created by the SLURM job user before Python script tries
mkdir -p "$FOLKTABLES_CACHE_PATH" 

PARAMS="--dataset $DATASET_CHOICE \
--acs_states $ACS_STATES_CHOICE \
--acs_year $ACS_YEAR_CHOICE \
--folktables_data_dir $FOLKTABLES_CACHE_PATH \
--domain_column $DOMAIN_COLUMN_CHOICE \
--domain_method $DOMAIN_METHOD_CHOICE \
--num_bins $NUM_BINS_CHOICE \
--min_samples_domain $MIN_SAMPLES_PER_DOMAIN_CHOICE \
--preprocessing $PREPROCESSING_CHOICE \
--verbose \
--n_features_plot $NUM_FEATURES_TO_PLOT \
--out_dir $SCRIPT_DIR/data_regression_plots/"

echo "----------------------------------------"
echo "Running $SCRIPT_NAME with parameters for $DATASET_CHOICE:"
echo "$PARAMS" | sed 's/\\//g'
echo "----------------------------------------"

python3 "$SCRIPT_PATH" $PARAMS

echo "----------------------------------------"
echo "ACS Regression data loading and plotting finished."
echo "----------------------------------------"