#!/bin/bash
#SBATCH --job-name="test_acs_regression_data"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/test_acs_regression_data_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/test_acs_regression_data_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00 # Increased time slightly for potential first download
#SBATCH --mem=64G
#SBATCH --partition=general

# --- Environment Setup ---
echo "Setting up environment..."
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv      

echo "Ensuring folktables is installed..."
# pip install folktables

export REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-bundle.crt"
echo "SLURM: Attempting to use REQUESTS_CA_BUNDLE=$REQUESTS_CA_BUNDLE"

# Optional: Check if this file actually exists on your system
if [ -f "$REQUESTS_CA_BUNDLE" ]; then
    echo "SLURM: CA Bundle file at $REQUESTS_CA_BUNDLE exists."
else
    echo "SLURM: WARNING - CA Bundle file at $REQUESTS_CA_BUNDLE does NOT exist on this node!"
    echo "SLURM: The folktables download will likely still fail if the path is incorrect."
    # As a fallback, you could try the other common name, though the one above is more promising given the context
    # export REQUESTS_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
    # echo "SLURM: Trying fallback REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt"
fi

# --- Script Execution ---
SCRIPT_DIR="/data/user_data/mswaroop/Subset-Selection-Code"
SCRIPT_NAME="data_acs.py"
SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"

# Define a specific, writable cache directory for folktables data
# This path will be absolute or resolved to absolute by the Python script
FOLKTABLES_CACHE_PATH="$SCRIPT_DIR/folktables_data_storage"

# Parameters for ACS data generation and plotting
ACS_STATES_CHOICE="CA NY FL"
FOLK_SURVEY="person"
FOLK_HORIZON="1-Year"
TARGET_COL="PINCP"
# FEATURES_TO_PLOT="AGEP WKHP"
PLOT_DIR="$SCRIPT_DIR/data_acs_plots"
YEAR="2018"

# Debug info
echo "SLURM: SCRIPT_DIR is $SCRIPT_DIR"
echo "SLURM: Python script to run is $SCRIPT_PATH"
echo "SLURM: Folktables cache path set to $FOLKTABLES_CACHE_PATH"

# Ensure the cache and plot directories exist
mkdir -p "$FOLKTABLES_CACHE_PATH"
mkdir -p "$PLOT_DIR"

PARAMS="--states $ACS_STATES_CHOICE \
    --year $YEAR \
    --survey $FOLK_SURVEY \
    --horizon $FOLK_HORIZON \
    --root_dir $FOLKTABLES_CACHE_PATH \
    --target $TARGET_COL \
    --plot \
    --plot_dir $PLOT_DIR"

echo "----------------------------------------"
echo "Running data_acs.py with parameters:"
echo "$PARAMS"
echo "----------------------------------------"

python3 "$SCRIPT_PATH" $PARAMS

echo "----------------------------------------"
echo "ACS data generation, preprocessing, and plotting finished."
echo "Plots saved to $PLOT_DIR"
echo "----------------------------------------"