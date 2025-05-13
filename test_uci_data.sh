#!/bin/bash
#SBATCH --job-name="explore_uci_adult"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/explore_uci_adult_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/explore_uci_adult_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --partition=general

# --- Environment Setup ---
echo "Setting up environment..."
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv

# --- Script Execution ---
SCRIPT_DIR="/data/user_data/mswaroop/Subset-Selection-Code"
SCRIPT_NAME="data_uci.py"
SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"

# Define paths for data and plots
UCI_CACHE_PATH="$SCRIPT_DIR/data_uci"
PLOT_DIR="$SCRIPT_DIR/data_uci_plots"

# Ensure the cache and plot directories exist
mkdir -p "$UCI_CACHE_PATH"
mkdir -p "$PLOT_DIR"

# Run the script with visualization and exploration parameters
python "$SCRIPT_PATH" \
  --populations Young Middle Senior \
  --target income_binary \
  --categorical_encoding onehot \
  --save_dir "$UCI_CACHE_PATH" \
  --plot \
  --plot_dir "$PLOT_DIR" \
  --subsample_fraction 0.2 \
  --debug

echo "Data exploration completed. Plots saved to $PLOT_DIR"