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
SCRIPT_NAME="test_uci.py"

SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"
# Define paths for data and plots

python "$SCRIPT_PATH"