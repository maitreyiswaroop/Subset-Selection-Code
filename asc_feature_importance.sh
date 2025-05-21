#!/bin/bash
#SBATCH --job-name="acs_features"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/acs_analysis_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/acs_analysis_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --partition=general

# Set up environment
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv

# Paths
SCRIPT_DIR="/data/user_data/mswaroop/Subset-Selection-Code"
OUTPUT_DIR="$SCRIPT_DIR/acs_analysis_output"
DATA_DIR="/data/user_data/mswaroop/Subset-Selection-Code/folktables_data_storage"

# Create dirs
mkdir -p "$OUTPUT_DIR" "$SCRIPT_DIR/logs"

# Run with only states available in STATE_FIPS dictionary
python -c "
import sys
sys.path.append('$SCRIPT_DIR')
from acs_feature_importances import analyze_acs_feature_importance

analyze_acs_feature_importance(
    states=['CA', 'NY', 'FL'],  # Removed TX since it's not in STATE_FIPS
    year=2018,
    target='PINCP',
    output_dir='$OUTPUT_DIR',
    root_dir='$DATA_DIR'
)
"

echo "Analysis complete. Results in $OUTPUT_DIR"