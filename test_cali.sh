#!/bin/bash
#SBATCH --job-name="explore_uci_adult"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/explore_cali_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/explore_cali_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --partition=general

# --- Environment Setup ---
echo "Setting up environment..."
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv

# Configuration
OUTPUT_DIR="/data/user_data/mswaroop/Subset-Selection-Code/cali_analysis_output"
SEED=42
PYTHON_CMD="python3"  # Change to "python" if needed for your environment

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

echo "======================================================"
echo "   Running UCI Feature Importance Divergence Analysis"
echo "======================================================"
echo "Output will be saved to: $OUTPUT_DIR"
echo "Using seed: $SEED"
echo "Starting analysis at: $(date)"
echo "======================================================" 

# Run the analysis script
python test_cali.py

echo "======================================================" 
echo "Analysis complete at: $(date)"
echo "Results are available in: $OUTPUT_DIR"
echo "======================================================" 

# Optional: open the output directory (macOS specific)
# open $OUTPUT_DIR

# Optional: Quick summary of results if available
if [ -f "$OUTPUT_DIR/cali_divergence_results.csv" ]; then
    echo "Top results:"
    head -n 5 "$OUTPUT_DIR/cali_divergence_results.csv"
fi