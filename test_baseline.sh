#!/bin/bash
#SBATCH --job-name=test_baseline
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/test_baselines/test_baseline_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/test_baselines/test_baseline_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --time=03:30:00
#SBATCH --mem=64G
#SBATCH --partition=general

set -e

echo "Loading conda..."
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv

echo "Running baseline test script..."
python3 /data/user_data/mswaroop/Subset-Selection-Code/test_baseline.py \
    --corr-strength 0.5 \
    --save-dir /data/user_data/mswaroop/Subset-Selection-Code/results/baseline/

echo "Baseline test complete."