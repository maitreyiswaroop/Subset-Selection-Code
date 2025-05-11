#!/bin/bash
#SBATCH --job-name="gd_pops_v7"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/gd_v7_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/gd_v7_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=05:00:00
#SBATCH --mem=32G
#SBATCH --partition=general

echo "Loading environment..."
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv   # or your env name

SCRIPT_DIR="/data/user_data/mswaroop/Subset-Selection-Code"
python3 "$SCRIPT_DIR/gd_pops_v7.py" "$@"