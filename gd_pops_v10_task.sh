#!/bin/bash
#SBATCH --job-name="v10"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/v10_task_gd_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/v10_task_gd_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1 
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
echo "Loading environment..."
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv   # or your env name


export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

SCRIPT_DIR="/data/user_data/mswaroop/Subset-Selection-Code"
python3 "$SCRIPT_DIR/gd_pops_v10.py" "$@"