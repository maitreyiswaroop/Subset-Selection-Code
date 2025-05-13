#!/bin/bash
#SBATCH --job-name="EYS_est"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/term2_est_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/term2_est_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --partition=debug


# Ensure that python3 is installed
command -v python3 >/dev/null 2>&1 || { echo >&2 "python3 is not installed. Exiting."; exit 1; }

# venv
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv 

# Path to the Python script
SCRIPT="E_Y_S_estimation.py"

PARAMS="--n-small 1000 \
    --n-large 50000 \
    --m 20 \
    --seeds 42 123 987 \
    --alphas 0.01 0.1 0.5 1.0 2.0 5.0 \
    --save-dir ./comparison_run1/"

echo "Running gd_populations_v3.py with the following parameters:"
echo "Using the OOF function"
echo $PARAMS

# Execute the experiment
python3 $SCRIPT $PARAMS