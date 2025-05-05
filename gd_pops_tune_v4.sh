#!/bin/bash
#SBATCH --job-name="v4_tune_gd_pops"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/v4_tune_gd_pops_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/v4_tune_gd_pops_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --partition=general
# run_experiment.sh
#
# This bash script runs the gd_populations_v3.py experiment
# with a predefined set of parameters.
# Adjust the parameters below as needed.

# Ensure that python3 is installed
command -v python3 >/dev/null 2>&1 || { echo >&2 "python3 is not installed. Exiting."; exit 1; }

# venv
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv 

# Path to the Python script
SCRIPT="gd_pops_tune_v4.py"

PARAMS="--n-trials 100 --study-name my_vss_tuning --storage sqlite:///my_study.db"

echo "Running gd_pops_tune_v4.py with the following parameters:"
echo "Using the OOF function"
echo "$PARAMS"

# Execute the experiment
python3 $SCRIPT $PARAMS