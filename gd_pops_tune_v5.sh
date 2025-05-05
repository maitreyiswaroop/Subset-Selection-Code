#!/bin/bash
#SBATCH --job-name="v5_tune_gd_pops"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/v5_tune_gd_pops_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/v5_tune_gd_pops_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --partition=debug
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
SCRIPT="gd_pops_tune_v5.py"

PARAMS="--n-trials 50 \
--study-name gd_pops_v5_tuning \
--storage sqlite:///tuning.db \
--base-save-path ./results_v5_tuning/ \
--m1 4 \
--m 20 \
--dataset-size 5000 \
--noise-scale 0.1 \
--corr-strength 0.0 \
--populations cubic_regression cubic_regression cubic_regression \
--num-epochs 100 \
--batch-size 5000 \
--alpha-init random_1 \
--patience 10 \
--param-freezing \
--use-baseline \
--estimator-type if \
--base-model-type rf \
--objective-value-estimator if \
--seed 42"

echo "Running gd_populations_v3.py with the following parameters:"
echo "Using the OOF function"
echo $PARAMS

# Execute the experiment
python3 $SCRIPT $PARAMS