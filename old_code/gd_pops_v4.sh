#!/bin/bash
#SBATCH --job-name="v4_gd_pops"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/v4_gd_pops_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/v4_gd_pops_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --partition=general
# run_experiment.sh
#
# This bash script runs the gd_populations_v4.py experiment
# with a predefined set of parameters.
# Adjust the parameters below as needed.

# Ensure that python3 is installed
command -v python3 >/dev/null 2>&1 || { echo >&2 "python3 is not installed. Exiting."; exit 1; }
# venv
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv 
# Path to the Python script
SCRIPT="gd_populations_v4.py"

PARAMS="--m1 4 \
--m 20 \
--dataset-size 5000 \
--noise-scale 0.1 \
--corr-strength 0 \
--num-epochs 50 \
--reg-type Max_Dev \
--reg-lambda 0.001 \
--learning-rate 0.01 \
--batch-size 5000 \
--optimizer-type adam \
--seed 17 \
--patience 50 \
--alpha-init random_2 \
--estimator-type if \
--base-model-type rf \
--smooth-minmax 100 \
--populations linear_regression \
--run-baseline"

echo "Running gd_populations_v3.py with the following parameters:"
echo "Using the OOF function"
echo $PARAMS

# Execute the experiment
python3 $SCRIPT $PARAMS