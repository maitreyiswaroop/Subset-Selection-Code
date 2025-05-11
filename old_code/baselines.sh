#!/bin/bash
# run_experiment.sh
#
# This bash script runs the gd_populations_v3.py experiment
# with a predefined set of parameters.
# Adjust the parameters below as needed.

# Ensure that python3 is installed
command -v python3 >/dev/null 2>&1 || { echo >&2 "python3 is not installed. Exiting."; exit 1; }

# Path to the Python script
SCRIPT="baselines.py"

# Default parameters
PARAMS="--m1 4 \
--m 20 \
--dataset-size 1000 \
--noise-scale 0.1 \
--corr-strength 0.5 \
--seed 17 \
--populations cubic_regression cubic_regression cubic_regression \
--save-path ./results/baselines/corr_5"

echo "Running baselines with the following parameters:"
echo $PARAMS

# Execute the experiment
python3 $SCRIPT $PARAMS