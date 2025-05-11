#!/bin/bash
# run_experiment.sh
#
# This bash script runs the gd_populations_v3.py experiment
# with a predefined set of parameters.
# Adjust the parameters below as needed.

# Ensure that python3 is installed
command -v python3 >/dev/null 2>&1 || { echo >&2 "python3 is not installed. Exiting."; exit 1; }

# Path to the Python script
SCRIPT="gd_populations_v3.py"

PARAMS="--m1 4 \
--m 20 \
--dataset-size 1000 \
--noise-scale 0.1 \
--corr-strength 0 \
--num-epochs 200 \
--reg-type Reciprocal_L1 \
--reg-lambda 0.05 \
--learning-rate 0.01 \
--batch-size 1000 \
--optimizer-type sgd \
--seed 17 \
--patience 10 \
--alpha-init random_5 \
--estimator-type plugin \
--base-model-type rf \
--populations resnet cubic_regression cubic_regression \
--param-freezing"

echo "Running gd_populations_v3.py with the following parameters:"
echo "Using the OOF function"
echo $PARAMS

# Execute the experiment
python3 $SCRIPT $PARAMS