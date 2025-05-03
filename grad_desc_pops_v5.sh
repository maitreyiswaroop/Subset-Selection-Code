#!/bin/bash
# run_experiment.sh
#
# This bash script runs the grad_desc_populations_v3.py experiment
# with a predefined set of parameters.
# Adjust the parameters below as needed.

# Ensure that python3 is installed
command -v python3 >/dev/null 2>&1 || { echo >&2 "python3 is not installed. Exiting."; exit 1; }

# Path to the Python script
SCRIPT="grad_desc_populations_v5.py"

PARAMS="--m1 4 \
--m 20 \
--dataset-size 5000 \
--noise-scale 0.1 \
--corr-strength 0 \
--populations cubic_regression cubic_regression cubic_regression \
--num-epochs 100 \
--penalty-type Reciprocal_L1 \
--penalty-lambda 0.001 \
--learning-rate 0.1 \
--batch-size 5000 \
--optimizer-type adam \
--alpha-init random_2 \
--seed 17 \
--patience 30 \
--estimator-type if \
--base-model-type rf \
--param-freezing"

echo "Running grad_desc_populations_v3.py with the following parameters:"
echo "Using the OOF function"
echo $PARAMS

# Execute the experiment
python3 $SCRIPT $PARAMS