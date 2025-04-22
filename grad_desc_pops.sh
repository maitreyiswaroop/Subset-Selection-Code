#!/bin/bash
# run_experiment.sh
#
# This bash script runs the grad_desc_populations_v3.py experiment
# with a predefined set of parameters.
# Adjust the parameters below as needed.

# Ensure that python3 is installed
command -v python3 >/dev/null 2>&1 || { echo >&2 "python3 is not installed. Exiting."; exit 1; }

# Path to the Python script
SCRIPT="grad_desc_populations_v1.py"

PARAMS="--m1 4 \
--m 20 \
--dataset-size 1000 \
--noise-scale 0.1 \
--num-epochs 1000 \
--reg-type Quadratic_Barrier \
--reg-lambda 0.0352965711480333 \
--learning-rate 1.0 \
--batch-size 256 \
--optimizer-type sgd \
--seed 17 \
--patience 10 \
--alpha-init random_5 \
--estimator-type plugin \
--base-model-type rf \
--populations resnet resnet resnet \
--param-freezing \
--scheduler warmup_cosine \
--warmup-epochs 100 \
--scheduler-min-lr 0.0001"

echo "Running grad_desc_populations_v3.py with the following parameters:"
echo $PARAMS

# Execute the experiment
python3 $SCRIPT $PARAMS