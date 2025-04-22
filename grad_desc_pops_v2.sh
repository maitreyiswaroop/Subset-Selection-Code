#!/bin/bash
# run_experiment.sh
#
# This bash script runs the python grad_desc_populations_v2.py experiment
# with a predefined set of parameters.
# Adjust the parameters below as needed.

# Ensure that python3 is installed
command -v python3 >/dev/null 2>&1 || { echo >&2 "python3 is not installed. Exiting."; exit 1; }

# Path to the python3 script
SCRIPT="grad_desc_populations_v2.py"

python3 grad_desc_populations_v2.py \
    --m1 4 \
    --m 20 \
    --dataset-size 1000 \
    --noise-scale 0.1 \
    --num-epochs 1000 \
    --reg-type Reciprocal_L1 \
    --reg-lambda 0.0352965711480333 \
   --learning-rate 10.0 \
    --batch-size 256 \
    --optimizer-type sgd \
    --seed 17 \
    --patience 20 \
    --alpha-init random_10 \
    --estimator-type plugin \
    --base-model-type rf \
    --populations resnet resnet resnet \
    --param-freezing

# Step learning rate scheduler (decay learning rate by half every 200 epochs)
python3 grad_desc_populations_v2.py \
    --m1 4 \
    --m 20 \
    --dataset-size 1000 \
    --noise-scale 0.1 \
    --num-epochs 1000 \
    --reg-type Reciprocal_L1 \
    --reg-lambda 0.0352965711480333 \
    --learning-rate 10.0 \
    --batch-size 256 \
    --optimizer-type sgd \
    --seed 17 \
    --patience 20 \
    --alpha-init random_10 \
    --estimator-type plugin \
    --base-model-type rf \
    --populations resnet resnet resnet \
    --param-freezing \
    --scheduler-type step \
    --step-size 200 \
    --gamma 0.5 \
    --min-lr 0.001

# Cosine annealing learning rate scheduler
python3 grad_desc_populations_v2.py \
    --m1 4 \
    --m 20 \
    --dataset-size 1000 \
    --noise-scale 0.1 \
    --num-epochs 1000 \
    --reg-type Reciprocal_L1 \
    --reg-lambda 0.0352965711480333 \
    --learning-rate 10.0 \
    --batch-size 256 \
    --optimizer-type sgd \
    --seed 17 \
    --patience 20 \
    --alpha-init random_10 \
    --estimator-type plugin \
    --base-model-type rf \
    --populations resnet resnet resnet \
    --param-freezing \
    --scheduler-type cosine \
    --min-lr 0.001

# Reduce on plateau learning rate scheduler (reduce when objective stops improving)
python3 grad_desc_populations_v2.py \
    --m1 4 \
    --m 20 \
    --dataset-size 1000 \
    --noise-scale 0.1 \
    --num-epochs 1000 \
    --reg-type Quadratic_Barrier \
    --reg-lambda 0.0352965711480333 \
    --learning-rate 10.0 \
    --batch-size 256 \
    --optimizer-type sgd \
    --seed 17 \
    --patience 20 \
    --alpha-init random_10 \
    --estimator-type plugin \
    --base-model-type rf \
    --populations resnet resnet resnet \
    --param-freezing \
    --scheduler-type plateau \
    --gamma 0.5 \
    --min-lr 0.001

echo "Running python grad_desc_populations_v2.py with the following parameters:"