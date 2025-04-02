#!/bin/bash
# filepath: /Users/mswaroop/Desktop/Projects/Bryan/Subset selection/Subset-Selection-Code/run_experiments.sh

# Make script executable
chmod +x gd_populations_run.sh

# Base configuration
BASE_ARGS="--m1 4 --m 15 --dataset-size 10000 --noise-scale 0.01 --num-epochs 100"
BASE_ARGS="$BASE_ARGS --reg-type Reciprocal_L1 --reg-lambda 0.001 --batch-size 256"

python3 grad_desc_populations_v2.py \
    --m1 4 \
    --m 15 \
    --dataset-size 10000 \
    --noise-scale 0.01 \
    --num-epochs 1000 \
    --reg-type Reciprocal_L1 \
    --reg-lambda 0.0001 \
    --learning-rate 0.05 \
    --batch-size 256 \
    --optimizer-type sgd \
    --seed 42 \
    --patience 10 \
    --alpha-init random \
    --estimator-type if \
    --populations linear_regression sinusoidal_regression \
    --param-freezing \
    --save-path ./results/multi_population/