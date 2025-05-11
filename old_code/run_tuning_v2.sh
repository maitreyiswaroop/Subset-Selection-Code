#!/bin/bash
# List of regularizer types to test
regularizers=("Reciprocal_L1" "Quadratic_Barrier" "Exponential" "Max_Dev")
dataset_types=("linear_regression")

# Define common hyperparameters
DATASET_SIZE=10000
M1=2
M=6
SEED=10
DATASET_TYPE="sinusoidal_regression"
NUM_EPOCHS=100
LEARNING_RATE=0.001
BATCH_SIZE=100
OPTIMIZER_TYPE="adam"
NOISE_SCALE=0.01

for dataset_type in "${dataset_types[@]}"
    do
        SAVE_PATH="./results/discrete/hyperparam_tuning/parse_1/$dataset_type/"
        for reg in "${regularizers[@]}"
            do
                echo "----------------------------------------"
                echo "Tuning for regularizer type: $reg"
                python3 tune_lambda_v2.py \
                --reg_type "$reg" \
                --dataset_size $DATASET_SIZE \
                --m1 $M1 \
                --m $M \
                --seed $SEED \
                --dataset_type $dataset_type \
                --num_epochs $NUM_EPOCHS \
                --learning_rate $LEARNING_RATE \
                --batch_size $BATCH_SIZE \
                --optimizer_type $OPTIMIZER_TYPE \
                --noise_scale $NOISE_SCALE \
                --save_path "$SAVE_PATH" 
            done
        done