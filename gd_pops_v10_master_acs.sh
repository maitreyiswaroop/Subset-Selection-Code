#!/bin/bash
#SBATCH --job-name="master_acs_pops_v10"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/master_acs_pops_v10_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/master_acs_pops_v10_%j.err

# baseline_failure_1 baseline_failure_2 baseline_failure_3 baseline_failure_4 baseline_failure_5
# populations=(linear_regression cubic_regression)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Experiment settings
SAVE_PATH="./results_v10_acs"
POPULATIONS=("acs")
ACS_DATA_FRACTION=0.05    # fraction of ACS data to use
M1=10
M=18
DATASET_SIZE=30000
NOISE_SCALE=0.0
CORR_STRENGTH=0.0
BUDGET=15
LEARNING_RATE=0.01
PENALTY_TYPE="Reciprocal_L1"
PENALTY_LAMBDA=0.0001
SEED=123

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

penalty_lambdas=(0.0001) # 0.00005)
lrs=(0.01) # 0.005 0.001)
seeds=(123 456 789 101112 131415)
for penalty_lambda in "${penalty_lambdas[@]}"; do
  for learning_rate in "${lrs[@]}"; do
    for seed in "${seeds[@]}"; do
      SAVE_PATH="/data/user_data/mswaroop/Subset-Selection-Code/results_v10/${t2}/final_may_14/${seed}/smooth_minmax/"
      mkdir -p "$SAVE_PATH" "logs"
      sbatch gd_pops_v10_task_general.sh \
        --populations ${POPULATIONS[@]} \
        --acs_data_fraction $ACS_DATA_FRACTION \
        --m1 $M1 \
        --m $M \
        --dataset_size $DATASET_SIZE \
        --noise_scale $NOISE_SCALE \
        --corr_strength $CORR_STRENGTH \
        --budget $BUDGET \
        --learning_rate $learning_rate \
        --penalty_type $PENALTY_TYPE \
        --penalty_lambda $penalty_lambda \
        --optimizer_type adam \
        --parameterization alpha \
        --alpha_init random_5 \
        --num_epochs 100 \
        --patience 10 \
        --gradient_mode autograd \
        --objective_value_estimator if \
        --t2_estimator_type mc_plugin \
        --N_grad_samples 10 \
        --estimator_type plugin \
        --base_model_type xgb \
        --seed $seed \
        --save_path $SAVE_PATH \
        --force_regenerate_data \
        --k_kernel 500 \
        --param_freezing
      sleep 10
      joc_count=$(squeue -u mswaroop | wc -l)
      while [ "$joc_count" -gt 12 ]; do
        sleep 60
        debug_count=$(squeue -u mswaroop | grep debug | wc -l)
      done
    done
  done
done
#  \
      # --param-freezing
echo "All jobs submitted."