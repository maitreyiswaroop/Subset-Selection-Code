#!/bin/bash
#SBATCH --job-name="regen_master_gd_pops_v10"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/v10_master_gd_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/v10_master_gd_%j.err

# baseline_failure_1 baseline_failure_2 baseline_failure_3 baseline_failure_4 baseline_failure_5
# populations=(linear_regression cubic_regression)
SAVE_PATH="./results_v10"
# POPULATIONS=("uci" "uci" "uci")
# POPULATIONS=("Male" "Female")
UCI_POPULATIONS=("Male" "Female")
POPULATIONS=("uci" "uci")
BUDGET=10
M1=10
M=100
DATASET_SIZE=48842
NOISE_SCALE=0.1
CORR_STRENGTH=0.0
PENALTY_TYPE="Reciprocal_L1"
PENALTY_LAMBDA=0.000001
# penalty_lambdas=(0.000005 0.0001)
penalty_lambdas=(0.000005)
LEARNING_RATE=0.005
lrs=(0.02 0.01)
lrs=(0.02)
SEED=26
seeds=(30)
# 26 30 29 
# seeds=(50 60)
t2=mc_plugin

# for pop in "${populations[@]}"; do
#   for t2 in "${t2_types[@]}"; do
# for learning_rate in "${lrs[@]}"; do
for penalty_lambda in "${penalty_lambdas[@]}"; do
  for learning_rate in "${lrs[@]}"; do
    for seed in "${seeds[@]}"; do
      SAVE_PATH="/data/user_data/mswaroop/Subset-Selection-Code/results_v10/${t2}/final_may_14/${seed}/"
      mkdir -p "$SAVE_PATH" "logs"
      sbatch gd_pops_v10_task_general.sh \
        --populations ${POPULATIONS[@]} \
        --budget $BUDGET \
        --m1 $M1 \
        --m $M \
        --dataset_size $DATASET_SIZE \
        --noise_scale $NOISE_SCALE \
        --corr_strength $CORR_STRENGTH \
        --penalty_type $PENALTY_TYPE \
        --penalty_lambda $penalty_lambda \
        --learning_rate $learning_rate \
        --parameterization theta \
        --alpha_init random_5 \
        --optimizer_type adam \
        --num_epochs 120 \
        --patience 15 \
        --gradient_mode autograd \
        --objective_value_estimator if \
        --t2_estimator_type mc_plugin \
        --N_grad_samples 25 \
        --estimator_type plugin \
        --base_model_type xgb \
        --seed $seed \
        --save_path $SAVE_PATH \
        --param_freezing \
        --uci_populations ${UCI_POPULATIONS[@]} \
        --force_regenerate_data \
        --resume /data/user_data/mswaroop/Subset-Selection-Code/results_v10/mc_plugin/final_may_14/30/run_0/ 
      sleep 10
      joc_count=$(squeue -u mswaroop | wc -l)
      while [ "$joc_count" -gt 10 ]; do
        sleep 60
        debug_count=$(squeue -u mswaroop | grep debug | wc -l)
      done
    done
  done
done
#  \
      # --param-freezing
echo "All jobs submitted."