#!/bin/bash
#SBATCH --job-name="master_uci"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/uci_master_gd_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/uci_master_gd_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=debug
# UCI Adult dataset with age-based populations
# populations=(uci uci uci)
pop=uci
t2_types=(mc_plugin)
t2=mc_plugin
seeds=(28 29 30 101 102) #26 17 9 30 
seed=26
penalty_lambdas=(0.0001)
penalty_lambda=0.0001
lrs=(0.001 0.0005 0.0002)
# lrs=(0.001)

for seed in "${seeds[@]}"; do
# for penalty_lambda in "${penalty_lambdas[@]}"; do
  # for t2 in "${t2_types[@]}"; do
  for lr in "${lrs[@]}"; do
    joc_count=$(squeue -u mswaroop | wc -l)
    while [ "$joc_count" -gt 10 ]; do
      echo "Job count $joc_count is greater than 10. Waiting..."
      sleep 60
      joc_count=$(squeue -u mswaroop | wc -l)
    done
    SAVE_PATH="/data/user_data/mswaroop/Subset-Selection-Code/results_v8/${t2}/uci_sex/tuning/"
    mkdir -p "$SAVE_PATH"
    sbatch gd_pops_v8_task.sh \
      --populations $pop $pop $pop \
      --budget 10 \
      --penalty-type Reciprocal_L1 \
      --penalty-lambda $penalty_lambda \
      --learning-rate $lr \
      --optimizer-type adam \
      --parameterization theta \
      --alpha-init random_2 \
      --num-epochs 5 \
      --patience 15 \
      --gradient-mode autograd \
      --t2-estimator-type $t2 \
      --N-grad-samples 25 \
      --estimator-type plugin \
      --base-model-type xgb \
      --objective-value-estimator if \
      --k-kernel 2000 \
      --scheduler-type CosineAnnealingLR \
      --scheduler-t-max 80 \
      --seed $seed \
      --save-path $SAVE_PATH \
      --verbose
    sleep 20
  done
done