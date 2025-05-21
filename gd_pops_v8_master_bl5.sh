#!/bin/bash
#SBATCH --job-name="master_gd_pops_v8"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/v8_master_gd_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/v8_master_gd_%j.err

# baseline_failure_1 baseline_failure_2 baseline_failure_3 baseline_failure_4 baseline_failure_5
# populations=(linear_regression cubic_regression)
t2_types=(mc_plugin)
t2=mc_plugin
# seeds=(17 30 29 9 26)
# seeds=(14 15 16)
# seeds=(100 101 102 103 104)
seeds=(7 10 2 26 29)
# seed=42
estimator=plugin
# population=quadratic_regression
populations=(baseline_failure_5)
N_GRAD_SAMPLES=5
learning_rates=(0.05)
# for baseline 3, smaller lr is better
# for pop in "${populations[@]}"; do
#   for t2 in "${t2_types[@]}"; do
# for t2 in "${t2_types[@]}"; do
for lr in "${learning_rates[@]}"; do
  for seed in "${seeds[@]}"; do
    for population in "${populations[@]}"; do
      joc_count=$(squeue -u mswaroop | wc -l)
      while [ "$joc_count" -gt 10 ]; do
        echo "Job count $joc_count is greater than 10. Waiting..."
        sleep 60
        joc_count=$(squeue -u mswaroop | wc -l)
      done
      SAVE_PATH="/Users/mswaroop/Desktop/Projects/Bryan/Subset_selection/Subset-Selection-Code/results_v8/${t2}/${population}/may15/"
      mkdir -p "$SAVE_PATH" "logs"
      bash gd_pops_v8_task.sh \
        --populations $population $population $population  \
        --m1 4 \
        --m 15 \
        --dataset-size 12000 \
        --baseline-data-size 30000 \
        --noise-scale 0.1 \
        --corr-strength 0.1 \
        --num-epochs 100 \
        --budget 10 \
        --penalty-type Reciprocal_L1 \
        --penalty-lambda 0.001 \
        --learning-rate $lr \
        --optimizer-type sgd \
        --parameterization theta \
        --alpha-init random_1 \
        --patience 20 \
        --gradient-mode autograd \
        --t2-estimator-type $t2 \
        --N-grad-samples $N_GRAD_SAMPLES \
        --estimator-type $estimator \
        --base-model-type xgb \
        --objective-value-estimator if \
        --k-kernel 1000 \
        --scheduler-type CosineAnnealingLR \
        --scheduler-t-max 180 \
        --scheduler-min-lr 1e-6 \
        --seed $seed \
        --save-path $SAVE_PATH \
        --verbose \
        --param-freezing
      sleep 10
    done
  done
done
#  \
      # --param-freezing
echo "All jobs submitted."