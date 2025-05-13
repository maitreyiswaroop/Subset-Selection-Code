#!/bin/bash
#SBATCH --job-name="master_gd_pops_v8"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/v8_master_gd_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/v8_master_gd_%j.err

# baseline_failure_1 baseline_failure_2 baseline_failure_3 baseline_failure_4 baseline_failure_5
# populations=(linear_regression cubic_regression)
t2_types=(mc_plugin)
seeds=(17 30 29 9 26)
# seed=42
estimator=plugin
# population=quadratic_regression
populations=(baseline_failure_2)
N_GRAD_SAMPLES=25

# for baseline 3, smaller lr is better
# for pop in "${populations[@]}"; do
#   for t2 in "${t2_types[@]}"; do
for t2 in "${t2_types[@]}"; do
  for seed in "${seeds[@]}"; do
    for population in "${populations[@]}"; do
      joc_count=$(squeue -u mswaroop | wc -l)
      while [ "$joc_count" -gt 10 ]; do
        echo "Job count $joc_count is greater than 10. Waiting..."
        sleep 60
        joc_count=$(squeue -u mswaroop | wc -l)
      done
      SAVE_PATH="/data/user_data/mswaroop/Subset-Selection-Code/results_v8/${t2}/${population}/"
      mkdir -p "$SAVE_PATH" "logs"
      sbatch gd_pops_v8_task.sh \
        --populations $population $population $population  \
        --m1 4 \
        --m 20 \
        --dataset-size 12000 \
        --baseline-data-size 25000 \
        --noise-scale 0.1 \
        --corr-strength 0.1 \
        --num-epochs 150 \
        --budget 5 \
        --penalty-type Reciprocal_L1 \
        --penalty-lambda 0.001 \
        --learning-rate 0.05 \
        --optimizer-type adam \
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
        --scheduler-t-max 150 \
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