#!/bin/bash
#SBATCH --job-name="master_gd_pops_v8"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/v8_master_gd_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/v8_master_gd_%j.err

# baseline_failure_1 baseline_failure_2 baseline_failure_3 baseline_failure_4 baseline_failure_5
# populations=(linear_regression cubic_regression)
t2_types=(kernel_if_like mc_plugin)
seeds=(17) # 17 30 29 9)
seed=17
population=asc
estimator=plugin
N_GRAD_SAMPLES=10
t2=mc_plugin
learning_rate=0.0001
learningrates=(0.0001 0.00001 0.000001)
# lr should be 1e-4 or lower
# for pop in "${populations[@]}"; do
#   for t2 in "${t2_types[@]}"; do
# for t2 in "${t2_types[@]}"; do
smooth_minmaxes=(5.0 10.0)
for learning_rate in "${learningrates[@]}"; do
# for smooth_minmax in "${smooth_minmaxes[@]}"; do
  # for seed in "${seeds[@]}"; do
  for t2 in "${t2_types[@]}"; do
    for penalty_lambda in 0.00001 0.0001 0.001; do
      SAVE_PATH="/data/user_data/mswaroop/Subset-Selection-Code/results_v8/${t2}/${population}/"
      mkdir -p "$SAVE_PATH" "logs"
      sbatch gd_pops_v8_task.sh \
        --populations $population $population $population  \
        --m1 4 \
        --m 20 \
        --dataset-size 5000 \
        --noise-scale 0.1 \
        --asc-data-fraction 0.5 \
        --corr-strength 0.1 \
        --num-epochs 50 \
        --budget 50 \
        --penalty-type Reciprocal_L1 \
        --penalty-lambda $penalty_lambda \
        --learning-rate $learning_rate \
        --optimizer-type adam \
        --parameterization theta \
        --alpha-init random_1 \
        --patience 15 \
        --gradient-mode autograd \
        --t2-estimator-type $t2 \
        --N-grad-samples $N_GRAD_SAMPLES \
        --estimator-type $estimator \
        --base-model-type xgb \
        --objective-value-estimator if \
        --k-kernel 1000 \
        --scheduler-type ReduceLROnPlateau \
        --scheduler-t-max 150 \
        --scheduler-min-lr 1e-6 \
        --seed $seed \
        --save-path $SAVE_PATH \
        --verbose \
        --param-freezing
      sleep 10
      joc_count=$(squeue -u mswaroop | wc -l)
      while [ "$joc_count" -gt 10 ]; do
        echo "Job count $joc_count is greater than 100. Waiting..."
        sleep 60
        joc_count=$(squeue -u mswaroop | wc -l)
      done
    done
  done
done

echo "All jobs submitted."