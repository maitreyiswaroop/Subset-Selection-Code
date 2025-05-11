#!/bin/bash
# baseline_failure_1 baseline_failure_2 baseline_failure_3 baseline_failure_4 baseline_failure_5
populations=(linear_regression cubic_regression)
t2_types=(kernel_if_like mc_plugin)

for pop in "${populations[@]}"; do
  for t2 in "${t2_types[@]}"; do
    SAVE_PATH="/data/user_data/mswaroop/Subset-Selection-Code/results_v7/${pop}_${t2}/"
    mkdir -p "$SAVE_PATH" "logs"
    sbatch gd_pops_v7_task.sh \
      --populations $pop $pop $pop \
      --m1 4 \
      --m 20 \
      --dataset-size 10000 \
      --noise-scale 0.1 \
      --corr-strength 0.0 \
      --num-epochs 150 \
      --budget 8 \
      --penalty-type Reciprocal_L1 \
      --penalty-lambda 0.005 \
      --learning-rate 0.05 \
      --optimizer-type adam \
      --parameterization theta \
      --alpha-init random_1 \
      --patience 20 \
      --gradient-mode autograd \
      --t2-estimator-type $t2 \
      --N-grad-samples 25 \
      --estimator-type if \
      --base-model-type xgb \
      --objective-value-estimator if \
      --k-kernel 1000 \
      --scheduler-type CosineAnnealingLR \
      --scheduler-t-max 150 \
      --scheduler-min-lr 1e-6 \
      --seed 42 \
      --save-path $SAVE_PATH \
      --verbose \
      --param-freezing
  done
done

echo "All jobs submitted."