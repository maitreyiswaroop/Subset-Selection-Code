#!/bin/bash
#SBATCH --job-name="gd_pops_v8"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/gd_pops/v8_gd_pops_%j.out  # CHANGE: Specify log directory
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/gd_pops/v8_gd_pops_%j.err   # CHANGE: Specify log directory
#SBATCH --gres=gpu:2                         # Request 1 GPU (adjust if needed)
#SBATCH --cpus-per-task=2                  # Request CPUs (adjust if needed)
#SBATCH --time=05:00:00                      # Time limit (e.g., 2 hours, adjust)
#SBATCH --mem=32G                          # Memory requirement (adjust if needed)
#SBATCH --partition=general                  # CHANGE: Specify your cluster partition

# --- Environment Setup ---
echo "Setting up environment..."
# Activate your conda environment (adjust path if needed)
source /home/$USER/miniconda/etc/profile.d/conda.sh # CHANGE: Specify conda path
conda activate venv                       # CHANGE: Specify your environment name

# Or load modules if not using conda
# module load python/3.9 cuda/11.x etc...         # CHANGE: Adapt to your cluster modules

# --- Script Execution ---
SCRIPT_DIR="/data/user_data/mswaroop/Subset-Selection-Code" # CHANGE: Set path to your script directory
SCRIPT_NAME="gd_pops_v8.py"
SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"

# Define parameters (Example using Autograd and CosineAnnealingLR)
# Modify these parameters as needed for your specific run
# Define parameters (Example using Theta Param, Autograd, and CosineAnnealingLR)
# Modify these parameters as needed for your specific run
# baseline_failure_1; baseline_failure_2; baseline_failure_3; baseline_failure_4; baseline_failure_5
# linear_regression; cubic_regression
# kernel_if_like / mc_plugin
# seeds: 42, 17, 30, 29, 9
t2_types=(mc_plugin)
populations=(quadratic_regression)
estimator=plugin
seed=9
# baseline_failure_2 baseline_failure_3 baseline_failure_4 baseline_failure_5 linear_regression cubic_regression)

for population in "${populations[@]}"; do
    for t2 in "${t2_types[@]}"; do
    # customize save-path per estimator
    SAVE_PATH="$SCRIPT_DIR/results/results_v8/theta_run_${t2}/quadratic_regression/$estimator/"
    mkdir -p "$SAVE_PATH"

    PARAMS="--m1 4 \
        --m 20 \
        --dataset-size 5000 \
        --noise-scale 0.1 \
        --corr-strength 0.0 \
        --populations $population $population $population \
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
        --param-freezing"

    echo "----------------------------------------"
    echo "Running with t2-estimator-type = $t2"
    echo "$PARAMS" | sed 's/\\//g'
    echo "----------------------------------------"

    python3 "$SCRIPT_PATH" $PARAMS
    done
    echo "----------------------------------------"
    echo "Job finished for population: $population"
    echo "----------------------------------------"
done
echo "----------------------------------------"

echo "All runs complete."