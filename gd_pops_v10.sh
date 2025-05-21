#!/bin/bash
#SBATCH --job-name="v8_task_gd_pops"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/v10_task_gd_%j.out
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/v10_task_gd_%j.err
#SBATCH --mem=64G
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=4  # Increased CPU count for CPU-only processing
#SBATCH --time=05:00:00
#SBATCH --partition=debug
#SBATCH --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Loading environment..."
source /home/$USER/miniconda/etc/profile.d/conda.sh
conda activate venv   # or your env name

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
PENALTY_LAMBDA=0.0001
LEARNING_RATE=0.005
SEED=26

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Run gd_pops_v10.py with specified parameters
python3 gd_pops_v10.py \
  --populations ${POPULATIONS[@]} \
  --budget $BUDGET \
  --m1 $M1 \
  --m $M \
  --dataset_size $DATASET_SIZE \
  --noise_scale $NOISE_SCALE \
  --corr_strength $CORR_STRENGTH \
  --penalty_type $PENALTY_TYPE \
  --penalty_lambda $PENALTY_LAMBDA \
  --learning_rate $LEARNING_RATE \
  --parameterization theta \
  --alpha_init random_2 \
  --optimizer_type adam \
  --num_epochs 100 \
  --patience 15 \
  --gradient_mode autograd \
  --objective_value_estimator if \
  --t2_estimator_type mc_plugin \
  --N_grad_samples 25 \
  --estimator_type plugin \
  --base_model_type xgb \
  --seed $SEED \
  --save_path $SAVE_PATH \
  --param_freezing \
  --uci_populations ${UCI_POPULATIONS[@]} \
  --force_regenerate_data \
  2>&1 | tee logs/gd_pops_v10_$(date +%Y%m%d_%H%M%S).log

# --resume "$SAVE_PATH/run_2"