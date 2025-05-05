#!/bin/bash
# run_optuna_tuning.sh
#
# This script runs the Optuna hyperparameter optimization for the
# gd_populations_v3.py experiment.

# Check if Optuna is installed
python3 -c "import optuna" 2>/dev/null || { 
    echo "Optuna is not installed. Installing now..."
    pip install optuna plotly kaleido
}

# Path to the Python optimization script
SCRIPT="optuna_tuning.py"

# Number of trials to run (adjust as needed)
N_TRIALS=200

# Database file for storing results
DB_FILE="sqlite:///optuna_gd_.db"

# Output directory for optimization results
OUTPUT_DIR="./optuna_results"

# Run the optimization
python3 "$SCRIPT" \
    --n-trials $N_TRIALS \
    --storage "$DB_FILE" \
    --m1 4 \
    --m 20 \
    --dataset-size 1000 \
    --num-epochs 150 \
    --populations resnet resnet resnet \
    --seed 17 \
    --base-model-type rf \
    --param-freezing \
    --output-dir "$OUTPUT_DIR"

echo "Optimization complete. Results saved to $OUTPUT_DIR"
echo "Best parameters saved as $OUTPUT_DIR/best_config.sh"