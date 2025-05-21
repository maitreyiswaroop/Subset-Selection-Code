# gd_pops_tune_v5.py: Hyperparameter tuning for grad_desc_populations_v5.py using Optuna

"""
This script uses Optuna to automatically tune hyperparameters for the
multi-population variable selection experiment defined in grad_desc_populations_v5.py.

It aims to find the hyperparameter combination that maximizes the F1 score
based on the selected variables compared to the true meaningful indices.
"""

import optuna
import argparse
import os
import json
import numpy as np
import torch
from functools import partial # To pass fixed arguments to objective function

# --- Import necessary components from the experiment script ---
# Ensure grad_desc_populations_v5.py is in the Python path or same directory
try:
    from grad_desc_populations_v5 import (
        run_experiment_multi_population,
        get_pop_data, # Needed to get meaningful indices for evaluation
        compute_population_stats,
        convert_numpy_to_python,
        get_latest_run_number
    )
    print("Successfully imported from grad_desc_populations_v5.py")
except ImportError as e:
    print(f"Error importing from grad_desc_populations_v5.py: {e}")
    print("Please ensure grad_desc_populations_v5.py is in the same directory or Python path.")
    exit(1)

# --- Define the Objective Function for Optuna ---

def objective(trial, fixed_args):
    """
    Optuna objective function.
    Takes an Optuna trial and fixed arguments, runs the experiment,
    and returns the metric to optimize (negative F1 score for minimization).
    """
    # --- Suggest Hyperparameters ---
    # Optimization related
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    optimizer_type = trial.suggest_categorical("optimizer_type", ["adam", "sgd"])

    # Penalty related
    penalty_type = trial.suggest_categorical("penalty_type", ["Reciprocal_L1", "Quadratic_Barrier", "Exponential", "None"])
    # Adjust lambda range based on penalty type? For now, use a wide log range.
    # If penalty_type is None, lambda is effectively ignored by run_experiment, but we still suggest it.
    penalty_lambda = trial.suggest_float("penalty_lambda", 1e-6, 1e-1, log=True)

    # REINFORCE related
    n_grad_samples = trial.suggest_categorical("N_grad_samples", [10, 25, 50, 75]) # Fewer options for speed

    # Other potential hyperparameters to tune (optional):
    # alpha_init = trial.suggest_categorical("alpha_init", ["random_1", "random_0.5", "ones"])
    # param_freezing = trial.suggest_categorical("param_freezing", [True, False])

    print(f"\n--- Trial {trial.number} ---")
    print(f"  Params: lr={lr:.5f}, opt={optimizer_type}, pen_type={penalty_type}, pen_lambda={penalty_lambda:.6f}, N_grad={n_grad_samples}")

    # --- Prepare arguments for run_experiment ---
    run_args = fixed_args.copy() # Start with fixed args
    run_args['learning_rate'] = lr
    run_args['optimizer_type'] = optimizer_type
    run_args['penalty_type'] = penalty_type if penalty_type != "None" else None
    run_args['penalty_lambda'] = penalty_lambda
    run_args['N_grad_samples'] = n_grad_samples
    # Add others if tuning them:
    # run_args['alpha_init'] = alpha_init
    # run_args['param_freezing'] = param_freezing

    # Ensure save_path is unique for each trial to avoid conflicts
    trial_save_path = os.path.join(fixed_args['save_path'], f"trial_{trial.number}")
    run_args['save_path'] = trial_save_path
    run_args['verbose'] = False # Keep trial runs quiet unless debugging

    # --- Run the Experiment ---
    try:
        results = run_experiment_multi_population(**run_args)

        # --- Calculate the Metric (F1 Score) ---
        selected_indices = results.get('selected_indices', [])
        meaningful_indices_list = results.get('meaningful_indices', [])

        if not meaningful_indices_list:
             print("Warning: Meaningful indices list is empty. Cannot calculate F1 score.")
             return 0.0 # Or some other indicator of failure

        all_meaningful_indices = set()
        for indices in meaningful_indices_list:
            all_meaningful_indices.update(indices)

        selected_set = set(selected_indices)
        intersection = selected_set.intersection(all_meaningful_indices)

        recall = len(intersection) / len(all_meaningful_indices) if len(all_meaningful_indices) > 0 else 0.0
        precision = len(intersection) / len(selected_set) if len(selected_set) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"  Trial {trial.number} Result: F1 Score = {f1_score:.4f}, Recall = {recall:.4f}, Precision = {precision:.4f}")

        # Optuna minimizes, so return negative F1 score if maximizing F1
        metric_to_optimize = f1_score

        # You might want to prune trials that look unpromising early
        # trial.report(metric_to_optimize, step=run_args.get('stopped_epoch', run_args['num_epochs']))
        # if trial.should_prune():
        #     raise optuna.exceptions.TrialPruned()

        return metric_to_optimize # Return F1 score directly for maximization

    except Exception as e:
        print(f"Error during trial {trial.number}: {e}")
        # traceback.print_exc() # Uncomment for detailed traceback
        # Return a value indicating failure (e.g., 0.0 F1 score)
        return 0.0


# --- Main Execution Block ---

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for Multi-population Variable Selection (v5)')

    # --- Arguments for the Tuning Study ---
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials to run')
    parser.add_argument('--study-name', type=str, default='gd_pops_v5_tuning', help='Name for the Optuna study')
    parser.add_argument('--storage', type=str, default=None, help='Database URL for Optuna storage (e.g., sqlite:///tuning.db)')
    parser.add_argument('--base-save-path', type=str, default='./results_v5_tuning/', help='Base directory to save trial results')

    # --- Fixed Arguments for the Experiment (passed to run_experiment_multi_population) ---
    # These are NOT tuned by Optuna in this setup
    parser.add_argument('--m1', type=int, default=4)
    parser.add_argument('--m', type=int, default=20) # Keep small for faster tuning
    parser.add_argument('--dataset-size', type=int, default=5000) # Keep small
    parser.add_argument('--noise-scale', type=float, default=0.1)
    parser.add_argument('--corr-strength', type=float, default=0.0)
    parser.add_argument('--populations', nargs='+', default=['linear_regression', 'sinusoidal_regression'])
    parser.add_argument('--num-epochs', type=int, default=100) # Shorter epochs for tuning
    parser.add_argument('--batch-size', type=int, default=5000)
    parser.add_argument('--alpha-init', type=str, default='random_1')
    parser.add_argument('--patience', type=int, default=10) # Shorter patience
    parser.add_argument('--param-freezing', action=argparse.BooleanOptionalAction, default=True) # Use --no-param-freezing to disable
    parser.add_argument('--use-baseline', action=argparse.BooleanOptionalAction, default=True) # Use --no-use-baseline to disable
    parser.add_argument('--estimator-type', type=str, default='if', choices=['plugin', 'if'])
    parser.add_argument('--base-model-type', type=str, default='rf', choices=['rf', 'krr'])
    parser.add_argument('--objective-value-estimator', type=str, default='if', choices=['if', 'mc'])
    parser.add_argument('--seed', type=int, default=42, help='Fixed seed for data generation across trials') # Important for comparability


    args = parser.parse_args()

    # --- Prepare Fixed Arguments Dictionary ---
    fixed_args = {
        'm1': args.m1,
        'm': args.m,
        'dataset_size': args.dataset_size,
        'noise_scale': args.noise_scale,
        'corr_strength': args.corr_strength,
        # 'populations': args.populations, # Need to convert to pop_configs inside objective? No, run_exp handles it.
        'pop_configs': [{'pop_id': i, 'dataset_type': args.populations[i]} for i in range(len(args.populations))], # Create pop_configs here
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'alpha_init': args.alpha_init,
        'early_stopping_patience': args.patience,
        'estimator_type': args.estimator_type,
        'base_model_type': args.base_model_type,
        'param_freezing': args.param_freezing,
        'use_baseline': args.use_baseline,
        'objective_value_estimator': args.objective_value_estimator,
        'seed': args.seed, # Use same seed for data generation
        'save_path': args.base_save_path # Base path, objective will add trial subfolder
        # budget is calculated inside run_experiment
    }

    # --- Create or Load Optuna Study ---
    # We want to MAXIMIZE F1 score
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True, # Resume study if it already exists in storage
        direction="maximize" # Optimize for higher F1 score
    )

    # --- Run the Optimization ---
    print(f"Starting Optuna study '{args.study_name}' with {args.n_trials} trials...")
    study.optimize(
        partial(objective, fixed_args=fixed_args), # Pass fixed args using partial
        n_trials=args.n_trials,
        timeout=None # No time limit unless specified
        # Add callbacks=[...] here if needed (e.g., for logging)
    )

    # --- Print Best Results ---
    print("\n--- Tuning Finished ---")
    print(f"Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial
    print(f"Best trial number: {best_trial.number}")
    print(f"Best F1 Score: {best_trial.value:.4f}")
    print("Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # --- Save Best Parameters ---
    best_params_file = os.path.join(args.base_save_path, "best_params.json")
    os.makedirs(args.base_save_path, exist_ok=True) # Ensure directory exists
    try:
        with open(best_params_file, 'w') as f:
            json.dump(best_trial.params, f, indent=4)
        print(f"Best parameters saved to: {best_params_file}")
    except Exception as e:
        print(f"Error saving best parameters: {e}")

    # You can also save the full study results if using storage
    if args.storage:
        print(f"Study results are stored in: {args.storage}")
    else:
        print("Study results are in-memory only (no storage specified).")


if __name__ == '__main__':
    main()