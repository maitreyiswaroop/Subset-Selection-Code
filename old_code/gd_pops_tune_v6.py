# gd_pops_tune_v6.py: Hyperparameter tuning for gd_pops_v6.py using Optuna
# Fixed: Removed base_save_path from fixed_args passed to objective

"""
This script uses Optuna to automatically tune hyperparameters for the
multi-population variable selection experiment defined in gd_pops_v6.py.

It aims to find the hyperparameter combination that maximizes the F1 score
based on the selected variables compared to the true meaningful indices.
"""

import optuna
import argparse
import os
import json
import numpy as np
import torch
import traceback
from functools import partial # To pass fixed arguments to objective function
from typing import Dict # Added typing

# --- Import necessary components from the experiment script ---
# Ensure gd_pops_v6.py is in the Python path or same directory
try:
    # Make sure you have the latest version saved as gd_pops_v6.py
    from gd_pops_v6 import (
        run_experiment_multi_population_v6,
        compute_population_stats,
        convert_numpy_to_python,
        get_latest_run_number
    )
    print("Successfully imported from gd_pops_v6.py")
except ImportError as e:
    print(f"Error importing from gd_pops_v6.py: {e}")
    print("Please ensure gd_pops_v6.py is in the same directory or Python path.")
    exit(1)
except Exception as e_other:
    print(f"An unexpected error occurred during import: {e_other}")
    exit(1)

# --- Define the Objective Function for Optuna ---

def objective(trial: optuna.Trial, fixed_args: Dict, base_save_path_for_trial: str): # Added base_save_path_for_trial
    """
    Optuna objective function.
    Takes an Optuna trial, fixed arguments, and the base save path, runs the experiment,
    and returns the metric to optimize (F1 score for maximization).
    """
    # --- Suggest Hyperparameters ---
    suggested_params = {
        'learning_rate': trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        'optimizer_type': trial.suggest_categorical("optimizer_type", ["adam", "sgd"]),
        'penalty_type': trial.suggest_categorical("penalty_type", ["Reciprocal_L1", "Quadratic_Barrier", "Exponential", "Max_Dev", "None"]),
        'penalty_lambda': trial.suggest_float("penalty_lambda", 1e-6, 1e-1, log=True),
        'parameterization': trial.suggest_categorical("parameterization", ['alpha', 'theta']),
        'alpha_init': trial.suggest_categorical("alpha_init", ["random_0.1", "random_1", "random_2", "random_5"]), # Example alpha init values
        'gradient_mode': trial.suggest_categorical("gradient_mode", ['autograd', 'reinforce']),
        'N_grad_samples': trial.suggest_int("N_grad_samples", 10, 50), # Reduced range for tuning speed
        'k_kernel': trial.suggest_int("k_kernel", 100, 3000),
        'base_model_type': trial.suggest_categorical("base_model_type", ["rf", "xgb"]), # Removed krr for simplicity? Add back if needed
        'estimator_type': trial.suggest_categorical("estimator_type", ["plugin", "if"]),
        # 'smooth_minmax': trial.suggest_float("smooth_minmax", 0.1, 10.0, log=True) # Optional: Tune smoothing factor
    }

    # If penalty is None, lambda is irrelevant, but Optuna needs a value.
    # The run function should handle penalty_type=None correctly.
    if suggested_params['penalty_type'] == "None":
        suggested_params['penalty_lambda'] = 0.0 # Ensure lambda is 0 if no penalty

    print(f"\n--- Optuna Trial {trial.number} ---")
    print("  Suggesting Parameters:")
    for key, value in suggested_params.items():
        print(f"    {key}: {value}")

    # --- Prepare arguments for run_experiment ---
    run_args = fixed_args.copy() # Start with fixed args
    run_args.update(suggested_params) # Overwrite fixed args with suggested ones

    # Ensure save_path is unique for each trial
    # Use the base_save_path passed specifically for this purpose
    trial_save_path = os.path.join(base_save_path_for_trial, f"trial_{trial.number}")
    run_args['save_path'] = trial_save_path # This is the argument the function expects
    run_args['verbose'] = False # Keep trial runs quiet unless debugging

    # --- Run the Experiment ---
    try:
        # Make sure run_experiment_multi_population_v6 doesn't expect 'base_save_path'
        results = run_experiment_multi_population_v6(**run_args)

        if 'error' in results:
            print(f"  Trial {trial.number} failed with error: {results['error']}")
            return 0.0 # Return poor score on failure

        # --- Calculate the Metric (F1 Score) ---
        selected_indices = results.get('selected_indices', [])
        meaningful_indices_list = results.get('meaningful_indices', [])

        if not meaningful_indices_list:
             print("Warning: Meaningful indices list is empty in results. Cannot calculate F1 score.")
             return 0.0

        all_meaningful_indices = set()
        for indices in meaningful_indices_list:
            if indices: # Ensure sublist is not empty/None
                all_meaningful_indices.update(indices)

        selected_set = set(selected_indices)
        intersection_size = len(selected_set.intersection(all_meaningful_indices))

        precision = intersection_size / len(selected_set) if selected_set else 0.0
        recall = intersection_size / len(all_meaningful_indices) if all_meaningful_indices else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"  Trial {trial.number} Result: F1={f1_score:.4f} (P={precision:.4f}, R={recall:.4f})")

        # Optuna Pruning (Optional)
        trial.report(f1_score, step=results.get('stopped_epoch', run_args['num_epochs']))
        if trial.should_prune():
            print(f"  Trial {trial.number} pruned.")
            raise optuna.exceptions.TrialPruned()

        return f1_score # Return F1 score directly for maximization

    except optuna.exceptions.TrialPruned:
        raise # Re-raise pruned exceptions
    except Exception as e:
        print(f"Error during trial {trial.number}: {e}")
        traceback.print_exc() # Print detailed traceback for debugging
        return 0.0 # Indicate failure with a low score

# --- Main Execution Block ---

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning for gd_pops_v6.py')

    # --- Arguments for the Tuning Study ---
    parser.add_argument('--n-trials', type=int, default=100, help='Number of Optuna trials to run')
    parser.add_argument('--study-name', type=str, default='gd_pops_v6_study', help='Name for the Optuna study')
    parser.add_argument('--storage', type=str, default=None, help='Database URL for Optuna storage (e.g., sqlite:///tuning_v6.db)')
    parser.add_argument('--base-save-path', type=str, default='./results_v6_tuning/', help='Base directory to save trial results')

    # --- Fixed Arguments for the Experiment (passed to run_experiment_multi_population_v6) ---
    parser.add_argument('--m1', type=int, default=4)
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--dataset-size', type=int, default=5000)
    parser.add_argument('--noise-scale', type=float, default=0.1)
    parser.add_argument('--corr-strength', type=float, default=0.0)
    parser.add_argument('--populations', nargs='+', default=['linear_regression', 'sinusoidal_regression'])
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--budget', type=int, default=None)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--param-freezing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use-baseline', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--objective-value-estimator', type=str, default='if', choices=['if', 'mc'])
    parser.add_argument('--smooth-minmax', type=float, default=float('inf'))
    parser.add_argument('--scheduler-type', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42, help='Fixed seed for data generation across trials')

    args = parser.parse_args()

    # --- Prepare Fixed Arguments Dictionary ---
    # *** REMOVE base_save_path from here ***
    fixed_args = {
        'm1': args.m1, 'm': args.m, 'dataset_size': args.dataset_size,
        'noise_scale': args.noise_scale, 'corr_strength': args.corr_strength,
        'pop_configs': [{'pop_id': i, 'dataset_type': dt} for i, dt in enumerate(args.populations)],
        'num_epochs': args.num_epochs, 'budget': args.budget,
        'early_stopping_patience': args.patience, 'param_freezing': args.param_freezing,
        'use_baseline': args.use_baseline,
        'objective_value_estimator': args.objective_value_estimator,
        'smooth_minmax': args.smooth_minmax,
        'scheduler_type': args.scheduler_type,
        'scheduler_kwargs': {}, # Populate if scheduler_type is fixed
        'seed': args.seed
        # 'base_save_path': args.base_save_path # <-- REMOVE THIS LINE
    }

    # --- Create or Load Optuna Study ---
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20, interval_steps=5)
    )

    # --- Run the Optimization ---
    print(f"Starting Optuna study '{args.study_name}' with {args.n_trials} trials...")
    print(f"Fixed Args: {fixed_args}")
    # Pass base_save_path separately using partial
    objective_with_paths = partial(objective, fixed_args=fixed_args, base_save_path_for_trial=args.base_save_path)
    try:
        study.optimize(
            objective_with_paths, # Use the partial function
            n_trials=args.n_trials,
            timeout=None
        )
    except KeyboardInterrupt:
         print("Optimization stopped manually.")

    # --- Print Best Results ---
    print("\n--- Tuning Finished ---")
    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    fail_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.FAIL])

    print(f"Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")
    print(f"  Number of failed trials: {len(fail_trials)}")

    if complete_trials:
        best_trial = study.best_trial
        print(f"\nBest trial number: {best_trial.number}")
        print(f"Best F1 Score: {best_trial.value:.4f}")
        print("Best hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")

        # --- Save Best Parameters ---
        best_params_file = os.path.join(args.base_save_path, "best_params_v6.json")
        os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
        try:
            best_config = {'best_value': best_trial.value, 'best_params': best_trial.params}
            with open(best_params_file, 'w') as f:
                json.dump(convert_numpy_to_python(best_config), f, indent=4)
            print(f"Best parameters saved to: {best_params_file}")
        except Exception as e:
            print(f"Error saving best parameters: {e}")
    else:
        print("\nNo trials completed successfully.")

    if args.storage:
        print(f"Study results are stored in: {args.storage}")
    else:
        print("Study results were stored in-memory only.")

if __name__ == '__main__':
    main()
