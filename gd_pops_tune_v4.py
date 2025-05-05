import optuna
import numpy as np
import torch
import os
import json
import argparse
import time

# --- Import necessary functions from your script ---
# Adjust path if your script is named differently or located elsewhere
try:
    from gd_populations_v4 import (
        get_pop_data,
        run_experiment_multi_population,
        compute_population_stats, # Assuming this is useful for calculating recall
        get_latest_run_number,
        convert_numpy_to_python,
        N_FOLDS, # Import constants if needed
        CLAMP_MIN, CLAMP_MAX, EPS, FREEZE_THRESHOLD
    )
    print("Successfully imported from gd_populations_v4.py")
except ImportError as e:
    print(f"Error importing from gd_populations_v4.py: {e}")
    print("Please ensure gd_populations_v4.py and its dependencies are in the Python path.")
    exit()

# --- Define Fixed Parameters for the Optuna Study ---
# These are parameters you are *not* tuning in this study
FIXED_PARAMS = {
    'm1': 4,
    'm': 100,
    'dataset_size': 5000, # Smaller size for faster tuning? Adjust as needed.
    'noise_scale': 0.1,
    'corr_strength': 0.0,
    'num_epochs': 100, # Fewer epochs for faster tuning? Adjust.
    'batch_size': 10000, # Use full batch based on dataset_size for tuning
    'alpha_init': "random",
    'early_stopping_patience': 10, # Maybe increase slightly for tuning
    'smooth_minmax': float('inf'), # Use hard max for simplicity during tuning
    'param_freezing': True, # Or False, depending on what you want to test
    'run_baseline': False,
    'populations': ['linear_regression', 'sinusoidal_regression'], # Example
    'base_save_path': './optuna_runs/' # Base directory for Optuna trial results
}

# Determine budget based on fixed params (as in your main script)
FIXED_PARAMS['budget'] = FIXED_PARAMS['m1'] // 2 + len(FIXED_PARAMS['populations']) * FIXED_PARAMS['m1'] // 2

# Global variable for pop_configs (or pass differently if preferred)
# Needs to be generated once if data generation is fixed across trials
# Or generated inside objective if you want variation (use trial.number for seed)
POP_CONFIGS = [
    {'pop_id': i, 'dataset_type': FIXED_PARAMS['populations'][i]}
    for i in range(len(FIXED_PARAMS['populations']))
]

# --- Optuna Objective Function ---

def objective(trial: optuna.Trial):
    """
    Objective function for Optuna hyperparameter optimization.
    Aims to maximize recall.
    """
    # --- Suggest Hyperparameters ---
    # Using trial.suggest_... methods
    suggested_params = {
        'learning_rate': trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-5, 1e-1, log=True),
        'reg_type': trial.suggest_categorical("reg_type", ["Reciprocal_L1", "Quadratic_Barrier", "Exponential", "None", "Max_Dev" ]),
        'optimizer_type': trial.suggest_categorical("optimizer_type", ["sgd", "adam"]),
        'k_kernel': trial.suggest_int("k_kernel", 100, 3000), # Adjust range based on dataset_size
        'num_mc_samples': trial.suggest_int("num_mc_samples", 10, 200), # Increase range
        'estimator_type': trial.suggest_categorical("estimator_type", ["plugin", "if"]),
        'base_model_type': trial.suggest_categorical("base_model_type", ["rf", "krr", "xgb"]),
        # Add other parameters you want to tune (e.g., alpha_init noise?)
    }

    # Combine fixed and suggested parameters
    run_params = {**FIXED_PARAMS, **suggested_params}

    # Unique save path for this trial
    run_no = trial.number # Use Optuna trial number
    save_path = os.path.join(run_params['base_save_path'], f'trial_{run_no}/')
    # Optuna can run in parallel, ensure directory creation is safe
    try:
        os.makedirs(save_path, exist_ok=True)
    except FileExistsError:
        pass # Another process might have created it

    # Use trial number to ensure different seeds if desired, or keep fixed seed
    current_seed = trial.number # Example: different seed per trial
    # current_seed = 42 # Example: fixed seed for all trials

    print(f"\n--- Starting Optuna Trial {run_no} ---")
    print(f"Params: {suggested_params}")

    try:
        # Run the experiment with the current set of hyperparameters
        results = run_experiment_multi_population(
            pop_configs=POP_CONFIGS,
            m1=run_params['m1'],
            m=run_params['m'],
            dataset_size=run_params['dataset_size'],
            budget=run_params['budget'],
            noise_scale=run_params['noise_scale'],
            corr_strength=run_params['corr_strength'],
            num_epochs=run_params['num_epochs'],
            reg_type=run_params['reg_type'],
            reg_lambda=run_params['reg_lambda'],
            learning_rate=run_params['learning_rate'],
            batch_size=run_params['batch_size'],
            optimizer_type=run_params['optimizer_type'],
            seed=current_seed, # Use trial-specific or fixed seed
            alpha_init=run_params['alpha_init'],
            k_kernel=run_params['k_kernel'],
            num_mc_samples=run_params['num_mc_samples'],
            estimator_type=run_params['estimator_type'],
            base_model_type=run_params['base_model_type'],
            early_stopping_patience=run_params['early_stopping_patience'],
            save_path=save_path, # Save results for inspection
            smooth_minmax=run_params['smooth_minmax'],
            param_freezing=run_params['param_freezing'],
            run_baseline=run_params['run_baseline'],
            verbose=False # Keep verbose off during Optuna runs unless debugging
        )

        # --- Calculate the Metric to Optimize (Recall) ---
        if results['final_alpha'] is None:
             print(f"Trial {run_no} failed to produce final alpha. Returning low recall.")
             return 0.0 # Assign a poor value if the run failed badly

        final_alpha = np.array(results['final_alpha'])
        # Use the budget defined for the run
        selected_indices = np.argsort(final_alpha)[:run_params['budget']]

        # Combine all true meaningful indices across populations
        all_meaningful_indices = set()
        for indices in results['meaningful_indices']:
            all_meaningful_indices.update(indices)

        if not all_meaningful_indices:
             print(f"Trial {run_no}: No meaningful indices found. Returning 0.0 recall.")
             return 0.0 # Or 1.0 if selecting nothing is correct? Assume 0.0

        # Calculate recall
        selected_set = set(selected_indices)
        true_positives = selected_set.intersection(all_meaningful_indices)
        recall = len(true_positives) / len(all_meaningful_indices)

        print(f"Trial {run_no} completed. Final Objective: {results.get('final_objective', 'N/A'):.4f}, Recall: {recall:.4f}")

        # --- Optuna Pruning (Optional) ---
        # Report intermediate results for potential pruning
        # Example: report final objective, Optuna can prune if it's already bad
        final_objective_val = results.get('final_objective', float('inf'))
        if np.isnan(final_objective_val) or np.isinf(final_objective_val):
             final_objective_val = float('inf') # Use a value indicating failure
        trial.report(final_objective_val, step=run_params['num_epochs'])

        # Check if trial should be pruned based on intermediate value
        if trial.should_prune():
            print(f"Trial {run_no} pruned.")
            raise optuna.TrialPruned()

        return recall # Optuna will maximize this value

    except Exception as e:
        print(f"Trial {run_no} failed with error: {e}")
        # Optionally: return a very bad value or re-raise depending on Optuna setup
        return 0.0 # Indicate failure with a low recall value

# --- Main Optuna Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Optimization')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of Optuna trials to run')
    parser.add_argument('--study-name', type=str, default='multi_pop_vss_study', help='Name for the Optuna study')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_study.db', help='Database storage URL for Optuna study')
    optuna_args = parser.parse_args()

    print(f"Starting Optuna study '{optuna_args.study_name}' with {optuna_args.n_trials} trials.")
    print(f"Using storage: {optuna_args.storage}")

    # Create or load the study
    # Use storage to allow resuming and parallel execution
    study = optuna.create_study(
        study_name=optuna_args.study_name,
        storage=optuna_args.storage,
        direction="maximize", # We want to maximize recall
        load_if_exists=True, # Resume study if it already exists
        pruner=optuna.pruners.MedianPruner() # Example pruner
    )

    # Start the optimization
    start_time = time.time()
    try:
        study.optimize(
            objective,
            n_trials=optuna_args.n_trials,
            # timeout=600 # Optional: set a time limit in seconds
            # n_jobs=-1 # Optional: run trials in parallel (requires care with seeding and file I/O)
        )
    except KeyboardInterrupt:
        print("Optimization stopped manually.")

    end_time = time.time()
    print(f"\nOptimization finished in {end_time - start_time:.2f} seconds.")

    # Print results
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    best_trial = study.best_trial
    print("Best trial:")
    print(f"  Value (Recall): {best_trial.value:.4f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # You can also explore other analysis plots provided by Optuna
    # e.g., optuna.visualization.plot_optimization_history(study)
    #       optuna.visualization.plot_param_importances(study)