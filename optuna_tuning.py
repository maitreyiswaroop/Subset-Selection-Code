#!/usr/bin/env python3
"""
Comprehensive hyperparameter tuning for gd_populations_v3.py
This script optimizes all the requested parameters:
- reg_type
- reg_lambda
- learning_rate
- batch_size
- optimizer_type
- alpha_init
- estimator_type
"""

import os
import json
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import subprocess
import re
import argparse
from datetime import datetime

# Parameter search spaces - comprehensive as requested
SEARCH_RANGES = {
    "reg_type": ["None", "Reciprocal_L1", "Quadratic_Barrier", "Exponential", "Max_Dev"],
    "reg_lambda": (0.00001, 0.1),
    "learning_rate": (0.001, 0.5),
    "batch_size": [64, 128, 256, 500],
    "optimizer_type": ["adam", "sgd"],
    "alpha_init": ["random", "random_5"],
    "estimator_type": ["plugin", "if"]
}

def parse_args():
    parser = argparse.ArgumentParser(description='Comprehensive Optuna hyperparameter optimization')
    parser.add_argument('--n-trials', type=int, default=50, 
                      help='Number of Optuna trials')
    parser.add_argument('--study-name', type=str, 
                      default=f'comprehensive_study_{datetime.now().strftime("%Y%m%d_%H%M%S")}', 
                      help='Name of the Optuna study')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_comprehensive.db', 
                      help='Storage URL for Optuna database')
    parser.add_argument('--output-dir', type=str, default='./optuna_results', 
                      help='Directory to save optimization results')
    # Fixed parameters
    parser.add_argument('--m1', type=int, default=4)
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--dataset-size', type=int, default=1000)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--populations', nargs='+', default=['resnet', 'resnet', 'resnet'])
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--base-model-type', type=str, default='rf')
    parser.add_argument('--param-freezing', action='store_true', default=True)
    return parser.parse_args()

def extract_recall_from_output(output):
    """Extract the recall value from script output."""
    match = re.search(r"Recall: ([\d\.]+)", output)
    if match:
        return float(match.group(1))
    else:
        print("Warning: Couldn't extract recall from output")
        return 0.0

def objective(trial, fixed_params):
    """Optuna objective function that runs the gradient descent populations script."""
    # Sample ALL hyperparameters from the search space
    reg_type = trial.suggest_categorical("reg_type", SEARCH_RANGES["reg_type"])
    reg_lambda = trial.suggest_float("reg_lambda", *SEARCH_RANGES["reg_lambda"], log=True)
    learning_rate = trial.suggest_float("learning_rate", *SEARCH_RANGES["learning_rate"], log=True)
    batch_size = trial.suggest_categorical("batch_size", SEARCH_RANGES["batch_size"])
    optimizer_type = trial.suggest_categorical("optimizer_type", SEARCH_RANGES["optimizer_type"])
    alpha_init = trial.suggest_categorical("alpha_init", SEARCH_RANGES["alpha_init"])
    estimator_type = trial.suggest_categorical("estimator_type", SEARCH_RANGES["estimator_type"])
    
    # Create a directory for this trial's results
    temp_dir = os.path.join(fixed_params["output_dir"], f"trial_{trial.number}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Construct command to run the script with the sampled parameters
    cmd = [
        "python3", "gd_populations_v3.py",
        "--m1", str(fixed_params["m1"]),
        "--m", str(fixed_params["m"]),
        "--dataset-size", str(fixed_params["dataset_size"]),
        "--num-epochs", str(fixed_params["num_epochs"]),
        "--reg-type", reg_type,
        "--reg-lambda", str(reg_lambda),
        "--learning-rate", str(learning_rate),
        "--batch-size", str(batch_size),
        "--optimizer-type", optimizer_type,
        "--seed", str(fixed_params["seed"]),
        "--alpha-init", alpha_init,
        "--estimator-type", estimator_type,
        "--populations", *fixed_params["populations"],
        "--base-model-type", fixed_params["base_model_type"],
        "--save-path", temp_dir
    ]
    
    # # Add populations
    # for pop in fixed_params["populations"]:
    #     cmd.append("--populations")
    #     cmd.append(pop)
    
    # Add parameter freezing if enabled
    if fixed_params["param_freezing"]:
        cmd.append("--param-freezing")
    
    # Execute the script and capture output
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    stdout, stderr = process.communicate()
    
    # Print some output for monitoring
    if process.returncode == 0:
        # Extract recall from output
        recall = extract_recall_from_output(stdout)
        print(f"Trial {trial.number}: reg_type={reg_type}, reg_lambda={reg_lambda:.6f}, "
              f"lr={learning_rate:.6f}, batch_size={batch_size}, optimizer={optimizer_type}, "
              f"alpha_init={alpha_init}, estimator={estimator_type}, Recall={recall:.4f}")
        
        # Save trial details
        with open(os.path.join(temp_dir, "trial_details.json"), "w") as f:
            json.dump({
                "trial_number": trial.number,
                "params": {
                    "reg_type": reg_type,
                    "reg_lambda": reg_lambda,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "optimizer_type": optimizer_type,
                    "alpha_init": alpha_init,
                    "estimator_type": estimator_type
                },
                "recall": recall
            }, f, indent=2)
        
        # Save stdout for reference
        with open(os.path.join(temp_dir, "stdout.txt"), "w") as f:
            f.write(stdout)
        
        return recall
    else:
        print(f"Trial {trial.number} failed with error: {stderr}")
        with open(os.path.join(temp_dir, "stderr.txt"), "w") as f:
            f.write(stderr)
        return 0.0  # Return lowest possible recall on failure

def save_results(study, output_dir):
    """Save the optimization results."""
    os.makedirs(output_dir, exist_ok=True)
    
    best_params = study.best_params
    best_value = study.best_value
    
    # Collect all trials data
    trials_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            data = {
                "number": trial.number,
                "params": trial.params,
                "recall": trial.value
            }
            trials_data.append(data)
    
    # Save best parameters
    with open(os.path.join(output_dir, 'best_params.json'), 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_recall': best_value
        }, f, indent=4)
    
    # Save all trials data
    with open(os.path.join(output_dir, 'all_trials.json'), 'w') as f:
        json.dump(trials_data, f, indent=4)
    
    # Create a bash script with the best parameters
    with open(os.path.join(output_dir, 'run_best_params.sh'), 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write('# Best hyperparameters found by Optuna\n')
        f.write(f'# Best recall: {best_value:.4f}\n\n')
        
        f.write('PARAMS="--m1 4 \\\n')
        f.write('--m 20 \\\n')
        f.write('--dataset-size 1000 \\\n')
        for param, value in best_params.items():
            param_name = param.replace('_', '-')
            f.write(f'--{param_name} {value} \\\n')
        f.write('--seed 17 \\\n')
        f.write('--base-model-type rf \\\n')
        f.write('--param-freezing"\n\n')
        
        f.write('echo "Running gd_populations_v3.py with optimized parameters:"\n')
        f.write('echo $PARAMS\n\n')
        f.write('python3 gd_populations_v3.py $PARAMS\n')
    
    # Make the script executable
    os.chmod(os.path.join(output_dir, 'run_best_params.sh'), 0o755)
    
    print(f"Best parameters (recall = {best_value:.4f}):")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Fixed parameters that won't be tuned
    fixed_params = {
        "m1": args.m1,
        "m": args.m,
        "dataset_size": args.dataset_size,
        "num_epochs": args.num_epochs,
        "populations": args.populations,
        "seed": args.seed,
        "base_model_type": args.base_model_type,
        "param_freezing": args.param_freezing,
        "output_dir": args.output_dir
    }
    
    # Configure the Optuna sampler and pruner
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    
    # Create or load the study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",  # Maximize recall
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    # Run optimization
    try:
        study.optimize(
            lambda trial: objective(trial, fixed_params), 
            n_trials=args.n_trials
        )
    except KeyboardInterrupt:
        print("Optimization interrupted! Saving current best results...")
    
    # Save results
    save_results(study, args.output_dir)
    
    # Generate visualizations if possible
    try:
        # Import visualization components
        from optuna.visualization import plot_optimization_history, plot_param_importances
        import matplotlib.pyplot as plt
        
        # Create visualization directory
        viz_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate plots
        plot_optimization_history(study).write_image(
            os.path.join(viz_dir, "optimization_history.png"))
        
        plot_param_importances(study).write_image(
            os.path.join(viz_dir, "param_importances.png"))
        
        print(f"Visualizations saved to {viz_dir}")
    except ImportError:
        print("Visualization libraries not available. Skipping visualization generation.")

if __name__ == "__main__":
    main()