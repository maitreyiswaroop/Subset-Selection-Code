# E_Y_S_estimation.py
"""
Compares different estimators for the functional E[(E[Y|S])^2], where
S = X + sqrt(alpha) * epsilon, epsilon ~ N(0, I).

Estimators Compared:
1. Plugin estimator applied to (S, Y).
2. Influence Function (IF) based estimator applied to (S, Y).
3. Kernel (KeOps) based estimator: mean( (E[Y|S])^2 ), where E[Y|S] is estimated
   using estimate_conditional_keops based on E[Y|X].

Baseline ("Ground Truth"):
- Plugin estimator applied to (S_large, Y_large) from a much larger dataset.

The script iterates over random seeds, function classes (data generating processes),
and noise levels (alpha). It calculates the Mean Squared Error (MSE) of each
estimator relative to the baseline.
"""

import os
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm # Progress bar
import json

# --- Import necessary functions ---
try:
    from data import generate_data_continuous
    # Assuming estimators.py has the following functions:
    from estimators import (
        plugin_estimator_squared_conditional,
        IF_estimator_squared_conditional,
        plugin_estimator_conditional_mean,
        estimate_conditional_keops
    )
    print("Successfully imported data and estimator functions.")
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure data.py and estimators.py are accessible.")
    exit(1)

# --- Constants ---
EPS = 1e-8 # Small constant for numerical stability (e.g., division by std dev)

# --- Helper Functions ---

def standardize_data(X, Y):
    """Standardizes X (features) and Y (outcome)."""
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    Y_mean = np.mean(Y)
    Y_std = np.std(Y)

    X_std[X_std < EPS] = EPS # Avoid division by zero
    if Y_std < EPS:
        Y_std = EPS

    X_stdized = (X - X_mean) / X_std
    Y_stdized = (Y - Y_mean) / Y_std

    return X_stdized, Y_stdized, X_mean, X_std, Y_mean, Y_std

def generate_noisy_data(X_std, alpha_scalar, device):
    """Generates S = X + sqrt(alpha) * noise."""
    if alpha_scalar < 0:
        raise ValueError("alpha_scalar must be non-negative.")
    # Ensure X_std is a torch tensor
    if isinstance(X_std, np.ndarray):
        X_std_torch = torch.tensor(X_std, dtype=torch.float32).to(device)
    else:
        X_std_torch = X_std.to(device)

    noise = torch.randn_like(X_std_torch)
    # Element-wise multiplication with sqrt(alpha)
    S = X_std_torch + torch.sqrt(torch.tensor(alpha_scalar, device=device)) * noise
    return S

# --- Main Evaluation Function ---

def evaluate_estimators(config):
    """
    Runs the evaluation loop for the given configuration.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = []

    base_model_type = config['base_model_type'] # For plugin/IF estimators

    # --- Outer loops: Seed and Function Class ---
    for seed in tqdm(config['seeds'], desc="Seeds"):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for func_class in tqdm(config['function_classes'], desc=f"Func Classes (Seed {seed})", leave=False):

            # --- Generate Base Data (Small and Large) ---
            try:
                X_small_orig, Y_small_orig, _, _ = generate_data_continuous(
                    pop_id=0, m1=config['m1'], m=config['m'],
                    dataset_type=func_class, dataset_size=config['n_small'],
                    noise_scale=config['noise_scale'], seed=seed
                )
                X_large_orig, Y_large_orig, _, _ = generate_data_continuous(
                    pop_id=0, m1=config['m1'], m=config['m'],
                    dataset_type=func_class, dataset_size=config['n_large'],
                    noise_scale=config['noise_scale'], seed=seed + 1000 # Different seed for large data
                )
            except Exception as e:
                print(f"Error generating data for {func_class} (Seed {seed}): {e}")
                continue

            # --- Standardize Data ---
            X_small_std, Y_small_std, _, _, _, _ = standardize_data(X_small_orig, Y_small_orig)
            X_large_std, Y_large_std, _, _, _, _ = standardize_data(X_large_orig, Y_large_orig)

            # Precompute E[Y|X] on small standardized data (needed for KeOps)
            try:
                E_Yx_small_std_np = plugin_estimator_conditional_mean(
                    X_small_std, Y_small_std, estimator_type=base_model_type
                )
                E_Yx_small_std_torch = torch.tensor(E_Yx_small_std_np, dtype=torch.float32).to(device)
            except Exception as e:
                 print(f"Error precomputing E[Y|X] for {func_class} (Seed {seed}): {e}")
                 continue

            # --- Inner loop: Alpha ---
            for alpha_scalar in tqdm(config['alphas'], desc=f"Alphas ({func_class}, Seed {seed})", leave=False):

                start_time = time.time()

                # --- Generate S (Noisy Data) ---
                try:
                    # Ensure X data is on the correct device before generating S
                    X_small_std_torch = torch.tensor(X_small_std, dtype=torch.float32).to(device)
                    X_large_std_torch = torch.tensor(X_large_std, dtype=torch.float32).to(device)

                    S_small_torch = generate_noisy_data(X_small_std_torch, alpha_scalar, device)
                    S_large_torch = generate_noisy_data(X_large_std_torch, alpha_scalar, device)

                    # Convert S to numpy for estimators if needed
                    S_small_np = S_small_torch.cpu().numpy()
                    S_large_np = S_large_torch.cpu().numpy()
                    # Keep Y standardized as numpy arrays
                    Y_small_std_np = Y_small_std
                    Y_large_std_np = Y_large_std

                except Exception as e:
                    print(f"Error generating S for alpha={alpha_scalar:.3f} ({func_class}, Seed {seed}): {e}")
                    continue

                # --- Calculate Baseline ("Ground Truth") ---
                baseline_val = np.nan
                try:
                    # baseline_val = plugin_estimator_squared_conditional(
                    #     S_large_np, Y_large_std_np, estimator_type=base_model_type
                    # )
                    baseline_val = IF_estimator_squared_conditional(
                        S_large_np, Y_large_std_np, estimator_type=base_model_type
                    )
                except Exception as e:
                    print(f"Error calculating baseline for alpha={alpha_scalar:.3f} ({func_class}, Seed {seed}): {e}")

                # --- Calculate Estimator Values ---
                plugin_val = np.nan
                if_val = np.nan
                kernel_val = np.nan

                # Plugin Estimator
                try:
                    plugin_val = plugin_estimator_squared_conditional(
                        S_small_np, Y_small_std_np, estimator_type=base_model_type
                    )
                except Exception as e:
                    print(f"Error calculating Plugin for alpha={alpha_scalar:.3f} ({func_class}, Seed {seed}): {e}")

                # IF Estimator
                try:
                    if_val = IF_estimator_squared_conditional(
                        S_small_np, Y_small_std_np, estimator_type=base_model_type
                    )
                except Exception as e:
                    print(f"Error calculating IF for alpha={alpha_scalar:.3f} ({func_class}, Seed {seed}): {e}")

                # Kernel (KeOps) Estimator
                try:
                    # estimate_conditional_keops needs torch tensors
                    # It also needs alpha as a vector for per-feature noise
                    alpha_vec = torch.full((config['m'],), alpha_scalar, dtype=torch.float32, device=device)
                    alpha_vec_clamped = torch.clamp(alpha_vec, min=1e-6) # Clamp alpha for stability if KeOps needs it

                    # Ensure inputs to KeOps are tensors on the correct device
                    X_small_std_dev = X_small_std_torch # Already on device
                    S_small_dev = S_small_torch         # Already on device
                    E_Yx_small_std_dev = E_Yx_small_std_torch # Already on device

                    E_Ys_small_torch = estimate_conditional_keops(
                        X_small_std_dev, S_small_dev, E_Yx_small_std_dev, alpha_vec_clamped
                    )
                    kernel_val = torch.mean(E_Ys_small_torch**2).item()
                except Exception as e:
                    print(f"Error calculating Kernel for alpha={alpha_scalar:.3f} ({func_class}, Seed {seed}): {e}")

                # --- Store Results ---
                elapsed_time = time.time() - start_time
                result_row = {
                    'seed': seed,
                    'function_class': func_class,
                    'alpha': alpha_scalar,
                    'baseline': baseline_val,
                    'plugin_est': plugin_val,
                    'if_est': if_val,
                    'kernel_est': kernel_val,
                    'plugin_sq_err': (plugin_val - baseline_val)**2 if not np.isnan(plugin_val) and not np.isnan(baseline_val) else np.nan,
                    'if_sq_err': (if_val - baseline_val)**2 if not np.isnan(if_val) and not np.isnan(baseline_val) else np.nan,
                    'kernel_sq_err': (kernel_val - baseline_val)**2 if not np.isnan(kernel_val) and not np.isnan(baseline_val) else np.nan,
                    'time': elapsed_time
                }
                results.append(result_row)

    return pd.DataFrame(results)

# --- Plotting Function ---

def plot_results(df, save_dir):
    """Generates and saves plots of MSE vs Alpha."""
    if df.empty:
        print("No results to plot.")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Melt dataframe for easier plotting with seaborn
    df_melt = pd.melt(df,
                      id_vars=['seed', 'function_class', 'alpha'],
                      value_vars=['plugin_sq_err', 'if_sq_err', 'kernel_sq_err'],
                      var_name='estimator_type',
                      value_name='squared_error')

    # Clean estimator names
    df_melt['estimator_type'] = df_melt['estimator_type'].str.replace('_sq_err', '')

    # Calculate Mean Squared Error (MSE) across seeds
    mse_df = df_melt.groupby(['function_class', 'alpha', 'estimator_type'], as_index=False)['squared_error'].mean()
    mse_df.rename(columns={'squared_error': 'MSE'}, inplace=True)


    # Plot MSE vs Alpha for each function class
    func_classes = df['function_class'].unique()
    num_classes = len(func_classes)

    plt.figure(figsize=(7 * num_classes, 6)) # Adjust figure size dynamically

    for i, func_class in enumerate(func_classes):
        ax = plt.subplot(1, num_classes, i + 1)
        sns.lineplot(data=mse_df[mse_df['function_class'] == func_class],
                     x='alpha', y='MSE', hue='estimator_type', marker='o', ax=ax)
        ax.set_title(f'MSE vs Alpha ({func_class})')
        ax.set_xlabel('Alpha (Noise Level)')
        ax.set_ylabel('Mean Squared Error (MSE)')
        ax.set_yscale('log') # Log scale often helpful for MSE
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(title='Estimator')

    plt.tight_layout()
    plot_filename = os.path.join(save_dir, "estimator_comparison_mse_vs_alpha.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"Comparison plot saved to {plot_filename}")

# --- Argument Parser ---

def parse_args():
    parser = argparse.ArgumentParser(description='Compare E[(E[Y|S])^2] Estimators')
    # Data parameters
    parser.add_argument('--m1', type=int, default=4, help='Number of meaningful features')
    parser.add_argument('--m', type=int, default=20, help='Total number of features')
    parser.add_argument('--n-small', type=int, default=1000, help='Sample size for estimators')
    parser.add_argument('--n-large', type=int, default=50000, help='Sample size for baseline') # Reduced default
    parser.add_argument('--noise-scale', type=float, default=0.1, help='Noise added to Y in data generation')
    parser.add_argument('--function-classes', nargs='+', default=['linear_regression', 'cubic_regression', 'sinusoidal_regression', 'quadratic_regression'], help='Function classes for data generation')

    # Evaluation parameters
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 987], help='List of random seeds')
    parser.add_argument('--alphas', type=float, nargs='+', default=np.linspace(0.01, 5.0, 10).tolist(), help='List of alpha values (noise levels)') # Start slightly > 0
    parser.add_argument('--base-model-type', type=str, default='rf', choices=['rf', 'krr'], help='Base model for plugin/IF estimators')

    # Output parameters
    parser.add_argument('--save-dir', type=str, default='./results/estimator_comparison_results/', help='Directory to save results CSV and plots')

    return parser.parse_args()

# --- Main Execution ---

if __name__ == "__main__":
    args = parse_args()

    config = vars(args) # Convert args to dictionary

    print("Starting estimator evaluation with configuration:")
    print(json.dumps(config, indent=2, default=str)) # Use default=str for numpy arrays in alphas

    # Run evaluation
    results_df = evaluate_estimators(config)

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    csv_filename = os.path.join(args.save_dir, "comparison_results.csv")
    results_df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to {csv_filename}")

    # Plot results
    plot_results(results_df, args.save_dir)

    print("\nEvaluation finished.")
