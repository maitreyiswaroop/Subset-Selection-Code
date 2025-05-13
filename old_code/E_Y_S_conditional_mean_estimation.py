# E_Y_S_conditional_mean_estimation_v2.py
"""
Compares different estimators for the conditional mean E[Y|S], where
S = X + sqrt(alpha) * epsilon, epsilon ~ N(0, I).

Evaluation Strategy v2:
1. Generate a single large dataset (X_large, Y_large).
2. For each alpha:
   a. Generate S_large from X_large.
   b. Calculate baseline predictions ("ground truth") E_base[Y|S] by training a
      plugin model on the full (S_large, Y_large).
   c. Subsample indices to define a small dataset.
   d. Extract X_small, Y_small, S_small, and the corresponding baseline predictions
      GT_preds_small from the large dataset arrays based on the subsampled indices.
   e. Precompute E[Y|X]_small needed for the Kernel estimator using (X_small, Y_small).
   f. Train/apply Plugin, IF, and Kernel estimators using only the small dataset
      (X_small, Y_small, S_small, E[Y|X]_small) to get their predictions at S_small.
   g. Calculate MSE between each estimator's predictions and GT_preds_small.

Estimators Compared:
1. Plugin estimator: Model trained on (S_small, Y_small), predicts E[Y|S=s].
2. Influence Function (IF) based estimator: IF_estimator_conditional_mean applied
   to (S_small, Y_small) to predict E[Y|S=s].
3. Kernel (KeOps) based estimator: estimate_conditional_keops based on
   (X_small, E[Y|X]_small, S_small, alpha) to predict E[Y|S=s].

Baseline ("Ground Truth"):
- Plugin estimator (model) trained on the full (S_large, Y_large), evaluated at S_small.

The script iterates over random seeds, function classes, and noise levels (alpha).
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
from sklearn.base import clone # To ensure fresh models
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split # For subsampling

# --- Import necessary functions ---
try:
    from data import generate_data_continuous
    # Assuming estimators.py has the following functions:
    from estimators import (
        plugin_estimator_conditional_mean,  # Used for precomputing E[Y|X]_small
        IF_estimator_conditional_mean,      # Used for IF estimator
        estimate_conditional_keops          # Used for Kernel estimator
    )
    print("Successfully imported data and estimator functions.")
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure data.py and estimators.py are accessible.")
    exit(1)

# --- Constants ---
EPS = 1e-8 # Small constant for numerical stability

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

def evaluate_conditional_mean_estimators_v2(config):
    """
    Runs the evaluation loop using the revised strategy (subsampling from large).
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

            # --- Generate ONE Large Base Dataset ---
            try:
                X_large_orig, Y_large_orig, _, meaningful_indices_large = generate_data_continuous(
                    pop_id=0, m1=config['m1'], m=config['m'],
                    dataset_type=func_class, dataset_size=config['n_large'],
                    noise_scale=config['noise_scale'], seed=seed
                )
                print(f"Generated large data for {func_class} (Seed {seed}). Meaningful indices: {meaningful_indices_large}")
            except Exception as e:
                print(f"Error generating large data for {func_class} (Seed {seed}): {e}")
                continue

            # --- Standardize Large Data ---
            X_large_std, Y_large_std_np, _, _, _, _ = standardize_data(X_large_orig, Y_large_orig)
            X_large_std_torch = torch.tensor(X_large_std, dtype=torch.float32).to(device)

            # --- Inner loop: Alpha ---
            for alpha_scalar in tqdm(config['alphas'], desc=f"Alphas ({func_class}, Seed {seed})", leave=False):

                start_time = time.time()

                # --- Generate S_large (Noisy Data for Large Dataset) ---
                try:
                    S_large_torch = generate_noisy_data(X_large_std_torch, alpha_scalar, device)
                    S_large_np = S_large_torch.cpu().numpy()
                except Exception as e:
                    print(f"Error generating S_large for alpha={alpha_scalar:.3f} ({func_class}, Seed {seed}): {e}")
                    continue

                # --- Calculate Baseline Predictions on Large Dataset ---
                baseline_preds_large = np.full(config['n_large'], np.nan) # Initialize
                model_base = None # Define outside try block
                try:
                    # Choose base model
                    if base_model_type == 'rf':
                        model_base = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=seed*10)
                    elif base_model_type == 'krr':
                        model_base = KernelRidge(kernel='rbf', alpha=0.1) # Example parameters
                    else:
                        raise ValueError(f"Unsupported base_model_type: {base_model_type}")

                    # Train baseline model on the full large (S, Y) data
                    baseline_model = clone(model_base) # Use a fresh model instance
                    baseline_model.fit(S_large_np, Y_large_std_np)
                    # Predict baseline for ALL points in S_large
                    baseline_preds_large = baseline_model.predict(S_large_np)

                except Exception as e:
                    print(f"Error calculating baseline model/preds for alpha={alpha_scalar:.3f} ({func_class}, Seed {seed}): {e}")
                    # If baseline fails, we cannot compare, skip this alpha
                    continue

                # --- Subsample Small Dataset Indices ---
                # Ensure consistent subsampling for this alpha/seed/func_class
                subsample_indices = np.random.choice(config['n_large'], config['n_small'], replace=False)

                # --- Extract Small Dataset Components ---
                X_small_std = X_large_std[subsample_indices]
                Y_small_std_np = Y_large_std_np[subsample_indices]
                S_small_np = S_large_np[subsample_indices]
                S_small_torch = S_large_torch[subsample_indices] # Get torch tensor view/copy
                GT_preds_small = baseline_preds_large[subsample_indices] # Ground truth for the small set

                # --- Precompute E[Y|X]_small (needed for Kernel estimator) ---
                # Use the small dataset (standardized X and Y) for this
                E_Yx_small_std_np = np.full(config['n_small'], np.nan)
                try:
                    E_Yx_small_std_np = plugin_estimator_conditional_mean(
                        X_small_std, Y_small_std_np, estimator_type=base_model_type,
                        seed=seed
                    )
                    E_Yx_small_std_torch = torch.tensor(E_Yx_small_std_np, dtype=torch.float32).to(device)
                except Exception as e:
                    print(f"Error precomputing E[Y|X]_small for alpha={alpha_scalar:.3f} ({func_class}, Seed {seed}): {e}")
                    # If this fails, Kernel estimator cannot run
                    E_Yx_small_std_torch = None # Signal failure

                # --- Calculate Estimator Predictions using SMALL Dataset ---
                plugin_preds_at_S_small = np.full(config['n_small'], np.nan)
                if_preds_at_S_small = np.full(config['n_small'], np.nan)
                kernel_preds_at_S_small = np.full(config['n_small'], np.nan)

                # Plugin Estimator Predictions (Train on small, predict on small)
                try:
                    if model_base is None: # Should have been defined above
                         print(f"Skipping Plugin: model_base not defined.")
                    else:
                        plugin_model = clone(model_base) # Use a fresh model instance
                        plugin_model.fit(S_small_np, Y_small_std_np)
                        plugin_preds_at_S_small = plugin_model.predict(S_small_np)
                except Exception as e:
                    print(f"Error calculating Plugin preds (small) for alpha={alpha_scalar:.3f} ({func_class}, Seed {seed}): {e}")

                # IF Estimator Predictions (Apply to small)
                try:
                    if_preds_at_S_small = IF_estimator_conditional_mean(
                        S_small_np, Y_small_std_np, estimator_type=base_model_type,
                        n_folds=config.get('if_n_folds', 5), # Allow configuring folds for IF
                        k_neighbors_factor=config.get('if_k_factor', 0.1),
                        min_k_neighbors=config.get('if_min_k', 10),
                        seed=seed
                    )
                except Exception as e:
                    print(f"Error calculating IF preds (small) for alpha={alpha_scalar:.3f} ({func_class}, Seed {seed}): {e}")

                # Kernel (KeOps) Estimator Predictions (Needs X_small, S_small, E[Y|X]_small)
                try:
                    if E_Yx_small_std_torch is None:
                        print(f"Skipping Kernel: E[Y|X]_small precomputation failed.")
                    else:
                        alpha_vec = torch.full((config['m'],), alpha_scalar, dtype=torch.float32, device=device)
                        alpha_vec_clamped = torch.clamp(alpha_vec, min=1e-6)

                        # Need X_small as torch tensor
                        X_small_std_torch = torch.tensor(X_small_std, dtype=torch.float32).to(device)

                        # S_small_torch is already available and on device
                        # E_Yx_small_std_torch is already available and on device

                        kernel_preds_torch = estimate_conditional_keops(
                            X_small_std_torch, S_small_torch, E_Yx_small_std_torch, alpha_vec_clamped
                        )
                        kernel_preds_at_S_small = kernel_preds_torch.cpu().numpy()
                except Exception as e:
                    print(f"Error calculating Kernel preds (small) for alpha={alpha_scalar:.3f} ({func_class}, Seed {seed}): {e}")

                # --- Calculate MSE vs Ground Truth (GT_preds_small) ---
                def calculate_mse(preds, baseline):
                    if np.isnan(preds).any() or np.isnan(baseline).any(): return np.nan
                    if preds.shape != baseline.shape:
                        print(f"Warning: Shape mismatch between preds ({preds.shape}) and baseline ({baseline.shape}).")
                        return np.nan
                    return np.mean((preds - baseline)**2)

                plugin_mse = calculate_mse(plugin_preds_at_S_small, GT_preds_small)
                if_mse = calculate_mse(if_preds_at_S_small, GT_preds_small)
                kernel_mse = calculate_mse(kernel_preds_at_S_small, GT_preds_small)

                # --- Store Results ---
                elapsed_time = time.time() - start_time
                result_row = {
                    'seed': seed,
                    'function_class': func_class,
                    'alpha': alpha_scalar,
                    'plugin_mse': plugin_mse,
                    'if_mse': if_mse,
                    'kernel_mse': kernel_mse,
                    'time': elapsed_time
                }
                results.append(result_row)

    return pd.DataFrame(results)

# --- Plotting Function ---

def plot_conditional_mean_results(df, save_dir):
    """
    Generates and saves plots of E[Y|S] prediction MSE vs Alpha,
    showing the mean and confidence interval across seeds.
    """
    if df.empty:
        print("No results to plot.")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Melt dataframe for easier plotting with seaborn
    # This dataframe contains MSE for each seed, alpha, func_class, estimator_type
    df_melt = pd.melt(df,
                      id_vars=['seed', 'function_class', 'alpha'],
                      value_vars=['plugin_mse', 'if_mse', 'kernel_mse'],
                      var_name='estimator_type',
                      value_name='MSE') # Already MSE

    # Clean estimator names
    df_melt['estimator_type'] = df_melt['estimator_type'].str.replace('_mse', '')

    # Drop rows with NaN MSE before plotting to avoid issues with seaborn aggregation
    df_melt.dropna(subset=['MSE'], inplace=True)

    # Plot MSE vs Alpha for each function class
    func_classes = df['function_class'].unique()
    num_classes = len(func_classes)

    plt.figure(figsize=(7 * num_classes, 6)) # Adjust figure size dynamically

    all_plots_generated = True
    for i, func_class in enumerate(func_classes):
        ax = plt.subplot(1, num_classes, i + 1)
        # Filter data for the current function class
        plot_data = df_melt[df_melt['function_class'] == func_class]

        if plot_data.empty:
             print(f"Warning: No valid data to plot for function class '{func_class}'. Skipping subplot.")
             ax.set_title(f'No Data for {func_class}')
             ax.text(0.5, 0.5, 'No valid data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
             all_plots_generated = False
             continue

        # Use seaborn lineplot directly on the melted data.
        # Seaborn automatically calculates the mean and 95% CI across seeds.
        sns.lineplot(data=plot_data,
                     x='alpha',
                     y='MSE',
                     hue='estimator_type',
                     marker='o',
                     errorbar=('ci', 95), # Show 95% confidence interval bands
                     ax=ax)

        ax.set_title(f'E[Y|S] Prediction MSE vs Alpha ({func_class})')
        ax.set_xlabel('Alpha (Noise Level)')
        ax.set_ylabel('Mean Squared Error (vs Large N Baseline)')
        ax.set_yscale('log') # Log scale often helpful for MSE
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(title='Estimator')

    if not all_plots_generated:
         print("Plot generation incomplete due to missing data for some function classes.")

    plt.tight_layout()
    plot_filename = os.path.join(save_dir, "conditional_mean_estimation_v2_mse_vs_alpha_with_ci.png") # Updated filename
    try:
        plt.savefig(plot_filename)
        print(f"Conditional Mean comparison plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()

# --- Argument Parser ---

def parse_args():
    parser = argparse.ArgumentParser(description='Compare E[Y|S] Conditional Mean Estimators (v2 Strategy)')
    # Data parameters
    parser.add_argument('--m1', type=int, default=4, help='Number of meaningful features')
    parser.add_argument('--m', type=int, default=20, help='Total number of features')
    parser.add_argument('--n-small', type=int, default=1000, help='Subsample size for estimators')
    parser.add_argument('--n-large', type=int, default=20000, help='Sample size for large dataset/baseline')
    parser.add_argument('--noise-scale', type=float, default=0.1, help='Noise added to Y in data generation')
    parser.add_argument('--function-classes', nargs='+', default=['linear_regression', 'cubic_regression', 'sinusoidal_regression', 'quadratic_regression'], help='Function classes for data generation')

    # Evaluation parameters
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123], help='List of random seeds')
    parser.add_argument('--alphas', type=float, nargs='+', default=np.linspace(0.01, 3.0, 8).tolist(), help='List of alpha values (noise levels)')
    parser.add_argument('--base-model-type', type=str, default='rf', choices=['rf', 'krr'], help='Base model for plugin/IF estimators')
    # Add IF specific args if needed
    parser.add_argument('--if-n-folds', type=int, default=5, help='Number of folds for IF estimator CV')
    parser.add_argument('--if-k-factor', type=float, default=0.1, help='k-neighbors factor for IF estimator')
    parser.add_argument('--if-min-k', type=int, default=10, help='Min k-neighbors for IF estimator')


    # Output parameters
    parser.add_argument('--save-dir', type=str, default='./cond_mean_comparison_v2_results/', help='Directory to save results CSV and plots')

    return parser.parse_args()

# --- Main Execution ---

if __name__ == "__main__":
    args = parse_args()

    config = vars(args) # Convert args to dictionary

    print("Starting E[Y|S] conditional mean estimator evaluation (v2 Strategy) with configuration:")
    print(json.dumps(config, indent=2, default=str)) # Use default=str for numpy arrays in alphas

    # Run evaluation
    results_df = evaluate_conditional_mean_estimators_v2(config)

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    csv_filename = os.path.join(args.save_dir, "cond_mean_comparison_v2_results.csv")
    results_df.to_csv(csv_filename, index=False)
    print(f"text Results saved to {csv_filename}")
    # Plot results
    plot_conditional_mean_results(results_df, args.save_dir)
    print("Plotting complete. Check the save directory for results and plots.")
    print("E[Y|S] conditional mean estimator evaluation completed.")
    print("All tasks completed successfully.")
