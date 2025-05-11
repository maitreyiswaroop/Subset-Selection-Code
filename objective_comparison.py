# objective_comparison.py
"""
Compares methods (MC+Kernel with alpha/theta param, IF) for computing
the objective L = T1 - T2 + P and its components against the analytical
solution for a linear data generating process.

Focuses on the accuracy of the *values* obtained via different methods.

Ground Truth (Linear Case Y = XA + noise, X ~ N(0,I)):
  T1 = E[E[Y|X]^2] = E[(XA)^2] = sum(A_i^2)
  T2(alpha) = E[E[Y|S]^2] = sum(A_i^2 / (1+alpha_i))
  L(alpha) - P(alpha) = T1 - T2(alpha) = sum(A_i^2 * alpha_i / (1+alpha_i))
  E[Y|S=s] = s @ (A / (1+alpha))
"""

import os
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
import json
import math
from typing import List, Dict, Any, Optional, Tuple

# --- Import project modules ---
try:
    from estimators import (
        estimate_conditional_keops, # Original kernel estimator
        plugin_estimator_conditional_mean,
        IF_estimator_squared_conditional, # IF estimator for T2
        IF_estimator_conditional_mean, # IF estimator for E[Y|X]
        estimate_conditional_keops_flexible, # Flexible kernel estimator
        estimate_T2_mc_flexible, # Flexible MC estimator for T2
        estimate_E_Y_S_kernel_flexible, # Flexible kernel estimator for E[Y|S]
    )
    from gd_pops_v6 import compute_penalty
    print("Successfully imported functions.")
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure estimators.py and gd_pops_v6.py are accessible.")
    exit(1)

from global_vars import *
# --- Constants ---
# Inherited from gd_pops_v6 via import

# --- Helper Functions ---
def standardize_data(X, Y):
    """Standardizes X (features) and Y (outcome). Returns standardized data and original means/stds."""
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

# --- Analytical Functions (Linear Case) ---
def analytical_T1(A: np.ndarray) -> float:
    """Calculates analytical T1 = E[E[Y|X]^2] = sum(A_i^2) for X~N(0,I)."""
    return np.sum(A**2)

def analytical_T2(A: np.ndarray, alpha: np.ndarray) -> float:
    """Calculates analytical T2 = E[E[Y|S]^2] = sum(A_i^2 / (1+alpha_i))."""
    alpha_safe = np.maximum(alpha, 0)
    return np.sum(A**2 / (1.0 + alpha_safe))

def analytical_objective(A: np.ndarray, alpha: np.ndarray) -> float:
    """Calculates analytical L - P = T1 - T2 = sum(A_i^2 * alpha_i / (1+alpha_i))."""
    alpha_safe = np.maximum(alpha, 0)
    return np.sum(A**2 * alpha_safe / (1.0 + alpha_safe))

def analytical_E_Y_S(S: np.ndarray, A: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Calculates analytical E[Y|S] = S·(A / (1 + alpha))."""
    alpha_safe = np.maximum(alpha, 1e-12)
    return S @ (A / (1.0 + alpha_safe))

def analytical_gradient_objective(A: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Calculates analytical gradient dL/d(alpha_i) = A_i^2 / (1+alpha_i)^2."""
    alpha_safe = np.maximum(alpha, 1e-12)
    return A**2 / (1.0 + alpha_safe)**2

# --- Estimator Functions ---

# --- Main Evaluation Function ---

def run_comparison(config: Dict):
    """Runs the comparison study."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = []
    m = config['m'] # Number of features

    # --- Loop over Seeds ---
    for seed in tqdm(config['seeds'], desc="Seeds"):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        # --- Generate Base Data ---
        n_base = config['n_base_t1']
        X_base_np = np.random.normal(0, 1, (n_base, m))
        A = np.random.uniform(-config['A_scale'], config['A_scale'], m)
        Y_base_np = X_base_np @ A + np.random.normal(0, config['y_noise_scale'], n_base)

        # --- Estimate T1 using the base data ---
        _, _, X_base_mean, X_base_std, Y_base_mean, Y_base_std = standardize_data(X_base_np, Y_base_np)
        E_Yx_base_orig_np = plugin_estimator_conditional_mean(
            X_base_np, Y_base_np, estimator_type=config['base_model_type'], n_folds=1
        )
        E_Yx_base_std_np = (E_Yx_base_orig_np - Y_base_mean) / max(Y_base_std, EPS)
        T1_estimated_std = np.mean(E_Yx_base_std_np**2)
        T1_analytical = analytical_T1(A)
        print(f"Seed {seed}: Analytical T1={T1_analytical:.4f}, Estimated T1_std={T1_estimated_std:.4f}")

        # --- Loop over Sample Sizes (N) ---
        for n in tqdm(config['sample_sizes'], desc=f"Sample Sizes (Seed {seed})", leave=False):
            indices = np.random.choice(n_base, n, replace=False)
            X_np = X_base_np[indices]
            Y_np = Y_base_np[indices]

            # Standardize the subsample
            X_std_np, Y_std_np, X_sub_mean, X_sub_std_vals, Y_sub_mean, Y_sub_std_val = standardize_data(X_np, Y_np) # Get means/stds
            X_std_torch = torch.tensor(X_std_np, dtype=torch.float32).to(device)
            Y_std_torch = torch.tensor(Y_std_np, dtype=torch.float32).to(device)

            # Precompute E[Y_std|X_std] for the subsample
            E_Yx_orig_np_sub = plugin_estimator_conditional_mean(
                X_np, Y_np, estimator_type=config['base_model_type'], n_folds=1
            )
            E_Yx_std_np_sub = (E_Yx_orig_np_sub - Y_sub_mean) / Y_sub_std_val
            E_Yx_std_torch_sub = torch.tensor(E_Yx_std_np_sub, dtype=torch.float32).to(device)

            # --- Loop over Alpha Values ---
            for alpha_val in tqdm(config['alpha_values'], desc=f"Alphas (N={n}, Seed {seed})", leave=False):
                alpha_np = np.full(m, alpha_val)
                alpha_torch = torch.tensor(alpha_np, dtype=torch.float32, device=device, requires_grad=False)
                alpha_safe_for_log = np.maximum(alpha_np, 1e-20)
                theta_np = np.log(alpha_safe_for_log)
                theta_torch = torch.tensor(theta_np, dtype=torch.float32, device=device, requires_grad=False)

                # --- Analytical Ground Truth (based on alpha) ---
                T2_true = analytical_T2(A, alpha_np)
                L_true = T1_analytical - T2_true
                grad_L_true = analytical_gradient_objective(A, alpha_np)

                # --- Estimate T2: MC+Kernel (alpha param) ---
                T2_est_mc_alpha = estimate_T2_mc_flexible(
                    X_std_torch, E_Yx_std_torch_sub, alpha_torch.clone().requires_grad_(True),
                    'alpha', config['n_mc_samples_obj'], config['k_kernel']
                ).item()
                L_est_mc_alpha = T1_estimated_std - T2_est_mc_alpha

                # --- Estimate T2: MC+Kernel (theta param) ---
                T2_est_mc_theta = estimate_T2_mc_flexible(
                    X_std_torch, E_Yx_std_torch_sub, theta_torch.clone().requires_grad_(True),
                    'theta', config['n_mc_samples_obj'], config['k_kernel']
                ).item()
                L_est_mc_theta = T1_estimated_std - T2_est_mc_theta

                # --- Estimate T2: IF Estimator ---
                alpha_val_detached = alpha_torch.detach().clone().clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
                epsilon = torch.randn_like(X_std_torch)
                S_std_torch = X_std_torch + epsilon * torch.sqrt(alpha_val_detached)
                S_std_np = S_std_torch.cpu().numpy()

                T2_est_if = IF_estimator_squared_conditional(
                    S_std_np, Y_std_np, estimator_type=config['base_model_type'], n_folds=config['if_n_folds']
                )
                L_est_if = T1_estimated_std - T2_est_if if not np.isnan(T2_est_if) else np.nan

                # --- Estimate E[Y|S] using Kernel (alpha & theta) vs Analytical ---
                # Analytical E[Y|S] (standardized)
                # FIX: Use correct std dev variable X_sub_std_vals
                S_orig_np = S_std_np * X_sub_std_vals + X_sub_mean # Need S on original scale
                E_Y_S_true_orig = analytical_E_Y_S(S_orig_np, A, alpha_np)
                # FIX: Use correct std dev variable Y_sub_std_val
                E_Y_S_true_std = (E_Y_S_true_orig - Y_sub_mean) / Y_sub_std_val

                # Kernel Estimate E[Y_std|S_std] (alpha param)
                E_Y_S_est_kernel_alpha = estimate_E_Y_S_kernel_flexible(
                    X_std_torch, E_Yx_std_torch_sub, alpha_torch, 'alpha', k_kernel=config['k_kernel']
                ).cpu().numpy()

                # Kernel Estimate E[Y_std|S_std] (theta param)
                E_Y_S_est_kernel_theta = estimate_E_Y_S_kernel_flexible(
                    X_std_torch, E_Yx_std_torch_sub, theta_torch, 'theta', k_kernel=config['k_kernel']
                ).cpu().numpy()

                # --- Calculate Errors ---
                t2_err_mc_alpha = abs(T2_est_mc_alpha - T2_true)
                t2_err_mc_theta = abs(T2_est_mc_theta - T2_true)
                t2_err_if = abs(T2_est_if - T2_true) if not np.isnan(T2_est_if) else np.nan

                l_err_mc_alpha = abs(L_est_mc_alpha - L_true)
                l_err_mc_theta = abs(L_est_mc_theta - L_true)
                l_err_if = abs(L_est_if - L_true) if not np.isnan(L_est_if) else np.nan

                # Compare shapes before MSE calculation
                eys_mse_alpha = np.nan
                if E_Y_S_est_kernel_alpha.shape == E_Y_S_true_std.shape:
                    eys_mse_alpha = np.mean((E_Y_S_est_kernel_alpha - E_Y_S_true_std)**2)
                else:
                    print(f"Shape mismatch! Kernel Alpha E[Y|S]: {E_Y_S_est_kernel_alpha.shape}, True E[Y|S]: {E_Y_S_true_std.shape}")

                eys_mse_theta = np.nan
                if E_Y_S_est_kernel_theta.shape == E_Y_S_true_std.shape:
                    eys_mse_theta = np.mean((E_Y_S_est_kernel_theta - E_Y_S_true_std)**2)
                else:
                     print(f"Shape mismatch! Kernel Theta E[Y|S]: {E_Y_S_est_kernel_theta.shape}, True E[Y|S]: {E_Y_S_true_std.shape}")


                # --- Analyze Penalties ---
                penalty_results = {}
                alpha_torch_param = alpha_torch.clone().detach().requires_grad_(True)

                for pen_type in config['penalty_types']:
                     if pen_type.lower() == 'none': continue
                     P_val = compute_penalty(alpha_torch_param, pen_type, config['penalty_lambda'])
                     grad_P = torch.zeros_like(alpha_torch_param)
                     if P_val.requires_grad:
                         try:
                             grad_P = torch.autograd.grad(P_val, alpha_torch_param, retain_graph=False)[0]
                         except RuntimeError as e:
                             print(f"Warning: Autograd failed for penalty {pen_type}. Error: {e}")
                             grad_P = torch.full_like(alpha_torch_param, float('nan'))

                     grad_L_true_norm = np.linalg.norm(grad_L_true)
                     penalty_results[pen_type] = {
                         'value': P_val.item(),
                         'grad_norm': torch.linalg.norm(grad_P).item(),
                         'grad_ratio_norm': torch.linalg.norm(grad_P).item() / (grad_L_true_norm + EPS),
                     }

                results.append({
                    'seed': seed, 'n_samples': n, 'alpha_val': alpha_val,
                    'T1_est_std': T1_estimated_std, 'T1_true': T1_analytical,
                    # T2 Estimates and Errors
                    'T2_est_mc_alpha': T2_est_mc_alpha, 'T2_est_mc_theta': T2_est_mc_theta,
                    'T2_est_if': T2_est_if, 'T2_true': T2_true,
                    'T2_abs_err_mc_alpha': t2_err_mc_alpha, 'T2_abs_err_mc_theta': t2_err_mc_theta,
                    'T2_abs_err_if': t2_err_if,
                    # L Estimates and Errors
                    'L_est_mc_alpha': L_est_mc_alpha, 'L_est_mc_theta': L_est_mc_theta,
                    'L_est_if': L_est_if, 'L_true': L_true,
                    'L_abs_err_mc_alpha': l_err_mc_alpha, 'L_abs_err_mc_theta': l_err_mc_theta,
                    'L_abs_err_if': l_err_if,
                    # E[Y|S] Errors
                    'E_Y_S_mse_alpha': eys_mse_alpha, 'E_Y_S_mse_theta': eys_mse_theta,
                    # Penalties
                    'penalties': penalty_results,
                    'analytical_grad_L_norm': grad_L_true_norm,
                })

    return pd.DataFrame(results)

# --- Plotting Function ---
def plot_objective_comparison_results(df: pd.DataFrame, config: Dict, save_dir: str):
    """Generates plots summarizing the objective comparison."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n--- Generating Plots in {save_dir} ---")

    # Melt data for easier plotting
    df_melt_t2 = pd.melt(df, id_vars=['seed', 'n_samples', 'alpha_val'],
                         value_vars=['T2_abs_err_mc_alpha', 'T2_abs_err_mc_theta', 'T2_abs_err_if'],
                         var_name='Estimator', value_name='T2 Abs Error')
    df_melt_t2['Estimator'] = df_melt_t2['Estimator'].str.replace('T2_abs_err_', '')

    df_melt_L = pd.melt(df, id_vars=['seed', 'n_samples', 'alpha_val'],
                        value_vars=['L_abs_err_mc_alpha', 'L_abs_err_mc_theta', 'L_abs_err_if'],
                        var_name='Estimator', value_name='L Abs Error')
    df_melt_L['Estimator'] = df_melt_L['Estimator'].str.replace('L_abs_err_', '')

    df_melt_eys = pd.melt(df, id_vars=['seed', 'n_samples', 'alpha_val'],
                          value_vars=['E_Y_S_mse_alpha', 'E_Y_S_mse_theta'],
                          var_name='Parametrization', value_name='E[Y|S] MSE')
    df_melt_eys['Parametrization'] = df_melt_eys['Parametrization'].str.replace('E_Y_S_mse_', '')


    # Plot 1: T2 Estimation Error vs Alpha (faceted by N)
    plt.figure()
    g = sns.relplot(data=df_melt_t2, x='alpha_val', y='T2 Abs Error', col='n_samples',
                    hue='Estimator', style='Estimator',
                    kind='line', marker='o', errorbar='sd', col_wrap=3,
                    height=4, aspect=1.1)
    g.set_titles("N = {col_name}")
    g.set_axis_labels("Alpha Value", "Abs Error T2 (vs True)")
    g.map(plt.axhline, y=0, color='grey', linestyle='--', lw=1)
    g.set(yscale='log')
    plt.suptitle('T2 Estimation Error Comparison', y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(save_dir, "t2_error_comparison.png"))
    plt.close()

    # Plot 2: Objective (L) Estimation Error vs Alpha (faceted by N)
    plt.figure()
    g = sns.relplot(data=df_melt_L, x='alpha_val', y='L Abs Error', col='n_samples',
                    hue='Estimator', style='Estimator',
                    kind='line', marker='o', errorbar='sd', col_wrap=3,
                    height=4, aspect=1.1)
    g.set_titles("N = {col_name}")
    g.set_axis_labels("Alpha Value", "Abs Error L (vs True)")
    g.map(plt.axhline, y=0, color='grey', linestyle='--', lw=1)
    g.set(yscale='log')
    plt.suptitle('Objective L Estimation Error Comparison', y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(save_dir, "L_error_comparison.png"))
    plt.close()

    # Plot 3: E[Y|S] Estimation MSE vs Alpha (faceted by N) - Kernel only
    plt.figure()
    g = sns.relplot(data=df_melt_eys, x='alpha_val', y='E[Y|S] MSE', col='n_samples',
                    hue='Parametrization', style='Parametrization',
                    kind='line', marker='o', errorbar='sd', col_wrap=3,
                    height=4, aspect=1.1)
    g.set_titles("N = {col_name}")
    g.set_axis_labels("Alpha Value", "MSE E[Y|S] (Kernel vs True)")
    g.map(plt.axhline, y=0, color='grey', linestyle='--', lw=1)
    g.set(yscale='log')
    plt.suptitle('E[Y|S] Estimation MSE (Kernel: Alpha vs Theta Param)', y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(save_dir, "eys_mse_kernel_comparison.png"))
    plt.close()

    # Plot 4 & 5: Penalty Analysis (remain the same as before)
    # Plot 4: Penalty Analysis - Value vs Alpha
    penalty_data = []
    for _, row in df.iterrows():
        if isinstance(row.get('penalties'), dict):
            for pen_type, pen_vals in row['penalties'].items():
                 if isinstance(pen_vals, dict) and 'value' in pen_vals:
                     penalty_data.append({
                         'n_samples': row['n_samples'],
                         'alpha_val': row['alpha_val'],
                         'penalty_type': pen_type,
                         'value': pen_vals['value']
                     })
    if penalty_data:
        pen_df = pd.DataFrame(penalty_data)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=pen_df, x='alpha_val', y='value', hue='penalty_type', style='n_samples', marker='o')
        plt.title(f'Penalty Value vs Alpha (Lambda={config["penalty_lambda"]})')
        plt.xlabel('Alpha Value')
        plt.ylabel('Penalty Value P(alpha)')
        plt.yscale('log')
        plt.grid(True, alpha=0.5)
        plt.legend(title='Penalty Type (Style=N Samples)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "penalty_value_vs_alpha.png"))
        plt.close()
    else: print("Skipping penalty value plot - no valid data.")

    # Plot 5: Penalty Analysis - Gradient Norm Ratio vs Alpha
    penalty_grad_data = []
    for _, row in df.iterrows():
         if isinstance(row.get('penalties'), dict):
            for pen_type, pen_vals in row['penalties'].items():
                 if isinstance(pen_vals, dict) and 'grad_ratio_norm' in pen_vals: # Removed 'grad_ratio_mean' check
                     penalty_grad_data.append({
                         'n_samples': row['n_samples'],
                         'alpha_val': row['alpha_val'],
                         'penalty_type': pen_type,
                         'grad_ratio_norm': pen_vals['grad_ratio_norm'],
                         # 'grad_ratio_mean': pen_vals['grad_ratio_mean'] # Removed for simplicity
                     })
    if penalty_grad_data:
        pen_grad_df = pd.DataFrame(penalty_grad_data)
        plt.figure(figsize=(8, 6)) # Adjusted size
        sns.lineplot(data=pen_grad_df, x='alpha_val', y='grad_ratio_norm', hue='penalty_type', style='n_samples', marker='o')
        plt.title('Penalty Grad Norm / Objective Grad Norm')
        plt.xlabel('Alpha Value')
        plt.ylabel('Ratio of Norms ||∇P|| / ||∇L||')
        plt.yscale('log')
        plt.grid(True, alpha=0.5)
        plt.legend(title='Penalty (Style=N)', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.suptitle('Penalty Gradient Magnitude Relative to Objective Gradient')
        plt.tight_layout(rect=[0, 0, 0.85, 0.96]) # Adjust rect for legend
        plt.savefig(os.path.join(save_dir, "penalty_gradient_ratios_vs_alpha.png"))
        plt.close()
    else: print("Skipping penalty gradient plot - no valid data.")

    print("Plots generated.")


# --- Arg Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description='Objective Computation Comparison (Extended)')
    # Data args
    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--A-scale', type=float, default=1.0)
    parser.add_argument('--y-noise-scale', type=float, default=0.1)
    # Estimation args
    parser.add_argument('--n-base-t1', type=int, default=50000)
    parser.add_argument('--sample-sizes', type=int, nargs='+', default=[1000, 5000, 10000])
    parser.add_argument('--alpha-values', type=float, nargs='+', default=np.logspace(-2, 1.3, 10).tolist()) # Wider range, more points
    parser.add_argument('--n-mc-samples-obj', type=int, default=30)
    parser.add_argument('--base-model-type', type=str, default='xgb', choices=['rf', 'krr', 'xgb'])
    parser.add_argument('--if-n-folds', type=int, default=5, help='Folds for IF T2 estimator')
    parser.add_argument('--k-kernel', type=int, default=500, help='k for kernel estimators')
    # Penalty args
    parser.add_argument('--penalty-types', nargs='+', default=['Reciprocal_L1', 'Quadratic_Barrier', 'Exponential', 'Max_Dev'])
    parser.add_argument('--penalty-lambda', type=float, default=0.01)
    # Other args
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123])
    parser.add_argument('--save-dir', type=str, default='./results_compare_objective/linear_case/')

    return parser.parse_args()

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()
    config = vars(args)

    print("--- Starting Objective Comparison Study (Extended) ---")
    print(json.dumps(config, indent=2, default=str))

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'config_v2.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)

    # Run the comparison
    results_df = run_comparison(config)

    # Save detailed results
    results_df.to_pickle(os.path.join(args.save_dir, "comparison_results_v2.pkl"))
    # Save summary to CSV
    summary_cols = [
        'seed', 'n_samples', 'alpha_val',
        'T2_abs_err_mc_alpha', 'T2_abs_err_mc_theta', 'T2_abs_err_if',
        'L_abs_err_mc_alpha', 'L_abs_err_mc_theta', 'L_abs_err_if',
        'E_Y_S_mse_alpha', 'E_Y_S_mse_theta'
    ]
    results_df[summary_cols].to_csv(os.path.join(args.save_dir, "comparison_summary_v2.csv"), index=False)

    # Generate plots
    plot_objective_comparison_results(results_df, config, args.save_dir)

    print(f"\n--- Study Finished ---")
    print(f"Results saved in: {args.save_dir}")