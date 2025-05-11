# gradient_comparison_v2.py
"""
Compares gradient computation methods (Autograd MC vs REINFORCE) for both
alpha and theta=log(alpha) parameterizations against the analytical gradient
for a linear model.

Focuses on the accuracy and variance of the estimated gradients dL/d(param).

Ground Truth (Linear Case Y = XA + noise, X ~ N(0,I)):
  L(alpha) = sum(A_i^2 * alpha_i / (1+alpha_i)) + P(alpha)
  dL/d(alpha_i) = A_i^2 / (1+alpha_i)^2 + dP/d(alpha_i)
  dL/d(theta_i) = dL/d(alpha_i) * d(alpha_i)/d(theta_i) = dL/d(alpha_i) * alpha_i
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
from typing import Dict, List, Optional, Any, Tuple

# --- Import project modules ---
try:
    # Assuming estimators.py and gd_pops_v6 are accessible
    from estimators import plugin_estimator_conditional_mean, estimate_conditional_keops_flexible, estimate_gradient_autograd_flexible, estimate_gradient_reinforce_flexible
    # Need flexible kernel estimator and penalty function
    from gd_pops_v6 import compute_penalty
    from global_vars import *
    print("Successfully imported functions.")
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure estimators.py, gd_pops_v6.py, and objective_comparison.py are accessible.")
    exit(1)

# --- Analytical Functions (Linear Case) ---
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

def analytical_gradient_objective_alpha(A: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Calculates analytical gradient d(T1-T2)/d(alpha_i) = A_i^2 / (1+alpha_i)^2."""
    alpha_safe = np.maximum(alpha, 1e-12) # Avoid division by zero
    return A**2 / (1.0 + alpha_safe)**2

def analytical_gradient_penalty_alpha(alpha_torch: torch.Tensor,
                                      penalty_type: Optional[str],
                                      penalty_lambda: float) -> torch.Tensor:
    """Calculates analytical gradient dP/d(alpha) using autograd."""
    if penalty_type is None or penalty_lambda == 0 or penalty_type.lower() == "none":
        return torch.zeros_like(alpha_torch)

    alpha_torch_param = alpha_torch.clone().detach().requires_grad_(True)
    penalty_val = compute_penalty(alpha_torch_param, penalty_type, penalty_lambda)

    if penalty_val.requires_grad:
        try:
            grad_P = torch.autograd.grad(penalty_val, alpha_torch_param, retain_graph=False)[0]
            return grad_P
        except RuntimeError as e:
            print(f"Warning: Autograd failed for penalty {penalty_type}. Error: {e}")
            return torch.full_like(alpha_torch, float('nan')) # Indicate failure
    else:
        return torch.zeros_like(alpha_torch)

def analytical_gradient_total_alpha(A: np.ndarray, alpha_np: np.ndarray,
                                    penalty_type: Optional[str], penalty_lambda: float,
                                    device='cpu') -> np.ndarray:
    """Calculates the total analytical gradient dL/d(alpha)."""
    grad_main = analytical_gradient_objective_alpha(A, alpha_np)
    alpha_torch = torch.tensor(alpha_np, dtype=torch.float32, device=device)
    grad_penalty_torch = analytical_gradient_penalty_alpha(alpha_torch, penalty_type, penalty_lambda)
    if torch.isnan(grad_penalty_torch).any():
         print("Warning: NaN detected in analytical penalty gradient.")
         return np.full_like(grad_main, np.nan)
    grad_penalty = grad_penalty_torch.cpu().numpy()
    return grad_main + grad_penalty # Note: Objective is min(T2+P) -> L = T2+P -> dL/da = dT2/da + dP/da. But T1-T2 derivation used dL/da = -dT2/da + dP/da. Let's stick to dL/da = A^2/(1+a)^2 + dP/da

def analytical_gradient_total_theta(A: np.ndarray, alpha_np: np.ndarray,
                                     penalty_type: Optional[str], penalty_lambda: float,
                                     device='cpu') -> np.ndarray:
    """Calculates the total analytical gradient dL/d(theta) using chain rule."""
    grad_L_alpha = analytical_gradient_total_alpha(A, alpha_np, penalty_type, penalty_lambda, device)
    if np.isnan(grad_L_alpha).any():
        return grad_L_alpha # Propagate NaN
    # dL/dtheta = dL/dalpha * dalpha/dtheta = dL/dalpha * alpha
    # Ensure alpha is positive for multiplication
    alpha_safe = np.maximum(alpha_np, 1e-12)
    return grad_L_alpha * alpha_safe


# --- Main Evaluation Function ---
def run_gradient_comparison(config: Dict):
    """Runs the gradient comparison study."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    results = []
    m = config['m']

    # --- Loop over Seeds ---
    for seed in tqdm(config['seeds'], desc="Seeds"):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        # Generate Base Data
        n_base = config['n_base_eyx']
        X_base_np = np.random.normal(0, 1, (n_base, m))
        A = np.random.uniform(-config['A_scale'], config['A_scale'], m)
        Y_base_np = X_base_np @ A + np.random.normal(0, config['y_noise_scale'], n_base)

        # Standardize base data and precompute E[Y|X]
        X_base_std_np, _, _, _, Y_base_mean, Y_base_std = standardize_data(X_base_np, Y_base_np)
        E_Yx_base_orig_np = plugin_estimator_conditional_mean(
            X_base_np, Y_base_np, estimator_type=config['base_model_type'], n_folds=1
        )
        E_Yx_base_std_np = (E_Yx_base_orig_np - Y_base_mean) / max(Y_base_std, EPS)
        E_Yx_base_std_torch = torch.tensor(E_Yx_base_std_np, dtype=torch.float32).to(device)
        X_base_std_torch = torch.tensor(X_base_std_np, dtype=torch.float32).to(device)

        # --- Loop over Sample Sizes (N) ---
        for n in tqdm(config['sample_sizes'], desc=f"Sample Sizes (Seed {seed})", leave=False):
            indices = np.random.choice(n_base, n, replace=False)
            X_std_torch_sub = X_base_std_torch[indices]
            E_Yx_std_torch_sub = E_Yx_base_std_torch[indices]

            # --- Loop over Alpha Values ---
            for alpha_val in tqdm(config['alpha_values'], desc=f"Alphas (N={n}, Seed {seed})", leave=False):
                alpha_np = np.full(m, alpha_val)
                alpha_safe_for_log = np.maximum(alpha_np, 1e-20)
                theta_np = np.log(alpha_safe_for_log)

                # --- Analytical Gradients ---
                grad_L_true_alpha = analytical_gradient_total_alpha(
                    A, alpha_np, config['penalty_type'], config['penalty_lambda'], device=device
                )
                grad_L_true_theta = analytical_gradient_total_theta(
                    A, alpha_np, config['penalty_type'], config['penalty_lambda'], device=device
                )
                grad_L_true_alpha_torch = torch.from_numpy(grad_L_true_alpha).to(device)
                grad_L_true_theta_torch = torch.from_numpy(grad_L_true_theta).to(device)


                # --- Loop over Number of MC/Gradient Samples ---
                for n_grad in config['n_grad_samples_list']:

                    # --- Estimate Gradients (Multiple Runs for Variance) ---
                    ag_grads_alpha, ag_grads_theta = [], []
                    rf_grads_alpha, rf_grads_theta = [], []

                    for _ in range(config['n_variance_runs']):
                        # Autograd Alpha
                        alpha_torch_ag = torch.tensor(alpha_np, dtype=torch.float32, device=device, requires_grad=True)
                        grad_ag_a = estimate_gradient_autograd_flexible(
                            X_std_torch_sub, E_Yx_std_torch_sub, alpha_torch_ag, 'alpha',
                            n_grad, config['k_kernel'], config['penalty_type'], config['penalty_lambda']
                        )
                        ag_grads_alpha.append(grad_ag_a.detach())

                        # Autograd Theta
                        theta_torch_ag = torch.tensor(theta_np, dtype=torch.float32, device=device, requires_grad=True)
                        grad_ag_t = estimate_gradient_autograd_flexible(
                            X_std_torch_sub, E_Yx_std_torch_sub, theta_torch_ag, 'theta',
                            n_grad, config['k_kernel'], config['penalty_type'], config['penalty_lambda']
                        )
                        ag_grads_theta.append(grad_ag_t.detach())

                        # REINFORCE Alpha
                        alpha_torch_rf = torch.tensor(alpha_np, dtype=torch.float32, device=device)
                        grad_rf_a = estimate_gradient_reinforce_flexible(
                            X_std_torch_sub, E_Yx_std_torch_sub, alpha_torch_rf, 'alpha',
                            n_grad, config['k_kernel'], config['penalty_type'], config['penalty_lambda'], config['use_baseline']
                        )
                        rf_grads_alpha.append(grad_rf_a.detach())

                        # REINFORCE Theta
                        theta_torch_rf = torch.tensor(theta_np, dtype=torch.float32, device=device)
                        grad_rf_t = estimate_gradient_reinforce_flexible(
                            X_std_torch_sub, E_Yx_std_torch_sub, theta_torch_rf, 'theta',
                            n_grad, config['k_kernel'], config['penalty_type'], config['penalty_lambda'], config['use_baseline']
                        )
                        rf_grads_theta.append(grad_rf_t.detach())

                    # --- Aggregate Variance Runs ---
                    def aggregate_grads(grads_list):
                        if not grads_list: return torch.zeros(m, device=device), torch.zeros(m, device=device)
                        grads_stack = torch.stack(grads_list)
                        mean_grad = torch.mean(grads_stack, dim=0)
                        var_grad = torch.var(grads_stack, dim=0)
                        return mean_grad, var_grad

                    ag_mean_a, ag_var_a = aggregate_grads(ag_grads_alpha)
                    ag_mean_t, ag_var_t = aggregate_grads(ag_grads_theta)
                    rf_mean_a, rf_var_a = aggregate_grads(rf_grads_alpha)
                    rf_mean_t, rf_var_t = aggregate_grads(rf_grads_theta)

                    # --- Calculate Metrics ---
                    def cosine_similarity(g1, g2):
                        if g1.shape != g2.shape:
                            print(f"Warning: Gradient shapes do not match: {g1.shape} vs {g2.shape}")
                            return 0.0
                        g2 = g2.to(dtype=g1.dtype)
                        g1_norm = torch.linalg.norm(g1); g2_norm = torch.linalg.norm(g2)
                        if g1_norm < EPS or g2_norm < EPS or torch.isnan(g1).any() or torch.isnan(g2).any(): return 0.0
                        return (torch.dot(g1, g2) / (g1_norm * g2_norm)).item()

                    def l2_rel_error(g_est, g_true):
                        true_norm = torch.linalg.norm(g_true)
                        if true_norm < EPS or torch.isnan(g_est).any() or torch.isnan(g_true).any(): return torch.linalg.norm(g_est).item() if not torch.isnan(g_est).any() else float('inf')
                        return (torch.linalg.norm(g_est - g_true) / true_norm).item()

                    results.append({
                        'seed': seed, 'n_samples': n, 'alpha_val': alpha_val, 'n_grad_samples': n_grad,
                        # Autograd Alpha
                        'ag_cos_sim_alpha': cosine_similarity(ag_mean_a, grad_L_true_alpha_torch),
                        'ag_l2_rel_err_alpha': l2_rel_error(ag_mean_a, grad_L_true_alpha_torch),
                        'ag_var_mean_alpha': torch.mean(ag_var_a).item(),
                        # Autograd Theta
                        'ag_cos_sim_theta': cosine_similarity(ag_mean_t, grad_L_true_theta_torch),
                        'ag_l2_rel_err_theta': l2_rel_error(ag_mean_t, grad_L_true_theta_torch),
                        'ag_var_mean_theta': torch.mean(ag_var_t).item(),
                        # REINFORCE Alpha
                        'rf_cos_sim_alpha': cosine_similarity(rf_mean_a, grad_L_true_alpha_torch),
                        'rf_l2_rel_err_alpha': l2_rel_error(rf_mean_a, grad_L_true_alpha_torch),
                        'rf_var_mean_alpha': torch.mean(rf_var_a).item(),
                        # REINFORCE Theta
                        'rf_cos_sim_theta': cosine_similarity(rf_mean_t, grad_L_true_theta_torch),
                        'rf_l2_rel_err_theta': l2_rel_error(rf_mean_t, grad_L_true_theta_torch),
                        'rf_var_mean_theta': torch.mean(rf_var_t).item(),
                        # True Gradient Norms
                        'true_grad_norm_alpha': torch.linalg.norm(grad_L_true_alpha_torch).item(),
                        'true_grad_norm_theta': torch.linalg.norm(grad_L_true_theta_torch).item()
                    })

    return pd.DataFrame(results)

# --- Plotting Function ---
def plot_gradient_comparison_results(df: pd.DataFrame, config: Dict, save_dir: str):
    """Generates plots summarizing the gradient comparison."""
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n--- Generating Gradient Plots (v2) in {save_dir} ---")

    metrics = ['cos_sim', 'l2_rel_err', 'var_mean']
    methods = ['ag', 'rf']
    params = ['alpha', 'theta']
    titles = {
        'cos_sim': 'Cosine Similarity vs True Gradient',
        'l2_rel_err': 'L2 Relative Error vs True Gradient',
        'var_mean': 'Mean Gradient Variance Across Dimensions'
    }
    ylabels = {
        'cos_sim': 'Cosine Similarity',
        'l2_rel_err': 'L2 Relative Error',
        'var_mean': 'Mean Variance'
    }

    for metric in metrics:
        plt.figure(figsize=(18, 6)) # Wider figure for side-by-side

        # Plot Autograd (Alpha vs Theta)
        plt.subplot(1, 2, 1)
        metric_alpha_col = f'ag_{metric}_alpha'
        metric_theta_col = f'ag_{metric}_theta'
        if metric_alpha_col in df.columns and metric_theta_col in df.columns:
            df_melt_ag = pd.melt(df, id_vars=['seed', 'n_samples', 'alpha_val', 'n_grad_samples'],
                                 value_vars=[metric_alpha_col, metric_theta_col],
                                 var_name='Parametrization', value_name='MetricValue')
            df_melt_ag['Parametrization'] = df_melt_ag['Parametrization'].str.replace(f'ag_{metric}_', '')

            sns.lineplot(data=df_melt_ag, x='alpha_val', y='MetricValue', hue='Parametrization',
                         style='n_grad_samples', col='n_samples', marker='o', errorbar='sd')
            plt.title(f'Autograd: {titles[metric]}')
            plt.ylabel(ylabels[metric])
            plt.xlabel('Alpha Value')
            if 'var' in metric or 'l2' in metric: plt.yscale('log')
            plt.grid(True, alpha=0.5)
            plt.legend(title='Param (Style=N Grad Samples)')
        else:
            plt.title(f'Autograd: {titles[metric]} (Data Missing)')

        # Plot REINFORCE (Alpha vs Theta)
        plt.subplot(1, 2, 2)
        metric_alpha_col = f'rf_{metric}_alpha'
        metric_theta_col = f'rf_{metric}_theta'
        if metric_alpha_col in df.columns and metric_theta_col in df.columns:
            df_melt_rf = pd.melt(df, id_vars=['seed', 'n_samples', 'alpha_val', 'n_grad_samples'],
                                 value_vars=[metric_alpha_col, metric_theta_col],
                                 var_name='Parametrization', value_name='MetricValue')
            df_melt_rf['Parametrization'] = df_melt_rf['Parametrization'].str.replace(f'rf_{metric}_', '')

            sns.lineplot(data=df_melt_rf, x='alpha_val', y='MetricValue', hue='Parametrization',
                         style='n_grad_samples', col='n_samples', marker='x', errorbar='sd')
            plt.title(f'REINFORCE: {titles[metric]}')
            plt.ylabel(ylabels[metric])
            plt.xlabel('Alpha Value')
            if 'var' in metric or 'l2' in metric: plt.yscale('log')
            plt.grid(True, alpha=0.5)
            plt.legend(title='Param (Style=N Grad Samples)')
        else:
             plt.title(f'REINFORCE: {titles[metric]} (Data Missing)')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"grad_compare_{metric}_alpha_vs_theta.png"))
        plt.close()

    print("Gradient comparison plots generated.")

# --- Arg Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description='Gradient Computation Comparison (Alpha vs Theta)')
    # Data args
    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--A-scale', type=float, default=1.0)
    parser.add_argument('--y-noise-scale', type=float, default=0.1)
    # Estimation args
    parser.add_argument('--n-base-eyx', type=int, default=20000)
    parser.add_argument('--sample-sizes', type=int, nargs='+', default=[1000, 5000])
    parser.add_argument('--alpha-values', type=float, nargs='+', default=np.logspace(-2, 1, 5).tolist())
    parser.add_argument('--n-grad-samples-list', type=int, nargs='+', default=[10, 25, 50])
    parser.add_argument('--base-model-type', type=str, default='xgb', choices=['rf', 'krr', 'xgb'])
    parser.add_argument('--n-variance-runs', type=int, default=15)
    parser.add_argument('--k-kernel', type=int, default=500, help='k for kernel estimators')
    # Penalty args
    parser.add_argument('--penalty-type', type=str, default='Reciprocal_L1')
    parser.add_argument('--penalty-lambda', type=float, default=0.01)
    # REINFORCE args
    parser.add_argument('--use-baseline', action=argparse.BooleanOptionalAction, default=True)
    # Other args
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123])
    parser.add_argument('--save-dir', type=str, default='./results_compare_gradient/linear_case/')

    return parser.parse_args()

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()
    config = vars(args)

    print("--- Starting Gradient Comparison Study (Alpha vs Theta) ---")
    print(json.dumps(config, indent=2, default=str))

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'config_v2.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)

    # Run the comparison
    results_df = run_gradient_comparison(config)

    # Save detailed results
    results_df.to_pickle(os.path.join(args.save_dir, "gradient_comparison_results_v2.pkl"))
    results_df.to_csv(os.path.join(args.save_dir, "gradient_comparison_summary_v2.csv"), index=False)

    # Generate plots
    plot_gradient_comparison_results(results_df, config, args.save_dir)

    print(f"\n--- Study Finished ---")
    print(f"Results saved in: {args.save_dir}")
