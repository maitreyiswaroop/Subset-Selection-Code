# test_baseline.py

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
import argparse
from torch.utils.data import DataLoader
import re
import time
import math
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

# --- Import project modules ---
try:
    from data import generate_data_continuous, generate_data_continuous_with_corr
    from data_baseline_failures import get_pop_data_baseline_failures
    from estimators import (
        plugin_estimator_conditional_mean,
        plugin_estimator_squared_conditional,
        IF_estimator_conditional_mean,
        IF_estimator_squared_conditional,
        estimate_conditional_keops_flexible,
        estimate_T2_mc_flexible,
        estimate_T2_kernel_IF_like_flexible, # <-- New Estimator
        estimate_gradient_reinforce_flexible
    )
    from torch.optim import lr_scheduler
    print("Successfully imported data, estimators (v7), and scheduler functions.")
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure data.py, estimators.py are accessible and updated for v7.")
    exit(1)

# Assuming global_vars.py defines these appropriately
try:
    from global_vars import *
except ImportError:
    print("Warning: global_vars.py not found. Using placeholder values for gd_pops_v7.py.")
    EPS = 1e-9
    CLAMP_MIN_ALPHA = 1e-5
    CLAMP_MAX_ALPHA = 1e5
    THETA_CLAMP_MIN = math.log(CLAMP_MIN_ALPHA) if CLAMP_MIN_ALPHA > 0 else -11.5
    THETA_CLAMP_MAX = math.log(CLAMP_MAX_ALPHA) if CLAMP_MAX_ALPHA > 0 else 11.5
    N_FOLDS = 5
    FREEZE_THRESHOLD_ALPHA = 1e-4
    THETA_FREEZE_THRESHOLD = math.log(FREEZE_THRESHOLD_ALPHA) if FREEZE_THRESHOLD_ALPHA > 0 else -9.2

from sklearn.linear_model import Lasso

def standardize_data(X, Y):
    X_mean = np.mean(X, axis=0); X_std = np.std(X, axis=0)
    Y_mean = np.mean(Y); Y_std = np.std(Y)
    X_std[X_std < EPS] = EPS
    if Y_std < EPS: Y_std = EPS
    return (X - X_mean) / X_std, (Y - Y_mean) / Y_std, X_mean, X_std, Y_mean, Y_std

def get_pop_data(pop_configs, m1, m, dataset_size=10000, noise_scale=0.0, corr_strength=0.0, common_meaningful_indices=None, estimator_type="plugin", device="cpu", base_model_type="rf", seed=None):
    if common_meaningful_indices is None: common_meaningful_indices = np.arange(max(1, m1 // 2))
    pop_data_list = []
    for i, pop_config in enumerate(pop_configs):
        pop_id = pop_config.get('pop_id', i); dataset_type = pop_config['dataset_type']
        current_seed = seed + pop_id if seed is not None else None
        gen_fn = generate_data_continuous_with_corr
        X_np, Y_np, _, meaningful_idx = gen_fn(pop_id=pop_id, m1=m1, m=m, dataset_type=dataset_type, dataset_size=dataset_size, noise_scale=noise_scale, corr_strength=corr_strength, seed=current_seed, common_meaningful_indices=common_meaningful_indices)
        X_std_np, Y_std_np, _, _, _, _ = standardize_data(X_np, Y_np)
        pop_data_list.append({
            'pop_id': pop_id, 'X_std': torch.tensor(X_std_np, dtype=torch.float32).to(device),
            'Y_std': torch.tensor(Y_std_np, dtype=torch.float32).to(device),
            'X_raw': X_np, 'Y_raw': Y_np,
            'meaningful_indices': meaningful_idx,
        })
    return pop_data_list

def baseline_lasso_comparison(
    pop_data: List[Dict[str, Any]],
    budget: int,
    alpha_lasso: Optional[float] = None,
    lasso_alphas_to_try: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Run Lasso baseline comparison on pooled pop_data.
    Returns the best baseline_results dict with keys:
      - alpha_value, selected_indices, baseline_coeffs,
        baseline_pop_stats, baseline_overall_stats,
        precision, recall, f1_score
    """
    # pool raw X, Y
    X_pooled = np.vstack([pop['X_raw'] for pop in pop_data])
    Y_pooled = np.hstack([pop['Y_raw'] for pop in pop_data])
    X_std, Y_std, _, _, _, _ = standardize_data(X_pooled, Y_pooled)

    # determine alphas to try
    if lasso_alphas_to_try is None:
        lasso_alphas_to_try = [alpha_lasso] if alpha_lasso is not None else [0.0001, 0.001, 0.01, 0.1]

    meaningful_indices_list = [pop['meaningful_indices'] for pop in pop_data]
    best_lasso_f1 = -1.0
    best_results = {}

    for current_alpha in lasso_alphas_to_try:
        model = Lasso(alpha=current_alpha, fit_intercept=False, max_iter=10000, tol=1e-4)
        model.fit(X_std, Y_std)
        coeffs = model.coef_
        selected_idx = np.argsort(np.abs(coeffs))[-budget:]
        
        sel_set  = set(selected_idx)
        true_set = set.union(*(set(mi) for mi in meaningful_indices_list))
        intersect = len(sel_set & true_set)
        prec = intersect / len(sel_set)   if sel_set  else 0.0
        rec  = intersect / len(true_set)  if true_set else 0.0
        f1   = 2*prec*rec/(prec+rec)      if (prec+rec)>0 else 0.0

        if f1 > best_lasso_f1:
            best_lasso_f1 = f1
            pop_stats, overall_stats = compute_population_stats(
                selected_idx.tolist(), meaningful_indices_list
            )
            best_results = {
                'alpha_value':       current_alpha,
                'selected_indices':  selected_idx.tolist(),
                'baseline_coeffs':   coeffs[selected_idx].tolist(),
                'baseline_pop_stats':    pop_stats,
                'baseline_overall_stats': overall_stats,
                'precision':         prec,
                'recall':            rec,
                'f1_score':          f1
            }

    return best_results

def run_experiment_multi_population(
    pop_configs: List[Dict], m1: int, m: int, dataset_size: int = 5000,
    noise_scale: float = 0.1, corr_strength: float = 0.0,
    parameterization: str = 'alpha', 
    smooth_minmax: float = float('inf'),
    gradient_mode: str = 'autograd', t2_estimator_type: str = 'mc_plugin', # New arg
    seed: Optional[int] = None, save_path: str = './results_v7/multi_population/',
    verbose: bool = False, implement_baseline_comparison: bool = True,
    alpha_lasso: float = None
    ) -> Dict[str, Any]:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Starting Experiment (v7.0) ---")
    print(f"Parameterization: {parameterization}, Gradient Mode: {gradient_mode}, T2 Estimator (Grad): {t2_estimator_type}")
    print(f"Using device: {device}")
    if smooth_minmax != float('inf'): print(f"SmoothMax Beta: {smooth_minmax}")
    os.makedirs(save_path, exist_ok=True)

    if seed is not None: np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available() and seed is not None: torch.cuda.manual_seed_all(seed)

    if budget is None: budget = min(m, max(1, m1 // 2) + len(pop_configs) * (m1 - max(1, m1 // 2)))
    else: budget = min(budget, m)
    
    if any('baseline' in pop_config['dataset_type'].lower() for pop_config in pop_configs):
        # Use baseline data generation function
        pop_data = get_pop_data_baseline_failures(
            pop_configs=pop_configs, dataset_size=dataset_size,
            n_features=m, 
            noise_scale=noise_scale, corr_strength=corr_strength, seed=seed
        )
    else:
        pop_data = get_pop_data(
            pop_configs=pop_configs, m1=m1, m=m, dataset_size=dataset_size,
            noise_scale=noise_scale, corr_strength=corr_strength, seed=seed
        )
    if not pop_data: return {'error': 'Data generation failed'}

    meaningful_indices_list = [pop['meaningful_indices'] for pop in pop_data]
    
    # Baseline Comparison (Lasso)
    if implement_baseline_comparison:
        baseline_results = baseline_lasso_comparison(
            pop_data=pop_data, budget=budget, alpha_lasso=alpha_lasso
        )
        if baseline_results:
            print(f"Baseline Lasso Results: {baseline_results}")
            baseline_pop_stats, baseline_overall_stats = compute_population_stats(
                baseline_results['selected_indices'], meaningful_indices_list
            )
            baseline_results.update({
                'baseline_pop_stats': baseline_pop_stats,
                'baseline_overall_stats': baseline_overall_stats
            })
        else:
            print("No valid results from Lasso baseline comparison.")

    return baseline_results

def convert_numpy_to_python(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int_)): return int(obj)
    if isinstance(obj, (np.floating, np.float_)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (torch.Tensor)): return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, dict): return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy_to_python(item) for item in obj]
    if isinstance(obj, set): return [convert_numpy_to_python(item) for item in obj] # Convert set to list
    if isinstance(obj, (bool, np.bool_)): return bool(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return "NaN" # Represent NaN/Inf as string
    return obj

def get_latest_run_number(save_path: str) -> int:
    if not os.path.exists(save_path): os.makedirs(save_path); return 0
    existing = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d)) and d.startswith('run_')]
    run_nums = [int(d.split('_')[1]) for d in existing if d.split('_')[1].isdigit()]
    return max(run_nums) + 1 if run_nums else 0

def compute_population_stats(selected_indices: List[int],
                             meaningful_indices_list: List[List[int]]) -> Tuple[List[Dict], Dict]:
    """Compute population-wise statistics for selected variables."""
    pop_stats = []
    percentages = []
    selected_set = set(selected_indices)

    for i, meaningful in enumerate(meaningful_indices_list):
        meaningful_set = set(meaningful)
        common = selected_set.intersection(meaningful_set)
        count = len(common)
        total = len(meaningful_set)
        percentage = (count / total * 100) if total > 0 else 0.0
        percentages.append(percentage)
        pop_stats.append({
            'population': i, 'selected_relevant_count': count,
            'total_relevant': total, 'percentage': percentage
        })

    min_perc = min(percentages) if percentages else 0.0
    max_perc = max(percentages) if percentages else 0.0
    median_perc = float(np.median(percentages)) if percentages else 0.0
    min_pop_idx = np.argmin(percentages) if percentages else -1
    max_pop_idx = np.argmax(percentages) if percentages else -1

    overall_stats = {
        'min_percentage': min_perc, 'max_percentage': max_perc,
        'median_percentage': median_perc,
        'min_population_details': pop_stats[min_pop_idx] if min_pop_idx != -1 else None,
        'max_population_details': pop_stats[max_pop_idx] if max_pop_idx != -1 else None,
    }
    return pop_stats, overall_stats

# =============================================================================
# Arg Parsing and Main Execution
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Multi-population VSS (v7.2: Theta Option)')
    # Data args
    parser.add_argument('--m1', type=int, default=4)
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--dataset-size_baseline', type=int, default=10000)
    parser.add_argument('--dataset-size', type=int, default=5000)
    parser.add_argument('--noise-scale', type=float, default=0.1)
    parser.add_argument('--corr-strength', type=float, default=0.0)
    parser.add_argument('--cases', type=str, default='all')
    # Other args
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save-path', type=str, default='./results/')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--lasso-alpha', type=float, default=None,
                   help='Regularization strength for Lasso baseline')
    parser.add_argument('--save-dir', type=str, default='./results/baseline/',
                   help='Directory to save results'),
    parser.add_argument('--num_pops', type=int, default=3,
                   help='Number of populations to generate data for')
    parser.add_argument('--budget', type=int, default=None,
                   help='Budget for variable selection')

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    baseline_cases = [
        "baseline_failure_1",
        "baseline_failure_2",
        "baseline_failure_3",
        "baseline_failure_4",
        "baseline_failure_5",
    ]
    regression_cases = [
        "linear_regression",
        "quadratic_regression",
        "cubic_regression",
        "sinusoidal_regression",
        "exponential_regression",
        "logarithmic_regression",
        "tanh_regression",
        "interaction_regression",
        "piecewise_regression",
    ]

    # select which cases to run
    if args.cases == 'all':
        cases = regression_cases + baseline_cases
    elif args.cases == 'baseline':
        cases = baseline_cases
    elif args.cases == 'regression':
        cases = regression_cases
    else:
        cases = [args.cases]

    seeds = [0, 1, 2, 3, 4]
    summary_records = []

    for case in cases:
        pop_configs = [{'pop_id': i, 'dataset_type': case} for i in range(args.num_pops)]
        # determine budget
        if case in baseline_cases:
            budget = args.num_pops
        else:
            budget = args.m1 // 2 + args.num_pops * (args.m1 - args.m1 // 2)

        # collect metrics per seed
        metrics = []
        for seed in seeds:
            print(f"Running case={case}, seed={seed}")
            if case in baseline_cases:
                pop_data = get_pop_data_baseline_failures(
                    pop_configs=pop_configs,
                    dataset_size=args.dataset_size_baseline,
                    n_features=args.m,
                    noise_scale=args.noise_scale,
                    corr_strength=args.corr_strength,
                    seed=seed
                )
            else:
                pop_data = get_pop_data(
                    pop_configs=pop_configs,
                    m1=args.m1, m=args.m,
                    dataset_size=args.dataset_size,
                    noise_scale=args.noise_scale,
                    corr_strength=args.corr_strength,
                    seed=seed
                )
            baseline_results = baseline_lasso_comparison(
                pop_data=pop_data,
                budget=budget,
                alpha_lasso=args.lasso_alpha
            )
            metrics.append(baseline_results)

        # aggregate across seeds
        precisions = [m['precision'] for m in metrics]
        recalls    = [m['recall']    for m in metrics]
        f1_scores  = [m['f1_score']  for m in metrics]

        summary_records.append({
            'case':          case,
            'alpha':         args.lasso_alpha,
            'dataset_size': args.dataset_size if case not in baseline_cases else args.dataset_size_baseline,
            'corr_strength': args.corr_strength,
            'precision_mean': float(np.mean(precisions)),
            'precision_min':  float(np.min(precisions)),
            'precision_max':  float(np.max(precisions)),
            'recall_mean':    float(np.mean(recalls)),
            'recall_min':     float(np.min(recalls)),
            'recall_max':     float(np.max(recalls)),
            'f1_mean':        float(np.mean(f1_scores)),
            'f1_min':         float(np.min(f1_scores)),
            'f1_max':         float(np.max(f1_scores)),
        })

    # save summary to CSV
    df = pd.DataFrame(summary_records)
    # check if the file already exists
    if os.path.exists(args.save_dir):
        # if it exists, increment the filename
        existing_files = [f for f in os.listdir(args.save_dir) if f.startswith('baseline_summary')]
        if existing_files:
            latest_file = max(existing_files, key=lambda x: int(re.search(r'(\d+)', x).group()))
            latest_num = int(re.search(r'(\d+)', latest_file).group())
            new_num = latest_num + 1
            out_csv = os.path.join(args.save_dir, f'baseline_summary_{new_num}.csv')
        else:
            out_csv = os.path.join(args.save_dir, 'baseline_summary_1.csv')
    else:
        # if it doesn't exist, create the file
        out_csv = os.path.join(args.save_dir, 'baseline_summary.csv')
    # out_csv = os.path.join(args.save_dir, 'baseline_summary.csv')
    df.to_csv(out_csv, index=False)
    print(f"Saved summary metrics to {out_csv}")

if __name__ == "__main__":
    main()