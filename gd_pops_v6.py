# gd_pops_v6.py: Variable Selection with Optional Theta Reparameterization

"""
Performs variable subset selection using gradient descent.
Combines objective backpropagation (v4) and REINFORCE (v5) gradient methods.
Includes learning rate scheduling and optional SmoothMax for robust objective.
Allows optimizing either alpha directly or theta = log(alpha).

Objective: Minimize L(param) = SmoothMax_pop [ T1_std - T2(param) + P(param) ]
                                (or Hard Max if smooth_minmax is inf)
  T1_std = E[(standardized E[Y|X])^2]
  T2(param) = E[(standardized E[Y|S(param)])^2]
  P(param) = Penalty Term applied to alpha (derived from theta if needed)
  S(param) = X_std + sqrt(variance(param)) * epsilon

Gradient Modes:
  - 'autograd': Computes gradient via automatic differentiation.
  - 'reinforce': Computes gradient of T2 using REINFORCE.
"""

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
        estimate_gradient_reinforce_flexible
    )
    # Use the flexible kernel estimator from objective_comparison
    from torch.optim import lr_scheduler # Using torch built-ins
    print("Successfully imported data, estimators, and scheduler functions.")
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure data.py, estimators.py, objective_comparison.py are accessible.")
    exit(1)

from global_vars import *
from sklearn.linear_model import Lasso

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
# =============================================================================
# Penalty Function (Takes alpha, computes penalty)
# =============================================================================

def compute_penalty(alpha: torch.Tensor, # Input is always alpha
                    penalty_type: Optional[str],
                    penalty_lambda: float,
                    epsilon: float = EPS) -> torch.Tensor:
    """
    Compute a penalty term P(alpha) designed to encourage large alpha values.
    We minimize L = (T1 - T2) + P(alpha).
    """
    # Clamp alpha within the function for calculation, ensuring gradients flow
    alpha_clamped = torch.clamp(alpha, min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)

    if penalty_type is None or penalty_lambda == 0 or penalty_type.lower() == "none":
        return torch.tensor(0.0, device=alpha.device, dtype=alpha.dtype, requires_grad=alpha.requires_grad)

    penalty_type_lower = penalty_type.lower()

    if penalty_type_lower == "reciprocal_l1":
        return penalty_lambda * torch.sum(1.0 / (alpha_clamped + epsilon))
    elif penalty_type_lower == "neg_l1":
        print("Warning: Using Neg_L1 penalty encourages small alpha.")
        return penalty_lambda * torch.sum(torch.abs(alpha_clamped))
    elif penalty_type_lower == "max_dev":
        target_val = torch.tensor(1.0, device=alpha.device) # Target alpha=1
        return penalty_lambda * torch.sum(torch.abs(target_val - alpha_clamped))
    elif penalty_type_lower == "quadratic_barrier":
        return penalty_lambda * torch.sum((alpha_clamped + epsilon) ** (-2))
    elif penalty_type_lower == "exponential":
        return penalty_lambda * torch.sum(torch.exp(-alpha_clamped))
    else:
        raise ValueError("Unknown penalty_type: " + str(penalty_type))

# =============================================================================
# Objective Value Computation (for tracking/selection, NOT gradient)
# =============================================================================

def compute_objective_value_mc(X: torch.Tensor,
                               E_Y_given_X_std: torch.Tensor,
                               term1_std: float,
                               param: torch.Tensor, # alpha or theta
                               param_type: str,
                               penalty_lambda: float = 0.0,
                               penalty_type: Optional[str] = None,
                               num_mc_samples: int = 25,
                               k_kernel: int = 1000) -> float:
    """
    Computes L = (T1_std - T2) + P value using Monte Carlo for T2.
    Uses flexible kernel estimator. Does NOT compute gradients.
    """
    device = param.device
    X = X.to(device)
    E_Y_given_X_std = E_Y_given_X_std.to(device)
    param_val = param.detach().clone() # Use detached value

    # Derive alpha for calculations, clamping appropriately
    if param_type == 'alpha':
        alpha_val = param_val.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
    elif param_type == 'theta':
        # Clamp theta before exp to prevent overflow, then clamp alpha
        theta_clamped = param_val.clamp(min=THETA_CLAMP_MIN, max=THETA_CLAMP_MAX)
        alpha_val = torch.exp(theta_clamped).clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
    else: raise ValueError("Invalid param_type")

    term1_std_tensor = torch.tensor(term1_std, dtype=alpha_val.dtype, device=device)

    avg_term2_std = 0.0
    with torch.no_grad():
        noise_var = alpha_val # Variance is always alpha
        for _ in range(num_mc_samples):
            epsilon = torch.randn_like(X)
            S_param = X + epsilon * torch.sqrt(noise_var)
            # Estimate E[Y_std|S] using the correct parameter type for the kernel
            E_Y_S_std = estimate_conditional_keops_flexible(
                X, S_param, E_Y_given_X_std, param_val, param_type, k=k_kernel
            )
            term2_sample_std = E_Y_S_std.pow(2).mean()
            avg_term2_std += term2_sample_std

    term2_value_std = avg_term2_std / num_mc_samples

    # Penalty is always computed based on alpha
    penalty_value = compute_penalty(alpha_val, penalty_type, penalty_lambda)

    objective_val = term1_std - term2_value_std + penalty_value
    # objective_val = term1_std_tensor - term2_value_std + penalty_value # Original T1-T2+P
    return objective_val.item()

def compute_objective_value_if(X: torch.Tensor,
                               Y_std: torch.Tensor,
                               term1_std: float,
                               param: torch.Tensor, # alpha or theta
                               param_type: str,
                               penalty_lambda: float = 0.0,
                               penalty_type: Optional[str] = None,
                               base_model_type: str = "rf",
                               n_folds: int = N_FOLDS) -> float:
    """
    Computes L = (T1_std - T2) + P value using IF estimator for T2.
    Does NOT compute gradients.
    """
    device = param.device
    X = X.to(device)
    Y_std = Y_std.to(device)
    param_val = param.detach().clone()

    # Derive alpha for calculations
    if param_type == 'alpha':
        alpha_val = param_val.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
    elif param_type == 'theta':
        theta_clamped = param_val.clamp(min=THETA_CLAMP_MIN, max=THETA_CLAMP_MAX)
        alpha_val = torch.exp(theta_clamped).clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
    else: raise ValueError("Invalid param_type")

    term1_std_tensor = torch.tensor(term1_std, dtype=alpha_val.dtype, device=device)

    term2_value_std_tensor = torch.tensor(0.0, device=device, dtype=alpha_val.dtype)
    try:
        with torch.no_grad():
            epsilon = torch.randn_like(X)
            S_alpha = X + epsilon * torch.sqrt(alpha_val) # S based on standardized X
            S_alpha_np = S_alpha.cpu().numpy()
            Y_std_np = Y_std.cpu().numpy()

            term2_value_float = IF_estimator_squared_conditional(
                S_alpha_np, Y_std_np, estimator_type=base_model_type, n_folds=n_folds
            )
            if np.isnan(term2_value_float):
                 print("Warning: IF Term 2 calculation resulted in NaN.")
                 return float('nan')
            term2_value_std_tensor = torch.tensor(term2_value_float, dtype=alpha_val.dtype, device=device)
    except Exception as e:
        print(f"Warning: IF_estimator_squared_conditional failed during value calculation: {e}")
        return float('nan')

    penalty_value = compute_penalty(alpha_val, penalty_type, penalty_lambda)

    # FIX: Objective T2 + P
    objective_val = term1_std_tensor - term2_value_std_tensor + penalty_value
    # objective_val = term1_std_tensor - term2_value_std_tensor + penalty_value # Original T1-T2+P
    return objective_val.item()

# =============================================================================
# Data Loading / Precomputation
# =============================================================================
# (Keep get_pop_data as is - it returns standardized data needed)
def get_pop_data(pop_configs: List[Dict], m1: int, m: int,
                 dataset_size: int = 10000,
                 noise_scale: float = 0.0,
                 corr_strength: float = 0.0,
                 common_meaningful_indices: Optional[np.ndarray] = None,
                 estimator_type: str = "plugin",
                 device: str = "cpu",
                 base_model_type: str = "rf",
                 seed: Optional[int] = None) -> List[Dict]:
    """
    Generates datasets and precomputes standardized terms for each population.
    Returns list of dicts, each containing standardized X, Y, E_Y_given_X, and term1_std.
    """
    if common_meaningful_indices is None:
        k_common = max(1, m1 // 2)
        common_meaningful_indices = np.arange(k_common)

    pop_data = []
    for i, pop_config in enumerate(pop_configs):
        pop_id = pop_config.get('pop_id', i)
        dataset_type = pop_config['dataset_type']
        current_seed = seed + pop_id if seed is not None else None

        print(f"\nPopulation {pop_id} ({dataset_type}): Generating data...")
        if corr_strength > 0:
            X_np, Y_np, _, meaningful_indices = generate_data_continuous_with_corr(
                pop_id=pop_id, m1=m1, m=m,
                dataset_type=dataset_type, dataset_size=dataset_size,
                noise_scale=noise_scale, corr_strength=corr_strength,
                seed=current_seed, common_meaningful_indices=common_meaningful_indices
            )
        else:
            X_np, Y_np, _, meaningful_indices = generate_data_continuous(
                pop_id=pop_id, m1=m1, m=m,
                dataset_type=dataset_type, dataset_size=dataset_size,
                noise_scale=noise_scale, seed=current_seed,
                common_meaningful_indices=common_meaningful_indices
            )

        # --- Standardize Data ---
        X_std_np, Y_std_np, _, _, Y_mean, Y_std = standardize_data(X_np, Y_np) # Get means/stds too

        print(f"Population {pop_id}: Precomputing E[Y|X] ({estimator_type}/{base_model_type})...")
        # --- Precompute E[Y|X] using ORIGINAL Y scale ---
        try:
            if estimator_type == "plugin":
                E_Yx_orig_np = plugin_estimator_conditional_mean(X_np, Y_np, base_model_type, n_folds=N_FOLDS)
            elif estimator_type == "if":
                E_Yx_orig_np = IF_estimator_conditional_mean(X_np, Y_np, base_model_type, n_folds=N_FOLDS)
            else:
                raise ValueError("estimator_type must be 'plugin' or 'if'")
        except Exception as e:
            print(f"ERROR: Failed to precompute E[Y|X] for pop {pop_id}. Skipping population. Error: {e}")
            continue

        # --- Standardize the E[Y|X] estimate ---
        E_Yx_std_np = (E_Yx_orig_np - Y_mean) / Y_std

        # --- Calculate Term 1 based on the STANDARDIZED estimate ---
        term1_std = np.mean(E_Yx_std_np ** 2)
        print(f"Population {pop_id}: Precomputed Term1_std = {term1_std:.4f}")

        # --- Convert to Tensors ---
        X_std_torch = torch.tensor(X_std_np, dtype=torch.float32).to(device)
        Y_std_torch = torch.tensor(Y_std_np, dtype=torch.float32).to(device)
        E_Yx_std_torch = torch.tensor(E_Yx_std_np, dtype=torch.float32).to(device)

        pop_data.append({
            'pop_id': pop_id,
            'X_std': X_std_torch,        # Standardized X
            'Y_std': Y_std_torch,        # Standardized Y
            'E_Yx_std': E_Yx_std_torch, # Standardized E[Y|X] estimate
            'term1_std': term1_std,     # Scalar Term 1 (from standardized E[Y|X])
            'meaningful_indices': meaningful_indices.tolist(), # Store as list
            'X_raw': X_np,            # Raw X (for reference)
            'Y_raw': Y_np,            # Raw Y (for reference)
        })
    return pop_data

# =============================================================================
# Parameter Initialization (Alpha or Theta)
# =============================================================================

def init_parameter(m: int,
                   parameterization: str,
                   alpha_init_config: str = "random_1", # Config string like "random_1", "ones" etc.
                   noise: float = 0.1,
                   device: str = "cpu") -> torch.nn.Parameter:
    """
    Initializes the parameter being optimized (alpha or theta).
    """
    # First, determine the initial alpha value based on alpha_init_config
    init_alpha_val = torch.ones(m, device=device) # Default base
    alpha_init_lower = alpha_init_config.lower()

    if alpha_init_lower.startswith("random_"):
        try:
            k_str = alpha_init_lower.split('_', 1)[1]
            k = float(k_str)
            print(f"Initializing alpha randomly around {k}")
            init_alpha_val = k * torch.ones(m, device=device) + noise * torch.abs(torch.randn(m, device=device))
        except (ValueError, IndexError):
             raise ValueError(f"Invalid numeric value in alpha_init_config: {alpha_init_config}")
    elif alpha_init_lower == "ones":
        print("Initializing alpha to ones + noise")
        init_alpha_val = torch.ones(m, device=device) + noise * torch.abs(torch.randn(m, device=device))
    elif alpha_init_lower == "random":
        print("Initializing alpha randomly around 1")
        init_alpha_val = torch.ones(m, device=device) + noise * torch.abs(torch.randn(m, device=device))
    else:
        raise ValueError("alpha_init_config must be 'ones', 'random', or 'random_k.k'")

    # Ensure initial alpha is within clamp bounds
    init_alpha_val.clamp_(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)

    # Initialize the parameter based on parameterization type
    if parameterization == 'alpha':
        print(f"Initializing parameter: alpha (Value based on {alpha_init_config})")
        param = torch.nn.Parameter(init_alpha_val)
    elif parameterization == 'theta':
        print(f"Initializing parameter: theta = log(alpha) (alpha based on {alpha_init_config})")
        # Ensure alpha is strictly positive before log
        init_alpha_safe = torch.clamp(init_alpha_val, min=EPS)
        init_theta_val = torch.log(init_alpha_safe)
        # Clamp initial theta based on derived limits
        init_theta_val.clamp_(min=THETA_CLAMP_MIN, max=THETA_CLAMP_MAX)
        param = torch.nn.Parameter(init_theta_val)
    else:
        raise ValueError("parameterization must be 'alpha' or 'theta'")

    return param

def create_results_dataframe(results, args, save_path=None):
    """Create a comprehensive DataFrame from experiment results."""
    
    # Extract key data
    data = {
        'Run_ID': os.path.basename(os.path.dirname(save_path)),
        'Dataset_Type': '_'.join(args.populations),
        'Dataset_Size': args.dataset_size,
        'm1': args.m1,
        'm': args.m,
        'Budget': len(results['selected_indices']),
        'Parameterization': args.parameterization,
        'Penalty_Type': args.penalty_type,
        'Penalty_Lambda': args.penalty_lambda,
        'Gradient_Mode': args.gradient_mode,
        
        # Ground truth indices
        'All_True_Indices': str(list(set().union(*[set(indices) for indices in results['meaningful_indices']]))),
    }
    
    # Add per-population true indices
    for i, indices in enumerate(results['meaningful_indices']):
        data[f'True_Indices_Pop_{i}'] = str(indices)
    
    # Our method results
    data.update({
        'Our_F1': results['overall_f1_score'],
        'Our_Precision': results['overall_precision'],
        'Our_Recall': results['overall_recall'],
        'Our_Selected_Indices': str(results['selected_indices']),
    })
    
    # Add per-population coverage percentages for our method
    for stat in results['population_stats']:
        pop_id = stat['population']
        data[f'Our_Pop{pop_id}_Coverage'] = stat['percentage']
    
    # Baseline results (if available)
    if results.get('baseline_results'):
        baseline = results['baseline_results']
        data.update({
            'Baseline_F1': baseline.get('f1_score', None),
            'Baseline_Precision': baseline.get('precision', None),
            'Baseline_Recall': baseline.get('recall', None),
            'Baseline_Selected_Indices': str(baseline['selected_indices']),
        })
        
        # Add per-population coverage for baseline
        if 'baseline_pop_stats' in baseline:
            for stat in baseline['baseline_pop_stats']:
                pop_id = stat['population']
                data[f'Baseline_Pop{pop_id}_Coverage'] = stat['percentage']
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # Save if path provided
    if save_path:
        csv_path = os.path.join(save_path, 'results_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results summary saved to: {csv_path}")
        
        # Also save as pickle for easier loading with types preserved
        pickle_path = os.path.join(save_path, 'results_summary.pkl')
        df.to_pickle(pickle_path)
        print(f"Results also saved as pickle: {pickle_path}")
    
    return df
# =============================================================================
# Main Experiment Runner (v6.2 - Theta Option)
# =============================================================================

def run_experiment_multi_population_v6(
    # Data args
    pop_configs: List[Dict], m1: int, m: int,
    dataset_size: int = 5000, noise_scale: float = 0.1, corr_strength: float = 0.0,
    # Optimization args
    num_epochs: int = 100, budget: Optional[int] = None,
    penalty_type: Optional[str] = None, penalty_lambda: float = 0.0,
    learning_rate: float = 0.01, optimizer_type: str = 'adam',
    parameterization: str = 'alpha', # 'alpha' or 'theta'
    alpha_init: str = "random_1", # Config for initial alpha value
    early_stopping_patience: int = 15,
    param_freezing: bool = True, smooth_minmax: float = float('inf'),
    # Gradient computation args
    gradient_mode: str = 'autograd',
    N_grad_samples: int = 25,
    use_baseline: bool = True,
    # Estimator args
    estimator_type: str = "if", base_model_type: str = "rf",
    objective_value_estimator: str = 'if', k_kernel: int = 1000,
    # Scheduler args
    scheduler_type: Optional[str] = None, scheduler_kwargs: Optional[Dict] = None,
    # Other args
    seed: Optional[int] = None, save_path: str = './results_v6/multi_population/', verbose: bool = False,
    implement_baseline_comparison: bool = True,
    alpha_lasso: float = None,
    ) -> Dict[str, Any]:
    """
    Main experiment runner with optional theta parameterization.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Starting Experiment (v6.2) ---")
    print(f"Parameterization: {parameterization}")
    print(f"Using device: {device}")
    print(f"Gradient Mode: {gradient_mode}")
    if smooth_minmax != float('inf'): print(f"SmoothMax Beta: {smooth_minmax}")
    os.makedirs(save_path, exist_ok=True)

    # --- Seed ---
    if seed is not None: np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available() and seed is not None: torch.cuda.manual_seed_all(seed)
    print(f"Using seed: {seed if seed is not None else 'Random'}")

    # --- Budget ---
    if budget is None:
        k_common = max(1, m1 // 2); k_pop_specific = m1 - k_common
        budget = k_common + len(pop_configs) * k_pop_specific
    budget = min(budget, m)
    print(f"Budget for variable selection: {budget}")

    # --- Get Population Data ---
    # check for any pop_config dataset_type starting with baseline
    if any('baseline' in pop_config['dataset_type'].lower() for pop_config in pop_configs):
        # Use baseline data generation function
        pop_data = get_pop_data_baseline_failures(
            pop_configs=pop_configs, dataset_size=dataset_size,
            n_features=m, 
            noise_scale=noise_scale, corr_strength=corr_strength,
            estimator_type=estimator_type, device=device,
            base_model_type=base_model_type, seed=seed
        )
    else:
        pop_data = get_pop_data(
            pop_configs=pop_configs, m1=m1, m=m, dataset_size=dataset_size,
            noise_scale=noise_scale, corr_strength=corr_strength,
            estimator_type=estimator_type, device=device,
            base_model_type=base_model_type, seed=seed
        )
    if not pop_data: return {'error': 'Data generation failed'}

    # --- Initialize Parameter (alpha or theta) and Optimizer ---
    param = init_parameter(m, parameterization, alpha_init, noise=0.1, device=device)
    print(f"Initialized param ({parameterization}, first 5): {param.data.cpu().numpy()[:5]}")

    if optimizer_type.lower() == 'adam': optimizer = optim.Adam([param], lr=learning_rate)
    elif optimizer_type.lower() == 'sgd': optimizer = optim.SGD([param], lr=learning_rate, momentum=0.9, nesterov=True)
    else: raise ValueError("Unsupported optimizer_type.")

    # --- Initialize Scheduler ---
    scheduler = None
    # (Scheduler initialization code remains the same, using 'optimizer')
    if scheduler_type:
        scheduler_kwargs = scheduler_kwargs or {}
        scheduler_class = getattr(lr_scheduler, scheduler_type, None)
        if scheduler_class:
            try:
                valid_kwargs = {}
                if scheduler_type == 'StepLR': valid_kwargs = {'step_size': scheduler_kwargs.get('step_size', 30), 'gamma': scheduler_kwargs.get('gamma', 0.1)}
                elif scheduler_type == 'MultiStepLR': valid_kwargs = {'milestones': scheduler_kwargs.get('milestones', [50, 80]), 'gamma': scheduler_kwargs.get('gamma', 0.1)}
                elif scheduler_type == 'ExponentialLR': valid_kwargs = {'gamma': scheduler_kwargs.get('gamma', 0.99)}
                elif scheduler_type == 'CosineAnnealingLR': valid_kwargs = {'T_max': scheduler_kwargs.get('T_max', num_epochs), 'eta_min': scheduler_kwargs.get('eta_min', 0)}
                elif scheduler_type == 'ReduceLROnPlateau': valid_kwargs = {'mode':'min', 'factor': scheduler_kwargs.get('factor', 0.1), 'patience': scheduler_kwargs.get('patience', 10), 'min_lr': scheduler_kwargs.get('min_lr', 1e-6)}
                scheduler = scheduler_class(optimizer, **valid_kwargs)
                print(f"Using scheduler: {scheduler_type} with kwargs: {valid_kwargs}")
            except Exception as e:
                print(f"Warning: Failed to initialize scheduler '{scheduler_type}'. Error: {e}. Continuing without scheduler.")
                scheduler = None
        else:
            print(f"Warning: Scheduler type '{scheduler_type}' not found in torch.optim.lr_scheduler. Continuing without scheduler.")

    # --- History Tracking ---
    param_history = [param.detach().cpu().numpy().copy()] # History of the optimized param
    objective_history = []
    gradient_history = []
    lr_history = [optimizer.param_groups[0]['lr']]

    best_objective_val = float('inf')
    best_param = param.detach().cpu().numpy().copy() # Store the best param (alpha or theta)
    early_stopping_counter = 0
    stopped_epoch = num_epochs

    print(f"\n--- Starting Optimization ({gradient_mode} mode, optimizing {parameterization}) ---")
    total_start_time = time.time()

    # --- Main Optimization Loop ---
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        optimizer.zero_grad()

        # --- Calculate Current Alpha (needed for penalty, value tracking, noise) ---
        with torch.no_grad(): # No gradient needed for this conversion
            if parameterization == 'alpha':
                current_alpha = param.data.clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
            else: # theta
                current_alpha = torch.exp(param.data).clamp(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)

        # --- Calculate Objective Per Population (for gradient) ---
        population_objectives_for_grad = []
        param.requires_grad_(True) # Ensure parameter requires grad for this block

        for pop in pop_data:
            X_std = pop['X_std']
            E_Yx_std = pop['E_Yx_std']
            term1_std = pop['term1_std']

            # Calculate T2 using MC + Flexible Kernel, passing the OPTIMIZED parameter
            term2_value = estimate_T2_mc_flexible(
                X_std, E_Yx_std, param, parameterization, N_grad_samples, k_kernel
            )

            # Calculate Penalty based on CURRENT alpha derived from param
            if parameterization == 'alpha':
                alpha_for_penalty = param
            else: # theta
                alpha_for_penalty = torch.exp(param) # Maintain graph

            penalty_value = compute_penalty(alpha_for_penalty, penalty_type, penalty_lambda)

            # Objective L = T1 - T2 + P
            pop_objective = term1_std - term2_value + penalty_value
            population_objectives_for_grad.append(pop_objective)

        objectives_tensor = torch.stack(population_objectives_for_grad)

        # --- Calculate Robust Objective for GRADIENT ---
        beta = smooth_minmax
        if torch.isfinite(torch.tensor(beta)) and beta > 0:
            with torch.no_grad(): M = torch.max(beta * objectives_tensor)
            logsumexp_val = M + torch.log(torch.sum(torch.exp(beta * objectives_tensor - M)) + EPS)
            robust_objective_for_grad = (1.0 / beta) * logsumexp_val
        else:
            robust_objective_for_grad, _ = torch.max(objectives_tensor, dim=0) # Hard Max

        # --- Calculate Objective VALUE (Hard Max) for Tracking/Stopping ---
        population_objective_values = []
        # Pass the *parameter* being optimized to the value function
        current_param_val_detached = param.detach().clone()
        for pop in pop_data:
             if objective_value_estimator == 'if':
                 # Value function needs alpha, derive it if optimizing theta
                 alpha_for_value = current_alpha # Use pre-calculated alpha
                 obj_val = compute_objective_value_if(
                     pop['X_std'], pop['Y_std'], pop['term1_std'],
                     alpha_for_value, # Pass alpha
                        parameterization, # Pass param and type
                     penalty_lambda, penalty_type, base_model_type=base_model_type
                 )
             else: # 'mc'
                 obj_val = compute_objective_value_mc(
                     pop['X_std'], pop['E_Yx_std'], pop['term1_std'],
                     current_param_val_detached, parameterization, # Pass param and type
                     penalty_lambda, penalty_type, num_mc_samples=N_grad_samples, k_kernel=k_kernel
                 )
             population_objective_values.append(obj_val)

        valid_obj_values = [v for v in population_objective_values if not math.isnan(v)]
        if not valid_obj_values:
             print(f"ERROR: All pop objective values NaN at epoch {epoch} for tracking. Stopping.")
             stopped_epoch = epoch
             break
        current_robust_objective_value = max(valid_obj_values) # Hard Max
        winning_pop_index_for_tracking = population_objective_values.index(current_robust_objective_value) # Index of min
        objective_history.append(current_robust_objective_value)

        # --- Calculate Gradient ---
        if gradient_mode == 'autograd':
            robust_objective_for_grad.backward()
            total_gradient = param.grad.clone() if param.grad is not None else torch.zeros_like(param)
            param.requires_grad_(False) # Detach after backward

        elif gradient_mode == 'reinforce':
             # REINFORCE calculates grad w.r.t the parameter being optimized
             winning_pop_data = pop_data[winning_pop_index_for_tracking]
             X_win_std = winning_pop_data['X_std']
             E_Yx_win_std = winning_pop_data['E_Yx_std']

             total_gradient = estimate_gradient_reinforce_flexible(
                 X_win_std, E_Yx_win_std, param, parameterization, # Pass param and type
                 N_grad_samples, k_kernel, penalty_type, penalty_lambda, use_baseline
             )
             # Assign gradient manually
             if param.grad is not None: param.grad.zero_()
             param.grad = total_gradient
        else:
            raise ValueError("gradient_mode must be 'autograd' or 'reinforce'")

        gradient_history.append(total_gradient.detach().cpu().numpy().copy() if total_gradient is not None else np.zeros(m))

        # --- Optional: Parameter Freezing & Clipping ---
        if param_freezing:
            with torch.no_grad():
                # Define threshold based on parameterization
                freeze_thresh = FREEZE_THRESHOLD_ALPHA if parameterization == 'alpha' else THETA_FREEZE_THRESHOLD
                # Apply freezing based on parameter value
                frozen_mask = param.data < freeze_thresh
                if param.grad is not None:
                    param.grad[frozen_mask] = 0
                    # Robustly clear optimizer state
                    for p_group in optimizer.param_groups:
                         if p_group['params'][0] is param:
                              for state_key, state_value in optimizer.state[param].items():
                                   if isinstance(state_value, torch.Tensor) and state_value.shape == param.shape:
                                        state_value[frozen_mask] = 0.0
                              break

        if param.grad is not None:
            grad_norm_before_clip = torch.linalg.norm(param.grad).item()
            torch.nn.utils.clip_grad_norm_([param], max_norm=10.0)
            grad_norm_after_clip = torch.linalg.norm(param.grad).item()
        else:
             print(f"Warning: No gradient computed for epoch {epoch}.")
             grad_norm_before_clip = 0.0; grad_norm_after_clip = 0.0

        # --- Optimizer Step ---
        optimizer.step() # Updates param based on param.grad

        # --- Clamp Parameter ---
        with torch.no_grad():
            if parameterization == 'alpha':
                param.data.clamp_(min=CLAMP_MIN_ALPHA, max=CLAMP_MAX_ALPHA)
            else: # theta
                param.data.clamp_(min=THETA_CLAMP_MIN, max=THETA_CLAMP_MAX)

        # --- Scheduler Step ---
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_robust_objective_value)
            else:
                scheduler.step()
        lr_history.append(optimizer.param_groups[0]['lr'])

        # --- Logging and History ---
        param_history.append(param.detach().cpu().numpy().copy())
        epoch_time = time.time() - epoch_start_time

        if verbose or (epoch % 10 == 0):
             print(f"Epoch {epoch}/{num_epochs} | EpTime: {epoch_time:.2f}s | WinPop: {pop_data[winning_pop_index_for_tracking]['pop_id']} "
                   f"| Robust Obj: {current_robust_objective_value:.4f} | LR: {current_lr:.6f} | GradNorm (B/A): {grad_norm_before_clip:.4f}/{grad_norm_after_clip:.4f}")

        # --- Early Stopping Check ---
        # FIX: Objective is MINIMIZED (T2+P)
        if current_robust_objective_value < best_objective_val - EPS:
            best_objective_val = current_robust_objective_value
            best_param = param.detach().cpu().numpy().copy()
            early_stopping_counter = 0
            if verbose: print(f"  New best objective: {best_objective_val:.4f}")
        else:
            early_stopping_counter += 1

        if epoch > 30 and early_stopping_counter >= early_stopping_patience:
            if abs(current_robust_objective_value - best_objective_val) < 0.01 * abs(best_objective_val):
                print(f"Early stopping triggered at epoch {epoch} due to lack of improvement.")
                stopped_epoch = epoch
                break

    # --- End of Training Loop ---
    total_time = time.time() - total_start_time
    print(f"\nOptimization finished in {total_time:.2f} seconds.")
    print(f"Best robust objective value achieved: {best_objective_val:.4f}")
    if stopped_epoch < num_epochs: print(f"Stopped at epoch: {stopped_epoch}")

    # --- Prepare Results ---
    # Always derive the final alpha values for selection and reporting
    if parameterization == 'alpha':
        final_alpha_np = best_param
    else: # theta
        final_alpha_np = np.exp(best_param)
    # Ensure final alpha is within bounds for selection logic
    final_alpha_np = np.clip(final_alpha_np, CLAMP_MIN_ALPHA, CLAMP_MAX_ALPHA)

    selected_indices = np.argsort(final_alpha_np)[:budget] # Select smallest alpha
    meaningful_indices_list = [pop['meaningful_indices'] for pop in pop_data]

    # --- Save Diagnostics Plot (if verbose) ---
    if verbose:
        try:
            plt.figure(figsize=(18, 6))
            # Objective
            plt.subplot(1, 3, 1)
            plt.plot(objective_history, label="Robust Objective L")
            plt.xlabel("Epoch"); plt.ylabel("Objective Value"); plt.title("Objective vs Epoch"); plt.grid(True)
            
            # Gradients
            plt.subplot(1, 3, 2)
            valid_gradients = [g for g in gradient_history if g is not None]
            grad_norms = [np.linalg.norm(g) for g in valid_gradients]
            if grad_norms: plt.plot(grad_norms, label="Total Gradient Norm")
            plt.xlabel("Epoch"); plt.ylabel("Gradient Norm"); plt.title("Gradient Norm vs Epoch"); plt.yscale('log'); plt.grid(True)
            
            # Parameters (Alpha or Theta)
            plt.subplot(1, 3, 3)
            param_hist_np = np.array(param_history)
            num_params_to_plot = min(budget, 10)
            # indices_to_plot = np.linspace(0, m - 1, num_params_to_plot, dtype=int)
            indices_to_plot = selected_indices
            param_label = r'$\alpha$' if parameterization == 'alpha' else r'$\theta$'
            for i in indices_to_plot:
                label = f'{param_label}_{{{i}}}'
                is_meaningful = any(i in pop_indices for pop_indices in meaningful_indices_list)
                if is_meaningful: label += " (True)"
                plt.plot(param_hist_np[:, i], label=label)
            plt.xlabel("Epoch"); plt.ylabel(f"{param_label} Value"); plt.title(f"{param_label} Trajectories (Top {num_params_to_plot})")
            if parameterization == 'alpha': plt.yscale('log') # Log scale often useful for alpha
            plt.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5)); plt.grid(True)

            plt.tight_layout()
            diag_path = os.path.join(save_path, f"optimization_diagnostics_v6_{parameterization}.png")
            plt.savefig(diag_path); plt.close()
            print(f"Diagnostic plots saved to {diag_path}")
        except Exception as e: print(f"Warning: Failed to generate diagnostic plots: {e}")

    baseline_results = None
    if implement_baseline_comparison:
        print("Running baseline comparison (Lasso regression)...")
        pooled_pop_data = []
        for pop in pop_data:
            X_raw = pop['X_raw']
            Y_raw = pop['Y_raw']
            pooled_pop_data.append((X_raw, Y_raw))
        X_pooled = np.vstack([data[0] for data in pooled_pop_data])
        Y_pooled = np.hstack([data[1] for data in pooled_pop_data])

        # Standardize the pooled data
        X_pooled, Y_pooled, _, _, Y_mean, Y_std = standardize_data(X_pooled, Y_pooled)
        
        if alpha_lasso is None:
            # Try multiple alpha values for Lasso
            alpha_values = [0.001, 0.01, 0.1]  # Different alpha values to try
            best_f1 = -1
            best_baseline_results = None
            
            for current_alpha in alpha_values:
                print(f"Trying Lasso with alpha={current_alpha}")
                model = Lasso(alpha=current_alpha, fit_intercept=False, max_iter=10000, tol=1e-4)
                model.fit(X_pooled, Y_pooled)
                
                # Get coefficients and select features
                baseline_coeffs = model.coef_
                
                # Selection logic - highest absolute coefficient values
                baseline_selected_indices = np.argsort(np.abs(baseline_coeffs))[-budget:]
                
                # compute baseline stats
                baseline_pop_stats, baseline_overall_stats = compute_population_stats(
                    selected_indices=baseline_selected_indices,
                    meaningful_indices_list=meaningful_indices_list
                )
                
                # Compute F1 score for this alpha
                selected_set = set(baseline_selected_indices)
                all_meaningful_indices_set = set()
                for indices in meaningful_indices_list:
                    all_meaningful_indices_set.update(indices)
                    
                intersection_size = len(selected_set & all_meaningful_indices_set)
                precision = intersection_size / len(selected_set) if selected_set else 0.0
                recall = intersection_size / len(all_meaningful_indices_set) if all_meaningful_indices_set else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                print(f"Lasso (alpha={current_alpha}) - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                
                # Save if best so far - decided by how many of true in the selected set
                if f1 > best_f1:
                    best_f1 = f1
                    best_baseline_results = {
                        'alpha_value': current_alpha,
                        'selected_indices': baseline_selected_indices.tolist(),
                        'baseline_coeffs': baseline_coeffs[baseline_selected_indices].tolist(),
                        'baseline_pop_stats': baseline_pop_stats,
                        'baseline_overall_stats': baseline_overall_stats,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    }
        else:
            # Replace linear regression with Lasso
            # alpha_lasso = 0.01  # L1 regularization strength (can be tuned)
            model = Lasso(alpha=alpha_lasso, fit_intercept=False, max_iter=10000, tol=1e-4)
            model.fit(X_pooled, Y_pooled)
            
            # Get coefficients and select features
            baseline_coeffs = model.coef_
        
            # Same selection logic - highest absolute coefficient values
            baseline_selected_indices = np.argsort(np.abs(baseline_coeffs))[-budget:]
            print(f"Lasso baseline selected indices: {baseline_selected_indices}")
            print(f"Lasso baseline coefficients: {baseline_coeffs[baseline_selected_indices]}")

            # compute baseline F1, Recall, Precision
            print("Computing baseline statistics...")
            baseline_pop_stats, baseline_overall_stats = compute_population_stats(
                selected_indices=baseline_selected_indices,
                meaningful_indices_list=meaningful_indices_list
            )
            print(f"Baseline population stats: {baseline_pop_stats}")
            print(f"Baseline overall stats: {baseline_overall_stats}")
            best_baseline_results = {
                'selected_indices': baseline_selected_indices.tolist(),
                'baseline_coeffs': baseline_coeffs[baseline_selected_indices].tolist(),
                'baseline_pop_stats': baseline_pop_stats,
                'baseline_overall_stats': baseline_overall_stats
            }
    
    results = {
    'parameterization': parameterization,
    'final_objective': best_objective_val if not math.isnan(best_objective_val) else None,
    'best_param': best_param.tolist(), # Store the optimized param (alpha or theta)
    'final_alpha': final_alpha_np.tolist(), # Always store final alpha
    'selected_indices': selected_indices.tolist(),
    'selected_alphas': final_alpha_np[selected_indices].tolist(),
    'objective_history': [o if not math.isnan(o) else None for o in objective_history],
    'param_history': [p.tolist() for p in param_history], # History of optimized param
    'lr_history': lr_history,
    'populations': [pop['pop_id'] for pop in pop_data],
    'meaningful_indices': meaningful_indices_list,
    'total_time_seconds': total_time,
    'stopped_epoch': stopped_epoch,
    'baseline_results': best_baseline_results,
    }
    return results

# =============================================================================
# Utility functions (JSON serialization, run numbering, stats)
# =============================================================================
# (Keep these functions as they were in the previous version:
#  convert_numpy_to_python, get_latest_run_number, compute_population_stats)
def convert_numpy_to_python(obj: Any) -> Any:
    """Convert NumPy/Torch types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int_)): return int(obj)
    if isinstance(obj, (np.floating, np.float_)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (torch.Tensor)): return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, dict): return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_numpy_to_python(item) for item in obj]
    if isinstance(obj, set): return [convert_numpy_to_python(item) for item in obj]
    if isinstance(obj, (bool, np.bool_)): return bool(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)): return None # Handle NaN/Inf
    return obj

def get_latest_run_number(save_path: str) -> int:
    """Determine the latest run number in the save path directory."""
    if not os.path.exists(save_path): os.makedirs(save_path); return 0
    existing = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d))]
    run_nums = []
    for d in existing:
        match = re.match(r'run_(\d+)', d)
        if match: run_nums.append(int(match.group(1)))
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
    parser = argparse.ArgumentParser(description='Multi-population VSS (v6.2: Theta Option)')
    # Data args
    parser.add_argument('--m1', type=int, default=4)
    parser.add_argument('--m', type=int, default=20)
    parser.add_argument('--dataset-size', type=int, default=5000)
    parser.add_argument('--noise-scale', type=float, default=0.1)
    parser.add_argument('--corr-strength', type=float, default=0.0)
    parser.add_argument('--populations', nargs='+', default=['linear_regression', 'sinusoidal_regression'])
    # Optimization args
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--budget', type=int, default=None, help="Selection budget (default: calculated)")
    parser.add_argument('--penalty-type', type=str, default='Reciprocal_L1', choices=['Reciprocal_L1', 'Neg_L1', 'Max_Dev', 'Quadratic_Barrier', 'Exponential', 'None'])
    parser.add_argument('--penalty-lambda', type=float, default=0.001)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--optimizer-type', type=str, default='adam', choices=['sgd', 'adam'])
    parser.add_argument('--parameterization', type=str, default='alpha', choices=['alpha', 'theta'], help='Parameter to optimize') # New
    parser.add_argument('--alpha-init', type=str, default='random_1', help='Config for initial alpha value (used to derive initial theta if needed)')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--param-freezing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--smooth-minmax', type=float, default=float('inf'), help='Beta param for SmoothMax objective (inf for hard max)')
    # Gradient args
    parser.add_argument('--gradient-mode', type=str, default='autograd', choices=['autograd', 'reinforce'])
    parser.add_argument('--N-grad-samples', type=int, default=25, help='MC samples for grad')
    parser.add_argument('--use-baseline', action=argparse.BooleanOptionalAction, default=True, help='Baseline for REINFORCE')
    # Estimator args
    parser.add_argument('--estimator-type', type=str, default='if', choices=['plugin', 'if'], help='For T1/E[Y|X] precomputation')
    parser.add_argument('--base-model-type', type=str, default='rf', choices=['rf', 'krr', 'xgb'])
    parser.add_argument('--objective-value-estimator', type=str, default='if', choices=['if', 'mc'], help='For tracking objective value')
    parser.add_argument('--k-kernel', type=int, default=500, help='k for kernel estimators') # Added k_kernel
    # Scheduler args
    parser.add_argument('--scheduler-type', type=str, default=None, choices=['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--scheduler-step-size', type=int, default=30)
    parser.add_argument('--scheduler-gamma', type=float, default=0.1)
    parser.add_argument('--scheduler-milestones', type=int, nargs='+', default=[50, 80])
    parser.add_argument('--scheduler-t-max', type=int, default=-1)
    parser.add_argument('--scheduler-min-lr', type=float, default=1e-6)
    parser.add_argument('--scheduler-patience', type=int, default=10)
    # Other args
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save-path', type=str, default='./results_v6/multi_population/')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--lasso-alpha', type=float, default=0.01, 
                   help='Regularization strength for Lasso baseline')

    return parser.parse_args()

def main():
    args = parse_args()
    if args.populations[0].lower().startswith('baseline_failure'):
        args.budget = 3
        base_save_path = os.path.join(args.save_path, f'{args.populations[0]}/')
    elif args.populations[0].lower().startswith('linear'):
        base_save_path = os.path.join(args.save_path, 'linear_regression/')
    elif args.populations[0].lower().startswith('sinusoidal'):  
        base_save_path = os.path.join(args.save_path, 'sinusoidal_regression/')
    elif args.populations[0].lower().startswith('cubic_regression'):
        base_save_path = os.path.join(args.save_path, 'cubic_regression/')
    else:
        base_save_path = args.save_path
    os.makedirs(base_save_path, exist_ok=True)
    run_no = get_latest_run_number(base_save_path)
    save_path = os.path.join(base_save_path, f'run_{run_no}/')
    os.makedirs(save_path, exist_ok=True)
    print(f"Results will be saved in: {save_path}")

    pop_configs = [{'pop_id': i, 'dataset_type': dt} for i, dt in enumerate(args.populations)]

    # Prepare scheduler kwargs
    scheduler_kwargs = {}
    if args.scheduler_type == 'StepLR': scheduler_kwargs = {'step_size': args.scheduler_step_size, 'gamma': args.scheduler_gamma}
    elif args.scheduler_type == 'MultiStepLR': scheduler_kwargs = {'milestones': args.scheduler_milestones, 'gamma': args.scheduler_gamma}
    elif args.scheduler_type == 'ExponentialLR': scheduler_kwargs = {'gamma': args.scheduler_gamma}
    elif args.scheduler_type == 'CosineAnnealingLR': scheduler_kwargs = {'T_max': args.num_epochs if args.scheduler_t_max < 0 else args.scheduler_t_max, 'eta_min': args.scheduler_min_lr}
    elif args.scheduler_type == 'ReduceLROnPlateau': scheduler_kwargs = {'factor': args.scheduler_gamma, 'patience': args.scheduler_patience, 'min_lr': args.scheduler_min_lr}

    # Save experiment parameters
    experiment_params = vars(args).copy()
    experiment_params['pop_configs'] = pop_configs
    experiment_params['script_version'] = 'v6.2_theta_option' # Updated version name
    experiment_params['final_save_path'] = save_path
    print("\n--- Running Experiment (v6.2 + Theta Option) ---")
    print(json.dumps(convert_numpy_to_python(experiment_params), indent=2))
    with open(os.path.join(save_path, 'params_v6.json'), 'w') as f:
        json.dump(convert_numpy_to_python(experiment_params), f, indent=2)

    # Run the experiment
    results = run_experiment_multi_population_v6(
        pop_configs=pop_configs, m1=args.m1, m=args.m, dataset_size=args.dataset_size,
        noise_scale=args.noise_scale, corr_strength=args.corr_strength, num_epochs=args.num_epochs,
        budget=args.budget, penalty_type=args.penalty_type, penalty_lambda=args.penalty_lambda,
        learning_rate=args.learning_rate, optimizer_type=args.optimizer_type,
        parameterization=args.parameterization, # Pass parameterization choice
        alpha_init=args.alpha_init,
        early_stopping_patience=args.patience, param_freezing=args.param_freezing,
        smooth_minmax=args.smooth_minmax,
        gradient_mode=args.gradient_mode, N_grad_samples=args.N_grad_samples, use_baseline=args.use_baseline,
        estimator_type=args.estimator_type, base_model_type=args.base_model_type,
        objective_value_estimator=args.objective_value_estimator,
        k_kernel=args.k_kernel, # Pass k_kernel
        scheduler_type=args.scheduler_type, scheduler_kwargs=scheduler_kwargs,
        seed=args.seed, save_path=save_path, verbose=args.verbose
    )

    # --- Post-processing and Saving ---
    print("\n--- Experiment Finished ---")
    if results.get('error'):
        print(f"Experiment failed: {results['error']}")
        return

    if results['final_objective'] is not None:
        print(f"Final Robust Objective (minimized): {results['final_objective']:.4f}")
    else:
        print("Final Robust Objective: NaN")

    final_alpha = np.array(results['final_alpha']) # Always derived alpha
    selected_indices = results['selected_indices']
    actual_budget = len(selected_indices)
    print(f"Final Alpha (derived from best param): {final_alpha}")
    print(f"Selected Variables (indices, budget={actual_budget}): {selected_indices}")
    print(f"Selected Variables (alpha values): {final_alpha[selected_indices]}")

    # --- Evaluate Selection ---
    all_meaningful_indices = set()
    for indices in results['meaningful_indices']:
        all_meaningful_indices.update(indices)

    selected_set = set(selected_indices)
    intersection_size = len(selected_set & all_meaningful_indices)
    precision = intersection_size / len(selected_set) if selected_set else 0.0
    recall = intersection_size / len(all_meaningful_indices) if all_meaningful_indices else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n--- Overall Performance ---")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1_score:.4f}")

    # --- Population Stats ---
    pop_stats, overall_stats = compute_population_stats(selected_indices, results['meaningful_indices'])
    print("\n--- Population-wise Statistics ---")
    stats_str = "Population-wise statistics for selected variables:\n"
    for stat in pop_stats:
        line = (f"  Pop {stat['population']}: {stat['selected_relevant_count']}/{stat['total_relevant']} relevant selected ({stat['percentage']:.2f}%)\n")
        print(line, end='')
        stats_str += line
    print("\nOverall Population Stats:")
    print(f"  Min %: {overall_stats['min_percentage']:.2f}%")
    print(f"  Max %: {overall_stats['max_percentage']:.2f}%")
    print(f"  Median %: {overall_stats['median_percentage']:.2f}%")
    stats_str += f"\nOverall Stats: {json.dumps(convert_numpy_to_python(overall_stats), indent=2)}\n"
    stats_str += f"\nOverall Precision: {precision:.4f}\nOverall Recall: {recall:.4f}\nOverall F1 Score: {f1_score:.4f}\n"

    # Add to the stats string, the baseline results if available
    if results.get('baseline_results'):
        baseline_results = results['baseline_results']
        stats_str += f"\nBaseline Results: {json.dumps(convert_numpy_to_python(baseline_results), indent=2)}\n"
        print(f"\nBaseline Results: {json.dumps(convert_numpy_to_python(baseline_results), indent=2)}")
    # --- Save Stats and Results ---
    stats_file_path = os.path.join(save_path, 'summary_stats_v6.txt')
    with open(stats_file_path, 'w') as f: f.write(stats_str)
    print(f"\nSummary statistics saved to: {stats_file_path}")

    # --- Save Full Results ---
    results_df = create_results_dataframe(results_to_save, args, save_path)
    print(f"\nResults DataFrame created with shape: {results_df.shape}")


    results_to_save = results.copy()
    results_to_save['overall_precision'] = precision
    results_to_save['overall_recall'] = recall
    results_to_save['overall_f1_score'] = f1_score
    results_to_save['population_stats'] = pop_stats
    results_to_save['overall_stats'] = overall_stats

    results_file_path = os.path.join(save_path, 'results_v6.json')
    try:
        serializable_results = convert_numpy_to_python(results_to_save)
        with open(results_file_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"Full results dictionary saved to: {results_file_path}")
    except Exception as e:
        print(f"ERROR saving results JSON: {e}")

if __name__ == '__main__':
    main()
