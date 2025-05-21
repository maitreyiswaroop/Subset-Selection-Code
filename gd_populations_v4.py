"""
gd_populations_v4.py: correlation structure in generated data

This script performs variable subset selection using gradient descent.
It precomputes the conditional expectation predictions E[Y|X] and the squared functional 
term (i.e. E[E[Y|X]^2]) in a K-fold manner for enhanced robustness.

There are two branches:
  - "plugin": uses standard model predictions computed out-of-fold.
  - "if": uses an influence-function (IF) correction on the out-of-fold predictions.
  
The fitted model is used only to compute predictions; only the predictions and precomputed
term1 are stored and passed downstream.
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
from sklearn.neighbors import BallTree
import argparse
from torch.utils.data import DataLoader
from baselines_v3 import pooled_lasso

# Import the data generation function from v2
from data import generate_data_continuous, generate_data_continuous_with_corr
from tune_estimator import find_best_estimator
from estimators import *
import re
# Global hyperparameters and clamping constants
CLAMP_MAX = 10.0
CLAMP_MIN = 1e-2
EPS = 1e-2
FREEZE_THRESHOLD = 0.1  # Threshold below which alpha values are frozen
N_FOLDS = 5             # Number of K-folds for precomputation

# =============================================================================
# Regularization penalty
# =============================================================================

def compute_reg_penalty(alpha, reg_type, reg_lambda, epsilon=1e-8):
    """
    Compute a regularization penalty for alpha.
    """
    # clamp all alpha values to the range [CLAMP_MIN, CLAMP_MAX]
    alpha = torch.clamp(alpha, min=CLAMP_MIN, max=CLAMP_MAX)
    if reg_type is None:
        return torch.tensor(0.0, device=alpha.device)
    elif reg_type == "Neg_L1":
        return -reg_lambda * torch.sum(torch.abs(alpha))
    elif reg_type == "Max_Dev":
        max_val = torch.tensor(2.0, device=alpha.device)
        return reg_lambda * torch.sum(torch.abs(max_val - alpha))
    elif reg_type == "Reciprocal_L1": 
        return reg_lambda * torch.sum(torch.abs(1.0 / (alpha + epsilon)))
    elif reg_type == "Quadratic_Barrier":
        return reg_lambda * torch.sum((alpha + epsilon) ** (-2))
    elif reg_type == "Exponential":
        return reg_lambda * torch.sum(torch.exp(-alpha))
    elif reg_type == "None":
        return torch.tensor(0.0, device=alpha.device)
    else:
        raise ValueError("Unknown reg_type: " + str(reg_type))

# =============================================================================
# Objective functions
# =============================================================================

def compute_objective_term2(X_ref, S_alpha, E_Y_X_ref, alpha, k_kernel):
    """Helper to compute Term2 = E[E[Y|S]^2] estimate."""
    # Estimate E[Y|S] for the given S_alpha sample
    E_Y_S = estimate_conditional_expectation_knn(
        X_ref=X_ref,
        S_query=S_alpha,
        E_Y_X_ref=E_Y_X_ref,
        alpha=alpha,
        k=k_kernel # Pass k for kernel
    )
    # Calculate term2 for THIS noise sample
    term2_sample = E_Y_S.pow(2).mean()
    return term2_sample

def compute_robust_objective(X_ref, E_Y_X_ref, term1,
                             alpha, k_kernel, # Added k_kernel
                             reg_lambda=0, reg_type=None,
                             num_mc_samples=1): # Removed chunk_size as full batch is used
    """
    Computes the objective Term1 - E[E[Y|S]^2] + Reg.
    Uses Monte Carlo sampling for the expectation over S.
    Returns the total objective, the main part (Term1 - Term2), and the regularization part.
    """
    # Ensure tensors are on the correct device (alpha's device)
    device = alpha.device
    X_ref = X_ref.to(device)
    E_Y_X_ref = E_Y_X_ref.to(device)
    # Clamp alpha within the function to ensure it happens before sqrt
    alpha_clamped = torch.clamp(alpha, min=CLAMP_MIN, max=CLAMP_MAX)

    # Ensure term1 is a tensor on the same device/dtype
    term1_tensor = torch.tensor(term1, dtype=alpha.dtype, device=device)

    # Estimate Term2 using Monte Carlo samples over noise epsilon
    avg_term2 = 0.0
    # Use sqrt(alpha_clamped) for stability if alpha is near zero
    sqrt_alpha = torch.sqrt(alpha_clamped)
    for _ in range(num_mc_samples):
        epsilon = torch.randn_like(X_ref)
        S_alpha = X_ref + sqrt_alpha * epsilon
        term2_sample = compute_objective_term2(X_ref, S_alpha, E_Y_X_ref, alpha, k_kernel)
        avg_term2 += term2_sample

    term2_estimate = avg_term2 / num_mc_samples

    # *** Calculate components separately ***
    main_objective_part = term1_tensor - term2_estimate
    reg_penalty_part = compute_reg_penalty(alpha, reg_type, reg_lambda)

    # Compute final objective
    total_objective = main_objective_part + reg_penalty_part

    # *** Return all three components ***
    return total_objective, main_objective_part, reg_penalty_part

# =============================================================================
# Experiment runner for multi-population variable selection
# =============================================================================

def get_pop_data(pop_configs, m1, m,
                  dataset_size=10000,
                  noise_scale=0.0,
                  corr_strength=0.0,
                  common_meaningful_indices=None,
                  estimator_type="plugin",
                  device="cpu",
                  base_model_type="rf",
                  batch_size=10000,
                  seed=None):
    """
    Generate datasets and precompute necessary terms for each population.
    Uses the appropriate E[Y|X] estimator based on estimator_type.
    """
    # Use default common indices if none provided (as in original code)
    if common_meaningful_indices is None:
        k_common = max(1, m1 // 2)
        common_meaningful_indices = np.arange(k_common)

    pop_data = []
    for pop_config in pop_configs:
        pop_id = pop_config['pop_id']
        dataset_type = pop_config['dataset_type']
        current_seed = seed + pop_id if seed is not None else None # Ensure different data per pop if seed is set

        # Generate Data
        if corr_strength > 0:
            new_X, Y, _, meaningful_indices = generate_data_continuous_with_corr( # Ignoring A
                pop_id=pop_id, m1=m1, m=m,
                dataset_type=dataset_type,
                dataset_size=dataset_size,
                noise_scale=noise_scale,
                corr_strength=corr_strength,
                seed=current_seed,
                common_meaningful_indices=common_meaningful_indices
            )
        else:
            new_X, Y, _, meaningful_indices = generate_data_continuous( # Ignoring A
                pop_id=pop_id, m1=m1, m=m,
                dataset_type=dataset_type,
                dataset_size=dataset_size,
                noise_scale=noise_scale,
                seed=current_seed,
                common_meaningful_indices=common_meaningful_indices
            )

        # Convert to Tensor and Standardize
        X_np = new_X if isinstance(new_X, np.ndarray) else new_X.numpy()
        Y_np = Y if isinstance(Y, np.ndarray) else Y.numpy()

        X = torch.tensor(X_np, dtype=torch.float32)
        X_mean = X.mean(dim=0)
        X_std = X.std(dim=0)
        X = (X - X_mean) / (X_std + EPS) # Standardize X

        Y = torch.tensor(Y_np, dtype=torch.float32)
        Y_mean = Y.mean()
        Y_std = Y.std()
        Y = (Y - Y_mean) / (Y_std + EPS)  # Standardize Y

        # Ensure numpy versions are standardized for estimators
        X_np_std = X.numpy()
        Y_np_std = Y.numpy()

        # Precompute Term1 and E[Y|X] based on estimator_type
        if estimator_type == "plugin":
            print(f"Pop {pop_id}: Computing Plugin Term1 and E[Y|X]...")
            term1 = plugin_estimator_squared_conditional(X_np_std, Y_np_std, estimator_type=base_model_type, n_folds=N_FOLDS, seed=current_seed)
            E_Y_given_X_np = plugin_estimator_conditional_mean(X_np_std, Y_np_std, estimator_type=base_model_type, n_folds=N_FOLDS, seed=current_seed)
        elif estimator_type == "if":
            print(f"Pop {pop_id}: Computing IF Term1 and E[Y|X]...")
            term1 = IF_estimator_squared_conditional(X_np_std, Y_np_std, estimator_type=base_model_type, n_folds=N_FOLDS, seed=current_seed)
            # *** FIXED: Use IF estimator for E[Y|X] when estimator_type is 'if' ***
            E_Y_given_X_np = IF_estimator_conditional_mean(X_np_std, Y_np_std, estimator_type=base_model_type, n_folds=N_FOLDS, seed=current_seed)
        else:
            raise ValueError("estimator_type must be 'plugin' or 'if'")

        E_Y_given_X = torch.tensor(E_Y_given_X_np, dtype=torch.float32)

        # Create DataLoader (optional, only if using mini-batches later, but included based on original code)
        dataset = torch.utils.data.TensorDataset(X, Y, E_Y_given_X) # Note: Storing standardized X, Y, E[Y|X]
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print(f"Pop {pop_id}: Done precomputing.")
        pop_data.append({
            'pop_id': pop_id,
            'X': X.to(device),  # Standardized X
            'Y': Y.to(device),  # Standardized Y
            'E_Y_given_X': E_Y_given_X.to(device), # Corresponding E[Y|X] estimate
            'dataloader': dataloader,
            'meaningful_indices': meaningful_indices,
            'term1': term1 # Term1 based on chosen estimator_type
        })
    return pop_data

def init_alpha(m, alpha_init="random", noise = 1.0, 
               device="cpu"):
    """
    Initialize the alpha parameter based on the specified initialization method.
    """
    if re.match(r"random_\d+", alpha_init):
        k = int(alpha_init.split('_')[1])
        return torch.nn.Parameter(k * torch.ones(m, device=device) + noise * torch.randn(m, device=device))
    elif alpha_init == "ones":
        return torch.nn.Parameter(torch.ones(m, device=device))
    elif alpha_init == "random":
        return torch.nn.Parameter(torch.ones(m, device=device) + noise * torch.randn(m, device=device))
    else:
        raise ValueError("alpha_init must be 'ones' or 'random'")
    
def run_experiment_multi_population(pop_configs, m1, m, 
                                    dataset_size=5000,
                                    budget=None,
                                    noise_scale=0.0, 
                                    corr_strength=0.0,
                                    num_epochs=30, 
                                    reg_type=None, reg_lambda=0,
                                    learning_rate=0.001,
                                    batch_size=10000,
                                    optimizer_type='sgd', seed=None, 
                                    alpha_init="random", 
                                    early_stopping_patience=3,
                                    save_path='./results/multi_population/',
                                    estimator_type="plugin",  # "plugin" or "if"
                                    base_model_type="rf",     # "rf" or "krr"
                                    looped=False,
                                    k_kernel=1000,
                                    num_mc_samples=10,
                                    smooth_minmax= float('inf'), # beta param for SmoothMax
                                    param_freezing=True,
                                    run_baseline=False,
                                    verbose=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = min(batch_size, dataset_size)
    os.makedirs(save_path, exist_ok=True)
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        seed = np.random.randint(0, 10000)
    print(f"Using seed: {seed}")
    
    if budget is None:
        budget = 2*m1
    print(f"Budget for variable selection: {budget}")
    # Define common meaningful indices (as in v2)
    k_common = max(1, m1 // 2)
    common_meaningful_indices = np.arange(k_common)
    # Generate datasets for each population
    pop_data = get_pop_data(
        pop_configs=pop_configs, 
        m1=m1, m=m, 
        dataset_size=dataset_size,
        noise_scale=noise_scale, 
        corr_strength=corr_strength,
        common_meaningful_indices=common_meaningful_indices,
        estimator_type=estimator_type,
        device=device,
        base_model_type=base_model_type,
        batch_size=batch_size,
        seed=seed
    )
    # Initialize alpha (the variable weight parameters)
    alpha = init_alpha(m, alpha_init=alpha_init, noise=0.01, device=device)
    print(f"Initialized alpha: {alpha.detach().cpu().numpy()}")
    optimizer = (optim.Adam([alpha], lr=learning_rate)
                if optimizer_type=='adam'
                else optim.SGD([alpha], lr=learning_rate, momentum=0.9, nesterov=True))
    
    alpha_history = [alpha.detach().cpu().numpy()]
    objective_history = []       # Stores the robust (max) total objective
    main_obj_part_history = []   # Stores Term1 - Term2 of the worst-case pop
    reg_penalty_history = []     # Stores Reg Penalty of the worst-case pop
    best_objective = float('inf')
    best_alpha = None
    early_stopping_counter = 0
    gradient_history = []
    
    beta = smooth_minmax
    for epoch in range(num_epochs):
        epoch_total_objectives = []
        epoch_main_obj_parts = []
        epoch_reg_penalties = []
        
        # Loop over populations to compute the objective
        for pop in pop_data:
            pop_total_obj, pop_main_obj, pop_reg = compute_robust_objective(
                X_ref=pop['X'], # Use the full population X as reference
                E_Y_X_ref=pop['E_Y_given_X'],
                term1=pop['term1'],
                alpha=alpha,
                k_kernel=k_kernel, # Pass k
                reg_lambda=reg_lambda,
                reg_type=reg_type,
                num_mc_samples=num_mc_samples # Pass num_mc_samples
            )
            epoch_total_objectives.append(pop_total_obj)
            # *** Store detached components for history ***
            epoch_main_obj_parts.append(pop_main_obj.detach())
            epoch_reg_penalties.append(pop_reg.detach())
        
        objectives_tensor = torch.stack(epoch_total_objectives) # Tensor of shape [num_populations]
        if torch.isfinite(torch.tensor(beta)) and beta > 0:
            # --- Use SmoothMax (LogSumExp) ---
            # Numerically stable LogSumExp: M + log(sum(exp(beta*Lp - M)))
            # where M = max(beta*Lp)
            with torch.no_grad(): # Find max without tracking gradient here
                M = torch.max(beta * objectives_tensor)
            # Calculate LogSumExp
            logsumexp_val = M + torch.log(torch.sum(torch.exp(beta * objectives_tensor - M)))
            # Calculate SmoothMax
            robust_objective = (1.0 / beta) * logsumexp_val
            # Ensure the result requires grad through the original objectives
            # This happens automatically as objectives_tensor elements require grad
        else:
            # --- Use Hard Max (original behavior) ---
            robust_objective = torch.max(objectives_tensor)
        # Robust objective: take the maximum across populations
        # robust_objective = torch.stack(epoch_objectives).max()
        
        optimizer.zero_grad()
        robust_objective.backward()
        # torch.nn.utils.clip_grad_norm_([alpha], max_norm=10.0)
        
        if param_freezing:
            with torch.no_grad():
                frozen_mask = alpha.detach() < FREEZE_THRESHOLD
                if alpha.grad is not None:
                    alpha.grad[frozen_mask] = 0
                for group in optimizer.param_groups:
                    for p in group['params']:
                        state = optimizer.state[p]
                        if 'momentum_buffer' in state:
                            buf = state['momentum_buffer']
                            buf[frozen_mask] = 0
        torch.nn.utils.clip_grad_norm_([alpha], max_norm=5.0)
        
        optimizer.step()
        with torch.no_grad():
            alpha.clamp_(min=CLAMP_MIN, max=CLAMP_MAX)
        gradient_history.append(alpha.grad.cpu().numpy() if alpha.grad is not None else None)
        current_total_obj_scalar = robust_objective.item()
        current_main_obj_scalar = torch.max(torch.stack(epoch_main_obj_parts)).item()
        current_reg_penalty_scalar = torch.max(torch.stack(epoch_reg_penalties)).item()

        objective_history.append(current_total_obj_scalar)
        main_obj_part_history.append(current_main_obj_scalar)
        reg_penalty_history.append(current_reg_penalty_scalar)
        alpha_history.append(alpha.detach().cpu().numpy())
        
        if verbose:
            print(f"Epoch {epoch}: Robust Obj={current_total_obj_scalar:.4f} "
                   f"(Main={current_main_obj_scalar:.4f}, Reg={current_reg_penalty_scalar:.4f})")

        
        if current_total_obj_scalar < best_objective:
            best_objective = current_total_obj_scalar
            best_alpha = alpha.detach().cpu().numpy()
            if verbose:
                print(f"New best objective: {best_objective:.4f} with alpha: {best_alpha}")
        
        # Early stopping logic
        if epoch > 15 and np.mean(objective_history[-3:]) >= np.mean(objective_history[-6:-3]):
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0
        if early_stopping_counter >= early_stopping_patience and verbose:
            print("Early stopping")
            break

        # clamping the alpha values to the range [CLAMP_MIN, CLAMP_MAX]
        alpha.data.clamp_(min=CLAMP_MIN, max=CLAMP_MAX)
    
    # post-processing, also seeing how the estimator performs
    if verbose:
        # selecting 10 random alphas from the history
        random_indices = np.random.choice(len(alpha_history), size=min(10, len(alpha_history)), replace=False)
        alpha_lists = [alpha_history[i] for i in random_indices]
        alpha_lists = sorted(alpha_lists, key=lambda x: np.max(x))
        # save the alphas used here
        np.save(os.path.join(save_path, "testing_alpha_history.npy"), alpha_lists)
        for pop in pop_data:
            test_estimator(seeds=[seed],
                           alpha_lists=[alpha_history[i] for i in random_indices],
                           X=pop['X'], Y=pop['Y'],
                           save_path = os.path.join(save_path, f"estimator_test_{pop['pop_id']}.png")
            )
                       
    # (Optional) Save diagnostics, for example plotting the objective history
    if verbose:
        plt.figure(figsize=(10, 6))
        plt.plot(objective_history, label="Robust Objective")
        plt.xlabel("Epoch")
        plt.ylabel("Objective Value")
        plt.title("Robust Objective vs Epoch")
        plt.legend()
        plt.savefig(os.path.join(save_path, "robust_objective_diagnostics.png"))
        plt.close()

        # also gradient norms
        plt.figure(figsize=(10, 6))
        plt.plot([np.linalg.norm(g) for g in gradient_history if g is not None], label="Gradient Norm")
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Norm")
        plt.title("Gradient Norm vs Epoch")
        plt.legend()
        plt.savefig(os.path.join(save_path, "gradient_norm_diagnostics.png"))
        plt.close()
    
    if run_baseline:
    # Convert pop_data to have CPU tensors before passing to pooled_lasso
        cpu_pop_data = []
        for pop in pop_data:
            cpu_pop = {
                'pop_id': pop['pop_id'],
                'X': pop['X'].cpu().numpy(),  # Move to CPU and convert to numpy
                'Y': pop['Y'].cpu().numpy(),  # Move to CPU and convert to numpy
                'meaningful_indices': pop['meaningful_indices'],
                # Only include fields needed by pooled_lasso
            }
            cpu_pop_data.append(cpu_pop)
            
        # pooled lasso
        pooled_lasso_results = pooled_lasso(
            pop_configs=cpu_pop_data, 
            budget=budget, 
            m=m, 
            m1=m1)
        
        # Save pooled lasso results
        with open(os.path.join(save_path, 'pooled_lasso_results.json'), 'w') as f:
            json.dump(convert_numpy_to_python(pooled_lasso_results), f, indent=4)
        if verbose:
            print("Pooled Lasso Results:", pooled_lasso_results)

    
    if looped:
        selected_vars = set(np.argsort(best_alpha)[:budget])
        true_var_indices = set(common_meaningful_indices)
        objective_curve_with_reg = []
        objective_curve = []
        for perturbation in np.linspace(0, 1, 100):
            perturbed_alpha = best_alpha + perturbation * (CLAMP_MAX - best_alpha)
            perturbed_alpha = np.clip(perturbed_alpha, CLAMP_MIN, CLAMP_MAX)
            perturbed_obj = compute_robust_objective(
                pop_data[0]['X'], pop_data[0]['E_Y_given_X'],
                pop_data[0]['term1'],
                torch.tensor(perturbed_alpha, dtype=torch.float32, device=device),
                reg_lambda=0, reg_type=reg_type,
                num_mc_samples=num_mc_samples, k_kernel=k_kernel
            )
            objective_curve.append(perturbed_obj.item())
            objective_curve_with_reg.append(compute_robust_objective(
                pop_data[0]['X'], pop_data[0]['E_Y_given_X'],
                pop_data[0]['term1'],
                torch.tensor(perturbed_alpha, dtype=torch.float32, device=device),
                reg_lambda=reg_lambda, reg_type=reg_type,
                num_mc_samples=num_mc_samples, k_kernel=k_kernel
            ).item())
        return {
            'objective_history': objective_history,
            'gradient_history': gradient_history,
            'alpha_history': alpha_history,
            'best_alpha': best_alpha,
            'selected_variables': list(selected_vars),
            'true_variable_index': list(true_var_indices),
            'objective_vs_perturbation': objective_curve,
            'objective_with_reg_vs_perturbation': objective_curve_with_reg,
            'meaningful_indices': [pop['meaningful_indices'] for pop in pop_data]
        }
    else:
        return {
            'final_objective': objective_history[-1],
            'final_alpha': best_alpha,
            'objective_history': objective_history,
            'alpha_history': alpha_history,
            'gradient_history': gradient_history,
            'populations': [pop['pop_id'] for pop in pop_data],
            'meaningful_indices': [pop['meaningful_indices'] for pop in pop_data]
        }

# =============================================================================
# Utility functions for JSON serialization and run numbering
# =============================================================================

def convert_numpy_to_python(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    return obj

def get_latest_run_number(save_path):
    """
    Determine the latest run number in the save path directory.
    """
    if not os.path.exists(save_path):
        return 0
    existing_runs = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d))]
    run_numbers = [int(d.split('_')[-1]) for d in existing_runs if d.startswith('run_') and d.split('_')[-1].isdigit()]
    if not run_numbers:
        return 0
    return max(run_numbers) + 1

def compute_population_stats(selected_indices, meaningful_indices_list):
    """
    Compute population-wise statistics for selected variables.
    
    Parameters:
        selected_indices : list or array
            Final selected variable indices.
        meaningful_indices_list : list of arrays/lists
            For each population, the list/array of meaningful indices.
            
    Returns:
        pop_stats : list of dicts
            Population-wise statistics containing:
              - population: population index
              - selected_relevant_count: number of relevant variables selected
              - total_relevant: total number of meaningful (relevant) variables for that population
              - percentage: percentage (%) of relevant variables selected
        overall_stats : dict
            Summary across populations: min, max, and median percentage.
    """
    pop_stats = []
    for i, meaningful in enumerate(meaningful_indices_list):
        meaningful_set = set(meaningful)
        selected_set = set(selected_indices)
        common = selected_set.intersection(meaningful_set)
        count = len(common)
        total = len(meaningful_set)
        percentage = (count / total * 100) if total > 0 else 0
        pop_stats.append({
            'population': i,
            'selected_relevant_count': count,
            'total_relevant': total,
            'percentage': percentage
        })
    
    percentages = [stat['percentage'] for stat in pop_stats]
    overall_stats = {
        'min_population': pop_stats[np.argmin(percentages)],
        'max_population': pop_stats[np.argmax(percentages)],
        'median_percentage': float(np.median(percentages))
    }
    
    return pop_stats, overall_stats

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-population variable selection')
    parser.add_argument('--m1', type=int, default=4, help='Number of meaningful features per population')
    parser.add_argument('--m', type=int, default=100, help='Total number of features')
    parser.add_argument('--dataset-size', type=int, default=10000)
    parser.add_argument('--noise-scale', type=float, default=0.1)
    parser.add_argument('--corr-strength', type=float, default=0.0)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--reg-type', type=str, default='Reciprocal_L1')
    parser.add_argument('--reg-lambda', type=float, default=0.001)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--optimizer-type', type=str, default='sgd')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--alpha-init', type=str, default='random')
    parser.add_argument('--k-kernel', type=int, default=1000,
                        help='Number of neighbors for KNN kernel')
    parser.add_argument('--num-mc-samples', type=int, default=10,
                        help='Number of Monte Carlo samples for estimating E[E[Y|S]^2]')
    parser.add_argument('--estimator-type', type=str, default='plugin', choices=['plugin', 'if'])
    parser.add_argument('--base-model-type', type=str, default='rf', choices=['rf', 'krr', 'xgb'])
    parser.add_argument('--populations', nargs='+', default=['linear_regression', 'sinusoidal_regression'])
    parser.add_argument('--run-baseline', action='store_true',
                        help='Run baseline Pooled Lasso for comparison')
    parser.add_argument('--smooth-minmax', type=float, default=float('inf'),
                        help='Beta param value for SmoothMax for the objective. Default is "Infinity", i.e. true max')
    parser.add_argument('--param-freezing', action='store_true',
                        help='Enable parameter freezing for alpha values below threshold')
    parser.add_argument('--save-path', type=str, default='./results/multi_population/')
    return parser.parse_args()

def main():
    args = parse_args()
    base_save_path = args.save_path
    run_no = get_latest_run_number(base_save_path)
    save_path = os.path.join(base_save_path, f'run_{run_no}/')
    os.makedirs(save_path, exist_ok=True)
    
    pop_configs = [
        {'pop_id': i, 'dataset_type': args.populations[i]}
        for i in range(len(args.populations))
    ]
    #     {'pop_id': 0, 'dataset_type': "linear_regression"},
    #     {'pop_id': 1, 'dataset_type': "sinusoidal_regression"},
    #     {'pop_id': 2, 'dataset_type': "quadratic_regression"},
    #     {'pop_id': 3, 'dataset_type': "cubic_regression"}
    # ]
    budget = args.m1 // 2 + len(args.populations) * args.m1 // 2
    # Save experiment parameters for reproducibility
    experiment_params = {
        'pop_configs': pop_configs,
        'm1': args.m1,
        'm': args.m,
        'dataset_size': args.dataset_size,
        'budget': budget,
        'noise_scale': args.noise_scale,
        'corr_strength': args.corr_strength,
        'num_epochs': args.num_epochs,
        'reg_type': args.reg_type,
        'reg_lambda': args.reg_lambda,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'alpha_init': args.alpha_init,
        'k_kernel': args.k_kernel,
        'num_mc_samples': args.num_mc_samples,
        'estimator-type': args.estimator_type,
        'optimizer_type': args.optimizer_type,
        'seed': args.seed,
        'early_stopping_patience': args.patience,
        'smooth-minmax': args.smooth_minmax,
        'run_baseline': args.run_baseline,
        'param_freezing': args.param_freezing,
    }
    print("Running multi-population experiment with parameters:")
    print(json.dumps(convert_numpy_to_python(experiment_params), indent=4))
    
    results = run_experiment_multi_population(
        pop_configs=pop_configs,
        m1=args.m1,
        m=args.m,
        dataset_size=args.dataset_size,
        budget=budget,
        noise_scale=args.noise_scale,
        corr_strength=args.corr_strength,
        num_epochs=args.num_epochs,
        reg_type=args.reg_type,
        reg_lambda=args.reg_lambda,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        optimizer_type=args.optimizer_type,
        seed=args.seed,
        alpha_init=args.alpha_init,
        num_mc_samples=args.num_mc_samples,
        k_kernel=args.k_kernel,
        estimator_type=args.estimator_type,  # "plugin" or "if"
        base_model_type=args.base_model_type,  # "rf" or "krr"
        early_stopping_patience=args.patience,
        save_path=save_path,
        smooth_minmax=args.smooth_minmax,
        param_freezing=args.param_freezing, 
        run_baseline=args.run_baseline,
        verbose=True
    )
    print("Final Robust Objective:", results['final_objective'])
    final_alpha = np.array(results['final_alpha'])
    print("Final Alpha:", final_alpha)
    selected_indices = np.argsort(final_alpha)[:budget]
    print("Selected Variables (indices):", selected_indices)
    print("Selected Variables (alpha values):", final_alpha[selected_indices])
    
    recall = len(set(selected_indices).intersection(set(np.concatenate(results['meaningful_indices'])))) / len(set(np.concatenate(results['meaningful_indices'])))
    print("Recall:", recall)
    
    print("\nPopulation-wise Important Parameters:")
    populations = results.get('populations', [])
    meaningful_indices = results.get('meaningful_indices', [])
    for i, pop_id in enumerate(populations):
        print(f"Population {pop_id} - Meaningful indices: {meaningful_indices[i]}")
    
    pop_stats, overall_stats = compute_population_stats(selected_indices, results['meaningful_indices'])
    print("\nPopulation-wise statistics for selected variables:")
    for stat in pop_stats:
        print(f"Population {stat['population']}: {stat['selected_relevant_count']} out of {stat['total_relevant']} "
            f"selected ({stat['percentage']:.2f}%)")
    print("Overall statistics:", overall_stats)

    # Write the statistics to a text file
    stats_file_path = os.path.join(save_path, 'population_stats.txt')
    with open(stats_file_path, 'w') as stats_file:
        stats_file.write("Population-wise statistics for selected variables:\n")
        for stat in pop_stats:
            stats_file.write(f"Population {stat['population']}: {stat['selected_relevant_count']} out of {stat['total_relevant']} "
                             f"selected ({stat['percentage']:.2f}%)\n")
        stats_file.write(f"\nOverall statistics: {overall_stats}\n")

    serializable_results = {
        'final_objective': float(results['final_objective']),
        'final_alpha': results['final_alpha'] if isinstance(results['final_alpha'], list) else results['final_alpha'].tolist(),
        'objective_history': [float(x) for x in results['objective_history']],
        'alpha_history': [a.tolist() for a in results['alpha_history']],
        'selected_indices': selected_indices.tolist(),
        'selected_alphas': final_alpha[selected_indices].tolist(),
        'populations': results.get('populations', []),
        'recall': recall,
        'meaningful_indices': [mi.tolist() if isinstance(mi, np.ndarray) else mi for mi in results['meaningful_indices']]
    }
    serializable_params = convert_numpy_to_python(experiment_params)
    
    with open(os.path.join(save_path, 'results.json'), 'w') as f:
        # json.dump(serializable_results, f, indent=4)
        json.dump(convert_numpy_to_python(serializable_results), f, indent=4)
    with open(os.path.join(save_path, 'experiment_params.json'), 'w') as f:
        json.dump(serializable_params, f, indent=4)

    # visualization for objective history, both with and without regularization
    plt.figure(figsize=(12, 7))
    epochs = range(len(results['objective_history']))

    # Plot total robust objective
    plt.plot(epochs, results['objective_history'], label="Robust Objective (Total)", linewidth=2)

    # Plot main objective part (Term1 - Term2) of the worst-case population
    plt.plot(epochs, results['main_obj_part_history'], label="Main Part (Worst Pop)", linestyle='--')

    # Plot regularization penalty part of the worst-case population
    plt.plot(epochs, results['reg_penalty_history'], label="Reg Penalty (Worst Pop)", linestyle=':')

    plt.xlabel("Epoch")
    plt.ylabel("Objective Value / Component Value")
    plt.title("Objective Components vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "objective_components_diagnostics.png"))
    plt.close()

    # Keep the original robust objective plot if desired (optional)
    # plt.figure(figsize=(10, 6))
    # plt.plot(results['objective_history'], label="Robust Objective")
    # plt.xlabel("Epoch")
    # plt.ylabel("Objective Value")
    # plt.title("Robust Objective vs Epoch")
    # plt.legend()
    # plt.savefig(os.path.join(save_path, "robust_objective_diagnostics.png"))
    # plt.close()

if __name__ == '__main__':
    main()
