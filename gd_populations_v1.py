"""
gd_populations_v3.py

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
import argparse
from torch.utils.data import DataLoader

# Import the data generation function from v2
from data import generate_data_continuous

# Global hyperparameters and clamping constants
CLAMP_MAX = 10.0
CLAMP_MIN = 1e-4
EPS = 1e-4
FREEZE_THRESHOLD = 0.1  # Threshold below which alpha values are frozen
N_FOLDS = 5             # Number of K-folds for precomputation

# =============================================================================
# K-fold based estimators for conditional means and squared functionals
# =============================================================================

def plugin_estimator_conditional_mean_Kfold(X, Y, estimator_type="rf", n_folds=N_FOLDS):
    """
    Compute out-of-fold plugin predictions for E[Y|X] using K-fold CV.
    """
    n_samples = X.shape[0]
    out_preds = np.zeros(n_samples)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train = Y[train_idx]
        if estimator_type == "rf":
            model = RandomForestRegressor(n_estimators=100,
                                          min_samples_leaf=5,
                                          n_jobs=-1,
                                          random_state=42)
        else:
            model = KernelRidge(kernel='rbf')
        model.fit(X_train, Y_train)
        out_preds[test_idx] = model.predict(X_test)
    return out_preds

def plugin_estimator_squared_conditional_Kfold(X, Y, estimator_type="rf", n_folds=N_FOLDS):
    """
    Compute the plugin estimator for E[E[Y|X]^2] using K-fold CV.
    Returns a scalar computed out-of-fold.
    """
    n_samples = X.shape[0]
    if n_folds == 1:
        if estimator_type == "rf":
            model = RandomForestRegressor(n_estimators=100,
                                          min_samples_leaf=5,
                                          n_jobs=-1,
                                          random_state=42)
        else:
            model = KernelRidge(kernel='rbf')
        model.fit(X, Y)
        mu_X = model.predict(X)
        return np.mean(mu_X ** 2)
    else:
        mu_X_all = np.zeros(n_samples)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train = Y[train_idx]
            if estimator_type == "rf":
                model = RandomForestRegressor(n_estimators=100,
                                              min_samples_leaf=5,
                                              n_jobs=-1,
                                              random_state=42)
            else:
                model = KernelRidge(kernel='rbf')
            model.fit(X_train, Y_train)
            mu_X_all[test_idx] = model.predict(X_test)
        return np.mean(mu_X_all ** 2)

def IF_estimator_conditional_mean_Kfold(X, Y, estimator_type="rf", n_folds=N_FOLDS):
    """
    Compute out-of-fold IF-corrected estimates for E[Y|X] using K-fold CV.
    For each fold, predictions are corrected using a kernel-weighted average of training residuals.
    """
    n_samples, n_features = X.shape
    out_preds = np.zeros(n_samples)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    bandwidth = 0.1 * np.sqrt(n_features)  # heuristic for kernel bandwidth
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train = Y[train_idx]
        if estimator_type == "rf":
            model = RandomForestRegressor(n_estimators=100,
                                          min_samples_leaf=5,
                                          n_jobs=-1,
                                          random_state=42)
        else:
            model = KernelRidge(kernel='rbf')
        model.fit(X_train, Y_train)
        mu_train = model.predict(X_train)
        mu_test = model.predict(X_test)
        residuals_train = Y_train - mu_train
        for i, x_test in enumerate(X_test):
            dists = np.sum((X_train - x_test)**2, axis=1)
            weights = np.exp(-dists / (2 * bandwidth**2))
            weights = weights / (np.sum(weights) + 1e-8)
            correction = np.sum(weights * residuals_train)
            out_preds[test_idx[i]] = mu_test[i] + correction
    return out_preds

def IF_estimator_squared_conditional_Kfold(X, Y, estimator_type="rf", n_folds=N_FOLDS):
    """
    Compute the IF-based estimator for E[E[Y|X]^2] using K-fold CV.
    """
    n_samples = X.shape[0]
    if n_folds == 1:
        if estimator_type == "rf":
            model = RandomForestRegressor(n_estimators=100,
                                          min_samples_leaf=5,
                                          n_jobs=-1,
                                          random_state=42)
        else:
            model = KernelRidge(kernel='rbf')
        model.fit(X, Y)
        mu_X = model.predict(X)
        plugin_estimate = np.mean(mu_X ** 2)
        residuals = Y - mu_X
        correction_term = 2 * np.mean(residuals * mu_X)
        return plugin_estimate + correction_term
    else:
        plugin_terms = []
        correction_terms = []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train = Y[train_idx]
            if estimator_type == "rf":
                model = RandomForestRegressor(n_estimators=100,
                                              min_samples_leaf=5,
                                              n_jobs=-1,
                                              random_state=42)
            else:
                model = KernelRidge(kernel='rbf')
            model.fit(X_train, Y_train)
            mu_X_test = model.predict(X_test)
            plugin_terms.append(np.mean(mu_X_test ** 2))
            residuals_test = Y[test_idx] - mu_X_test
            correction_terms.append(2 * np.mean(residuals_test * mu_X_test))
        plugin_estimate = np.mean(plugin_terms)
        correction_term = np.mean(correction_terms)
        return plugin_estimate + correction_term

# =============================================================================
# Kernel reweighting function (unchanged)
# =============================================================================

def estimate_conditional_expectation(X_batch, S_batch, E_Y_given_X_batch, alpha):
    """
    Estimate E[E[Y|X]|S] via a kernel method.
    """
    X_expanded = X_batch.unsqueeze(1)  # (batch_size, 1, n_features)
    S_expanded = S_batch.unsqueeze(0)  # (1, batch_size, n_features)
    alpha_clamped = torch.clamp(alpha, min=CLAMP_MIN)
    alpha_expanded = alpha_clamped.unsqueeze(0).unsqueeze(0)
    
    squared_distances = ((X_expanded - S_expanded) * torch.sqrt(1/(alpha_expanded + 1e-2)))**2
    kernel_matrix = torch.exp(-0.5 * torch.clamp(torch.sum(squared_distances, dim=2), max=100))
    weights = kernel_matrix / (kernel_matrix.sum(dim=0, keepdim=True) + 1e-8)
    E_Y_given_S = torch.mv(weights.t(), E_Y_given_X_batch)
    return E_Y_given_S

# =============================================================================
# Regularization penalty (unchanged)
# =============================================================================

def compute_reg_penalty(alpha, reg_type, reg_lambda, epsilon=1e-8):
    """
    Compute a regularization penalty for alpha.
    """
    if reg_type is None:
        return torch.tensor(0.0, device=alpha.device)
    elif reg_type == "Neg_L1":
        return -reg_lambda * torch.sum(torch.abs(alpha))
    elif reg_type == "Max_Dev":
        max_val = torch.tensor(1.0, device=alpha.device)
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

def compute_full_batch_objective(X, E_Y_given_X, term1,
                                 alpha, reg_lambda=0, reg_type=None, chunk_size=1000):
    """
    Compute the objective for one population:
      E[E[Y|X]^2] - E[E[Y|S(alpha)]^2] + regularization.
    """
    X = X.to(alpha.device)
    E_Y_given_X = E_Y_given_X.to(alpha.device)
    alpha = torch.clamp(alpha, min=CLAMP_MIN, max=CLAMP_MAX)
    
    batch_size = X.size(0)
    if batch_size < chunk_size:
        chunk_size = batch_size
    num_chunks = (batch_size + chunk_size - 1) // chunk_size
    total_obj = term1
    
    term2 = 0.0
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, batch_size)
        X_chunk = X[start_idx:end_idx]
        E_Y_given_X_chunk = E_Y_given_X[start_idx:end_idx]
        
        S_alpha = X_chunk + torch.randn_like(X_chunk) * torch.sqrt(alpha)
        E_Y_given_S = estimate_conditional_expectation(X_chunk, S_alpha, E_Y_given_X_chunk, alpha)
        # chunk_obj = torch.mean(E_Y_given_X_chunk**2) - torch.mean(E_Y_given_S**2)
        # total_obj += chunk_obj * (end_idx - start_idx) / batch_size
        chunk_obj = torch.mean(E_Y_given_S**2)
        term2 += chunk_obj * (end_idx - start_idx) / batch_size
    
    objective = total_obj - term2 + compute_reg_penalty(alpha, reg_type, reg_lambda)
    return objective

def compute_full_batch_objective_IF_optimized(X, E_Y_given_X_pre, term1, alpha, 
                                              reg_lambda=0, reg_type=None, chunk_size=1000):
    """
    Optimized IF-based objective computation using precomputed predictions.

    Instead of storing the fitted model, we use the precomputed predictions (E_Y_given_X_pre).
    """
    X = X.to(alpha.device)
    alpha = torch.clamp(alpha, min=CLAMP_MIN, max=CLAMP_MAX)
    
    # Use the precomputed predictions directly
    E_Y_given_X = E_Y_given_X_pre.to(alpha.device)
    
    batch_size = X.size(0)
    if batch_size < chunk_size:
        chunk_size = batch_size
    num_chunks = (batch_size + chunk_size - 1) // chunk_size
    total_term2 = 0.0
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, batch_size)
        X_chunk = X[start_idx:end_idx]
        E_Y_given_X_chunk = E_Y_given_X[start_idx:end_idx]
        
        S_alpha = X_chunk + torch.randn_like(X_chunk) * torch.sqrt(alpha)
        E_Y_given_S = estimate_conditional_expectation(X_chunk, S_alpha, E_Y_given_X_chunk, alpha)
        chunk_term2 = torch.mean(E_Y_given_S**2)
        total_term2 += chunk_term2 * (end_idx - start_idx) / batch_size
    
    objective = term1 - total_term2 + compute_reg_penalty(alpha, reg_type, reg_lambda)
    return objective

# =============================================================================
# Experiment runner for multi-population variable selection
# =============================================================================

def run_experiment_multi_population(pop_configs, m1, m, 
                                    dataset_size=10000,
                                    budget=None,
                                    noise_scale=0.0, num_epochs=30, 
                                    reg_type=None, reg_lambda=0,
                                    learning_rate=0.001,
                                    batch_size=100,
                                    optimizer_type='sgd', seed=None, 
                                    alpha_init="random", 
                                    early_stopping_patience=3,
                                    save_path='./results/multi_population/',
                                    estimator_type="plugin",  # "plugin" or "if"
                                    base_model_type="rf",     # "rf" or "krr"
                                    looped=False,
                                    param_freezing=True,
                                    verbose=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_path, exist_ok=True)
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    if budget is None:
        budget = 2*m1
    print(f"Budget for variable selection: {budget}")
    # Define common meaningful indices (as in v2)
    k_common = max(1, m1 // 2)
    common_meaningful_indices = np.arange(k_common)
    
    # Generate datasets for each population
    pop_data = []
    for pop_config in pop_configs:
        pop_id = pop_config['pop_id']
        dataset_type = pop_config['dataset_type']
        new_X, Y, A, meaningful_indices = generate_data_continuous(
            pop_id=pop_id, m1=m1, m=m, 
            dataset_type=dataset_type, 
            dataset_size=dataset_size,
            noise_scale=noise_scale, 
            seed=seed, 
            common_meaningful_indices=common_meaningful_indices
        )
        X = torch.tensor(new_X, dtype=torch.float32)
        # Y = torch.tensor(Y, dtype=torch.float32)
        Y = torch.tensor(Y.clone().detach(), dtype=torch.float32)
        
        # normalizing X
        X_mean = X.mean(dim=0)
        X_std = X.std(dim=0)
        X = (X - X_mean) / (X_std + EPS)
        # Precompute predictions and term1 using the chosen estimator method
        if estimator_type == "plugin":
            # if base_model_type == "rf":
            #     model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42)
            # else:
            #     model = KernelRidge(kernel='rbf')
            # model.fit(X.numpy(), Y.numpy())
            # # Store the predictions only
            # E_Y_given_X = torch.tensor(model.predict(X.numpy()), dtype=torch.float32)
            # term1 = None  # Not used in plugin branch
            term1 = plugin_estimator_squared_conditional_Kfold(X.numpy(), Y.numpy(), estimator_type=base_model_type)
            E_Y_given_X = torch.tensor(plugin_estimator_conditional_mean_Kfold(X.numpy(), Y.numpy(), estimator_type=base_model_type), dtype=torch.float32)
            if verbose:
                print(f"Precomputed term1 = {term1}")
        elif estimator_type == "if":
            # if verbose:
            #     print(f"Precomputing IF predictions and term1 for population {pop_id}...")
            # if base_model_type == "rf":
            #     model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42)
            # else:
            #     model = KernelRidge(kernel='rbf')
            # model.fit(X.numpy(), Y.numpy())
            # E_Y_given_X = torch.tensor(model.predict(X.numpy()), dtype=torch.float32)
            # term1 = IF_estimator_squared_conditional_Kfold(X.numpy(), Y.numpy(), estimator_type=base_model_type)
            # if verbose:
            #     print(f"Precomputed term1 = {term1}")
            term1 = IF_estimator_squared_conditional_Kfold(X.numpy(), Y.numpy(), estimator_type=base_model_type)
            E_Y_given_X = torch.tensor(plugin_estimator_conditional_mean_Kfold(X.numpy(), Y.numpy(), estimator_type=base_model_type), dtype=torch.float32)
        else:
            raise ValueError("estimator_type must be 'plugin' or 'if'")
        
        dataset = torch.utils.data.TensorDataset(X, Y, E_Y_given_X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        pop_data.append({
            'pop_id': pop_id,
            'X': X.to(device),
            'Y': Y.to(device),
            'E_Y_given_X': E_Y_given_X.to(device),
            'dataloader': dataloader,
            'meaningful_indices': meaningful_indices,
            'term1': term1
        })
    
    # Optionally visualize population covariances (if desired)
    # (e.g., via plot_population_covariances(pop_data, save_path, meaningful_indices))
    
    # Initialize alpha (the variable weight parameters)
    if alpha_init == "ones":
        alpha = torch.nn.Parameter(torch.ones(m, device=device))
    elif alpha_init == "random":
        alpha = torch.nn.Parameter(torch.ones(m, device=device) + 0.1 * torch.randn(m, device=device))
    elif alpha_init == "random_1":
        alpha = torch.nn.Parameter(torch.ones(m, device=device) + 0.1 * torch.randn(m, device=device))
    elif alpha_init == "random_5":
        alpha = torch.nn.Parameter(5*torch.ones(m, device=device) + 0.1 * torch.randn(m, device=device))
    else:
        raise ValueError("alpha_init must be 'ones' or 'random'")
    
    optimizer = (optim.Adam([alpha], lr=learning_rate)
                if optimizer_type=='adam'
                else optim.SGD([alpha], lr=learning_rate, momentum=0.9, nesterov=True))
    
    alpha_history = [alpha.detach().cpu().numpy()]
    objective_history = []
    best_objective = float('inf')
    best_alpha = None
    early_stopping_counter = 0
    gradient_history = []
    
    for epoch in range(num_epochs):
        epoch_objectives = []
        
        # Loop over populations to compute the objective
        for pop in pop_data:
            if estimator_type == "plugin":
                pop_obj = compute_full_batch_objective(
                    pop['X'], pop['E_Y_given_X'], pop['term1'], alpha, 
                    reg_lambda, reg_type, chunk_size=batch_size)
            elif estimator_type == "if":
                pop_obj = compute_full_batch_objective_IF_optimized(
                    pop['X'], pop['E_Y_given_X'], pop['term1'], alpha, 
                    reg_lambda, reg_type, chunk_size=batch_size)
            epoch_objectives.append(pop_obj)
        
        # Robust objective: take the maximum across populations
        robust_objective = torch.stack(epoch_objectives).max()
        
        optimizer.zero_grad()
        robust_objective.backward()
        torch.nn.utils.clip_grad_norm_([alpha], max_norm=1.0)
        
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
        
        optimizer.step()
        with torch.no_grad():
            alpha.clamp_(min=CLAMP_MIN, max=CLAMP_MAX)
        gradient_history.append(alpha.grad.cpu().numpy() if alpha.grad is not None else None)
        current_obj = robust_objective.item()
        objective_history.append(current_obj)
        alpha_history.append(alpha.detach().cpu().numpy())
        
        if verbose:
            print(f"Epoch {epoch}: Robust Objective = {current_obj}, Alpha = {alpha.detach().cpu().numpy()}")
        
        if current_obj < best_objective:
            best_objective = current_obj
            best_alpha = alpha.detach().cpu().numpy()
        
        # Early stopping logic
        if epoch > 15 and np.mean(objective_history[-3:]) >= np.mean(objective_history[-6:-3]):
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0
        if early_stopping_counter >= early_stopping_patience and verbose:
            print("Early stopping")
            break
    
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
    
    if looped:
        selected_vars = set(np.argsort(best_alpha)[:budget])
        true_var_indices = set(common_meaningful_indices)
        objective_curve = []
        for perturbation in np.linspace(0, 1, 100):
            perturbed_alpha = best_alpha + perturbation * (CLAMP_MAX - best_alpha)
            perturbed_alpha = np.clip(perturbed_alpha, CLAMP_MIN, CLAMP_MAX)
            perturbed_obj = compute_full_batch_objective(
                pop_data[0]['X'], pop_data[0]['E_Y_given_X'],
                pop_data[0]['term1'],
                torch.tensor(perturbed_alpha, dtype=torch.float32, device=device),
                reg_lambda=reg_lambda, reg_type=reg_type, chunk_size=batch_size
            )
            objective_curve.append(perturbed_obj.item())
        return {
            'objective_history': objective_history,
            'gradient_history': gradient_history,
            'alpha_history': alpha_history,
            'best_alpha': best_alpha,
            'selected_variables': list(selected_vars),
            'true_variable_index': list(true_var_indices),
            'objective_vs_perturbation': objective_curve,
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
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--reg-type', type=str, default='Reciprocal_L1')
    parser.add_argument('--reg-lambda', type=float, default=0.001)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--optimizer-type', type=str, default='sgd')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--alpha-init', type=str, default='random')
    parser.add_argument('--estimator-type', type=str, default='plugin', choices=['plugin', 'if'])
    parser.add_argument('--base-model-type', type=str, default='rf', choices=['rf', 'krr'])
    parser.add_argument('--populations', nargs='+', default=['linear_regression', 'sinusoidal_regression'])
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
        'num_epochs': args.num_epochs,
        'reg_type': args.reg_type,
        'reg_lambda': args.reg_lambda,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'alpha_init': args.alpha_init,
        'estimator-type': args.estimator_type,
        'optimizer_type': args.optimizer_type,
        'seed': args.seed,
        'early_stopping_patience': args.patience,
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
        num_epochs=args.num_epochs,
        reg_type=args.reg_type,
        reg_lambda=args.reg_lambda,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        optimizer_type=args.optimizer_type,
        seed=args.seed,
        alpha_init=args.alpha_init,
        estimator_type=args.estimator_type,  # "plugin" or "if"
        base_model_type=args.base_model_type,  # "rf" or "krr"
        early_stopping_patience=args.patience,
        save_path=save_path,
        param_freezing=args.param_freezing, 
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

if __name__ == '__main__':
    main()
