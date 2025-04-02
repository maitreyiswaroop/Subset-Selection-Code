# grad_desc_populations.py
import json
import numpy as np
import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from visualisers import plot_variable_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
import argparse


# Global variables for clamping
CLAMP_MAX = 10.0
CLAMP_MIN = 1e-4
EPS = 1e-4

def plugin_estimator(X, Y, estimator_type="rf"):
    """
    Plugin estimator for E[Y|X] using either random forest or kernel regression.
    """
    if estimator_type == "rf":
        model = RandomForestRegressor(n_estimators=100, 
                                      min_samples_leaf=5,
                                      n_jobs=-1)
    else:
        model = KernelRidge(kernel='rbf')
    
    model.fit(X, Y)
    return model.predict

def IF_estimator_squared_conditional(X, Y, estimator_type="rf"):
    """
    True influence function-based estimator for E[E[Y|X]^2] without using K-fold cross-validation.
    
    This estimator fits a model to estimate mu(X) = E[Y|X] on the entire dataset,
    computes the plugin estimate as:
      plugin_estimate = mean(mu(X)^2),
    and then applies the influence function correction:
      correction_term = 2 * mean((Y - mu(X)) * mu(X)).
    
    The final IF-corrected estimate is:
      IF_estimate = plugin_estimate + correction_term.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input features, shape (n_samples, n_features)
    Y : numpy.ndarray
        Target values, shape (n_samples,)
    estimator_type : str, optional
        Type of base estimator to use ('rf' for Random Forest, 'krr' for Kernel Ridge)
        
    Returns:
    --------
    float
        The bias-corrected estimate of E[E[Y|X]^2]
    """
    
    # Choose the model and parameters
    if estimator_type == "rf":
        model_class = RandomForestRegressor
        model_params = {'n_estimators': 100, 'min_samples_leaf': 5, 'n_jobs': -1, 'random_state': 42}
    else:
        model_class = KernelRidge
        model_params = {'kernel': 'rbf'}
    
    # Fit the model on the full dataset to estimate mu(X) = E[Y|X]
    model = model_class(**model_params)
    model.fit(X, Y)
    
    # Compute the estimated conditional mean for all observations.
    mu_X = model.predict(X)
    
    # Plugin estimator: estimate of E[mu(X)^2]
    plugin_estimate = np.mean(mu_X ** 2)
    
    # Compute residuals for the correction term
    residuals = Y - mu_X
    
    # Influence function correction term: 2 * E[(Y - mu(X)) * mu(X)]
    correction_term = 2 * np.mean(residuals * mu_X)
    
    # IF-corrected estimate:
    if_estimate = plugin_estimate + correction_term
    
    return if_estimate


def generate_data_continuous(pop_id, m1, m, dataset_type="linear_regression", 
                             dataset_size=10000,
                             noise_scale=0.0, seed=None, common_meaningful_indices=None):
    """
    Generate continuous data for a given population.
    
    For each population:
    - A set of "common" meaningful variables (provided as common_meaningful_indices) is used.
    - Additional unique meaningful indices are selected (if m1 > len(common_meaningful_indices)).
    - Y is generated using the specified dataset_type for that population.
    """
    if seed is not None:
        np.random.seed(seed + pop_id*50)  # Different seed per population

    # Determine meaningful indices for this population
    k_common = len(common_meaningful_indices)
    if m1 > k_common:
        remaining = [i for i in range(m) if i not in common_meaningful_indices]
        unique_indices = np.random.choice(remaining, size=m1 - k_common, replace=False)
        meaningful_indices = np.sort(np.concatenate([common_meaningful_indices, unique_indices]))
    else:
        meaningful_indices = np.array(common_meaningful_indices[:m1])
    
    # Generate meaningful features
    X_meaningful = np.random.normal(0, 1, (dataset_size, len(meaningful_indices)))
    A_meaningful = np.random.randn(len(meaningful_indices))
    AX = X_meaningful.dot(A_meaningful)
    
    if dataset_type == "linear_regression":
        Y = AX + noise_scale * np.random.randn(dataset_size)
    elif dataset_type == "quadratic_regression":
        Y = AX**2 + noise_scale * np.random.randn(dataset_size)
    elif dataset_type == "cubic_regression":
        Y = AX**3 + noise_scale * np.random.randn(dataset_size)
    elif dataset_type == "sinusoidal_regression":
        Y = np.sin(AX) + noise_scale * np.random.randn(dataset_size)
    else:
        raise ValueError("Unknown dataset_type for population ", pop_id)
    
    # Create full X by filling the non-meaningful columns with noise
    X = np.random.normal(0, 1, (dataset_size, m))
    # Place the meaningful features at the specified indices
    X[:, meaningful_indices] = X_meaningful
    
    print(f"Population {pop_id} - Meaningful indices: {meaningful_indices}")
    return X, Y, A_meaningful, meaningful_indices

def estimate_conditional_expectation(X_batch, S_batch, E_Y_given_X_batch, alpha):
    """
    Estimate E[E[Y|X]|S] via a kernel method for continuous data.
    """
    # Expand X and S for pairwise comparison:
    X_expanded = X_batch.unsqueeze(1)  # (batch_size, 1, n_features)
    S_expanded = S_batch.unsqueeze(0)  # (1, batch_size, n_features)
    
    alpha_clamped = torch.clamp(alpha, min=CLAMP_MIN)
    alpha_expanded = alpha_clamped.unsqueeze(0).unsqueeze(0)
    
    squared_distances = ((X_expanded - S_expanded) * torch.sqrt(1/(alpha_expanded + 1e-2)))**2
    kernel_matrix = torch.exp(-0.5 * torch.clamp(torch.sum(squared_distances, dim=2), max=100))
    weights = kernel_matrix / (kernel_matrix.sum(dim=0, keepdim=True))
    E_Y_given_S = torch.mv(weights.t(), E_Y_given_X_batch)
    return E_Y_given_S

def compute_reg_penalty(alpha, reg_type, reg_lambda, epsilon=1e-8):
    """
    Compute the regularization penalty for alpha.
    """
    if reg_type is None:
        return torch.tensor(0.0, device=alpha.device)
    elif reg_type == "Neg_L1":
        return - reg_lambda * torch.sum(torch.abs(alpha))
    elif reg_type == "Max_Dev":
        max_val = torch.tensor(1.0, device=alpha.device)
        return reg_lambda * torch.sum(torch.abs(max_val - alpha))
    elif reg_type == "Reciprocal_L1":
        return reg_lambda * torch.sum(torch.abs(1.0 / (alpha + epsilon)))
    elif reg_type == "Quadratic_Barrier":
        return reg_lambda * torch.sum((alpha + epsilon) ** (-2))
    elif reg_type == "Exponential":
        return reg_lambda * torch.sum(torch.exp(-alpha))
    else:
        raise ValueError("Unknown reg_type: " + str(reg_type))

def compute_full_batch_objective(X, E_Y_given_X, alpha, reg_lambda=0, reg_type=None, chunk_size=1000):
    """
    Compute objective for one population:
      E[E[Y|X]^2] - E[E[Y|S(alpha)]^2] + regularization.
    """
    X = X.to(alpha.device)
    E_Y_given_X = E_Y_given_X.to(alpha.device)
    alpha = torch.clamp(alpha, min=CLAMP_MIN, max=CLAMP_MAX)
    
    batch_size = X.size(0)
    if batch_size < chunk_size:
        chunk_size = batch_size
    num_chunks = (batch_size + chunk_size - 1) // chunk_size
    total_obj = 0.0
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, batch_size)
        X_chunk = X[start_idx:end_idx]
        E_Y_given_X_chunk = E_Y_given_X[start_idx:end_idx]
        
        S_alpha = X_chunk + torch.randn_like(X_chunk) * torch.sqrt(alpha)
        E_Y_given_S = estimate_conditional_expectation(X_chunk, S_alpha, E_Y_given_X_chunk, alpha)
        chunk_obj = torch.mean(E_Y_given_X_chunk**2) - torch.mean(E_Y_given_S**2)
        total_obj += chunk_obj * (end_idx - start_idx) / batch_size
    
    objective = total_obj + compute_reg_penalty(alpha, reg_type, reg_lambda)
    return objective

def run_experiment_multi_population(pop_configs, m1, m, 
                                    dataset_size=10000,
                                    noise_scale=0.0, num_epochs=30, 
                                    reg_type=None, reg_lambda=0,
                                    learning_rate=0.001,
                                    batch_size=100,
                                    optimizer_type='sgd', seed=None, 
                                    alpha_init = "random", # cdan be "ones" or "random"
                                    early_stopping_patience=3,
                                    save_path='./results/multi_population/',
                                    estimator_type="plugin",
                                    looped=False,
                                    verbose=False):
    """
    Run the robust optimization experiment over multiple populations.
    
    pop_configs: List of dictionaries for each population configuration. Each dict should have:
        - 'pop_id': integer id for the population.
        - 'dataset_type': string specifying the model type for Y (e.g., "linear_regression", "quadratic_regression", etc.)
    
    All populations share the same common meaningful indices (for overlap).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(save_path, exist_ok=True)
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Define common meaningful indices (for overlap across populations)
    # For example, assume the first half of m1 are common.
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
            noise_scale=noise_scale, 
            seed=seed, 
            common_meaningful_indices=common_meaningful_indices
        )
        X = torch.tensor(new_X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        # Estimate E[Y|X] using the plugin estimator.
        if estimator_type == "plugin":
            estimator = plugin_estimator(X.numpy(), Y.numpy())
            E_Y_given_X = torch.tensor(estimator(X.numpy()), dtype=torch.float32)
        elif estimator_type == "if":
            # Use the influence function estimator
            E_Y_given_X = torch.tensor(IF_estimator_squared_conditional(X.numpy(), Y.numpy(), estimator_type="rf"), dtype=torch.float32)
        # estimator = plugin_estimator(X.numpy(), Y.numpy())
        # E_Y_given_X = torch.tensor(estimator(X.numpy()), dtype=torch.float32)
        
        dataset = torch.utils.data.TensorDataset(X, Y, E_Y_given_X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        pop_data.append({
            'pop_id': pop_id,
            'X': X.to(device),
            'Y': Y.to(device),
            'E_Y_given_X': E_Y_given_X.to(device),
            'dataloader': dataloader,
            'meaningful_indices': meaningful_indices
        })
    
    # Initialize alpha (shared across populations, one per feature)
    # alpha = torch.nn.Parameter(torch.ones(m, device=device))
    if alpha_init == "ones":
        alpha = torch.nn.Parameter(torch.ones(m, device=device))
    elif alpha_init == "random":
        # alpha = torch.nn.Parameter(torch.randn(m, device=device))
        alpha = torch.nn.Parameter(torch.ones(m, device=device) + EPS * torch.randn(m, device=device))
    else:
        raise ValueError("alpha_init must be 'ones' or 'random'")
        # alpha = torch.nn.Parameter(torch.full((m,), 10.0, device=device))
    optimizer = optim.Adam([alpha], lr=learning_rate) if optimizer_type=='adam' else optim.SGD([alpha], lr=learning_rate)
    
    alpha_history = [alpha.detach().cpu().numpy()]
    objective_history = []
    best_objective = float('inf')
    best_alpha = None
    patience = early_stopping_patience
    early_stopping_counter = 0
    gradient_history = []
    
    for epoch in range(num_epochs):
        epoch_objectives = []
        
        # Loop over populations and compute objectives
        for pop in pop_data:
            # Here we compute the objective on the full batch for simplicity.
            pop_obj = compute_full_batch_objective(pop['X'], pop['E_Y_given_X'], alpha, reg_lambda, reg_type)
            epoch_objectives.append(pop_obj)
        
        # Robust objective: worst-case (maximum) over populations
        robust_objective = torch.stack(epoch_objectives).max()
        
        optimizer.zero_grad()
        robust_objective.backward()
        optimizer.step()
        with torch.no_grad():
            alpha.clamp_(min=CLAMP_MIN, max=CLAMP_MAX)
        gradient_history.append(alpha.grad.cpu().numpy())
        
        current_obj = robust_objective.item()
        objective_history.append(current_obj)
        alpha_history.append(alpha.detach().cpu().numpy())
        
        if verbose:
            print(f"\tEpoch {epoch}: Robust Objective = {current_obj}, Alpha = {alpha.detach().cpu().numpy()}")
        
        if current_obj < best_objective:
            best_objective = current_obj
            best_alpha = alpha.detach().cpu().numpy()
        # Simple early stopping based on lack of improvement
        if epoch > 15 and np.mean(objective_history[-3:]) >= np.mean(objective_history[-6:-3]):
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0
        if early_stopping_counter >= patience and verbose:
            print("Early stopping")
            break
    
    # Save diagnostics if needed
    if verbose:
        plt.figure(figsize=(10, 6))
        plt.plot(objective_history, label="Robust Objective")
        plt.xlabel("Epoch")
        plt.ylabel("Objective Value")
        plt.title("Robust Objective vs Epoch")
        plt.legend()
        plt.savefig(os.path.join(save_path, f"robust_objective_diagnostics.png"))
        plt.close()
    
    if looped:
        selected_vars = set(np.argsort(best_alpha)[:2*m1])
        true_var_indices = set(common_meaningful_indices)
        objective_curve = []
        for perturbation in np.linspace(0, 1, 100):
            perturbed_alpha = best_alpha + perturbation * (10.0 - best_alpha)
            perturbed_alpha = np.clip(perturbed_alpha, CLAMP_MIN, CLAMP_MAX)
            perturbed_obj = compute_full_batch_objective(pop_data[0]['X'], pop_data[0]['E_Y_given_X'], 
                                                         torch.tensor(perturbed_alpha, dtype=torch.float32, device=device), 
                                                         reg_lambda, reg_type)
            objective_curve.append(perturbed_obj.item())
        return {
            'objective_history': objective_history,        # shape (num_epochs,)
            'gradient_history': gradient_history,          # shape (num_epochs, n_params) or so
            'alpha_history': alpha_history,                # shape (num_epochs, n_pop, n_params_per_pop) maybe
            'best_alpha': best_alpha,                      # shape (n_pop, n_params_per_pop)
            'selected_variables': selected_vars,           # a python list or set
            'true_variable_index': true_var_indices,       # a python list or set
            'objective_vs_perturbation': objective_curve   # shape (n_points,)
        }
    else:
        return {
            'final_objective': objective_history[-1],
            'final_alpha': alpha.detach().cpu().numpy(),
            'objective_history': objective_history,
            'alpha_history': alpha_history,
            'populations': [pop['pop_id'] for pop in pop_data]
        }

def convert_numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
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

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-population variable selection')
    parser.add_argument('--m1', type=int, default=4, help='Number of meaningful features per population')
    parser.add_argument('--m', type=int, default=15, help='Total number of features')
    parser.add_argument('--dataset-size', type=int, default=10000)
    parser.add_argument('--noise-scale', type=float, default=0.01)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--reg-type', type=str, default='Reciprocal_L1')
    parser.add_argument('--reg-lambda', type=float, default=0.001)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--optimizer-type', type=str, default='sgd')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--alpha-init', type=str, default='random')
    parser.add_argument('--estimator-type', type=str, default='plugin')
    parser.add_argument('--populations', nargs='+', default=['linear_regression', 'sinusoidal_regression'])
    parser.add_argument('--save-path', type=str, default='./results/multi_population/')
    return parser.parse_args()

# Example main routine for multi-population experiment
def main():
    args = parse_args()
    save_path='./results/multi_population/'
    run_no = get_latest_run_number(save_path)
    save_path = os.path.join(save_path, f'run_{run_no}/')
    os.makedirs(save_path, exist_ok=True)

    pop_configs = [
        {'pop_id': 0, 'dataset_type': "linear_regression"},
        # {'pop_id': 1, 'dataset_type': "quadratic_regression"},
        {'pop_id': 1, 'dataset_type': "sinusoidal_regression"}
    ]
    results = run_experiment_multi_population(
        pop_configs=pop_configs,
        m1=args.m1,
        m=args.m,
        dataset_size=args.dataset_size,
        noise_scale=args.noise_scale,
        num_epochs=args.num_epochs,
        reg_type=args.reg_type,
        reg_lambda=args.reg_lambda,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        optimizer_type=args.optimizer_type,
        seed=args.seed,
        alpha_init=args.alpha_init,
        estimator_type=args.estimator_type,
        early_stopping_patience=args.patience,
        save_path=save_path,
        verbose=True
    )
    print("Final Robust Objective:", results['final_objective'])
    # Print the final variables selected (2*m1 variables with minimum alpha values)
    final_alpha = results['final_alpha']
    print("Final Alpha:", final_alpha)
    selected_indices = np.argsort(final_alpha)[:2*args.m1]  # Select the top 2*m1 variables
    print("Selected Variables (indices):", selected_indices)
    print("Selected Variables (alpha values):", final_alpha[selected_indices])

    # also save all the experiment params in a json file for reproducibility
    experiment_params = {
        'pop_configs': pop_configs,
        'm1': args.m1,
        'm': args.m,
        'dataset_size': args.dataset_size,
        'noise_scale': args.noise_scale,
        'num_epochs': args.num_epochs,
        'reg_type': args.reg_type,
        'reg_lambda': args.reg_lambda,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'alpha_init': args.alpha_init,
        'estimator_type': args.estimator_type,
        'optimizer_type': args.optimizer_type,
        'seed': args.seed,
        'early_stopping_patience': args.patience
    }
    # Convert results before saving
    serializable_results = {
        'final_objective': float(results['final_objective']),  # Convert to native Python float
        'final_alpha': results['final_alpha'].tolist(),  # Convert numpy array to list
        'objective_history': [float(x) for x in results['objective_history']],
        'alpha_history': [a.tolist() for a in results['alpha_history']],
        'selected_indices': selected_indices.tolist(),
        'selected_alphas': final_alpha[selected_indices].tolist(),
        'populations': results.get('populations', [])  # Handle optional key
    }
    serializable_params = convert_numpy_to_python(experiment_params)
    
    # Save the results
    with open(os.path.join(save_path, 'results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=4)
    with open(os.path.join(save_path, 'experiment_params.json'), 'w') as f:
        json.dump(serializable_params, f, indent=4)

    # # Save the results as well
    # with open(os.path.join(save_path, 'results.json'), 'w') as f:
    #     json.dump(results, f, indent=4)
    # Optionally, visualize the variable importance
    # plot_variable_importance(final_alpha, selected_indices, save_path=save_path, optimizer_type)


if __name__ == '__main__':
    main()
