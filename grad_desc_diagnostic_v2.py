# grad_desc_diagnostic_v2.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
import os
import torch.nn.functional as F
from visualisers import plot_variable_importance

# Global variables for clamping
CLAMP_MAX = 10.0
CLAMP_MIN = 1e-6
EPS = 1e-4

def plugin_estimator(X, Y, estimator_type="rf"):
    """
    Plugin estimator for E[Y|X] using either random forest or kernel regression.
    """
    if estimator_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, 
                                      min_samples_leaf=5,
                                      n_jobs=-1)
    else:
        from sklearn.kernel_ridge import KernelRidge
        model = KernelRidge(kernel='rbf')
    
    model.fit(X, Y)
    return model.predict

def generate_data(m1, m, n_samples, dataset_type="linear_regression", noise_scale=0.0, seed=None, discrete=False):
    """
    Generate data for continuous or discrete features.
    If discrete is True, each feature is an integer in {0,...,9} which is one-hot encoded.
    For the discrete case, m1 features are “meaningful” and the remaining m-m1 are noise.
    """
    if discrete:
        # Continuous case (as before)
        np.random.seed(0)
        indices = np.random.permutation(m)
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(0)
        # Generate discrete values in [0, 9]
        X_meaningful = np.random.randint(0, 10, size=(n_samples, m1))
        X_noise = np.random.randint(0, 10, size=(n_samples, m - m1))
        X_raw = np.concatenate([X_meaningful, X_noise], axis=1)  # shape (n_samples, m)
        X_raw = X_raw[:, indices]
        # One-hot encode: each entry becomes a 10-dim vector.
        # X_onehot will have shape (n_samples, m, 10)
        X_onehot = np.eye(10)[X_raw]
        # Reshape to (n_samples, m*10)
        X_encoded = X_onehot.reshape(n_samples, m * 10)
        # For Y, use a simple linear relation on the meaningful (raw) features.
        # Generate Y using one-hot encoded meaningful features
        X_meaningful_onehot = np.eye(10)[X_meaningful]  # shape: (n_samples, m1, 10)
        # Generate coefficients for each category of each meaningful feature
        A = [np.random.randn(10) for _ in range(m1)]  # List of coefficients for each feature
        
        # Compute Y using categorical effects
        Y = np.zeros(n_samples)
        for i in range(m1):
            category_effects = X_meaningful_onehot[:, i, :].dot(A[i])
            Y += category_effects
        Y += noise_scale * np.random.randn(n_samples)
        # Identify which original columns are meaningful (after shuffling).
        meaningful_indices = np.where(indices < m1)[0]
        print(f"\tMeaningful indices: {meaningful_indices}") 
        return X_encoded, Y, A, indices, meaningful_indices
    else:
        # Continuous case (as before)
        np.random.seed(0)
        indices = np.random.permutation(m)
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(0)
        X = np.random.normal(0, 1, (n_samples, m1))
        A = np.random.randn(m1)
        AX = X.dot(A)
        if dataset_type == "linear_regression":
            Y = AX + noise_scale * np.random.randn(n_samples)
        elif dataset_type == "quadratic_regression":
            Y = AX**2 + noise_scale * np.random.randn(n_samples)
        elif dataset_type == "cubic_regression":
            Y = AX**3 + noise_scale * np.random.randn(n_samples)
        elif dataset_type == "sinusoidal_regression":
            Y = np.sin(AX) + noise_scale * np.random.randn(n_samples)
        elif dataset_type == "classification":
            AX = (AX - np.mean(AX)) / np.std(AX)
            Y = (AX > 0.5).astype(int)
        new_X = np.zeros((n_samples, m))
        new_X[:, :m1] = X
        new_X[:, m1:] = np.random.normal(0, 1, (n_samples, m - m1))
        new_X = new_X[:, indices]
        print(f"\tMeaningful indices: {np.where(indices < m1)[0]}")
        return new_X, Y, A, indices

def estimate_conditional_expectation(X_batch, S_batch, E_Y_given_X_batch, alpha, discrete=False):
    """
    Estimate E[E[Y|X]|S] via a kernel method.
    For discrete data, alpha is group-level (one value per original variable),
    so we repeat it to match the one-hot dimension.
    """

    # Expand X and S for pairwise comparison:
    X_expanded = X_batch.unsqueeze(1)  # (batch_size, 1, n_features)
    S_expanded = S_batch.unsqueeze(0)  # (1, batch_size, n_features)
    
    if discrete:
        # alpha has shape (m,); we need to expand it to (m*10,) so that it aligns with one-hot encoded features.
        alpha_repeated = alpha.repeat_interleave(10)  # now shape (m*10,)
        alpha_clamped = torch.clamp(alpha_repeated, min=CLAMP_MIN)
        alpha_expanded = alpha_clamped.unsqueeze(0).unsqueeze(0)
        # alpha_expanded = alpha_repeated.unsqueeze(0).unsqueeze(0)  # shape (1,1, m*10)
    else:
        alpha_clamped = torch.clamp(alpha, min=CLAMP_MIN)
        alpha_expanded = alpha_clamped.unsqueeze(0).unsqueeze(0)
        # alpha_expanded = alpha.unsqueeze(0).unsqueeze(0)  # shape (1,1, n_features)
    
    # Check for NaNs in S_expanded
    # if torch.isnan(S_expanded).any():
    #     print("\tWarning: NaN values found in S_expanded")
    #     return torch.tensor(float('nan'))
    if torch.isnan(S_batch).any():
        print("\tWarning: NaN values found in S_batch")
        return torch.tensor(float('nan'))

    # Compute the weighted squared distances
    squared_distances = ((X_expanded - S_expanded) * torch.sqrt(1/(alpha_expanded + 1e-2)))**2
    if torch.isnan(squared_distances).any():
        print("\tWarning: NaN values found in squared distances")
        if (alpha_expanded < 0).any():
            print("\Reason: Negative alpha values found")
        if torch.any(torch.abs(X_expanded - S_expanded) > 1e6):
            print("\tWarning: Very large values found in X_expanded - S_expanded")
    if torch.any(squared_distances > 1e6):
        print("\tWarning: Extremely large squared distances detected.")
    # Sum over features and apply the Gaussian kernel
    # kernel_matrix = torch.exp(-0.5 * torch.sum(squared_distances, dim=2))
    kernel_matrix = torch.exp(-0.5 * torch.clamp(torch.sum(squared_distances, dim=2), max=100))
    # Normalize to obtain weights
    weights = kernel_matrix / (kernel_matrix.sum(dim=0, keepdim=True))
    if torch.isnan(weights).any():
        print("\tWarning: NaN values found in weights")
        # checking kernel_matrix for NaN values or inf values
        if torch.isnan(kernel_matrix).any():
            print("\tWarning: NaN values found in kernel matrix")
        if torch.isinf(kernel_matrix).any():
            print("\tWarning: Inf values found in kernel matrix")
        if (kernel_matrix.sum(dim=0, keepdim=True)==0).any():
            print("\tWarning: Sum of kernel matrix is zero")
        return torch.tensor(float('nan'))
    E_Y_given_S = torch.mv(weights.t(), E_Y_given_X_batch)
    if torch.isnan(E_Y_given_S).any():
        print("\tWarning: NaN values found in E_Y_given_S")
        return torch.tensor(float('nan'))
    return E_Y_given_S

def compute_reg_penalty(alpha, reg_type, reg_lambda, epsilon=1e-8):
    """
    Compute the regularization penalty for alpha.
    In the discrete case, alpha is group-level (one per original variable).
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

# Function to compute the full-batch objective for diagnostics.
def compute_full_batch_objective(X, E_Y_given_X, alpha_val, reg_lambda=0, reg_type=None, discrete=False, device='cpu', chunk_size=1000):
    """Compute full batch objective with chunked processing to save memory."""
    device = "cpu"
    X = X.to(device)
    E_Y_given_X = E_Y_given_X.to(device)
    alpha_val = alpha_val.to(device)
    alpha_val = torch.clamp(alpha_val, min=CLAMP_MIN, max=CLAMP_MAX)
    
    # Process in chunks
    batch_size = X.size(0)
    if X.size(0) < chunk_size:
        chunk_size = X.size(0)
    num_chunks = (batch_size + chunk_size - 1) // chunk_size
    total_obj = 0.0
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, batch_size)
        X_chunk = X[start_idx:end_idx]
        E_Y_given_X_chunk = E_Y_given_X[start_idx:end_idx]
        
        # Generate noisy observations for this chunk
        if discrete:
            S_alpha = X_chunk + torch.randn_like(X_chunk) * torch.sqrt(alpha_val.repeat_interleave(10))
        else:
            S_alpha = X_chunk + torch.randn_like(X_chunk) * torch.sqrt(alpha_val)
        
        # Compute conditional expectation for this chunk
        E_Y_given_S = estimate_conditional_expectation(
            X_chunk, S_alpha, E_Y_given_X_chunk, 
            alpha_val, discrete=discrete
        )
        
        # Accumulate objective
        chunk_obj = torch.mean(E_Y_given_X_chunk**2) - torch.mean(E_Y_given_S**2)
        total_obj += chunk_obj * (end_idx - start_idx) / batch_size
    
    # Add regularization penalty
    objective = total_obj + compute_reg_penalty(alpha_val, reg_type, reg_lambda)
    return objective

def run_experiment_with_diagnostics(dataset_size, m1, m, 
                                    dataset_type='linear_regression',
                                    estimator_fn=plugin_estimator,
                                    noise_scale=0.0, num_epochs=30, 
                                    reg_type=None, reg_lambda=0,
                                    learning_rate=0.001,
                                    batch_size=100,
                                    optimizer_type='sgd', seed=None, 
                                    early_stopping_patience=3,
                                    save_path=None,
                                    verbose=False,
                                    discrete=False,
                                    device='mps'):
    
    device = (
            "mps" 
            if torch.backends.mps.is_available() 
            else "cuda" 
            if torch.cuda.is_available() 
            else "cpu"
        )
    print(f"Using device: {device}")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    os.makedirs(save_path, exist_ok=True)
    
    if discrete:
        # Generate discrete data (one-hot encoded)
        new_X, Y, A, indices, meaningful_indices = generate_data(m1, m, dataset_size, 
                                                                  dataset_type=dataset_type, 
                                                                  noise_scale=noise_scale, 
                                                                  seed=seed, discrete=True)
    else:
        new_X, Y, A, indices = generate_data(m1, m, dataset_size, 
                                              dataset_type=dataset_type, 
                                              noise_scale=noise_scale, 
                                              seed=seed, discrete=False)
        meaningful_indices = np.where(indices < m1)[0]
        meaningful_indices.sort()
    
    X = torch.tensor(new_X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    batch_size = min(100, dataset_size)
    
    # Estimate E[Y|X] using the plugin estimator.
    E_Y_given_X = torch.tensor(estimator_fn(X.numpy(), Y.numpy())(X.numpy()), 
                               dtype=torch.float32).to(device)
    X = X.to(device)
    Y = Y.to(device)
    
    # For discrete data, initialize alpha with group-level dimensions (m,)
    if discrete:
        alpha = torch.nn.Parameter(torch.ones(m, device=device))
    else:
        alpha = torch.nn.Parameter(torch.ones(X.size(1), device=device))
    
    alpha_history = [alpha.detach().cpu().numpy()]
    gradient_history = []
    objective_history = []
    
    dataset = torch.utils.data.TensorDataset(X, Y, E_Y_given_X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam([alpha], lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD([alpha], lr=learning_rate)
    
    best_objective = float('inf')
    best_alpha = None

    
    patience = early_stopping_patience
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        epoch_gradients = []
        epoch_objectives = []
        
        for batch_X, batch_Y, batch_E_Y_given_X in dataloader:
            optimizer.zero_grad()
            if discrete:
                S_alpha = batch_X + torch.randn_like(batch_X, device=device) * torch.sqrt(alpha.repeat_interleave(10))
                E_Y_given_S = estimate_conditional_expectation(batch_X, S_alpha, batch_E_Y_given_X, alpha, discrete=True)
            else:
                S_alpha = batch_X + torch.randn_like(batch_X, device=device) * torch.sqrt(alpha)
                E_Y_given_S = estimate_conditional_expectation(batch_X, S_alpha, batch_E_Y_given_X, alpha, discrete=False)
            
            obj = torch.mean(batch_E_Y_given_X**2) - torch.mean(E_Y_given_S**2)
            if torch.isnan(obj) or torch.isinf(obj):
                print("Warning: Invalid objective value encountered")
                if verbose:
                    print(f"Objective: {obj}, Alpha: {alpha}")
                continue
            reg_penalty = compute_reg_penalty(alpha, reg_type, reg_lambda)
            objective = obj + reg_penalty
            
            if torch.isnan(objective) or torch.isinf(objective):
                print("Warning: Invalid total objective value encountered")
                if verbose:
                    print(f"Objective: {objective}, Alpha: {alpha}")
                continue
            
            if objective.item() < best_objective:
                best_objective = objective.item()
                best_alpha = alpha.detach().cpu().numpy()
            
            objective.backward()
            epoch_gradients.append(alpha.grad.detach().cpu().numpy())
            epoch_objectives.append(objective.item())
            optimizer.step()
            with torch.no_grad():
                alpha.data.clamp_(min=CLAMP_MIN, max=CLAMP_MAX)
        
        alpha_history.append(alpha.detach().cpu().numpy())
        gradient_history.append(np.mean([g for g in epoch_gradients], axis=0))
        objective_history.append(np.mean(epoch_objectives))
        
        if epoch % 5 == 0 and verbose:
            print(f"\tEpoch {epoch}")
            print(f"\tAverage objective: {np.mean(epoch_objectives):.4f}")
            print(f"\tAlpha values: {alpha_history[-1]}")
            print("\t----")
        
        if epoch > 15 and np.mean(objective_history[-3:]) >= np.mean(objective_history[-6:-3]):
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0
        if early_stopping_counter >= patience and verbose:
            print("Early stopping")
            break
    
    if verbose:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(objective_history)
        plt.legend(['Total', 'Variance', f'Reg (λ={reg_lambda})'])
        plt.title('Objective Value vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Objective Value')
        
        plt.subplot(2, 2, 2)
        gradient_magnitudes = [np.linalg.norm(grad) for grad in gradient_history]
        plt.plot(gradient_magnitudes)
        plt.title('Gradient Magnitude vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Magnitude')
        
        plt.subplot(2, 2, 3)
        alpha_hist_arr = np.array(alpha_history)
        for i in range(alpha_hist_arr.shape[1]):
            plt.plot(alpha_hist_arr[:, i], label=f'α_{i}')
        plt.title('Alpha Values vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Alpha Value')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        final_alpha = alpha.detach().cpu().numpy()
        delta = np.linspace(-0.5, 0.5, 20)
        landscape = []
        for d in delta:
            perturbed_alpha = torch.tensor(final_alpha + d, dtype=torch.float32)
            obj_val = compute_full_batch_objective(X, E_Y_given_X, perturbed_alpha, 
                                                   reg_lambda=reg_lambda, 
                                                   reg_type=reg_type, 
                                                   discrete=discrete, device=device)
            landscape.append(obj_val)
        plt.plot(delta, landscape)
        plt.title('Main Objective Landscape Around Final Point')
        plt.xlabel('Perturbation')
        plt.ylabel('Objective Value')
        
        plt.tight_layout()
        plt.savefig(save_path+f'/{dataset_size}_gradient_descent_diagnostics.png')
        plt.close()
    
    return {
        'final_objective': objective_history[-1],
        'final_alpha': alpha.detach().cpu().numpy(),
        'objective_history': objective_history,
        'gradient_history': gradient_history,
        'alpha_history': alpha_history,
        'true_variable_index': meaningful_indices if discrete else np.where(indices < m1)[0],
        'learned_variable_indices': np.argsort(alpha.detach().cpu().numpy())[:m1]
    }

def main():
    results_list = []
    optimizer_type = 'adam'
    # dataset_sizes = [100, 500, 1000, 2000, 5000, 7000]
    dataset_sizes = [10000]
    m1 = 2
    m = 6
    save_path = './results/gradient_descent_diagnostics/discrete/' + ('single_alpha/' if m1 == 1 else 'multiple_alpha/')
    # Here we use discrete=True so that our data is generated as discrete (integers 0-9, one-hot encoded)
    for dataset_size in dataset_sizes:
        results = run_experiment_with_diagnostics(dataset_size=dataset_size, m1=m1, m=m, num_epochs=100, 
                                                  reg_lambda=0,
                                                  seed=10,
                                                  learning_rate=0.001,
                                                  batch_size=dataset_size,
                                                  optimizer_type=optimizer_type,
                                                  noise_scale=0.01,
                                                  verbose=True,
                                                  save_path=save_path,
                                                  dataset_type="linear_regression",
                                                  discrete=True)
        print(f"Dataset size: {dataset_size}")
        print(f"Final objective: {results['final_objective']}")
        print(f"Final alpha: {results['final_alpha']}")
        print("===")
        results_list.append(results)
    
    df = pd.DataFrame(results_list)
    df.to_csv(save_path+f'results_summary.csv', index=False)
    plot_variable_importance(results_list, dataset_sizes, optimizer_type)

if __name__ == '__main__':
    main()
