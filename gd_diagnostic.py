# gd_diagnostic.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.neighbors import KernelDensity
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
import os
import torch.nn.functional as F
from visualisers import plot_variable_importance
# Global variable for clamping maximum value
CLAMP_MAX = 10.0
CLAMP_MIN = 1e-8

def plugin_estimator(X, Y, estimator_type="rf"):
    """
    Plugin estimator for E[Y|X] using either random forest or kernel regression
    
    Args:
        X: Input features
        Y: Target values
        estimator_type: 'rf' for random forest or 'kernel' for kernel regression
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


def generate_data(m1, m, n_samples, dataset_type = "linear_regression", noise_scale=0.0, seed=None):
    # TODO: find a better way of doing this
    # first determining the shuffled indices using a fixed seed for consistency across expts
    np.random.seed(0)
    indices = np.random.permutation(m)

    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(0)

    # X from normal distribution N(0,1)
    X = np.random.normal(0, 1, (n_samples, m1))
    
    # random linear map A
    A = np.random.randn(m1)
    
    AX = X.dot(A)
    # AX = (AX - np.mean(AX)) / np.std(AX)
    
    # Y based on AX
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
    
    # new X with m dimensions
    new_X = np.zeros((n_samples, m))
    
    # copy m1 dimensions from original X
    new_X[:, :m1] = X
    
    # fill remaining dimensions with random values from U[0,1]
    # new_X[:, m1:] = np.random.uniform(0, 1, (n_samples, m - m1))
    new_X[:, m1:] = np.random.normal(0, 1, (n_samples, m - m1))
    # shuffle the indices of the columns
    # indices = np.random.permutation(m)
    new_X = new_X[:, indices]
    
    print(f"\tMeaningful indices: {np.where(indices < m1)[0]}")
    return new_X, Y, A, indices

def estimate_conditional_expectation(X_batch, S_batch, E_Y_given_X_batch, alpha):
    """
    Vectorized estimation of E[E[Y|X]|S] using kernel density estimation for P(X|S)
    
    Args:
        X_batch: (batch_size, n_features)
        S_batch: (batch_size, n_features)
        E_Y_given_X_batch: (batch_size,)
        alpha: (n_features,)
    Returns:
        E_Y_given_S: (batch_size,)
    """
    # alpha = F.relu(alpha) + 1e-8
    # Compute pairwise squared distances with alpha scaling
    # Reshape to enable broadcasting
    X_expanded = X_batch.unsqueeze(1)  # (batch_size, 1, n_features)
    S_expanded = S_batch.unsqueeze(0)  # (1, batch_size, n_features)
    alpha_expanded = alpha.unsqueeze(0).unsqueeze(0)  # (1, 1, n_features)
    
    # Compute weighted distances
    # squared_distances = ((X_expanded - S_expanded) / torch.sqrt(alpha_expanded))**2
    # adding epsilon to avoid numerical instability
    squared_distances = ((X_expanded - S_expanded) * torch.sqrt(1/(alpha_expanded + 1e-2)))**2

    # Check for NaNs in squared distances
    if torch.isnan(squared_distances).any():
        print("\tWarning: NaN values found in squared distances")
    
    # Sum over features dimension and compute kernel
    kernel_matrix = torch.exp(-0.5 * torch.sum(squared_distances, dim=2))  # (batch_size, batch_size)
    
    # Normalize weights
    weights = kernel_matrix / (kernel_matrix.sum(dim=0, keepdim=True))  # (batch_size, batch_size)
    
    # Compute weighted average
    E_Y_given_S = torch.mv(weights.t(), E_Y_given_X_batch)  # (batch_size,)
    
    return E_Y_given_S

def compute_reg_penalty(alpha, reg_type, reg_lambda, epsilon=1e-8):
    """
    Compute the regularization penalty for the given alpha.
    
    Parameters:
      alpha      : The parameter vector (torch tensor).
      reg_type   : String indicating the regularization type.
                   Options: None, "Neg_L1", "Max_Dev", "Reciprocal_L1",
                            "Quadratic_Barrier", "Exponential"
      reg_lambda : Regularization strength.
      epsilon    : A small constant to avoid division by zero.
    
    Returns:
      A torch scalar representing the regularization penalty.
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
        # Heavily penalizes alpha values near 0.
        return reg_lambda * torch.sum((alpha + epsilon) ** (-2))
    elif reg_type == "Exponential":
        # Softly penalizes small alpha values.
        return reg_lambda * torch.sum(torch.exp(-alpha))
    else:
        raise ValueError("Unknown reg_type: " + str(reg_type))


def run_experiment_with_diagnostics(dataset_size, m1, m, 
                                    dataset_type='linear_regression',
                                    estimator_fn=plugin_estimator,
                                    noise_scale=0.0, num_epochs=30, 
                                    reg_type=None, reg_lambda=0,
                                    learning_rate=0.001,
                                    batch_size=100,
                                    optimizer_type='sgd', seed=None, 
                                    early_stopping_patience=3,
                                    save_path='./results/gradient_descent_diagnostics/',
                                    verbose=False):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    os.makedirs(save_path + f'{optimizer_type}/', exist_ok=True)
    
    # Generate data and convert to torch tensors
    new_X, Y, A, indices = generate_data(m1, m, dataset_size, dataset_type= dataset_type,
                                       noise_scale=noise_scale, seed=seed)
    X = torch.tensor(new_X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    batch_size = min(100, dataset_size)

    # meaningful_indices = indices[:m1]
    meaningful_indices = np.where(indices < m1)[0]
    meaningful_indices.sort()
    
    # Estimate E[Y|X] once, outside the training loop
    E_Y_given_X = torch.tensor(estimator_fn(X.numpy(), Y.numpy())(X.numpy()), 
                              dtype=torch.float32)

    # Initialize alpha and tracking variables
    alpha = torch.nn.Parameter(torch.ones(X.size(1)))
    alpha_history = [alpha.detach().clone().numpy()]
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
    
    # Track full batch objectives for smoothness analysis
    def compute_full_batch_objective(alpha_val, reg_lambda=0, reg_type=None):
        # reg_max = torch.ones_like(alpha_val) * CLAMP_MAX
        S_alpha = X + torch.randn_like(X) * torch.sqrt(alpha_val)
        E_Y_given_S = estimate_conditional_expectation(X, S_alpha, E_Y_given_X, alpha_val)
        obj = torch.mean(E_Y_given_X**2) - torch.mean(E_Y_given_S**2)
        objective = obj + compute_reg_penalty(alpha_val, reg_type, reg_lambda)
        return objective
    
    patience = early_stopping_patience
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        epoch_gradients = []
        epoch_objectives = []
        
        for batch_X, batch_Y, batch_E_Y_given_X in dataloader:
            optimizer.zero_grad()
            # relu to ensure positivity
            # alpha_pos = F.softplus(alpha) + 1e-8
            
            # Generate noisy observations S
            # alpha_pos = F.relu(alpha) + 1e-8
            S_alpha = batch_X + torch.randn_like(batch_X) * torch.sqrt(alpha)
            # S_alpha = batch_X * alpha_pos + torch.randn_like(batch_X) * torch.sqrt(alpha_pos)
            
            # Estimate E[E[Y|X]|S] using kernel method
            E_Y_given_S = estimate_conditional_expectation(batch_X, S_alpha, batch_E_Y_given_X, alpha)
            # E_Y_given_S = estimate_conditional_expectation(batch_X, S_alpha, batch_E_Y_given_X, alpha_pos)
            
            # Compute objective
            # reg_max = torch.ones_like(alpha) * CLAMP_MAX
            obj = torch.mean(batch_E_Y_given_X**2) - torch.mean(E_Y_given_S**2)
            reg_penalty = compute_reg_penalty(alpha, reg_type, reg_lambda)
            objective = obj + reg_penalty
            if torch.isnan(objective) or torch.isinf(objective):
                print("Warning: Invalid objective value encountered")
                continue
            
            if objective.item() < best_objective:
                best_objective = objective.item()
                best_alpha = alpha.detach().clone()
            
            # Gradient step
            objective.backward()
            epoch_gradients.append(alpha.grad.detach().clone().numpy())
            epoch_objectives.append(objective.item())
            optimizer.step()
            with torch.no_grad():
                alpha.data.clamp_(min=CLAMP_MIN, max=CLAMP_MAX)
        
        # Store diagnostics
        alpha_history.append(alpha.detach().clone().numpy())
        gradient_history.append(np.mean(epoch_gradients, axis=0))
        objective_history.append(np.mean(epoch_objectives))
        
        if epoch % 5 == 0 and verbose:
            print(f"\tEpoch {epoch}")
            print(f"\tAverage objective: {np.mean(epoch_objectives):.4f}")
            print(f"\tAlpha values: {alpha_history[-1]}")
            print("\t----")
        
        # early stopping
        if epoch > 10 and np.mean(objective_history[-3:]) >= np.mean(objective_history[-6:-3]):
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0
        if early_stopping_counter >= patience and verbose:
            print("Early stopping")
            break
    if verbose:
        # Create diagnostic plots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Objective value over epochs
        plt.subplot(2, 2, 1)
        plt.plot(objective_history)
        plt.legend(['Total', 'Variance', f'L1 Reg (λ={reg_lambda})'])
        plt.title('Objective Value vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Objective Value')
        
        # Plot 2: Gradient magnitude over epochs
        plt.subplot(2, 2, 2)
        gradient_magnitudes = [np.linalg.norm(grad) for grad in gradient_history]
        plt.plot(gradient_magnitudes)
        plt.title('Gradient Magnitude vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Magnitude')
        
        # Plot 3: Alpha trajectories
        plt.subplot(2, 2, 3)
        alpha_history = np.array(alpha_history)
        for i in range(alpha_history.shape[1]):
            plt.plot(alpha_history[:, i], label=f'α_{i}')
        plt.title('Alpha Values vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Alpha Value')
        plt.legend()
        
        # Plot 4: Objective landscape around final point
        plt.subplot(2, 2, 4)
        final_alpha = alpha.detach().numpy()
        delta = np.linspace(-0.5, 0.5, 20)
        landscape = []
        for d in delta:
            perturbed_alpha = torch.tensor(final_alpha + d, dtype=torch.float32)
            obj = compute_full_batch_objective(perturbed_alpha).item()
            landscape.append(obj)
        plt.plot(delta, landscape)
        plt.title('Main Objective Landscape Around Final Point')
        plt.xlabel('Perturbation')
        plt.ylabel('Objective Value')
        
        plt.tight_layout()
        plt.savefig(save_path+f'{optimizer_type}/{dataset_size}_gradient_descent_diagnostics.png')
        plt.close()
    
    return {
        'best_objective': best_objective,
        'best_alpha': best_alpha.numpy(),
        'final_alpha': alpha.detach().numpy(),
        'objective_history': objective_history,
        'gradient_history': gradient_history,
        'alpha_history': alpha_history,
        'true_variable_index': meaningful_indices,
        # 'learned_variable_index': np.argmin(alpha.detach().numpy())
        'learned_variable_indices': np.argsort(alpha.detach().numpy())[:m1]
    }


def main():
    results_list = []
    optimizer_type='adam'
    dataset_sizes = [100, 500, 1000, 2000, 5000, 7000]#, 10000, 20000, 50000, 100000]# 200000, 500000]
    m1 = 1
    save_path = './results/gradient_descent_diagnostics/' + ('single_alpha/' if m1 == 1 else 'multiple_alpha/')
    for dataset_size in dataset_sizes:
        results = run_experiment_with_diagnostics(dataset_size=dataset_size, m1=m1, m=3, num_epochs=100, 
                                                  reg_lambda=0,
                                                  seed=10,
                                                  learning_rate=0.001,
                                                  batch_size=dataset_size,
                                                  optimizer_type=optimizer_type,
                                                  noise_scale=0.01,
                                                  verbose=True,
                                                  save_path=save_path,
                                                  dataset_type="quadratic_regression")
        print(f"Dataset size: {dataset_size}")
        print(f"Best objective: {results['best_objective']}")
        print(f"Best alpha: {results['best_alpha']}")
        print("===")
        results_list.append(results)
    
    # Save results in df
    df = pd.DataFrame(results_list)
    df.to_csv(f'./results/gradient_descent_diagnostics/{optimizer_type}/gradient_descent_diagnostics_results.csv', index=False)

    plot_variable_importance(results_list, dataset_sizes, optimizer_type)

if __name__ == '__main__':
    main()