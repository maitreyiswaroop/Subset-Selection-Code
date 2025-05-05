# gd_diagnostic_bike.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
import os
import torch.nn.functional as F

# Import the bike sharing dataset loader
from data_bike import load_bike_sharing_dataset, prepare_data_for_variable_selection

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

def estimate_conditional_expectation(X_batch, S_batch, E_Y_given_X_batch, alpha, feature_group_sizes):
    """
    Estimate E[E[Y|X]|S] via a kernel method.
    For grouped features, alpha is group-level (one value per original variable),
    so we repeat it to match each feature in the group.
    
    Parameters:
    -----------
    feature_group_sizes : list
        A list containing the size of each feature group.
    """
    # Expand X and S for pairwise comparison:
    X_expanded = X_batch.unsqueeze(1)  # (batch_size, 1, n_features)
    S_expanded = S_batch.unsqueeze(0)  # (1, batch_size, n_features)
    
    # Expand alpha to match feature dimensions in a differentiable manner
    alpha_expanded = torch.repeat_interleave(alpha, torch.tensor(feature_group_sizes, device=alpha.device))
    alpha_expanded = torch.clamp(alpha_expanded, min=CLAMP_MIN)
    alpha_expanded = alpha_expanded.unsqueeze(0).unsqueeze(0)  # shape (1, 1, n_features)
    
    # Check for NaNs in S_batch
    if torch.isnan(S_batch).any():
        print("\tWarning: NaN values found in S_batch")
        return torch.tensor(float('nan'))

    # Compute the weighted squared distances
    squared_distances = ((X_expanded - S_expanded) * torch.sqrt(1/(alpha_expanded + 1e-2)))**2
    
    # Handle potential numerical issues
    if torch.isnan(squared_distances).any():
        print("\tWarning: NaN values found in squared distances")
        if (alpha_expanded < 0).any():
            print("\tReason: Negative alpha values found")
        if torch.any(torch.abs(X_expanded - S_expanded) > 1e6):
            print("\tWarning: Very large values found in X_expanded - S_expanded")
    
    if torch.any(squared_distances > 1e6):
        print("\tWarning: Extremely large squared distances detected.")
    
    # Sum over features and apply the Gaussian kernel with clamping for stability
    kernel_matrix = torch.exp(-0.5 * torch.clamp(torch.sum(squared_distances, dim=2), max=100))
    
    # Normalize to obtain weights
    weights = kernel_matrix / (kernel_matrix.sum(dim=0, keepdim=True) + 1e-10)
    
    if torch.isnan(weights).any():
        print("\tWarning: NaN values found in weights")
        # checking kernel_matrix for NaN or inf values
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

def compute_full_batch_objective(X, E_Y_given_X, alpha_val, feature_group_sizes, 
                                 reg_lambda=0, reg_type=None, device='cpu', chunk_size=1000):
    """
    Compute full batch objective with chunked processing to save memory.
    
    Parameters:
    -----------
    feature_group_sizes : list
        A list containing the size of each feature group.
    """
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
        
        # Expand alpha to match feature dimensions in a differentiable manner
        alpha_expanded = torch.repeat_interleave(alpha_val, torch.tensor(feature_group_sizes, device=alpha_val.device))
        
        # Generate noisy observations for this chunk
        S_alpha = X_chunk + torch.randn_like(X_chunk) * torch.sqrt(alpha_expanded)
        
        # Compute conditional expectation for this chunk
        E_Y_given_S = estimate_conditional_expectation(
            X_chunk, S_alpha, E_Y_given_X_chunk, 
            alpha_val, feature_group_sizes
        )
        
        # Accumulate objective
        chunk_obj = torch.mean(E_Y_given_X_chunk**2) - torch.mean(E_Y_given_S**2)
        total_obj += chunk_obj * (end_idx - start_idx) / batch_size
    
    # Add regularization penalty
    objective = total_obj + compute_reg_penalty(alpha_val, reg_type, reg_lambda)
    return objective

def run_bike_sharing_experiment(dataset_size=None, 
                               n_noise_features=4,
                               estimator_fn=plugin_estimator,
                               num_epochs=30, 
                               reg_type=None, 
                               reg_lambda=0,
                               learning_rate=0.001,
                               batch_size=100,
                               optimizer_type='sgd', 
                               seed=None, 
                               early_stopping_patience=3,
                               early_stopping=True,
                               save_path=None,
                               verbose=False,
                               device='cpu'):
    """
    Run the variable selection experiment on the Bike Sharing dataset.
    
    Parameters:
    -----------
    dataset_size : int or None
        Number of samples to use. If None, use all samples.
    n_noise_features : int
        Number of noise categorical features to add.
    """
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
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    # Load the bike sharing dataset
    bike_data = load_bike_sharing_dataset(
        n_samples=dataset_size,
        n_noise_features=n_noise_features,
        random_state=seed,
        file_path="./datasets/bike_sharing_dataset/hour.csv"  # Update this to match your actual file path
    )
    
    # Prepare data for variable selection
    X, Y, feature_group_sizes, meaningful_indices = prepare_data_for_variable_selection(bike_data)
    
    # Print dataset information
    print(f"Dataset loaded:")
    print(f"  - Number of samples: {X.shape[0]}")
    print(f"  - Number of features: {X.shape[1]}")
    print(f"  - Number of feature groups: {len(feature_group_sizes)}")
    print(f"  - Meaningful feature groups: {meaningful_indices}")
    
    batch_size = min(batch_size, X.shape[0])
    
    # Estimate E[Y|X] using the plugin estimator
    E_Y_given_X = torch.tensor(estimator_fn(X.numpy(), Y.numpy())(X.numpy()), 
                             dtype=torch.float32).to(device)
    X = X.to(device)
    Y = Y.to(device)
    
    # Initialize alpha with one parameter per feature group
    # alpha = torch.nn.Parameter(torch.ones(len(feature_group_sizes), device=device))
    alpha = torch.nn.Parameter(torch.full((len(feature_group_sizes),), 10.0, device=device))
    
    alpha_history = [alpha.detach().cpu().numpy()]
    gradient_history = []
    objective_history = []
    
    dataset = torch.utils.data.TensorDataset(X, Y, E_Y_given_X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam([alpha], lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD([alpha], lr=learning_rate, momentum=0.9)
    
    best_objective = float('inf')
    best_alpha = None
    
    patience = early_stopping_patience
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        epoch_gradients = []
        epoch_objectives = []
        
        for batch_X, batch_Y, batch_E_Y_given_X in dataloader:
            optimizer.zero_grad()
            
            # Expand alpha in a differentiable manner
            alpha_expanded = torch.repeat_interleave(alpha, torch.tensor(feature_group_sizes, device=alpha.device))
            
            # Generate noisy observations
            S_alpha = batch_X + torch.randn_like(batch_X, device=device) * torch.sqrt(alpha_expanded)
            
            # Estimate conditional expectation
            E_Y_given_S = estimate_conditional_expectation(
                batch_X, S_alpha, batch_E_Y_given_X, alpha, feature_group_sizes
            )
            
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
        gradient_history.append(np.mean(epoch_gradients, axis=0))
        objective_history.append(np.mean(epoch_objectives))
        
        if epoch % 5 == 0 and verbose:
            print(f"\tEpoch {epoch}")
            print(f"\tAverage objective: {np.mean(epoch_objectives):.4f}")
            print(f"\tAlpha values: {alpha_history[-1]}")
            print("\t----")
        
        if early_stopping:
            if epoch > 50 and np.mean(objective_history[-3:]) >= np.mean(objective_history[-6:-3]):
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0
            
            if early_stopping_counter >= patience and verbose:
                print("Early stopping")
                break
    
    # TODO: remove post debugging
    print(f"Alpha: {alpha}")
    alpha_expanded = torch.repeat_interleave(alpha, torch.tensor(feature_group_sizes, device=alpha.device))
    print(f"Expanded alpha values: {alpha_expanded}")
    # print alpha by groups
    for i, group in enumerate(bike_data['feature_groups']):
        print(f"Group {i}: {group}, alpha: {[alpha_expanded[j].item() for j in group]}")

    # Create visualization for results
    if verbose and save_path:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(objective_history)
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
        
        # Get feature names for the legend
        feature_names = []
        for i, group in enumerate(bike_data['feature_groups']):
            # Get the first feature name in the group as representative
            name = bike_data['all_feature_names'][group[0]]
            # Check if this is a meaningful feature
            is_meaningful = i in bike_data['meaningful_groups']
            status = " (meaningful)" if is_meaningful else " (noise)"
            feature_names.append(f"{name}{status}")
        
        for i in range(alpha_hist_arr.shape[1]):
            plt.plot(alpha_hist_arr[:, i], label=feature_names[i])
        
        plt.title('Alpha Values vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Alpha Value')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        final_alpha = alpha.detach().cpu().numpy()
        
        # Create sorted bar chart of alpha values
        sorted_indices = np.argsort(final_alpha)
        sorted_alphas = final_alpha[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        bars = plt.barh(range(len(sorted_alphas)), sorted_alphas)
        plt.yticks(range(len(sorted_alphas)), sorted_names)
        plt.title('Final Alpha Values (Smaller = More Important)')
        plt.xlabel('Alpha Value')
        
        # Color bars based on whether they're meaningful or noise
        for i, idx in enumerate(sorted_indices):
            is_meaningful = idx in bike_data['meaningful_groups']
            bars[i].set_color('blue' if is_meaningful else 'red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'bike_sharing_experiment_results.png'))
        plt.close()
    
    # Determine feature selection accuracy
    final_alpha = alpha.detach().cpu().numpy()
    learned_variable_indices = np.argsort(final_alpha)[:len(meaningful_indices)]
    
    # Calculate precision, recall, F1
    true_positives = len(set(learned_variable_indices) & set(meaningful_indices))
    precision = true_positives / len(learned_variable_indices) if learned_variable_indices.size > 0 else 0
    recall = true_positives / len(meaningful_indices) if meaningful_indices.size > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    results = {
        'final_objective': objective_history[-1],
        'final_alpha': alpha.detach().cpu().numpy(),
        'objective_history': objective_history,
        'gradient_history': gradient_history,
        'alpha_history': alpha_history,
        'true_variable_indices': meaningful_indices,
        'learned_variable_indices': learned_variable_indices,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'feature_names': feature_names
    }
    
    if verbose:
        print("\nFeature Selection Results:")
        print(f"True meaningful features: {meaningful_indices}")
        print(f"Selected features: {learned_variable_indices}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
    
    return results

def main():
    results = run_bike_sharing_experiment(
        dataset_size=8000,
        n_noise_features=20,
        num_epochs=100,
        reg_lambda=0.001,
        reg_type="Neg_L1",
        seed=42,
        learning_rate=0.001,
        batch_size=100,
        optimizer_type='sgd',
        verbose=True,
        early_stopping=False,
        save_path='./results/bike_sharing_experiment'
    )
    
    # Save results to CSV
    if not os.path.exists('./results/bike_sharing_experiment'):
        os.makedirs('./results/bike_sharing_experiment')
    
    # Save alpha values with feature names
    alpha_df = pd.DataFrame({
        'feature': results['feature_names'],
        'alpha': results['final_alpha'],
        'is_meaningful': [i in results['true_variable_indices'] for i in range(len(results['final_alpha']))]
    })
    alpha_df.sort_values('alpha', inplace=True)
    alpha_df.to_csv('./results/bike_sharing_experiment/feature_importance.csv', index=False)
    
    # Save summary metrics
    summary_df = pd.DataFrame({
        'metric': ['precision', 'recall', 'f1_score', 'final_objective'],
        'value': [
            results['precision'],
            results['recall'],
            results['f1_score'],
            results['final_objective']
        ]
    })
    summary_df.to_csv('./results/bike_sharing_experiment/summary_metrics.csv', index=False)
    
    print("Results saved to ./results/bike_sharing_experiment/")

if __name__ == '__main__':
    main()
