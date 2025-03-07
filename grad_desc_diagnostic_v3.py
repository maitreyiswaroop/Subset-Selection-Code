# grad_desc_diagnostic_v3.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader
from estimators_v3 import PlugInConditionalVarianceEstimator
import torch.optim as optim
import pandas as pd
import os

def generate_data(m1, m, n_samples, task = "regression", noise_scale=0.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(0)
    indices = np.random.permutation(m)
    # X from uniform distribution U[0,1]
    # X = np.random.uniform(0, 1, (n_samples, m1))

    # X from normal distribution N(0,1)
    X = np.random.normal(0, 1, (n_samples, m1))
    
    # random linear map A
    A = np.random.randn(m1)
    
    AX = X.dot(A)
    # AX = (AX - np.mean(AX)) / np.std(AX)
    
    # Y based on AX
    if task == "regression":
        Y = AX + noise_scale * np.random.randn(n_samples)
    elif task == "classification":
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
    
    return new_X, Y, A, indices


def run_experiment_with_diagnostics(dataset_size, m1, m, noise_scale=0.0,
                                    num_epochs=30, lambda_val=0.02, 
                                    optimizer_type='sgd',
                                    seed=None, 
                                    save_path='./results/gradient_descent_diagnostics/',
                                    verbose=False,
                                    task = "regression"):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    os.makedirs(save_path + f'{optimizer_type}/', exist_ok=True)
    
    new_X, Y, A, indices = generate_data(m1, m, dataset_size, task=task, noise_scale=noise_scale, seed=seed)
    X = torch.tensor(new_X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    batch_size = min(100, dataset_size)

    meaningful_indices = indices[:m1]
    meaningful_indices.sort()
    if verbose:
        print(f"Meaningful indices: {meaningful_indices}")

    # alpha as param
    # alpha = torch.nn.Parameter(torch.tensor(np.random.uniform(0, 1, (X.size(1),)), dtype=torch.float32))
    alpha = torch.nn.Parameter(1.0 * torch.ones(X.size(1))) 
    alpha_history = [alpha.detach().clone().numpy()]
    gradient_history = []
    objective_history = []
    var_objective_history = []
    l1_objective_history = []
    
    # defining model
    model = LinearRegression().fit(X.numpy(), Y.numpy())
    E_Y_given_X = torch.tensor(model.predict(X.numpy()), dtype=torch.float32)
    # E_Y_given_X = (E_Y_given_X - torch.min(E_Y_given_X)) / (torch.max(E_Y_given_X) - torch.min(E_Y_given_X))
    if task == "classification":
        E_Y_given_X = (E_Y_given_X > 0.5).float()

    dataset = torch.utils.data.TensorDataset(X, Y, E_Y_given_X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if optimizer_type == 'adam':
        optimizer = optim.Adam([alpha], lr=0.001)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD([alpha], lr=0.001)
    
    best_objective = float('inf')
    best_alpha = None
    
    # Track full batch objectives for smoothness analysis
    def compute_full_batch_objective(alpha_val, lambda_val=lambda_val):
        # Z = torch.tensor(np.random.uniform(0, 1, (X.size(0), m)), dtype=torch.float32)
        S_alpha = X + torch.randn_like(X) * torch.sqrt(alpha_val)
        
        XtX = S_alpha.T @ S_alpha
        XtY = S_alpha.T @ Y
        beta = torch.linalg.solve(XtX, XtY)
        E_Y_given_S_alpha = S_alpha @ beta
        return torch.mean(E_Y_given_X**2) - torch.mean(E_Y_given_S_alpha**2) + lambda_val * torch.norm(alpha_val, p=1)
    
    patience = 5
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        epoch_gradients = []
        epoch_objectives = []
        epoch_var_objectives = []
        epoch_l1_objectives = []

        for batch_idx, (batch_X, batch_Y, batch_E_Y_given_X) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # S_alpha = batch_X + torch.randn_like(batch_X) * torch.sqrt(alpha)
            S_alpha = batch_X + torch.randn_like(batch_X) * alpha
            # deal with negative S_alpha
            # S_alpha = torch.clamp(S_alpha, min=0)
            # model2 = LinearRegression().fit(S_alpha.detach().numpy(), batch_Y.detach().numpy())
            # E_Y_given_S_alpha = torch.tensor(model2.predict(S_alpha.detach().numpy()), dtype=torch.float32)
            XtX = S_alpha.T @ S_alpha
            XtY = S_alpha.T @ batch_Y
            beta = torch.linalg.solve(XtX, XtY)
            E_Y_given_S_alpha = S_alpha @ beta
            
            obj1 = torch.mean(batch_E_Y_given_X**2) - torch.mean(E_Y_given_S_alpha**2)
            # dealing with negative obj1
            if obj1.item() < 0:
                print("\tNegative obj1")
                # raise ValueError("Negative obj1 encountered")
            penalty = torch.exp(-obj1) if obj1 < 0 else 0

            objective =  obj1 + lambda_val * torch.norm(alpha, p=1) + penalty
            
            if objective.item() < best_objective:
                best_objective = objective.item()
                best_alpha = alpha.detach().clone()
            
            objective.backward()
            epoch_gradients.append(alpha.grad.detach().clone().numpy())
            epoch_objectives.append(objective.item())
            epoch_var_objectives.append(obj1.item())
            epoch_l1_objectives.append(lambda_val * torch.norm(alpha, p=1).item())
            
            optimizer.step()
            # with torch.no_grad():
            #     alpha.data.clamp_(0, 1)  # In-place clamping
        
        # Store diagnostics
        alpha_history.append(alpha.detach().clone().numpy())
        gradient_history.append(np.mean(epoch_gradients, axis=0))
        objective_history.append(np.mean(epoch_objectives))
        var_objective_history.append(np.mean(epoch_var_objectives))
        l1_objective_history.append(np.mean(epoch_l1_objectives))
        
        # Compute gradient noise scale
        # gradient_variance = np.var(epoch_gradients, axis=0)
        # gradient_mean = np.mean(epoch_gradients, axis=0)
        
        if epoch % 5 == 0 and verbose:
            print(f"\tEpoch {epoch}")
            print(f"\tAverage objective: {np.mean(epoch_objectives):.4f}")
            # print(f"Gradient noise scale: {gradient_noise_scale:.4f}")
            print(f"\tAlpha values: {alpha_history[-1]}")
            print("\t----")
        
        # Early stopping
        if epoch > 5 and objective_history[-1] > objective_history[-2]:
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
        plt.plot(var_objective_history)
        plt.plot(l1_objective_history)
        plt.legend(['Total', 'Variance', f'L1 Reg (λ={lambda_val})'])
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
        plt.title('Objective Landscape Around Final Point')
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
        'learned_variable_index': torch.argsort(best_alpha, descending=True)[:m1].sort()[0].numpy(),
    }


def main():
    results_list = []
    optimizer_type='adam'
    dataset_sizes = [100, 500, 1000, 2000, 5000, 7000, 10000, 20000, 50000, 100000]# 200000, 500000]
    for dataset_size in dataset_sizes:
        results = run_experiment_with_diagnostics(dataset_size=dataset_size, m1=1, m=3, num_epochs=50, 
                                                  lambda_val=0, 
                                                  seed=0,
                                                  optimizer_type=optimizer_type,
                                                  verbose = True,
                                                  task = "regression")
        print(f"Dataset size: {dataset_size}")
        print(f"Best objective: {results['best_objective']}")
        print(f"Best alpha: {results['best_alpha']}")
        print("===")
        results_list.append(results)
    
    # Save results in df
    df = pd.DataFrame(results_list)
    df.to_csv(f'./results/gradient_descent_diagnostics/{optimizer_type}/gradient_descent_diagnostics_results.csv', index=False)

    meaningful_weights = []

    for result in results_list:
        meaningful_idx = result['true_variable_index'][0]  # assuming m1=1
        final_alpha = result['final_alpha']
        meaningful_weights.append(final_alpha[meaningful_idx])

    plt.figure(figsize=(8, 6))
    plt.plot(dataset_sizes, meaningful_weights, 'bo-')
    plt.xscale('log')
    plt.xlabel('Dataset Size')
    plt.ylabel('Weight of Meaningful Variable (α)')
    plt.grid(True)
    plt.title('Noise of Meaningful Variable vs Dataset Size')
    plt.savefig(f'./results/gradient_descent_diagnostics/{optimizer_type}/meaningful_weight_vs_size.png')
    plt.close()

if __name__ == '__main__':
    main()