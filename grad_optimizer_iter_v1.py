import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from estimators_v3 import PlugInConditionalVarianceEstimator
import pandas as pd
import matplotlib.pyplot as plt


def generate_data(m1, m, n_samples):
    # X from uniform distribution U[0,1]
    X = np.random.uniform(0, 1, (n_samples, m1))
    
    # random linear map A
    A = np.random.randn(m1)
    
    AX = X.dot(A)
    AX = (AX - np.mean(AX)) / np.std(AX)
    
    # Y based on AX
    Y = (AX > 0.5).astype(int)
    
    # new X with m dimensions
    new_X = np.zeros((n_samples, m))
    
    # copy m1 dimensions from original X
    new_X[:, :m1] = X
    
    # fill remaining dimensions with random values from U[0,1]
    new_X[:, m1:] = np.random.uniform(0, 1, (n_samples, m - m1))
    # shuffle the indices of the columns
    indices = np.random.permutation(m)
    new_X = new_X[:, indices]
    
    return new_X, Y, A, indices


def run_experiment(dataset_size, m1, m, num_epochs=30, lambda_val=0.02, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    new_X, Y, A, indices = generate_data(m1, m, dataset_size)
    X = torch.tensor(new_X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    batch_size = min(32, dataset_size)

    meaningful_indices = indices[:m1]
    meaningful_indices.sort()

    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    alpha = torch.nn.Parameter(torch.tensor(np.random.uniform(0, 1, (X.size(1),)), dtype=torch.float32))
    
    model = LinearRegression().fit(X.numpy(), Y.numpy())
    E_Y_given_X = torch.tensor(model.predict(X.numpy()), dtype=torch.float32)
    E_Y_given_X = (E_Y_given_X - torch.min(E_Y_given_X)) / (torch.max(E_Y_given_X) - torch.min(E_Y_given_X))
    E_Y_given_X = (E_Y_given_X > 0.5).float()

    optimizer = optim.SGD([alpha], lr=0.01)
    
    best_objective = float('inf')
    best_alpha = None
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        for batch_X, batch_Y in dataloader:
            batch_Z = torch.tensor(np.random.uniform(0, 1, (batch_X.size(0), m)), dtype=torch.float32)
            S_alpha = alpha * batch_X + (1 - alpha) * batch_Z
            
            batch_indices = range(batch_X.size(0))
            E_Y_given_X_batch = E_Y_given_X[batch_indices]
            
            plugin_cv_estimator = PlugInConditionalVarianceEstimator()
            plugin_cv_estimator.fit(S_alpha.detach().numpy(), E_Y_given_X_batch.detach().numpy())
            V_X_given_S = torch.tensor(plugin_cv_estimator.predict_variance(S_alpha.detach().numpy()), dtype=torch.float32)
            
            objective = torch.mean(V_X_given_S) + lambda_val * torch.norm(alpha, p=1)
            
            if objective.item() < best_objective:
                best_objective = objective.item()
                best_alpha = alpha.detach().clone()
            
            objective.backward()
            optimizer.step()
    
    # Analyze results
    top_indices = torch.argsort(best_alpha, descending=True)[:m1].sort()[0]
    
    # Check if true features are discovered (alpha > 0.5)
    true_alpha_in_best_alpha = sum([1 for idx in meaningful_indices if best_alpha[idx] > 0.5])
    
    # Check if only true features are selected and no others
    exclusive_alpha_in_best_alpha = (
        sum([1 for idx in meaningful_indices if best_alpha[idx] > 0.5]) == m1 and
        sum([1 for idx in range(m) if idx not in meaningful_indices and best_alpha[idx] > 0.5]) == 0
    )
    
    return {
        'best_objective': best_objective,
        'best_alpha': best_alpha.numpy(),
        'true_alpha_in_best_alpha': true_alpha_in_best_alpha/m1,
        'exclusive_alpha_in_best_alpha': float(exclusive_alpha_in_best_alpha),
        'sparsity': np.sum(best_alpha.numpy() < 1e-5),
    }

def main():
    m1 = 1
    m = 3
    dataset_sizes = [100, 500, 1000, 5000, 10000, 20000]
    num_seeds = 5
    lambda_val = 0.02
    
    results = []
    
    for dataset_size in dataset_sizes:
        print(f"Running experiments for dataset size {dataset_size}")
        seeds = list(range(num_seeds))
        dataset_results = []
        
        for seed in seeds:
            result = run_experiment(dataset_size, m1, m, lambda_val=lambda_val, seed=seed)
            dataset_results.append(result)
        
        # Calculate aggregated metrics
        aggregated_result = {
            'dataset_size': dataset_size,
            'm1': m1,
            'm': m,
            'lambda': lambda_val,
            'avg_objective': np.mean([r['best_objective'] for r in dataset_results]),
            'true_alpha_success_rate': np.mean([r['true_alpha_in_best_alpha'] for r in dataset_results]),
            'exclusive_alpha_success_rate': np.mean([r['exclusive_alpha_in_best_alpha'] for r in dataset_results]),
            'avg_sparsity': np.mean([r['sparsity'] for r in dataset_results]),
            'seeds': seeds
        }
        results.append(aggregated_result)
        
        # Save intermediate results
        df = pd.DataFrame(results)
        df.to_csv('grad_desc_feature_discovery_results_temp.csv', index=False)
    
    # Save final results
    df = pd.DataFrame(results)
    df.to_csv('grad_desc_feature_discovery_results.csv', index=False)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(df['dataset_size'], df['true_alpha_success_rate'], 'b-', label='True Feature Discovery Rate')
    plt.plot(df['dataset_size'], df['exclusive_alpha_success_rate'], 'r--', label='Exclusive Correct Discovery Rate')
    plt.xscale('log')
    plt.xlabel('Dataset Size')
    plt.ylabel('Success Rate')
    plt.title('Feature Discovery Performance vs Dataset Size')
    plt.legend()
    plt.grid(True)
    plt.savefig('grad_desc_feature_discovery_performance.png')
    plt.close()
    
    return df

if __name__ == '__main__':
    main()