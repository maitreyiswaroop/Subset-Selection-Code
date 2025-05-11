# gd_looped.py
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from copy import deepcopy
from gd_diagnostic import run_experiment_with_diagnostics


def run_multiple_seeds_experiment(dataset_sizes, num_seeds=5, **experiment_kwargs):
    """
    Run experiments across multiple seeds and aggregate results
    
    Args:
        dataset_sizes: List of dataset sizes to test
        num_seeds: Number of different random seeds to use
        **experiment_kwargs: Additional arguments to pass to run_experiment_with_diagnostics
    """
    all_results = []
    seeds = range(num_seeds)
    
    for dataset_size in dataset_sizes:
        dataset_results = []
        for seed in seeds:
            print(f"Running experiment with dataset size {dataset_size}, seed {seed}")
            results = run_experiment_with_diagnostics(
                dataset_size=dataset_size,
                seed=seed,
                **experiment_kwargs
            )
            dataset_results.append(results)
        
        # For each seed's result, get the alpha value for its meaningful variable
        meaningful_alphas = []
        for result in dataset_results:
            true_indices = result['true_variable_index']
            # meaningful_idx = result;'f'
            meaningful_alphas.append(result['final_alpha'][true_indices])
        
        meaningful_alphas = np.vstack(meaningful_alphas)
        
        # Aggregate results for this dataset size
        aggregated_result = {
            'dataset_size': dataset_size,
            'objective_histories': [r['objective_history'] for r in dataset_results],
            'gradient_histories': [r['gradient_history'] for r in dataset_results],
            'alpha_histories': [r['alpha_history'] for r in dataset_results],
            'best_objectives': [r['best_objective'] for r in dataset_results],
            'meaningful_alphas': meaningful_alphas,  # Store just the meaningful alpha values
            'all_final_alphas': [r['final_alpha'] for r in dataset_results],
            'true_variable_index': [r['true_variable_index'] for r in dataset_results]
        }
        all_results.append(aggregated_result)
    
    return all_results

def plot_aggregated_results(all_results, save_path, optimizer_type):
    """
    Create plots showing averaged results across seeds with error bands
    """
    os.makedirs(save_path + f'{optimizer_type}/', exist_ok=True)
    
    for result in all_results:
        dataset_size = result['dataset_size']
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Average objective value over epochs with error bands
        plt.subplot(1, 3, 1)
        objective_histories = result['objective_histories']
        
        # Find the maximum length among all histories
        max_length = max(len(history) for history in objective_histories)
        
        # Pad shorter sequences with NaN
        padded_histories = []
        for history in objective_histories:
            padded_history = np.pad(
                history, 
                (0, max_length - len(history)), 
                mode='constant', 
                constant_values=np.nan
            )
            padded_histories.append(padded_history)
        
        objective_histories = np.array(padded_histories)
        mean_objective = np.nanmean(objective_histories, axis=0)
        std_objective = np.nanstd(objective_histories, axis=0)
        epochs = range(len(mean_objective))
        
        plt.plot(epochs, mean_objective, label='Mean Objective')
        plt.fill_between(
            epochs, 
            mean_objective - std_objective, 
            mean_objective + std_objective, 
            alpha=0.3
        )
        
        # Similar padding for gradient and alpha histories
        gradient_histories = result['gradient_histories']
        alpha_histories = result['alpha_histories']
        
        # Plot 2: Average gradient magnitude over epochs
        plt.subplot(1, 3, 2)
        padded_gradients = []
        max_grad_length = max(len(history) for history in gradient_histories)
        
        for history in gradient_histories:
            padded_history = np.pad(
                history, 
                ((0, max_grad_length - len(history)), (0, 0)), 
                mode='constant', 
                constant_values=np.nan
            )
            padded_gradients.append(padded_history)
        
        gradient_histories = np.array(padded_gradients)
        gradient_magnitudes = np.linalg.norm(gradient_histories, axis=2)
        mean_gradient = np.nanmean(gradient_magnitudes, axis=0)
        std_gradient = np.nanstd(gradient_magnitudes, axis=0)
        
        plt.plot(range(len(mean_gradient)), mean_gradient)
        plt.fill_between(
            range(len(mean_gradient)), 
            mean_gradient - std_gradient,
            mean_gradient + std_gradient, 
            alpha=0.3
        )
        
        # Plot 3: Average alpha trajectories for all variables
        plt.subplot(1, 3, 3)
        padded_alphas = []
        max_alpha_length = max(len(history) for history in alpha_histories)

        for history in alpha_histories:
            padded_history = np.pad(
                history, 
                ((0, max_alpha_length - len(history)), (0, 0)), 
                mode='constant', 
                constant_values=np.nan
            )
            padded_alphas.append(padded_history)

        alpha_histories = np.array(padded_alphas)
        mean_alpha = np.nanmean(alpha_histories, axis=0)
        std_alpha = np.nanstd(alpha_histories, axis=0)

        for i in range(mean_alpha.shape[1]):
            true_indices = result['true_variable_index'][0]
            label = f'Variable {i+1}' + (' (true)' if i in true_indices else '')
            plt.plot(range(len(mean_alpha)), mean_alpha[:, i], label=label)
            plt.fill_between(
                range(len(mean_alpha)), 
                mean_alpha[:, i] - std_alpha[:, i],
                mean_alpha[:, i] + std_alpha[:, i], 
                alpha=0.3
            )
        plt.title('Average Alpha Trajectories for All Variables')
        plt.xlabel('Epoch')
        plt.ylabel('Alpha Value')
        plt.legend()

        # # Plot 4: Distribution of final alpha values
        # plt.subplot(2, 2, 4)
        # meaningful_alphas = result['meaningful_alphas']
        # plt.boxplot([meaningful_alphas], tick_labels=['Meaningful Variable'])
        # plt.title('Distribution of Final Alpha Values\nfor Meaningful Variable')
        # plt.ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(save_path + f'{optimizer_type}/{dataset_size}_averaged_diagnostics.png')
        plt.close()

def plot_meaningful_weights_summary(all_results, dataset_sizes, save_path, optimizer_type):
    """
    Plot the evolution of meaningful weights across dataset sizes.
    For multiple meaningful variables (m1>1), we flatten them or take means.
    """
    meaningful_weights = []
    meaningful_stds = []
    for result in all_results:
        # shape = (num_seeds, m1)
        meaningful_alphas = result['meaningful_alphas']
        # Flatten across all seeds and all meaningful vars:
        flattened = meaningful_alphas.flatten()
        meaningful_weights.append(np.mean(flattened))
        meaningful_stds.append(np.std(flattened))
    plt.figure(figsize=(8, 6))
    meaningful_weights = np.array(meaningful_weights)
    meaningful_stds = np.array(meaningful_stds)
    plt.errorbar(dataset_sizes, meaningful_weights, yerr=meaningful_stds,
                 fmt='bo-', capsize=5, label='Mean ± Std')
    plt.xscale('log')
    plt.xlabel('Dataset Size')
    plt.ylabel('Noise of Meaningful Variables (α)')
    plt.grid(True)
    plt.title('Average Noise of Meaningful Variables vs Dataset Size')
    plt.legend()
    plt.savefig(f'{save_path}{optimizer_type}/avg_meaningful_weight_vs_size.png')
    plt.close()

def save_results_summary(all_results, save_path, optimizer_type):
    """
    Save aggregated results to CSV
    """
    summary_data = []
    for result in all_results:
        summary_data.append({
            'dataset_size': result['dataset_size'],
            'mean_best_objective': np.mean(result['best_objectives']),
            'std_best_objective': np.std(result['best_objectives']),
            'mean_meaningful_alpha': np.mean(result['meaningful_alphas']),
            'std_meaningful_alpha': np.std(result['meaningful_alphas'])
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(f'{save_path}{optimizer_type}/averaged_results_summary.csv', index=False)

def main():
    # Configuration
    optimizer_type = 'adam'
    dataset_sizes = [100, 500, 1000, 2000, 5000, 7000]
    dataset_type="quadratic_regression"
    save_path = './results/gradient_descent_diagnostics/multiple_alpha/across_seeds/'+dataset_type+'/'
    num_seeds = 5
    
    # Run experiments
    all_results = run_multiple_seeds_experiment(
        dataset_sizes=dataset_sizes,
        num_seeds=num_seeds,
        m1=2,
        m=6,
        num_epochs=100,
        lambda_val=0,
        learning_rate=0.001,
        optimizer_type=optimizer_type,
        noise_scale=0.01,
        verbose=True,
        dataset_type=dataset_type
    )
    
    # Generate plots and save results
    plot_aggregated_results(all_results, save_path, optimizer_type)
    plot_meaningful_weights_summary(all_results, dataset_sizes, save_path, optimizer_type)
    save_results_summary(all_results, save_path, optimizer_type)

if __name__ == '__main__':
    main()