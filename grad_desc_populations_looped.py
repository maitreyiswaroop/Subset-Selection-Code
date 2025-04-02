# grad_desc_populations_looped.py
import numpy as np
import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
from grad_desc_populations import generate_data_continuous, plugin_estimator, estimate_conditional_expectation
from grad_desc_populations import compute_reg_penalty, compute_full_batch_objective, CLAMP_MIN, CLAMP_MAX

def run_experiment_multi_population_with_seed(pop_configs, m1, m, 
                                              dataset_size=10000,
                                              seed=0, **experiment_kwargs):
    """
    Run the robust optimization experiment for a single seed
    
    Args:
        pop_configs: List of dictionaries for each population configuration
        m1: Number of meaningful features per population 
        m: Total number of features
        seed: Random seed
        **experiment_kwargs: Other arguments for the experiment
    """
    verbose = experiment_kwargs.pop('verbose', False)
    save_path = experiment_kwargs.pop('save_path', './results/multi_population_seed/')
    seed_save_path = os.path.join(save_path, f'seed_{seed}')
    os.makedirs(seed_save_path, exist_ok=True)
    
    # Run the experiment with the given seed
    from grad_desc_populations import run_experiment_multi_population
    results = run_experiment_multi_population(
        pop_configs=pop_configs,
        m1=m1,
        m=m,
        seed=seed,
        save_path=seed_save_path,
        verbose=verbose,
        **experiment_kwargs
    )
    
    # Add a perturbation analysis for this seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    final_alpha = torch.tensor(results['final_alpha'], device=device)
    
    # Perturbation analysis around final alpha
    delta_vals = np.linspace(-0.5, 0.5, 20)
    perturb_objectives = []
    
    # For each population, compute objectives at perturbed alphas
    pop_perturb_objectives = []
    for pop_idx, pop_config in enumerate(pop_configs):
        pop_id = pop_config['pop_id']
        pop_objectives = []
        
        # Load population data
        k_common = max(1, m1 // 2)
        common_meaningful_indices = np.arange(k_common)
        dataset_type = pop_config['dataset_type']
        new_X, Y, A, meaningful_indices = generate_data_continuous(
            pop_id=pop_id, m1=m1, m=m, 
            dataset_type=dataset_type, 
            noise_scale=experiment_kwargs.get('noise_scale', 0.01), 
            seed=seed, 
            common_meaningful_indices=common_meaningful_indices
        )
        X = torch.tensor(new_X, dtype=torch.float32).to(device)
        Y = torch.tensor(Y, dtype=torch.float32).to(device)
        estimator = plugin_estimator(X.cpu().numpy(), Y.cpu().numpy())
        E_Y_given_X = torch.tensor(estimator(X.cpu().numpy()), dtype=torch.float32).to(device)
        
        # For each perturbation, compute objective
        for delta in delta_vals:
            perturbed_alpha = final_alpha + delta
            perturbed_alpha = torch.clamp(perturbed_alpha, min=CLAMP_MIN, max=CLAMP_MAX)
            obj = compute_full_batch_objective(
                X, E_Y_given_X, perturbed_alpha, 
                experiment_kwargs.get('reg_lambda', 0), 
                experiment_kwargs.get('reg_type', None)
            ).item()
            pop_objectives.append(obj)
        
        pop_perturb_objectives.append(pop_objectives)
    
    # Compute robust (max) perturbation objective across populations
    robust_perturb_objectives = np.max(np.array(pop_perturb_objectives), axis=0)
    
    # Add perturbation analysis to results
    results['perturbation_deltas'] = delta_vals
    results['perturbation_objectives'] = robust_perturb_objectives
    results['pop_perturbation_objectives'] = pop_perturb_objectives
    
    # Calculate selected variables and accuracy
    final_alpha = results['final_alpha']
    selected_indices = np.argsort(final_alpha)[:2*m1]
    
    # Collect all meaningful indices across populations
    all_meaningful_indices = set()
    for pop_idx, pop_config in enumerate(pop_configs):
        pop_id = pop_config['pop_id']
        k_common = max(1, m1 // 2)
        common_meaningful_indices = np.arange(k_common)
        dataset_type = pop_config['dataset_type']
        _, _, _, meaningful_indices = generate_data_continuous(
            pop_id=pop_id, m1=m1, m=m, 
            dataset_type=dataset_type, 
            noise_scale=experiment_kwargs.get('noise_scale', 0.01), 
            seed=seed, 
            common_meaningful_indices=common_meaningful_indices
        )
        all_meaningful_indices.update(meaningful_indices)
    
    # Calculate selection accuracy
    true_positives = len(set(selected_indices) & all_meaningful_indices)
    selection_accuracy = true_positives / len(all_meaningful_indices) * 100
    
    results['selected_indices'] = selected_indices
    results['all_meaningful_indices'] = list(all_meaningful_indices)
    results['selection_accuracy'] = selection_accuracy
    
    # Calculate gradients for each epoch
    gradient_history = []
    alpha_history = np.array(results['alpha_history'])
    for i in range(1, len(alpha_history)):
        # Approximate gradient from alpha changes
        gradient = (alpha_history[i] - alpha_history[i-1]) / experiment_kwargs.get('learning_rate', 0.001)
        gradient_history.append(gradient)
    
    results['gradient_history'] = gradient_history
    
    return results

def run_multiple_seeds_experiment(pop_configs, m1, m, 
                                  dataset_size=10000,
                                  num_seeds=5, **experiment_kwargs):
    """
    Run experiments across multiple seeds and aggregate results
    
    Args:
        pop_configs: List of population configurations
        m1: Number of meaningful features per population
        m: Total number of features
        num_seeds: Number of different random seeds to use
        **experiment_kwargs: Additional arguments to pass to the experiment
    """
    save_path = experiment_kwargs.pop('save_path', './results/multi_population_multi_seeds/')
    os.makedirs(save_path, exist_ok=True)
    
    seeds = range(num_seeds)
    all_seed_results = []
    
    for seed in seeds:
        print(f"Running experiment with seed {seed}")
        seed_results = run_experiment_multi_population_with_seed(
            pop_configs=pop_configs,
            m1=m1,
            m=m,
            dataset_size=dataset_size,
            seed=seed,
            save_path=save_path,
            **experiment_kwargs
        )
        all_seed_results.append(seed_results)
    
    return all_seed_results

def plot_aggregated_results(all_seed_results, save_path, pop_configs, m1, m):
    """
    Create plots showing averaged results across seeds with error bands
    
    Args:
        all_seed_results: List of results from each seed
        save_path: Path to save the plots
        pop_configs: List of population configurations
        m1: Number of meaningful features per population
        m: Total number of features
    """
    os.makedirs(save_path, exist_ok=True)
    num_seeds = len(all_seed_results)
    
    # Plot 1: Average objective value over epochs with error bands
    plt.figure(figsize=(10, 6))
    objective_histories = [result['objective_history'] for result in all_seed_results]
    
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
    plt.title('Average Robust Objective vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Robust Objective Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'avg_objective_vs_epoch.png'))
    plt.close()
    
    # Plot 2: Alpha trajectories for each seed separately
    for seed_idx, result in enumerate(all_seed_results):
        plt.figure(figsize=(12, 8))
        alpha_history = np.array(result['alpha_history'])
        meaningful_indices = result['all_meaningful_indices']
        
        for i in range(alpha_history.shape[1]):
            label = f'Variable {i}'
            if i in meaningful_indices:
                label += ' (true)'
                plt.plot(range(len(alpha_history)), alpha_history[:, i], label=label, linewidth=2)
            else:
                plt.plot(range(len(alpha_history)), alpha_history[:, i], label=label, linestyle='--', alpha=0.5)
        
        plt.title(f'Alpha Trajectories for Seed {seed_idx}')
        plt.xlabel('Epoch')
        plt.ylabel('Alpha Value')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'seed_{seed_idx}_alpha_trajectories.png'))
        plt.close()
    
    # Plot 3: Best seed alpha trajectories
    best_seed_idx = np.argmax([result['selection_accuracy'] for result in all_seed_results])
    best_result = all_seed_results[best_seed_idx]
    best_alpha_history = np.array(best_result['alpha_history'])
    meaningful_indices = best_result['all_meaningful_indices']
    
    plt.figure(figsize=(12, 8))
    for i in range(best_alpha_history.shape[1]):
        label = f'Variable {i}'
        if i in meaningful_indices:
            label += ' (true)'
            plt.plot(range(len(best_alpha_history)), best_alpha_history[:, i], label=label, linewidth=2)
        else:
            plt.plot(range(len(best_alpha_history)), best_alpha_history[:, i], label=label, linestyle='--', alpha=0.5)
    
    plt.title(f'Alpha Trajectories for Best Seed (Seed {best_seed_idx})')
    plt.xlabel('Epoch')
    plt.ylabel('Alpha Value')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'best_seed_alpha_trajectories.png'))
    plt.close()
    
    # Plot 4: Frequency of variable selection across seeds
    plt.figure(figsize=(10, 6))
    selection_counts = np.zeros(m)
    for result in all_seed_results:
        for idx in result['selected_indices']:
            selection_counts[idx] += 1
    
    selection_frequency = selection_counts / len(all_seed_results) * 100
    
    # Get all unique meaningful indices from all seeds
    all_meaningful_indices = set()
    for result in all_seed_results:
        all_meaningful_indices.update(result['all_meaningful_indices'])
    
    bar_colors = ['blue' if i in all_meaningful_indices else 'gray' for i in range(m)]
    
    plt.bar(range(m), selection_frequency, color=bar_colors)
    plt.axhline(y=50, color='r', linestyle='--', label='50% Selection Rate')
    plt.xticks(range(m))
    plt.xlabel('Variable Index')
    plt.ylabel('Selection Frequency (%)')
    plt.title('Variable Selection Frequency Across Seeds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'variable_selection_frequency.png'))
    plt.close()
    
    # Plot 4: Average perturbation analysis
    plt.figure(figsize=(10, 6))
    all_deltas = all_seed_results[0]['perturbation_deltas']
    perturbation_objectives = np.array([result['perturbation_objectives'] for result in all_seed_results])
    mean_perturb_obj = np.mean(perturbation_objectives, axis=0)
    std_perturb_obj = np.std(perturbation_objectives, axis=0)
    
    plt.plot(all_deltas, mean_perturb_obj, label='Mean Objective')
    plt.fill_between(
        all_deltas, 
        mean_perturb_obj - std_perturb_obj, 
        mean_perturb_obj + std_perturb_obj, 
        alpha=0.3
    )
    
    # Add population-specific perturbation curves
    for pop_idx, pop_config in enumerate(pop_configs):
        pop_id = pop_config['pop_id']
        pop_perturb_objs = np.array([result['pop_perturbation_objectives'][pop_idx] for result in all_seed_results])
        mean_pop_perturb_obj = np.mean(pop_perturb_objs, axis=0)
        plt.plot(all_deltas, mean_pop_perturb_obj, linestyle='--', 
                 label=f'Population {pop_id} ({pop_config["dataset_type"]})')
    
    plt.title('Objective Landscape Around Final Alpha')
    plt.xlabel('Perturbation')
    plt.ylabel('Objective Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'avg_perturbation_analysis.png'))
    plt.close()
    
    # Plot 5: Average gradient magnitude over epochs
    plt.figure(figsize=(10, 6))
    gradient_histories = [result['gradient_history'] for result in all_seed_results]
    
    # Pad shorter sequences with NaN
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
    
    plt.title('Average Gradient Magnitude vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Magnitude')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, 'avg_gradient_magnitude.png'))
    plt.close()
    
    # Save summary statistics
    summary_data = {
        'num_seeds': num_seeds,
        'avg_final_objective': np.mean([result['final_objective'] for result in all_seed_results]),
        'std_final_objective': np.std([result['final_objective'] for result in all_seed_results]),
        'avg_selection_accuracy': np.mean([result['selection_accuracy'] for result in all_seed_results]),
        'std_selection_accuracy': np.std([result['selection_accuracy'] for result in all_seed_results]),
    }
    
    # Compute frequency of each variable being selected
    selection_counts = np.zeros(m)
    for result in all_seed_results:
        for idx in result['selected_indices']:
            selection_counts[idx] += 1
    
    # Add selection frequencies to summary
    for i in range(m):
        selection_frequency = selection_counts[i] / num_seeds * 100
        summary_data[f'var_{i}_selection_pct'] = selection_frequency
        is_meaningful = i in all_meaningful_indices
        summary_data[f'var_{i}_is_meaningful'] = is_meaningful
    
    # Add average alpha values
    final_alphas = np.array([result['final_alpha'] for result in all_seed_results])
    mean_final_alpha = np.mean(final_alphas, axis=0)
    std_final_alpha = np.std(final_alphas, axis=0)
    
    for i in range(m):
        summary_data[f'var_{i}_avg_alpha'] = mean_final_alpha[i]
        summary_data[f'var_{i}_std_alpha'] = std_final_alpha[i]
    
    # Save summary as CSV
    pd.DataFrame([summary_data]).to_csv(os.path.join(save_path, 'experiment_summary.csv'), index=False)
    
    # Return composite metric: true variable selection rate
    true_var_selection_rate = np.mean([result['selection_accuracy'] for result in all_seed_results])
    return true_var_selection_rate

def main():
    pop_configs = [
        {'pop_id': 0, 'dataset_type': "linear_regression"},
        {'pop_id': 1, 'dataset_type': "sinusoidal_regression"}
    ]
    m1 = 4  # Number of meaningful features per population
    m = 10  # Total number of features
    num_seeds = 5
    
    save_path = './results/multi_population_looped/'
    os.makedirs(save_path, exist_ok=True)
    
    # Run experiments across multiple seeds
    all_seed_results = run_multiple_seeds_experiment(
        pop_configs=pop_configs,
        m1=m1,
        m=m,
        dataset_size = 10000,
        num_seeds=num_seeds,
        noise_scale=0.01,
        num_epochs=100,
        reg_type="Neg_L1",
        reg_lambda=0.01,
        learning_rate=0.01,
        batch_size=256,
        optimizer_type='adam',
        early_stopping_patience=10,
        verbose=True,
        save_path=save_path
    )
    
    # Generate aggregated plots and statistics
    selection_accuracy = plot_aggregated_results(
        all_seed_results=all_seed_results,
        save_path=save_path,
        pop_configs=pop_configs,
        m1=m1,
        m=m
    )
    
    print(f"Average selection accuracy across {num_seeds} seeds: {selection_accuracy:.2f}%")
    
    # Find best performing seed and its selected variables
    best_seed_idx = np.argmax([result['selection_accuracy'] for result in all_seed_results])
    best_result = all_seed_results[best_seed_idx]
    
    print(f"Best seed: {best_seed_idx}")
    print(f"Best selection accuracy: {best_result['selection_accuracy']:.2f}%")
    print(f"Selected variables: {best_result['selected_indices']}")
    print(f"All meaningful variables: {best_result['all_meaningful_indices']}")
    print(f"Final alpha values: {best_result['final_alpha']}")

if __name__ == '__main__':
    main()