import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score # Not used, but kept import
import json
import os
from copy import deepcopy # Not used, but kept import
from data import generate_data_continuous, generate_data_continuous_with_corr
import argparse
from sklearn.base import clone # For stability selection cloning

# Helper function to get data for a single population config
def _get_or_generate_pop_data(pop_config, m1, m, dataset_size, noise_scale, corr_strength, seed):
    """Checks for existing X, Y in pop_config, else generates data."""
    if 'X' in pop_config and 'Y' in pop_config and pop_config['X'] is not None and pop_config['Y'] is not None:
        # Use provided data
        X = pop_config['X']
        Y = pop_config['Y']
        # Ensure meaningful_indices exists, even if None initially
        meaningful_indices = pop_config.get('meaningful_indices', None)
        # If meaningful_indices are missing, we can't evaluate properly later, maybe raise error or return None?
        # For now, assume if X,Y are provided, meaningful_indices should also be (or handled downstream)
        if meaningful_indices is None:
             print(f"Warning: Provided data for pop_id {pop_config.get('pop_id', 'N/A')} is missing 'meaningful_indices'.")
             # Attempt to regenerate just to get indices? Or handle in evaluation?
             # Let's assume it should be present if X, Y are.

        print(f"Using provided data for pop_id {pop_config.get('pop_id', 'N/A')}")
        return X, Y, meaningful_indices
    else:
        # Generate data
        pop_id = pop_config.get('pop_id', 0) # Default pop_id if missing
        dataset_type = pop_config.get('dataset_type', 'linear_regression') # Default type if missing
        current_seed = seed + pop_id if seed is not None else None

        # Define common meaningful indices (needed for generation)
        k_common = max(1, m1 // 2)
        common_meaningful_indices = np.arange(k_common)

        print(f"Generating data for pop_id {pop_id} (type: {dataset_type}, seed: {current_seed})...")
        if corr_strength > 0:
            X, Y, _, meaningful_indices = generate_data_continuous_with_corr(
                pop_id=pop_id, m1=m1, m=m,
                dataset_type=dataset_type,
                dataset_size=dataset_size,
                noise_scale=noise_scale,
                seed=current_seed,
                common_meaningful_indices=common_meaningful_indices,
                corr_strength=corr_strength
            )
        else:
            X, Y, _, meaningful_indices = generate_data_continuous(
                pop_id=pop_id, m1=m1, m=m,
                dataset_type=dataset_type,
                dataset_size=dataset_size,
                noise_scale=noise_scale,
                seed=current_seed,
                common_meaningful_indices=common_meaningful_indices
            )
        return X, Y, meaningful_indices


def pooled_lasso(pop_configs, m1, m, budget, dataset_size=10000, noise_scale=0.01,
                 corr_strength=0.5,
                 alpha=0.01, seed=None): # Removed data param
    """
    Baseline 1: Pool all populations and use Lasso regression to select variables.
    Uses data from pop_configs if available, otherwise generates it.
    """
    all_X = []
    all_Y = []
    all_meaningful_indices = []

    # Iterate through configs, get/generate data for each
    for i, pop_config in enumerate(pop_configs):
        # Use a consistent seed for generation if needed across calls within a run
        gen_seed = seed + i if seed is not None else None
        X, Y, meaningful_indices = _get_or_generate_pop_data(
            pop_config, m1, m, dataset_size, noise_scale, corr_strength, gen_seed
        )
        if X is None: # Handle potential generation failure if added in helper
             print(f"Skipping population {pop_config.get('pop_id', i)} due to data error.")
             continue
        all_X.append(X)
        all_Y.append(Y)
        if meaningful_indices is not None:
             all_meaningful_indices.append(meaningful_indices)
        else:
             # If indices are still None, we have an issue for evaluation
             print(f"Warning: Meaningful indices missing for pop {pop_config.get('pop_id', i)}.")


    if not all_X: # Check if any data was successfully processed
         print("Error: No data available for Pooled Lasso.")
         # Return empty/error structure
         return {'selected_indices': [], 'true_indices': [], 'method': 'pooled_lasso',
                 'recall': 0, 'precision': 0, 'f1_score': 0, 'coef_values': []}

    # Pool all data
    pooled_X = np.vstack(all_X)
    pooled_Y = np.concatenate(all_Y)

    # Set seed for Lasso reproducibility
    lasso_seed = seed

    # Train Lasso model
    lasso = Lasso(alpha=alpha, random_state=lasso_seed)
    lasso.fit(pooled_X, pooled_Y)

    # Select top features based on absolute coefficient values
    coef_abs = np.abs(lasso.coef_)
    actual_budget = min(budget, m)
    selected_indices = np.argsort(-coef_abs)[:actual_budget]

    # Evaluate the selection
    true_indices = set()
    for indices in all_meaningful_indices:
        if indices is not None: true_indices.update(indices)

    selected_set = set(selected_indices)
    intersection_size = len(selected_set & true_indices)
    precision = intersection_size / len(selected_set) if selected_set else 0
    recall = intersection_size / len(true_indices) if true_indices else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'selected_indices': selected_indices.tolist(),
        'true_indices': list(true_indices), # Store the combined true indices used
        'method': 'pooled_lasso',
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'coef_values': coef_abs[selected_indices].tolist()
    }

def population_wise_regression(pop_configs, m1, m, budget, dataset_size=10000, noise_scale=0.01,
                               corr_strength=0.5,
                               model_type='linear', voting='frequency', seed=None): # Removed data param
    """
    Baseline 2: Perform regression on each population and use a voting mechanism.
    Uses data from pop_configs if available, otherwise generates it.
    """
    all_meaningful_indices = []
    pop_data_list = [] # To store tuples (X, Y)

    # Iterate through configs, get/generate data for each
    for i, pop_config in enumerate(pop_configs):
        gen_seed = seed + i if seed is not None else None
        X, Y, meaningful_indices = _get_or_generate_pop_data(
            pop_config, m1, m, dataset_size, noise_scale, corr_strength, gen_seed
        )
        if X is None:
             print(f"Skipping population {pop_config.get('pop_id', i)} due to data error.")
             continue
        pop_data_list.append((X, Y)) # Store X, Y tuple
        if meaningful_indices is not None:
             all_meaningful_indices.append(meaningful_indices)
        else:
             print(f"Warning: Meaningful indices missing for pop {pop_config.get('pop_id', i)}.")

    if not pop_data_list:
         print("Error: No data available for Population-wise Regression.")
         return {'selected_indices': [], 'true_indices': [], 'method': f'population_wise_{model_type}_{voting}',
                 'recall': 0, 'precision': 0, 'f1_score': 0, 'votes': [], 'rank_scores': [], 'importance_scores': []}

    # Set seed for model reproducibility
    model_seed = seed

    # Initialize structures for voting
    votes = np.zeros(m)
    rank_scores = np.zeros(m)
    importance_scores = np.zeros(m)

    for X, Y in pop_data_list: # Iterate through collected data
        # Choose model based on model_type
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=model_seed)
        elif model_type == 'lasso':
            model = Lasso(alpha=0.01, random_state=model_seed)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X, Y)

        # Extract feature importance
        if hasattr(model, 'feature_importances_'): importance = model.feature_importances_
        elif hasattr(model, 'coef_'): importance = np.abs(model.coef_)
        else: importance = np.zeros(m)

        if np.sum(importance) < 1e-9: # Check for near-zero importance
             print(f"Warning: Near-zero importance scores for a population. Skipping voting updates.")
             continue

        # Select top features for this population
        num_features_to_consider = min(budget, m)
        sorted_indices = np.argsort(-importance)
        top_indices = sorted_indices[:num_features_to_consider]

        # Update voting
        votes[top_indices] += 1

        # Update rank scores (Borda count)
        for i, idx in enumerate(sorted_indices):
            rank_scores[idx] += m - i

        # Update importance scores (normalized)
        # Avoid division by zero if sum is zero (already checked above)
        importance_scores += importance / np.sum(importance)

    # Select final variables based on voting mechanism
    actual_budget = min(budget, m)
    if voting == 'frequency':
        selected_indices = np.argsort(-votes)[:actual_budget]
    elif voting == 'rank_sum' or voting == 'borda':
        selected_indices = np.argsort(-rank_scores)[:actual_budget]
    elif voting == 'weighted':
        selected_indices = np.argsort(-importance_scores)[:actual_budget]
    else:
        raise ValueError(f"Unknown voting mechanism: {voting}")

    # Evaluate the selection
    true_indices = set()
    for indices in all_meaningful_indices:
        if indices is not None: true_indices.update(indices)

    selected_set = set(selected_indices)
    intersection_size = len(selected_set & true_indices)
    precision = intersection_size / len(selected_set) if selected_set else 0
    recall = intersection_size / len(true_indices) if true_indices else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'selected_indices': selected_indices.tolist(),
        'true_indices': list(true_indices),
        'method': f'population_wise_{model_type}_{voting}',
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'votes': votes[selected_indices].tolist(),
        'rank_scores': rank_scores[selected_indices].tolist(),
        'importance_scores': importance_scores[selected_indices].tolist()
    }

def mutual_information_selection(pop_configs, m1, m, budget, dataset_size=10000, noise_scale=0.01,
                                 corr_strength=0.5,
                                 pooling='union', seed=None): # Removed data param
    """
    Baseline 3: Use mutual information to select variables.
    Uses data from pop_configs if available, otherwise generates it.
    """
    all_meaningful_indices = []
    pop_data_list = [] # To store tuples (X, Y)

    # Iterate through configs, get/generate data for each
    for i, pop_config in enumerate(pop_configs):
        gen_seed = seed + i if seed is not None else None
        X, Y, meaningful_indices = _get_or_generate_pop_data(
            pop_config, m1, m, dataset_size, noise_scale, corr_strength, gen_seed
        )
        if X is None:
             print(f"Skipping population {pop_config.get('pop_id', i)} due to data error.")
             continue
        pop_data_list.append((X, Y))
        if meaningful_indices is not None:
             all_meaningful_indices.append(meaningful_indices)
        else:
             print(f"Warning: Meaningful indices missing for pop {pop_config.get('pop_id', i)}.")

    if not pop_data_list:
         print("Error: No data available for Mutual Information Selection.")
         return {'selected_indices': [], 'true_indices': [], 'method': f'mutual_information_{pooling}',
                 'recall': 0, 'precision': 0, 'f1_score': 0, 'mi_scores': []}

    # Set seed for MI reproducibility
    mi_seed = seed

    # Compute mutual information for each population
    pop_selected = []
    mi_scores_all = np.zeros((len(pop_data_list), m))

    for i, (X, Y) in enumerate(pop_data_list):
        # Compute mutual information
        mi = mutual_info_regression(X, Y, random_state=mi_seed)
        mi_scores_all[i] = mi

        # Select top features for this population
        num_features_to_consider = min(budget, m)
        top_indices = np.argsort(-mi)[:num_features_to_consider]
        pop_selected.append(set(top_indices))

    # Aggregate MI scores
    mi_scores_agg = np.mean(mi_scores_all, axis=0)

    # Combine results based on pooling method
    actual_budget = min(budget, m)
    selected_indices = np.array([], dtype=int) # Initialize

    if pooling == 'union':
        selected_set = set().union(*pop_selected)
    elif pooling == 'intersection':
        selected_set = pop_selected[0].intersection(*pop_selected[1:]) if pop_selected else set()
    elif pooling == 'weighted':
        selected_indices = np.argsort(-mi_scores_agg)[:actual_budget]
        selected_set = set(selected_indices)
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")

    if pooling in ['union', 'intersection']:
        if len(selected_set) > actual_budget:
            selected_list = list(selected_set)
            scores_subset = mi_scores_agg[selected_list]
            top_indices_subset = np.argsort(-scores_subset)[:actual_budget]
            selected_indices = np.array(selected_list)[top_indices_subset]
        elif len(selected_set) < actual_budget:
            current_indices = list(selected_set)
            remaining_needed = actual_budget - len(current_indices)
            potential_indices = [idx for idx in range(m) if idx not in selected_set]
            potential_scores = mi_scores_agg[potential_indices]
            top_remaining_indices = np.argsort(-potential_scores)[:remaining_needed]
            selected_indices = np.concatenate((current_indices, np.array(potential_indices)[top_remaining_indices]))
        else:
            selected_indices = np.array(list(selected_set))
        selected_set = set(selected_indices) # Update set based on final indices

    # Evaluate the selection
    true_indices = set()
    for indices in all_meaningful_indices:
        if indices is not None: true_indices.update(indices)

    intersection_size = len(selected_set & true_indices)
    precision = intersection_size / len(selected_set) if selected_set else 0
    recall = intersection_size / len(true_indices) if true_indices else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'selected_indices': selected_indices.tolist(),
        'true_indices': list(true_indices),
        'method': f'mutual_information_{pooling}',
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'mi_scores': mi_scores_agg[selected_indices].tolist()
    }

def stability_selection(pop_configs, m1, m, budget, dataset_size=10000, noise_scale=0.01,
                          corr_strength=0.5,
                          n_bootstraps=50, sample_fraction=0.75, alpha_range=None,
                          threshold=0.7, seed=None): # Removed data param
    """
    Baseline 4: Stability selection with Lasso.
    Uses data from pop_configs if available, otherwise generates it.
    """
    # Prepare data (generate if needed) - only need pooled data here
    all_X_list = []
    all_Y_list = []
    all_meaningful_indices = []
    for i, pop_config in enumerate(pop_configs):
        gen_seed = seed + i if seed is not None else None
        X, Y, meaningful_indices = _get_or_generate_pop_data(
            pop_config, m1, m, dataset_size, noise_scale, corr_strength, gen_seed
        )
        if X is None:
             print(f"Skipping population {pop_config.get('pop_id', i)} due to data error.")
             continue
        all_X_list.append(X)
        all_Y_list.append(Y)
        if meaningful_indices is not None:
             all_meaningful_indices.append(meaningful_indices)
        else:
             print(f"Warning: Meaningful indices missing for pop {pop_config.get('pop_id', i)}.")


    if not all_X_list:
         print("Error: No data available for Stability Selection.")
         return {'selected_indices': [], 'true_indices': [], 'method': 'stability_selection',
                 'recall': 0, 'precision': 0, 'f1_score': 0, 'selection_probabilities': []}

    if alpha_range is None:
        alpha_range = [0.001, 0.005, 0.01, 0.05, 0.1]

    # Pool all data
    pooled_X = np.vstack(all_X_list)
    pooled_Y = np.concatenate(all_Y_list)

    n_samples = pooled_X.shape[0]
    if n_samples == 0:
         print("Error: Pooled dataset is empty for Stability Selection.")
         return {'selected_indices': [], 'true_indices': [], 'method': 'stability_selection',
                 'recall': 0, 'precision': 0, 'f1_score': 0, 'selection_probabilities': []}

    bootstrap_size = int(n_samples * sample_fraction)
    if bootstrap_size < 1: bootstrap_size = 1 # Ensure at least 1 sample

    # Set seed for bootstrapping reproducibility
    bootstrap_seed = seed
    if bootstrap_seed is not None:
        np.random.seed(bootstrap_seed)

    # Initialize selection probability matrix
    selection_probability = np.zeros((len(alpha_range), m))

    # Run stability selection
    for i, alpha in enumerate(alpha_range):
        feature_counts = np.zeros(m)

        for b in range(n_bootstraps):
            # Create bootstrap sample
            if n_samples == 0: continue # Skip if no samples
            bootstrap_indices = np.random.choice(n_samples, bootstrap_size, replace=True)
            # Handle potential empty bootstrap sample if n_samples is very small
            if len(bootstrap_indices) == 0: continue
            X_bootstrap = pooled_X[bootstrap_indices]
            Y_bootstrap = pooled_Y[bootstrap_indices]

            # Ensure bootstrap sample is not empty
            if X_bootstrap.shape[0] == 0: continue

            # Fit Lasso
            lasso_bootstrap_seed = bootstrap_seed + b if bootstrap_seed is not None else None
            lasso = Lasso(alpha=alpha, random_state=lasso_bootstrap_seed)
            try:
                 lasso.fit(X_bootstrap, Y_bootstrap)
                 selected = np.where(np.abs(lasso.coef_) > 1e-6)[0]
                 feature_counts[selected] += 1
            except Exception as e:
                 print(f"Warning: Lasso fit failed for bootstrap {b}, alpha {alpha}. Error: {e}")

        # Calculate selection probability
        selection_probability[i] = feature_counts / n_bootstraps

    # Aggregate over all alpha values
    max_probability = np.max(selection_probability, axis=0)

    # Select features that exceed the threshold
    stable_features_indices = np.where(max_probability >= threshold)[0]
    stable_features_probs = max_probability[stable_features_indices]

    # Sort stable features by probability (descending)
    sorted_stable_indices = stable_features_indices[np.argsort(-stable_features_probs)]

    # Select based on budget
    actual_budget = min(budget, m)
    selected_indices = np.array([], dtype=int) # Initialize
    if len(sorted_stable_indices) >= actual_budget:
        selected_indices = sorted_stable_indices[:actual_budget]
    else:
        current_selection = list(sorted_stable_indices)
        remaining_needed = actual_budget - len(current_selection)
        if remaining_needed > 0:
            potential_indices = [idx for idx in range(m) if idx not in current_selection]
            if potential_indices: # Check if there are any indices left
                 potential_probs = max_probability[potential_indices]
                 top_remaining_indices = np.argsort(-potential_probs)[:remaining_needed]
                 selected_indices = np.concatenate((current_selection, np.array(potential_indices)[top_remaining_indices]))
            else: # No more indices left to add
                 selected_indices = np.array(current_selection)
        else: # Exactly budget number were stable
             selected_indices = sorted_stable_indices

    selected_set = set(selected_indices)

    # Evaluate the selection
    true_indices = set()
    for indices in all_meaningful_indices:
        if indices is not None: true_indices.update(indices)

    intersection_size = len(selected_set & true_indices)
    precision = intersection_size / len(selected_set) if selected_set else 0
    recall = intersection_size / len(true_indices) if true_indices else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'selected_indices': selected_indices.tolist(),
        'true_indices': list(true_indices),
        'method': 'stability_selection',
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'selection_probabilities': max_probability[selected_indices].tolist()
    }


def group_lasso_selection(pop_configs, m1, m, budget, dataset_size=10000, noise_scale=0.01,
                            corr_strength=0.5,
                            alpha=0.01, seed=None): # Removed data param
    """
    Baseline 5: Group Lasso approach (approximated).
    Uses data from pop_configs if available, otherwise generates it.
    """
    all_meaningful_indices = []
    pop_data_list = [] # To store tuples (X, Y)

    # Iterate through configs, get/generate data for each
    for i, pop_config in enumerate(pop_configs):
        gen_seed = seed + i if seed is not None else None
        X, Y, meaningful_indices = _get_or_generate_pop_data(
            pop_config, m1, m, dataset_size, noise_scale, corr_strength, gen_seed
        )
        if X is None:
             print(f"Skipping population {pop_config.get('pop_id', i)} due to data error.")
             continue
        pop_data_list.append((X, Y))
        if meaningful_indices is not None:
             all_meaningful_indices.append(meaningful_indices)
        else:
             print(f"Warning: Meaningful indices missing for pop {pop_config.get('pop_id', i)}.")


    if not pop_data_list:
         print("Error: No data available for Group Lasso.")
         return {'selected_indices': [], 'true_indices': [], 'method': 'group_lasso',
                 'recall': 0, 'precision': 0, 'f1_score': 0, 'group_norms': []}

    # Set seed for Lasso reproducibility
    lasso_seed = seed

    # Run Lasso on each population and get coefficients
    all_coefs = np.zeros((len(pop_data_list), m))

    for i, (X, Y) in enumerate(pop_data_list):
        lasso = Lasso(alpha=alpha, random_state=lasso_seed)
        lasso.fit(X, Y)
        all_coefs[i] = lasso.coef_

    # Compute group norms
    group_norms = np.sqrt(np.sum(all_coefs**2, axis=0))

    # Select top features based on group norms
    actual_budget = min(budget, m)
    selected_indices = np.argsort(-group_norms)[:actual_budget]

    # Evaluate the selection
    true_indices = set()
    for indices in all_meaningful_indices:
        if indices is not None: true_indices.update(indices)

    selected_set = set(selected_indices)
    intersection_size = len(selected_set & true_indices)
    precision = intersection_size / len(selected_set) if selected_set else 0
    recall = intersection_size / len(true_indices) if true_indices else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'selected_indices': selected_indices.tolist(),
        'true_indices': list(true_indices),
        'method': 'group_lasso',
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'group_norms': group_norms[selected_indices].tolist()
    }

def condorcet_voting(pop_configs, m1, m, budget, dataset_size=10000, noise_scale=0.01,
                        corr_strength=0.5,
                        model_type='rf', seed=None): # Removed data param
    """
    Baseline 6: Condorcet voting method for variable selection.
    Uses data from pop_configs if available, otherwise generates it.
    """
    all_meaningful_indices = []
    pop_data_list = [] # To store tuples (X, Y)

    # Iterate through configs, get/generate data for each
    for i, pop_config in enumerate(pop_configs):
        gen_seed = seed + i if seed is not None else None
        X, Y, meaningful_indices = _get_or_generate_pop_data(
            pop_config, m1, m, dataset_size, noise_scale, corr_strength, gen_seed
        )
        if X is None:
             print(f"Skipping population {pop_config.get('pop_id', i)} due to data error.")
             continue
        pop_data_list.append((X, Y))
        if meaningful_indices is not None:
             all_meaningful_indices.append(meaningful_indices)
        else:
             print(f"Warning: Meaningful indices missing for pop {pop_config.get('pop_id', i)}.")

    if not pop_data_list:
         print("Error: No data available for Condorcet Voting.")
         return {'selected_indices': [], 'true_indices': [], 'method': f'condorcet_{model_type}',
                 'recall': 0, 'precision': 0, 'f1_score': 0, 'copeland_scores': []}

    # Set seed for model reproducibility
    model_seed = seed

    # Get feature rankings for each population
    pop_rankings = []

    for X, Y in pop_data_list:
        # Choose model based on model_type
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=model_seed)
        elif model_type == 'lasso':
            model = Lasso(alpha=0.01, random_state=model_seed)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X, Y)

        # Extract feature importance
        if hasattr(model, 'feature_importances_'): importance = model.feature_importances_
        elif hasattr(model, 'coef_'): importance = np.abs(model.coef_)
        else: importance = np.zeros(m)

        # Get ranking
        ranking = np.argsort(-importance)
        pop_rankings.append(ranking)

    # Initialize Condorcet matrix
    condorcet_matrix = np.zeros((m, m))

    # Fill Condorcet matrix
    for ranking in pop_rankings:
        for i in range(m):
            feature_i = ranking[i]
            for j in range(i+1, m):
                feature_j = ranking[j]
                condorcet_matrix[feature_i, feature_j] += 1

    # Compute Copeland score
    wins = np.sum(condorcet_matrix > condorcet_matrix.T, axis=1)
    losses = np.sum(condorcet_matrix < condorcet_matrix.T, axis=1)
    copeland_scores = wins - losses

    # Select top features based on Copeland scores
    actual_budget = min(budget, m)
    selected_indices = np.argsort(-copeland_scores)[:actual_budget]

    # Evaluate the selection
    true_indices = set()
    for indices in all_meaningful_indices:
        if indices is not None: true_indices.update(indices)

    selected_set = set(selected_indices)
    intersection_size = len(selected_set & true_indices)
    precision = intersection_size / len(selected_set) if selected_set else 0
    recall = intersection_size / len(true_indices) if true_indices else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'selected_indices': selected_indices.tolist(),
        'true_indices': list(true_indices),
        'method': f'condorcet_{model_type}',
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
        'copeland_scores': copeland_scores[selected_indices].tolist()
    }

# --- Utility for JSON Serialization ---
def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        if np.isnan(obj): return None
        if np.isinf(obj): return None
        return float(obj)
    elif isinstance(obj, (np.bool_)):
         return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, set):
        return list(obj)
    return obj

# --- Main Runner Function ---
def run_all_baselines(pop_configs, m1, m, budget, dataset_size=10000, noise_scale=0.01, corr_strength=0.0,
                      seed=42, save_path='./results/baselines/'):
    """
    Run all baseline methods and return/save the results.
    Baseline methods handle their own data loading/generation based on pop_configs.
    """
    os.makedirs(save_path, exist_ok=True)

    # Get meaningful indices from pop_configs if data is provided,
    # otherwise they will be generated inside baselines.
    # We need them for the summary function later.
    # Let's extract/generate them once here for consistency in summary.
    print("--- Preparing Data / Extracting Meaningful Indices ---")
    pop_configs_with_indices = []
    all_true_indices_combined = set()
    for i, pop_config in enumerate(pop_configs):
        gen_seed = seed + i if seed is not None else None
        # Call helper to get data OR just indices if data exists
        # Modify helper slightly? Or just call it? Let's call it.
        # This might generate data if not present, which is inefficient if
        # baselines also generate, but ensures we have indices for summary.
        # A better way would be to ensure baselines return the indices they used.
        # Let's assume baselines return 'true_indices' correctly.
        # We just need the config structure for the summary function.
        temp_config = pop_config.copy() # Avoid modifying original
        if 'meaningful_indices' not in temp_config:
             # Need to generate temporarily to get indices if not provided
             _, _, temp_indices = _get_or_generate_pop_data(
                 temp_config, m1, m, dataset_size, noise_scale, corr_strength, gen_seed
             )
             temp_config['meaningful_indices'] = temp_indices.tolist() if temp_indices is not None else []
        elif temp_config['meaningful_indices'] is not None:
             # Ensure it's a list for JSON
             temp_config['meaningful_indices'] = list(temp_config['meaningful_indices'])
        else:
             temp_config['meaningful_indices'] = []

        pop_configs_with_indices.append(temp_config)
        all_true_indices_combined.update(temp_config['meaningful_indices'])

    print(f"--- Total unique true indices across populations: {len(all_true_indices_combined)} ---")


    ######## Run Baselines ########
    results = {}
    # Store pop_configs with indices for later summary use
    results['pop_configs_with_indices'] = pop_configs_with_indices

    # Define actual budget, ensuring it's not > m
    actual_budget = min(budget, m)
    print(f"Using budget: {actual_budget} (Requested: {budget}, Max Features: {m})")

    # --- Pass pop_configs directly to each baseline ---
    # Each baseline will now use _get_or_generate_pop_data internally

    print("\n--- Running Baseline: Pooled Lasso ---")
    results['pooled_lasso'] = pooled_lasso(
        pop_configs, m1, m, actual_budget, dataset_size, noise_scale, corr_strength,
        alpha=0.01, seed=seed # Pass seed for internal generation/reproducibility
    )

    model_types = ['linear', 'rf', 'lasso']
    voting_methods = ['frequency', 'rank_sum', 'borda', 'weighted']
    print("\n--- Running Baseline: Population-wise Regression ---")
    for model_type in model_types:
        for voting in voting_methods:
            key = f'population_wise_{model_type}_{voting}'
            print(f"  Running {key}...")
            results[key] = population_wise_regression(
                pop_configs, m1, m, actual_budget, dataset_size, noise_scale, corr_strength,
                model_type=model_type, voting=voting, seed=seed
            )

    pooling_methods = ['union', 'intersection', 'weighted']
    print("\n--- Running Baseline: Mutual Information ---")
    for pooling in pooling_methods:
        key = f'mutual_information_{pooling}'
        print(f"  Running {key}...")
        results[key] = mutual_information_selection(
            pop_configs, m1, m, actual_budget, dataset_size, noise_scale, corr_strength,
            pooling=pooling, seed=seed
        )

    print("\n--- Running Baseline: Stability Selection ---")
    results['stability_selection'] = stability_selection(
        pop_configs, m1, m, actual_budget, dataset_size, noise_scale, corr_strength,
        n_bootstraps=50, sample_fraction=0.75, threshold=0.7, seed=seed
    )

    print("\n--- Running Baseline: Group Lasso (Approx) ---")
    results['group_lasso'] = group_lasso_selection(
        pop_configs, m1, m, actual_budget, dataset_size, noise_scale, corr_strength,
        alpha=0.01, seed=seed
    )

    print("\n--- Running Baseline: Condorcet Voting ---")
    for model_type in model_types:
        key = f'condorcet_{model_type}'
        print(f"  Running {key}...")
        results[key] = condorcet_voting(
            pop_configs, m1, m, actual_budget, dataset_size, noise_scale, corr_strength,
            model_type=model_type, seed=seed
        )

    # Convert results before saving
    print("\n--- Converting results to serializable format ---")
    serializable_results = convert_to_serializable(results)

    # Save results
    results_filepath = os.path.join(save_path, 'baseline_results.json')
    print(f"--- Saving baseline results to {results_filepath} ---")
    try:
        with open(results_filepath, 'w') as f:
            json.dump(serializable_results, f, indent=4)
    except Exception as e:
        print(f"Error saving results JSON: {e}")

    return results


def summarize_results(results, save_path='./results/baselines/'):
    """
    Summarize the results of all baseline methods.
    """
    # Use the pop_configs stored within the results dict
    pop_configs_with_indices = results.get('pop_configs_with_indices')
    if not pop_configs_with_indices:
         print("Warning: 'pop_configs_with_indices' not found in results. Cannot calculate population-wise summary stats.")

    summary = []

    for method_name, method_results in results.items():
        if method_name == 'pop_configs_with_indices': continue
        if not isinstance(method_results, dict) or 'precision' not in method_results:
             print(f"Skipping summary for '{method_name}': Invalid results format.")
             continue

        summary_row = {
            'method': method_name,
            'precision': method_results.get('precision', np.nan),
            'recall': method_results.get('recall', np.nan),
            'f1_score': method_results.get('f1_score', np.nan),
            'min_selected_percentage': np.nan,
            'max_selected_percentage': np.nan,
            'mean_selected_percentage': np.nan,
            'median_selected_percentage': np.nan,
        }

        # Calculate population-wise stats only if pop_configs available
        if pop_configs_with_indices:
             selected_indices_set = set(method_results.get('selected_indices', []))
             pop_percentages = []
             # Use the meaningful indices stored in pop_configs_with_indices
             for pop_config in pop_configs_with_indices:
                 true_indices_set = set(pop_config.get('meaningful_indices', []))
                 if not true_indices_set:
                      pop_percentages.append(np.nan)
                      continue
                 intersection = selected_indices_set.intersection(true_indices_set)
                 pop_percentages.append(len(intersection) / len(true_indices_set) * 100)

             valid_pop_percentages = [p for p in pop_percentages if not np.isnan(p)]
             if valid_pop_percentages:
                  summary_row['min_selected_percentage'] = np.min(valid_pop_percentages)
                  summary_row['max_selected_percentage'] = np.max(valid_pop_percentages)
                  summary_row['mean_selected_percentage'] = np.mean(valid_pop_percentages)
                  summary_row['median_selected_percentage'] = np.median(valid_pop_percentages)

        summary.append(summary_row)

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('f1_score', ascending=False)

    summary_filepath = os.path.join(save_path, 'baseline_summary.csv')
    print(f"--- Saving baseline summary to {summary_filepath} ---")
    summary_df.to_csv(summary_filepath, index=False)

    return summary_df

def compare_with_gradient_descent(baseline_results, gd_results_path, save_path='./results/comparison/'):
    """
    Compare baseline methods with gradient descent approach loaded from a file.
    """
    os.makedirs(save_path, exist_ok=True)

    # --- Load Grad Desc Results ---
    try:
        with open(gd_results_path, 'r') as f:
            gd_results = json.load(f)
    except FileNotFoundError:
        print(f"Gradient descent results file not found: {gd_results_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding gradient descent results file: {gd_results_path}")
        return None
    except Exception as e:
        print(f"Error loading gradient descent results file: {e}")
        return None

    # --- Calculate metrics for grad desc ---
    gd_selected = set(gd_results.get('selected_indices', []))
    # Try to get true indices from GD results, might be stored differently
    gd_true = set()
    # Use pop_configs_with_indices if available in GD results, otherwise fall back
    if 'pop_configs_with_indices' in gd_results:
         for pop_cfg in gd_results['pop_configs_with_indices']:
              gd_true.update(pop_cfg.get('meaningful_indices', []))
    elif 'meaningful_indices' in gd_results: # Check list of lists
         for indices in gd_results['meaningful_indices']:
             gd_true.update(indices)
    elif 'true_variable_index' in gd_results: # Check old key
         gd_true.update(gd_results['true_variable_index'])


    if not gd_true:
         print("Warning: Could not determine true indices from gradient descent results file.")

    gd_precision = len(gd_selected & gd_true) / len(gd_selected) if gd_selected else 0
    gd_recall = len(gd_selected & gd_true) / len(gd_true) if gd_true else 0
    gd_f1 = 2 * (gd_precision * gd_recall) / (gd_precision + gd_recall) if (gd_precision + gd_recall) > 0 else 0

    # --- Create comparison dataframe ---
    comparison = []

    # Add grad desc results
    comparison.append({
        'method': 'gradient_descent', # Assuming one GD result file
        'precision': gd_precision,
        'recall': gd_recall,
        'f1_score': gd_f1
    })

    # Add baseline results
    for method_name, method_results in baseline_results.items():
        if method_name == 'pop_configs_with_indices': continue # Skip config entry
        if isinstance(method_results, dict): # Ensure it's a result dict
            comparison.append({
                'method': method_name,
                'precision': method_results.get('precision', np.nan),
                'recall': method_results.get('recall', np.nan),
                'f1_score': method_results.get('f1_score', np.nan)
            })

    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('f1_score', ascending=False)

    # Save comparison
    comparison_filepath = os.path.join(save_path, 'method_comparison.csv')
    print(f"--- Saving comparison results to {comparison_filepath} ---")
    comparison_df.to_csv(comparison_filepath, index=False)

    return comparison_df


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run baseline variable selection methods')
    parser.add_argument('--m1', type=int, default=4, help='Number of meaningful features per population')
    parser.add_argument('--m', type=int, default=20, help='Total number of features')
    parser.add_argument('--dataset-size', type=int, default=1000, help='Number of samples per population')
    parser.add_argument('--noise-scale', type=float, default=1.0, help='Scale of noise in the data')
    parser.add_argument('--corr-strength', type=float, default=0.0)
    parser.add_argument('--populations', nargs='+', default=['linear_regression', 'sinusoidal_regression'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-path', type=str, default='./results/baselines/', help='Base path to save results') # Changed default
    parser.add_argument('--compare-with-grad-desc', type=str, default=None,
                        help='Path to gradient descent results JSON file for comparison')

    args = parser.parse_args()

    # Create save path if it doesn't exist
    run_save_path = os.path.join(args.save_path, f"run_{args.seed}") # Add seed to path
    os.makedirs(run_save_path, exist_ok=True)

    # Define population configurations
    pop_configs = [{'pop_id': i, 'dataset_type': dt} for i, dt in enumerate(args.populations)]

    # Calculate budget (consistent with grad desc script)
    k_common = max(1, args.m1 // 2)
    k_pop_specific = args.m1 - k_common
    budget = k_common + len(pop_configs) * k_pop_specific

    # Save args used for this run
    args_filepath = os.path.join(run_save_path, 'args.json')
    try:
        with open(args_filepath, 'w') as f:
            json.dump(vars(args), f, indent=4)
        print(f"Run arguments saved to {args_filepath}")
    except Exception as e:
        print(f"Error saving arguments JSON: {e}")


    # Run all baselines
    # pop_configs passed here might contain data if loaded from elsewhere,
    # or just the config settings if running standalone.
    results = run_all_baselines(
        pop_configs=pop_configs,
        m1=args.m1,
        m=args.m,
        budget=budget, # Pass calculated budget
        dataset_size=args.dataset_size,
        noise_scale=args.noise_scale,
        corr_strength=args.corr_strength,
        seed=args.seed,
        save_path=run_save_path # Pass run-specific path
    )

    # Summarize results
    summary = summarize_results(results, save_path=run_save_path)
    print("\nBaseline methods summary:")
    print(summary)

    # Compare with gradient descent if requested
    if args.compare_with_grad_desc:
        comparison_save_path = os.path.join(run_save_path, 'comparison') # Subfolder for comparison results
        comparison = compare_with_gradient_descent(
            results, args.compare_with_grad_desc,
            save_path=comparison_save_path
        )
        if comparison is not None:
            print("\nComparison with gradient descent method:")
            print(comparison)

    print(f"\nAll baseline results saved in: {run_save_path}")
