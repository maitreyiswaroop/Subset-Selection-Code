import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.metrics import r2_score
import json
import os
from copy import deepcopy
from data import generate_data_continuous, generate_data_continuous_with_corr
import argparse

def pooled_lasso(pop_configs, m1, m, budget, dataset_size=10000, noise_scale=0.01, 
                 corr_strength=0.5, 
                alpha=0.01, seed=None):
    """
    Baseline 1: Pool all populations and use Lasso regression to select variables.
    
    Parameters:
    -----------
    pop_configs : list
        List of population configurations
    m1 : int
        Number of meaningful features per population
    m : int
        Total number of features
    dataset_size : int
        Number of samples per population
    noise_scale : float
        Scale of noise in the data
    alpha : float
        L1 regularization parameter for Lasso
    seed : int
        Random seed
        
    Returns:
    --------
    dict
        Dictionary with results including selected variables and scores
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Define common meaningful indices
    k_common = max(1, m1 // 2)
    common_meaningful_indices = np.arange(k_common)
    
    # Generate data for each population
    all_X = []
    all_Y = []
    all_meaningful_indices = []
    
    for pop_config in pop_configs:
        pop_id = pop_config['pop_id']
        dataset_type = pop_config['dataset_type']
        
        if corr_strength > 0:
            X, Y, A, meaningful_indices = generate_data_continuous_with_corr(
                pop_id=pop_id, m1=m1, m=m, 
                dataset_type=dataset_type, 
                dataset_size=dataset_size,
                noise_scale=noise_scale, 
                seed=seed, 
                common_meaningful_indices=common_meaningful_indices,
                corr_strength=corr_strength
            )
        else:
            X, Y, A, meaningful_indices = generate_data_continuous(
                pop_id=pop_id, m1=m1, m=m, 
                dataset_type=dataset_type, 
                dataset_size=dataset_size,
                noise_scale=noise_scale, 
                seed=seed, 
                common_meaningful_indices=common_meaningful_indices
            )
            
        all_X.append(X)
        all_Y.append(Y)
        all_meaningful_indices.append(meaningful_indices)
    
    # Pool all data
    pooled_X = np.vstack(all_X)
    pooled_Y = np.concatenate(all_Y)
    
    # Train Lasso model
    lasso = Lasso(alpha=alpha, random_state=seed)
    lasso.fit(pooled_X, pooled_Y)
    
    # Select top 2*m1 features based on absolute coefficient values
    coef_abs = np.abs(lasso.coef_)
    selected_indices = np.argsort(-coef_abs)[:budget]
    
    # Evaluate the selection
    true_indices = set()
    for indices in all_meaningful_indices:
        true_indices.update(indices)
    print('True indices:', true_indices)
    print('Selected indices:', selected_indices)
    
    recall = len(set(selected_indices) & true_indices) / len(true_indices)
    precision = len(set(selected_indices) & true_indices) / len(selected_indices)
    
    return {
        'selected_indices': selected_indices.tolist(),
        'true_indices': list(true_indices),
        'method': 'pooled_lasso',
        'recall': recall,
        'precision': precision,
        'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
        'coef_values': coef_abs[selected_indices].tolist()
    }

def population_wise_regression(pop_configs, m1, m, budget, dataset_size=10000, noise_scale=0.01, 
                               corr_strength=0.5,
                              model_type='linear', voting='frequency', seed=None):
    """
    Baseline 2: Perform regression on each population and use a voting mechanism to select variables.
    
    Parameters:
    -----------
    pop_configs : list
        List of population configurations
    m1 : int
        Number of meaningful features per population
    m : int
        Total number of features
    dataset_size : int
        Number of samples per population
    noise_scale : float
        Scale of noise in the data
    model_type : str
        Type of regression model ('linear', 'rf', 'lasso')
    voting : str
        Voting mechanism ('frequency', 'rank_sum', 'borda', 'weighted')
    seed : int
        Random seed
        
    Returns:
    --------
    dict
        Dictionary with results including selected variables and scores
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Define common meaningful indices
    k_common = max(1, m1 // 2)
    common_meaningful_indices = np.arange(k_common)
    
    # Generate data for each population
    pop_data = []
    all_meaningful_indices = []
    
    for pop_config in pop_configs:
        pop_id = pop_config['pop_id']
        dataset_type = pop_config['dataset_type']
        
        if corr_strength > 0:
            X, Y, A, meaningful_indices = generate_data_continuous_with_corr(
                pop_id=pop_id, m1=m1, m=m, 
                dataset_type=dataset_type, 
                dataset_size=dataset_size,
                noise_scale=noise_scale, 
                seed=seed, 
                common_meaningful_indices=common_meaningful_indices,
                corr_strength=corr_strength
            )
        else:
            X, Y, A, meaningful_indices = generate_data_continuous(
                pop_id=pop_id, m1=m1, m=m, 
                dataset_type=dataset_type, 
                dataset_size=dataset_size,
                noise_scale=noise_scale, 
                seed=seed, 
                common_meaningful_indices=common_meaningful_indices
            )
        
        pop_data.append((X, Y, pop_id))
        all_meaningful_indices.append(meaningful_indices)
    
    # Initialize structures for voting
    votes = np.zeros(m)
    rank_scores = np.zeros(m)
    importance_scores = np.zeros(m)
    
    for X, Y, pop_id in pop_data:
        # Choose model based on model_type
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=seed)
        elif model_type == 'lasso':
            model = Lasso(alpha=0.01, random_state=seed)
        # elif model_type == 'elasticnet':
        #     model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=seed)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        model.fit(X, Y)
        
        # Extract feature importance
        if model_type == 'rf':
            importance = model.feature_importances_
        else:
            importance = np.abs(model.coef_)
        
        # Select top features for this population
        top_indices = np.argsort(-importance)[:budget]
        
        # Update voting
        votes[top_indices] += 1
        
        # Update rank scores (Borda count)
        for i, idx in enumerate(np.argsort(-importance)):
            rank_scores[idx] += m - i
        
        # Update importance scores
        importance_scores += importance / np.sum(importance)  # Normalize
    
    # Select final variables based on voting mechanism
    if voting == 'frequency':
        # Select based on frequency of appearance
        selected_indices = np.argsort(-votes)[:budget]
    elif voting == 'rank_sum':
        # Select based on sum of ranks
        selected_indices = np.argsort(-rank_scores)[:budget]
    elif voting == 'borda':
        # Borda count
        selected_indices = np.argsort(-rank_scores)[:budget]
    elif voting == 'weighted':
        # Weighted by importance
        selected_indices = np.argsort(-importance_scores)[:budget]
    else:
        raise ValueError(f"Unknown voting mechanism: {voting}")
    
    # Evaluate the selection
    true_indices = set()
    for indices in all_meaningful_indices:
        true_indices.update(indices)
    
    recall = len(set(selected_indices) & true_indices) / len(true_indices)
    precision = len(set(selected_indices) & true_indices) / len(selected_indices)
    
    return {
        'selected_indices': selected_indices.tolist(),
        'true_indices': list(true_indices),
        'method': f'population_wise_{model_type}_{voting}',
        'recall': recall,
        'precision': precision,
        'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
        'votes': votes[selected_indices].tolist(),
        'rank_scores': rank_scores[selected_indices].tolist(),
        'importance_scores': importance_scores[selected_indices].tolist()
    }

def mutual_information_selection(pop_configs, m1, m, budget, dataset_size=10000, noise_scale=0.01, 
                                 corr_strength=0.5,
                               pooling='union', seed=None):
    """
    Baseline 3: Use mutual information to select variables.
    
    Parameters:
    -----------
    pop_configs : list
        List of population configurations
    m1 : int
        Number of meaningful features per population
    m : int
        Total number of features
    dataset_size : int
        Number of samples per population
    noise_scale : float
        Scale of noise in the data
    pooling : str
        How to combine results from different populations ('union', 'intersection', 'weighted')
    seed : int
        Random seed
        
    Returns:
    --------
    dict
        Dictionary with results including selected variables and scores
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Define common meaningful indices
    k_common = max(1, m1 // 2)
    common_meaningful_indices = np.arange(k_common)
    
    # Generate data for each population
    pop_data = []
    all_meaningful_indices = []
    
    for pop_config in pop_configs:
        pop_id = pop_config['pop_id']
        dataset_type = pop_config['dataset_type']
        
        if corr_strength > 0:
            X, Y, A, meaningful_indices = generate_data_continuous_with_corr(
                pop_id=pop_id, m1=m1, m=m, 
                dataset_type=dataset_type, 
                dataset_size=dataset_size,
                noise_scale=noise_scale, 
                seed=seed, 
                common_meaningful_indices=common_meaningful_indices,
                corr_strength=corr_strength
            )
        else:
            X, Y, A, meaningful_indices = generate_data_continuous(
                pop_id=pop_id, m1=m1, m=m, 
                dataset_type=dataset_type, 
                dataset_size=dataset_size,
                noise_scale=noise_scale, 
                seed=seed, 
                common_meaningful_indices=common_meaningful_indices
            )
        
        pop_data.append((X, Y, pop_id))
        all_meaningful_indices.append(meaningful_indices)
    
    # Compute mutual information for each population
    pop_selected = []
    mi_scores = np.zeros(m)
    
    for X, Y, pop_id in pop_data:
        # Compute mutual information
        mi = mutual_info_regression(X, Y, random_state=seed)
        
        # Select top features for this population
        top_indices = np.argsort(-mi)[:budget]
        pop_selected.append(top_indices)
        
        # Update MI scores
        mi_scores += mi / np.sum(mi)  # Normalize
    
    # Combine results based on pooling method
    if pooling == 'union':
        # Union of selected features across populations
        selected_set = set()
        for indices in pop_selected:
            selected_set.update(indices)
        
        # If more than 2*m1 features, take the top by MI score
        if len(selected_set) > 2*m1:
            selected_indices = np.array(list(selected_set))
            selected_scores = mi_scores[selected_indices]
            selected_indices = selected_indices[np.argsort(-selected_scores)[:budget]]
        else:
            # If less than 2*m1, fill with top MI features
            remaining = 2*m1 - len(selected_set)
            if remaining > 0:
                selected_set_list = list(selected_set)
                remaining_indices = [i for i in range(m) if i not in selected_set]
                remaining_scores = mi_scores[remaining_indices]
                top_remaining = np.array(remaining_indices)[np.argsort(-remaining_scores)[:remaining]]
                selected_indices = np.concatenate([selected_set_list, top_remaining])
            else:
                selected_indices = np.array(list(selected_set))
    
    elif pooling == 'intersection':
        # Intersection of selected features (with a minimum)
        selected_set = set(pop_selected[0])
        for indices in pop_selected[1:]:
            selected_set = selected_set.intersection(set(indices))
        
        # If too few features, add more based on MI scores
        if len(selected_set) < 2*m1:
            remaining = 2*m1 - len(selected_set)
            selected_set_list = list(selected_set)
            remaining_indices = [i for i in range(m) if i not in selected_set]
            remaining_scores = mi_scores[remaining_indices]
            top_remaining = np.array(remaining_indices)[np.argsort(-remaining_scores)[:remaining]]
            selected_indices = np.concatenate([selected_set_list, top_remaining])
        else:
            # If more than 2*m1, take top by MI score
            selected_indices = np.array(list(selected_set))
            selected_scores = mi_scores[selected_indices]
            selected_indices = selected_indices[np.argsort(-selected_scores)[:budget]]
    
    elif pooling == 'weighted':
        # Simply take top 2*m1 features by combined MI scores
        selected_indices = np.argsort(-mi_scores)[:budget]
    
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")
    
    # Evaluate the selection
    true_indices = set()
    for indices in all_meaningful_indices:
        true_indices.update(indices)
    
    recall = len(set(selected_indices) & true_indices) / len(true_indices)
    precision = len(set(selected_indices) & true_indices) / len(selected_indices)
    
    return {
        'selected_indices': selected_indices.tolist(),
        'true_indices': list(true_indices),
        'method': f'mutual_information_{pooling}',
        'recall': recall,
        'precision': precision,
        'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
        'mi_scores': mi_scores[selected_indices].tolist()
    }

def stability_selection(pop_configs, m1, m, budget, dataset_size=10000, noise_scale=0.01, 
                          corr_strength=0.5,
                       n_bootstraps=50, sample_fraction=0.75, alpha_range=None, 
                       threshold=0.7, seed=None):
    """
    Baseline 4: Stability selection with Lasso.
    
    Parameters:
    -----------
    pop_configs : list
        List of population configurations
    m1 : int
        Number of meaningful features per population
    m : int
        Total number of features
    dataset_size : int
        Number of samples per population
    noise_scale : float
        Scale of noise in the data
    n_bootstraps : int
        Number of bootstrap samples
    sample_fraction : float
        Fraction of samples to use in each bootstrap
    alpha_range : list or None
        Range of alpha values for Lasso
    threshold : float
        Threshold for stability selection
    seed : int
        Random seed
        
    Returns:
    --------
    dict
        Dictionary with results including selected variables and scores
    """
    if seed is not None:
        np.random.seed(seed)
    
    if alpha_range is None:
        alpha_range = [0.001, 0.005, 0.01, 0.05, 0.1]
    
    # Define common meaningful indices
    k_common = max(1, m1 // 2)
    common_meaningful_indices = np.arange(k_common)
    
    # Generate data for each population
    all_X = []
    all_Y = []
    all_meaningful_indices = []
    
    for pop_config in pop_configs:
        pop_id = pop_config['pop_id']
        dataset_type = pop_config['dataset_type']
        
        if corr_strength > 0:
            X, Y, A, meaningful_indices = generate_data_continuous_with_corr(
                pop_id=pop_id, m1=m1, m=m, 
                dataset_type=dataset_type, 
                dataset_size=dataset_size,
                noise_scale=noise_scale, 
                seed=seed, 
                common_meaningful_indices=common_meaningful_indices,
                corr_strength=corr_strength
            )
        else:
            X, Y, A, meaningful_indices = generate_data_continuous(
                pop_id=pop_id, m1=m1, m=m, 
                dataset_type=dataset_type, 
                dataset_size=dataset_size,
                noise_scale=noise_scale, 
                seed=seed, 
                common_meaningful_indices=common_meaningful_indices
            )
        
        all_X.append(X)
        all_Y.append(Y)
        all_meaningful_indices.append(meaningful_indices)
    
    # Pool all data
    pooled_X = np.vstack(all_X)
    pooled_Y = np.concatenate(all_Y)
    
    n_samples = pooled_X.shape[0]
    bootstrap_size = int(n_samples * sample_fraction)
    
    # Initialize selection probability matrix
    selection_probability = np.zeros((len(alpha_range), m))
    
    # Run stability selection
    for i, alpha in enumerate(alpha_range):
        feature_counts = np.zeros(m)
        
        for b in range(n_bootstraps):
            # Create bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, bootstrap_size, replace=True)
            X_bootstrap = pooled_X[bootstrap_indices]
            Y_bootstrap = pooled_Y[bootstrap_indices]
            
            # Fit Lasso
            lasso = Lasso(alpha=alpha, random_state=seed+b)
            lasso.fit(X_bootstrap, Y_bootstrap)
            
            # Count selected features
            selected = np.where(np.abs(lasso.coef_) > 0)[0]
            feature_counts[selected] += 1
        
        # Calculate selection probability
        selection_probability[i] = feature_counts / n_bootstraps
    
    # Aggregate over all alpha values
    max_probability = np.max(selection_probability, axis=0)
    
    # Select features that exceed the threshold
    stable_features = np.where(max_probability >= threshold)[0]
    
    # If we have more than 2*m1 stable features, take the top ones
    if len(stable_features) > 2*m1:
        selected_indices = stable_features[np.argsort(-max_probability[stable_features])[:budget]]
    # If we have fewer, add more based on probability
    elif len(stable_features) < 2*m1:
        remaining = 2*m1 - len(stable_features)
        unstable_features = np.array([i for i in range(m) if i not in stable_features])
        top_unstable = unstable_features[np.argsort(-max_probability[unstable_features])[:remaining]]
        selected_indices = np.concatenate([stable_features, top_unstable])
    else:
        selected_indices = stable_features
    
    # Evaluate the selection
    true_indices = set()
    for indices in all_meaningful_indices:
        true_indices.update(indices)
    
    recall = len(set(selected_indices) & true_indices) / len(true_indices)
    precision = len(set(selected_indices) & true_indices) / len(selected_indices)
    
    return {
        'selected_indices': selected_indices.tolist(),
        'true_indices': list(true_indices),
        'method': 'stability_selection',
        'recall': recall,
        'precision': precision,
        'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
        'selection_probabilities': max_probability[selected_indices].tolist()
    }

def group_lasso_selection(pop_configs, m1, m, budget, dataset_size=10000, noise_scale=0.01, 
                            corr_strength=0.5,
                         alpha=0.01, seed=None):
    """
    Baseline 5: Group Lasso approach (by treating populations as groups and selecting features
    that are important across groups).
    
    Note: This is an approximation of group lasso since sklearn doesn't have it built-in.
    We use separate Lasso models and then combine the results.
    
    Parameters:
    -----------
    pop_configs : list
        List of population configurations
    m1 : int
        Number of meaningful features per population
    m : int
        Total number of features
    dataset_size : int
        Number of samples per population
    noise_scale : float
        Scale of noise in the data
    alpha : float
        L1 regularization parameter
    seed : int
        Random seed
        
    Returns:
    --------
    dict
        Dictionary with results including selected variables and scores
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Define common meaningful indices
    k_common = max(1, m1 // 2)
    common_meaningful_indices = np.arange(k_common)
    
    # Generate data for each population
    pop_data = []
    all_meaningful_indices = []
    
    for pop_config in pop_configs:
        pop_id = pop_config['pop_id']
        dataset_type = pop_config['dataset_type']
        
        if corr_strength > 0:
            X, Y, A, meaningful_indices = generate_data_continuous_with_corr(
                pop_id=pop_id, m1=m1, m=m, 
                dataset_type=dataset_type, 
                dataset_size=dataset_size,
                noise_scale=noise_scale, 
                seed=seed, 
                common_meaningful_indices=common_meaningful_indices,
                corr_strength=corr_strength
            )
        else:
            X, Y, A, meaningful_indices = generate_data_continuous(
                pop_id=pop_id, m1=m1, m=m, 
                dataset_type=dataset_type, 
                dataset_size=dataset_size,
                noise_scale=noise_scale, 
                seed=seed, 
                common_meaningful_indices=common_meaningful_indices
            )
        
        pop_data.append((X, Y, pop_id))
        all_meaningful_indices.append(meaningful_indices)
    
    # Run Lasso on each population and get coefficients
    all_coefs = np.zeros((len(pop_data), m))
    
    for i, (X, Y, _) in enumerate(pop_data):
        lasso = Lasso(alpha=alpha, random_state=seed)
        lasso.fit(X, Y)
        all_coefs[i] = np.abs(lasso.coef_)
    
    # Compute group norms (L2 norm across populations)
    group_norms = np.sqrt(np.sum(all_coefs**2, axis=0))
    
    # Select top features based on group norms
    selected_indices = np.argsort(-group_norms)[:budget]
    
    # Evaluate the selection
    true_indices = set()
    for indices in all_meaningful_indices:
        true_indices.update(indices)
    
    recall = len(set(selected_indices) & true_indices) / len(true_indices)
    precision = len(set(selected_indices) & true_indices) / len(selected_indices)
    
    return {
        'selected_indices': selected_indices.tolist(),
        'true_indices': list(true_indices),
        'method': 'group_lasso',
        'recall': recall,
        'precision': precision,
        'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
        'group_norms': group_norms[selected_indices].tolist()
    }

def condorcet_voting(pop_configs, m1, m, budget, dataset_size=10000, noise_scale=0.01, 
                        corr_strength=0.5,
                    model_type='rf', seed=None):
    """
    Baseline 6: Condorcet voting method for variable selection.
    
    Parameters:
    -----------
    pop_configs : list
        List of population configurations
    m1 : int
        Number of meaningful features per population
    m : int
        Total number of features
    dataset_size : int
        Number of samples per population
    noise_scale : float
        Scale of noise in the data
    model_type : str
        Type of regression model ('linear', 'rf', 'lasso')
    seed : int
        Random seed
        
    Returns:
    --------
    dict
        Dictionary with results including selected variables and scores
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Define common meaningful indices
    k_common = max(1, m1 // 2)
    common_meaningful_indices = np.arange(k_common)
    
    # Generate data for each population
    pop_data = []
    all_meaningful_indices = []
    
    for pop_config in pop_configs:
        pop_id = pop_config['pop_id']
        dataset_type = pop_config['dataset_type']
        
        if corr_strength > 0:
            X, Y, A, meaningful_indices = generate_data_continuous_with_corr(
                pop_id=pop_id, m1=m1, m=m, 
                dataset_type=dataset_type, 
                dataset_size=dataset_size,
                noise_scale=noise_scale, 
                seed=seed, 
                common_meaningful_indices=common_meaningful_indices,
                corr_strength=corr_strength
            )
        else:
            X, Y, A, meaningful_indices = generate_data_continuous(
                pop_id=pop_id, m1=m1, m=m, 
                dataset_type=dataset_type, 
                dataset_size=dataset_size,
                noise_scale=noise_scale, 
                seed=seed, 
                common_meaningful_indices=common_meaningful_indices
            )
        
        pop_data.append((X, Y, pop_id))
        all_meaningful_indices.append(meaningful_indices)
    
    # Get feature rankings for each population
    pop_rankings = []
    
    for X, Y, _ in pop_data:
        # Choose model based on model_type
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=seed)
        elif model_type == 'lasso':
            model = Lasso(alpha=0.01, random_state=seed)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        model.fit(X, Y)
        
        # Extract feature importance
        if model_type == 'rf':
            importance = model.feature_importances_
        else:
            importance = np.abs(model.coef_)
        
        # Get ranking (higher importance = better rank)
        ranking = np.argsort(-importance)
        pop_rankings.append(ranking)
    
    # Initialize Condorcet matrix (pairwise comparisons)
    condorcet_matrix = np.zeros((m, m))
    
    # Fill Condorcet matrix
    for ranking in pop_rankings:
        for i in range(m):
            feature_i = ranking[i]
            # All features ranked after feature_i lose to feature_i
            for j in range(i+1, m):
                feature_j = ranking[j]
                condorcet_matrix[feature_i, feature_j] += 1
    
    # Compute Copeland score (wins - losses)
    wins = np.sum(condorcet_matrix > condorcet_matrix.T, axis=1)
    losses = np.sum(condorcet_matrix < condorcet_matrix.T, axis=1)
    copeland_scores = wins - losses
    
    # Select top features based on Copeland scores
    selected_indices = np.argsort(-copeland_scores)[:budget]
    
    # Evaluate the selection
    true_indices = set()
    for indices in all_meaningful_indices:
        true_indices.update(indices)
    
    recall = len(set(selected_indices) & true_indices) / len(true_indices)
    precision = len(set(selected_indices) & true_indices) / len(selected_indices)
    
    return {
        'selected_indices': selected_indices.tolist(),
        'true_indices': list(true_indices),
        'method': f'condorcet_{model_type}',
        'recall': recall,
        'precision': precision,
        'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
        'copeland_scores': copeland_scores[selected_indices].tolist()
    }

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, set):
        return list(obj)
    return obj

def run_all_baselines(pop_configs, m1, m, budget, dataset_size=10000, noise_scale=0.01, corr_strength=0.0,
                      seed=42, save_path='./results/baselines/'):
    """
    Run all baseline methods and return/save the results.
    
    Parameters:
    -----------
    pop_configs : list
        List of population configurations
    m1 : int
        Number of meaningful features per population
    m : int
        Total number of features
    dataset_size : int
        Number of samples per population
    noise_scale : float
        Scale of noise in the data
    seed : int
        Random seed
    save_path : str
        Path to save results
        
    Returns:
    --------
    dict
        Dictionary with results for all methods
    """
    os.makedirs(save_path, exist_ok=True)
    
    # generate dummy dataset to get population configurations
    if seed is not None:
        np.random.seed(seed)
    
    # Define common meaningful indices
    k_common = max(1, m1 // 2)
    common_meaningful_indices = np.arange(k_common)
    
    # Generate data for each population
    pop_configs_temp = []
    all_meaningful_indices = []
    
    for pop_config in pop_configs:
        pop_id = pop_config['pop_id']
        dataset_type = pop_config['dataset_type']
        
        if corr_strength > 0:
            X, Y, A, meaningful_indices = generate_data_continuous_with_corr(
                pop_id=pop_id, m1=m1, m=m, 
                dataset_type=dataset_type, 
                dataset_size=dataset_size,
                noise_scale=noise_scale, 
                seed=seed, 
                common_meaningful_indices=common_meaningful_indices,
                corr_strength=corr_strength
            )
        else:
            X, Y, A, meaningful_indices = generate_data_continuous(
                pop_id=pop_id, m1=m1, m=m, 
                dataset_type=dataset_type, 
                dataset_size=dataset_size,
                noise_scale=noise_scale, 
                seed=seed, 
                common_meaningful_indices=common_meaningful_indices
            )
        
        # pop_data.append((X, Y, pop_id, meaningful_indices))
        pop_configs_temp.append({
            'pop_id': pop_id,
            'dataset_type': dataset_type,
            'meaningful_indices': meaningful_indices
        })

        all_meaningful_indices.append(meaningful_indices)
    ########
    results = {}
    results['pop_configs'] = pop_configs_temp
    budget = m1//2 + len(pop_configs) * (m1//2)
    # Run all baselines
    results['pooled_lasso'] = pooled_lasso(pop_configs, m1, m, budget, dataset_size, noise_scale, alpha=0.01, seed=seed)
    
    # Population-wise regression with different models and voting mechanisms
    model_types = ['linear', 'rf', 'lasso']
    voting_methods = ['frequency', 'rank_sum', 'borda', 'weighted']
    
    for model_type in model_types:
        for voting in voting_methods:
            key = f'population_wise_{model_type}_{voting}'
            results[key] = population_wise_regression(
                pop_configs, m1, m, budget, dataset_size, noise_scale, 
                model_type=model_type, voting=voting, seed=seed
            )
    
    # Mutual information with different pooling strategies
    pooling_methods = ['union', 'intersection', 'weighted']
    for pooling in pooling_methods:
        key = f'mutual_information_{pooling}'
        results[key] = mutual_information_selection(
            pop_configs, m1, m, budget, dataset_size, noise_scale, 
            pooling=pooling, seed=seed
        )
    
    # # Stability selection
    # results['stability_selection'] = stability_selection(
    #     pop_configs, m1, m, dataset_size, noise_scale, 
    #     n_bootstraps=50, sample_fraction=0.75, threshold=0.7, seed=seed
    # )
    
    # Group Lasso approach
    results['group_lasso'] = group_lasso_selection(
        pop_configs, m1, m, budget, dataset_size, noise_scale, alpha=0.01, seed=seed
    )
    
    # Condorcet voting with different base models
    for model_type in model_types:
        key = f'condorcet_{model_type}'
        results[key] = condorcet_voting(
            pop_configs, m1, m, budget, dataset_size, noise_scale, 
            model_type=model_type, seed=seed
        )
    
    # Convert results before saving
    serializable_results = convert_to_serializable(results)

    # Save results
    with open(os.path.join(save_path, 'baseline_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    return results

def summarize_results(results, save_path='./results/baselines/'):
    """
    Summarize the results of all baseline methods.
    
    Parameters:
    -----------
    results : dict
        Dictionary with results for all methods
    save_path : str
        Path to save summary
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with summarized results
    """
    pop_configs = results['pop_configs']
    summary = []
    
    for method_name, method_results in results.items():
        if method_name == 'pop_configs':
            continue
        # print(f"Processing method: {method_name}")
        # print(pop_configs)
        # find min selected percentage across populations
        min_selected_percentage = min([
            len(set(method_results['selected_indices']) & set(pop_config['meaningful_indices'])) / len(pop_config['meaningful_indices'])
            for pop_config in pop_configs
        ])
        # find max selected percentage across populations
        max_selected_percentage = max([
            len(set(method_results['selected_indices']) & set(pop_config['meaningful_indices'])) / len(pop_config['meaningful_indices'])
            for pop_config in pop_configs
        ])
        # find mean selected percentage across populations  
        mean_selected_percentage = np.mean([
            len(set(method_results['selected_indices']) & set(pop_config['meaningful_indices'])) / len(pop_config['meaningful_indices'])
            for pop_config in pop_configs
        ])
        # # find std selected percentage across populations
        # std_selected_percentage = np.std([
        #     len(set(method_results['selected_indices']) & set(pop_config['meaningful_indices'])) / len(pop_config['meaningful_indices'])
        #     for pop_config in pop_configs
        # ])
        # find median selected percentage across populations
        median_selected_percentage = np.median([
            len(set(method_results['selected_indices']) & set(pop_config['meaningful_indices'])) / len(pop_config['meaningful_indices'])
            for pop_config in pop_configs
        ])
        summary.append({
            'method': method_name,
            'precision': method_results['precision'],
            'recall': method_results['recall'],
            'f1_score': method_results['f1_score'],
            'min_selected_percentage': min_selected_percentage,
            'max_selected_percentage': max_selected_percentage,
            'mean_selected_percentage': mean_selected_percentage,
            # 'std_selected_percentage': std_selected_percentage,
            'median_selected_percentage': median_selected_percentage,
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('f1_score', ascending=False)
    
    # Save summary
    summary_df.to_csv(os.path.join(save_path, 'baseline_summary.csv'), index=False)
    
    return summary_df

def compare_with_gradient_descent(baseline_results, grad_desc_results, save_path='./results/comparison/'):
    """
    Compare baseline methods with gradient descent approach.
    
    Parameters:
    -----------
    baseline_results : dict
        Dictionary with results for all baseline methods
    grad_desc_results : dict
        Dictionary with results for gradient descent method
    save_path : str
        Path to save comparison
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with comparison results
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Extract grad desc results
    grad_desc_selected = set(grad_desc_results.get('selected_indices', []))
    grad_desc_true = set(grad_desc_results.get('true_variable_index', []))
    
    if not grad_desc_true and 'meaningful_indices' in grad_desc_results:
        # Combine all meaningful indices from all populations
        for indices in grad_desc_results['meaningful_indices']:
            grad_desc_true.update(indices)
    
    # Calculate metrics for grad desc
    grad_desc_precision = len(grad_desc_selected & grad_desc_true) / len(grad_desc_selected) if grad_desc_selected else 0
    grad_desc_recall = len(grad_desc_selected & grad_desc_true) / len(grad_desc_true) if grad_desc_true else 0
    grad_desc_f1 = 2 * (grad_desc_precision * grad_desc_recall) / (grad_desc_precision + grad_desc_recall) if (grad_desc_precision + grad_desc_recall) > 0 else 0
    
    # Create comparison dataframe
    comparison = []
    
    # Add grad desc results
    comparison.append({
        'method': 'gradient_descent',
        'precision': grad_desc_precision,
        'recall': grad_desc_recall,
        'f1_score': grad_desc_f1
    })
    
    # Add baseline results
    for method_name, method_results in baseline_results.items():
        comparison.append({
            'method': method_name,
            'precision': method_results['precision'],
            'recall': method_results['recall'],
            'f1_score': method_results['f1_score']
        })
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('f1_score', ascending=False)
    
    # Save comparison
    comparison_df.to_csv(os.path.join(save_path, 'method_comparison.csv'), index=False)
    
    return comparison_df

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Run baseline variable selection methods')
    parser.add_argument('--m1', type=int, default=4, help='Number of meaningful features per population')
    parser.add_argument('--m', type=int, default=20, help='Total number of features')
    parser.add_argument('--dataset-size', type=int, default=1000, help='Number of samples per population')
    parser.add_argument('--noise-scale', type=float, default=1.0, help='Scale of noise in the data')
    parser.add_argument('--corr-strength', type=float, default=0.0)
    parser.add_argument('--populations', nargs='+', default=['linear_regression', 'sinusoidal_regression'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-path', type=str, default='./results/baselines/num_true_vars', help='Path to save results')
    parser.add_argument('--compare-with-grad-desc', type=str, default=None, 
                        help='Path to gradient descent results JSON file for comparison')
    
    args = parser.parse_args()
    
    # create save path if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)

    # Define population configurations (similar to grad_desc_populations_v2.py)
    pop_configs = [
        {'pop_id': i, 'dataset_type': args.populations[i]}
        for i in range(len(args.populations))
    ]
    
    budget = args.m1//2 + len(pop_configs) * (args.m1//2)
    print(f"Running with budget: {budget}")
    # Run all baselines
    results = run_all_baselines(
        pop_configs=pop_configs,
        m1=args.m1,
        m=args.m,
        budget=budget,
        dataset_size=args.dataset_size,
        noise_scale=args.noise_scale,
        corr_strength=args.corr_strength,
        seed=args.seed,
        save_path=args.save_path
    )
    
    # append the pop configs to the results
    # Summarize results
    summary = summarize_results(results, save_path=args.save_path)
    print("Baseline methods summary:")
    print(summary)

    # also save the args
    with open(os.path.join(args.save_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Compare with gradient descent if requested
    if args.compare_with_grad_desc:
        try:
            with open(args.compare_with_grad_desc, 'r') as f:
                grad_desc_results = json.load(f)
            
            comparison = compare_with_gradient_descent(
                results, grad_desc_results, 
                save_path=os.path.join(args.save_path, 'comparison')
            )
            print("\nComparison with gradient descent method:")
            print(comparison)
        except FileNotFoundError:
            print(f"Gradient descent results file not found: {args.compare_with_grad_desc}")
        except json.JSONDecodeError:
            print(f"Error decoding gradient descent results file: {args.compare_with_grad_desc}")