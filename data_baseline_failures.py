# data_baseline_failures.py
import numpy as np
from global_vars import EPS
from estimators import *
import torch
# EPS = 1e-8 # Not strictly needed here, but good practice if doing divisions.

def standardize_data(X, Y):
    """Standardizes X (features) and Y (outcome). Returns standardized data and original means/stds."""
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    Y_mean = np.mean(Y)
    Y_std = np.std(Y)

    X_std[X_std < EPS] = EPS # Avoid division by zero
    if Y_std < EPS:
        Y_std = EPS

    X_stdized = (X - X_mean) / X_std
    Y_stdized = (Y - Y_mean) / Y_std

    return X_stdized, Y_stdized, X_mean, X_std, Y_mean, Y_std

def generate_baseline_failure_1_heterogeneous_importance(
    dataset_size: int = 10000,
    n_features: int = 20,
    noise_scale: float = 0.1,
    corr_strength: float = 0.4, 
    seed: int = None
):
    """
    Scenario 1: Heterogeneous Feature Importance - Enhanced Version.
    Pools three internal populations (A, B, C).
    Pop A (Very Large, 65%): Y ~ 3*X1 + 0.5*X2 + epsilon (low noise)
    Pop B (Medium, 25%): Y ~ 0.5*X1 + 3*X2 + epsilon (low noise)
    Pop C (Small but Critical, 10%): Y ~ 0.1*X1 + 0.1*X2 + 5*X3 + epsilon (high noise)
    
    Features beyond X3 have complex correlation patterns with varying strength.
    X3 (critical for minority population) has strong correlations with several noise features.
    """
    if n_features < 4:
        raise ValueError("Enhanced Scenario 1 requires at least 4 features.")
    if seed is not None:
        np.random.seed(seed)

    # More severe population imbalance
    n_a = int(0.65 * dataset_size)  # Very large population
    n_b = int(0.25 * dataset_size)  # Medium population
    n_c = dataset_size - n_a - n_b  # Small critical population (10%)

    # Generate base features
    X_all = np.random.randn(dataset_size, n_features)
    Y_all = np.zeros(dataset_size)

    # Pop A - Mostly relies on X1
    idx_a_end = n_a
    X_pop_a = X_all[:idx_a_end, :]
    
    # Create complex correlation patterns for Pop A
    for i in range(3, min(n_features, 8)):  # Correlate with first few noise features
        weight_x1 = 0.4 + 0.1 * (i % 3)  # Varies between 0.4-0.6
        weight_x2 = 0.6 - 0.1 * (i % 3)  # Varies between 0.6-0.4
        X_pop_a[:, i] = (weight_x1 * corr_strength * X_pop_a[:, 0] + 
                          weight_x2 * corr_strength * X_pop_a[:, 1] + 
                          (1 - corr_strength) * np.random.normal(0, 1, n_a))
    
    # Lower noise for dominant population
    Y_all[:idx_a_end] = 3.0 * X_pop_a[:, 0] + 0.5 * X_pop_a[:, 1] + np.random.normal(0, noise_scale * 0.8, n_a)

    # Pop B - Mostly relies on X2
    idx_b_start = n_a
    idx_b_end = n_a + n_b
    X_pop_b = X_all[idx_b_start:idx_b_end, :]
    
    # Create complex correlation patterns for Pop B
    for i in range(3, min(n_features, 8)):
        weight_x1 = 0.3 + 0.15 * (i % 3)  # Varies between 0.3-0.6
        weight_x2 = 0.7 - 0.15 * (i % 3)  # Varies between 0.7-0.4
        X_pop_b[:, i] = (weight_x1 * corr_strength * X_pop_b[:, 0] + 
                          weight_x2 * corr_strength * X_pop_b[:, 1] + 
                          (1 - corr_strength) * np.random.normal(0, 1, n_b))
    
    # Lower noise for medium population
    Y_all[idx_b_start:idx_b_end] = 0.5 * X_pop_b[:, 0] + 3.0 * X_pop_b[:, 1] + np.random.normal(0, noise_scale * 0.8, n_b)

    # Pop C - Critically relies on X3
    idx_c_start = n_a + n_b
    X_pop_c = X_all[idx_c_start:, :]
    
    # Stronger correlations for Pop C - particularly between X3 and noise features
    stronger_corr = min(0.7, corr_strength * 1.5)
    for i in range(8, min(n_features, 15)):  # More extensive correlations
        X_pop_c[:, i] = stronger_corr * X_pop_c[:, 2] + (1 - stronger_corr) * np.random.normal(0, 1, n_c)
    
    # Additional mixed correlations to create a complex structure
    if n_features > 15:
        for i in range(15, n_features):
            mix_weight = 0.4 + 0.2 * np.random.random()  # Random weight between 0.4-0.6
            X_pop_c[:, i] = (mix_weight * stronger_corr * X_pop_c[:, 2] + 
                             (1 - mix_weight) * corr_strength * X_pop_c[:, 0] + 
                             (1 - stronger_corr) * np.random.normal(0, 1, n_c))
    
    # Higher noise for minority population + stronger signal for X3
    Y_all[idx_c_start:] = (0.1 * X_pop_c[:, 0] + 
                           0.1 * X_pop_c[:, 1] + 
                           5.0 * X_pop_c[:, 2] + 
                           np.random.normal(0, noise_scale * 1.5, n_c))

    # Define meaningful indices (the truly relevant features)
    meaningful_indices = np.array([0, 1, 2])  # X1, X2, X3
    
    # Create final population data structures
    pop_data = [
        {
            'pop_id': 'A',
            'X_raw': X_all[:idx_a_end, :],
            'Y_raw': Y_all[:idx_a_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'B',
            'X_raw': X_all[idx_b_start:idx_b_end, :],
            'Y_raw': Y_all[idx_b_start:idx_b_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'C',
            'X_raw': X_all[idx_c_start:, :],
            'Y_raw': Y_all[idx_c_start:],
            'meaningful_indices': meaningful_indices
        }
    ]
    return pop_data

def generate_baseline_failure_2_opposing_effects(
    dataset_size: int = 10000,
    n_features: int = 5,
    noise_scale: float = 0.1,
    corr_strength: float = 0.2, 
    seed: int = None
):
    """
    Scenario 2: Opposing Effects.
    Pools three internal populations.
    Pop A (40%): Y ~ 3*X1 + X2 + epsilon
    Pop B (40%): Y ~ -3*X1 + X2 + epsilon
    Pop C (20%, neutral or reinforcing X1 weakly, uses X3): Y ~ 0.5*X1 + X2 + 1.5*X3 + epsilon
    """
    if n_features < 2:
        raise ValueError("Scenario 2 requires at least 2 features for X1, X2.")
    if seed is not None:
        np.random.seed(seed)

    n_a = int(0.40 * dataset_size)
    n_b = int(0.40 * dataset_size)
    n_c = dataset_size - n_a - n_b

    X_all = np.random.randn(dataset_size, n_features)
    Y_all = np.zeros(dataset_size)
    current_meaningful = {0, 1} # X1, X2 are primary due to Pop A & B

    # Pop A
    idx_a_end = n_a
    X_pop_a = X_all[:idx_a_end, :]
    Y_all[:idx_a_end] = 3 * X_pop_a[:, 0] + X_pop_a[:, 1] + np.random.normal(0, noise_scale, n_a)

    # Pop B
    idx_b_start = n_a
    idx_b_end = n_a + n_b
    X_pop_b = X_all[idx_b_start:idx_b_end, :]
    Y_all[idx_b_start:idx_b_end] = -3 * X_pop_b[:, 0] + X_pop_b[:, 1] + np.random.normal(0, noise_scale, n_b)

    # Pop C
    idx_c_start = n_a + n_b
    X_pop_c = X_all[idx_c_start:, :]
    if n_features >= 3:
        Y_all[idx_c_start:] = 0.5 * X_pop_c[:, 0] + X_pop_c[:, 1] + 1.5 * X_pop_c[:, 2] + np.random.normal(0, noise_scale, n_c)
        current_meaningful.add(2) # X3 for pop C
    else:
        Y_all[idx_c_start:] = 0.5 * X_pop_c[:, 0] + X_pop_c[:, 1] + np.random.normal(0, noise_scale, n_c)

    meaningful_indices = np.array(sorted(list(current_meaningful)))
    
    pop_data = [
        {
            'pop_id': 'A',
            'X_raw' : X_pop_a,
            'Y_raw' : Y_all[:idx_a_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'B',
            'X_raw' : X_pop_b,
            'Y_raw' : Y_all[idx_b_start:idx_b_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'C',
            'X_raw' : X_pop_c,
            'Y_raw' : Y_all[idx_c_start:],
            'meaningful_indices': meaningful_indices
        }
    ]

    return pop_data


def generate_baseline_failure_3_non_linearity_subgroup(
    dataset_size: int = 10000,
    n_features: int = 5,
    noise_scale: float = 0.1,
    corr_strength: float = 0.2, 
    seed: int = None
):
    """
    Scenario 3: Non-Linearity in a Subgroup.
    Pools three internal populations.
    Pop A (Dominant, 60%): Y ~ X1 + X2 + epsilon (linear)
    Pop B (Subgroup, 30%): Y ~ 3*X1^2 - 2*X1 + X2 + epsilon (non-linear for X1, centered for impact)
    Pop C (Small, 10%): Y ~ 0.5*X1 + X2 + 2*X3 + epsilon (linear, introduces X3)
    """
    if n_features < 2:
        raise ValueError("Scenario 3 requires at least 2 features for X1, X2.")
    if seed is not None:
        np.random.seed(seed)

    n_a = int(0.60 * dataset_size)
    n_b = int(0.30 * dataset_size)
    n_c = dataset_size - n_a - n_b
    current_meaningful = {0, 1}

    X_all = np.random.randn(dataset_size, n_features)
    # For Pop B, X1 values around +/-1 make X1^2 distinct from X1. randn is fine.
    Y_all = np.zeros(dataset_size)

    # Pop A
    idx_a_end = n_a
    X_pop_a = X_all[:idx_a_end, :]
    Y_all[:idx_a_end] = X_pop_a[:, 0] + X_pop_a[:, 1] + np.random.normal(0, noise_scale, n_a)

    # Pop B
    idx_b_start = n_a
    idx_b_end = n_a + n_b
    X_pop_b = X_all[idx_b_start:idx_b_end, :]
    # Y_all[idx_b_start:idx_b_end] = (X_pop_b[:, 0]**2) + X_pop_b[:, 1] + np.random.normal(0, noise_scale, n_b)
    Y_all[idx_b_start:idx_b_end] = 3*(X_pop_b[:, 0]**2) - 2*X_pop_b[:,0] + X_pop_b[:, 1] + np.random.normal(0, noise_scale, n_b)


    # Pop C
    idx_c_start = n_a + n_b
    X_pop_c = X_all[idx_c_start:, :]
    if n_features >= 3:
        Y_all[idx_c_start:] = 0.5 * X_pop_c[:, 0] + X_pop_c[:, 1] + 2 * X_pop_c[:, 2] + np.random.normal(0, noise_scale, n_c)
        current_meaningful.add(2)
    else:
        Y_all[idx_c_start:] = 0.5 * X_pop_c[:, 0] + X_pop_c[:, 1] + np.random.normal(0, noise_scale, n_c)

    meaningful_indices = np.array(sorted(list(current_meaningful)))
    
    pop_data = [
        {
            'pop_id': 'A',
            'X_raw' : X_pop_a,
            'Y_raw' : Y_all[:idx_a_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'B',
            'X_raw' : X_pop_b,
            'Y_raw' : Y_all[idx_b_start:idx_b_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'C',
            'X_raw' : X_pop_c,
            'Y_raw' : Y_all[idx_c_start:],
            'meaningful_indices': meaningful_indices
        }
    ]
    return pop_data

def generate_baseline_failure_4_different_noise_structures_features(
    dataset_size: int = 10000,
    n_features: int = 5,
    noise_scale_y: float = 0.5,
    x1_noise_stds: dict = None, # e.g. {'A': 0.1, 'B': 2.0, 'C': 0.5}
    corr_strength: float = 0.2, 
    seed: int = None
):
    """
    Scenario 4: Different Noise Structures in Features. Y ~ 2*X1_true + X2_clean [+ X3_clean for Pop C] + eps_y
    Pop A (33%): X1_obs = X1_true + noise_A_std
    Pop B (33%): X1_obs = X1_true + noise_B_std
    Pop C (34%): X1_obs = X1_true + noise_C_std
    X2 (idx 1) is clean. X3 (idx 2) is clean and used by Pop C if available.
    """
    if n_features < 1:
        raise ValueError("Scenario 4 requires at least 1 feature for X1.")
    if seed is not None:
        np.random.seed(seed)

    _x1_noise_stds = {'A': 0.1, 'B': 2.0, 'C': 0.5} # Defaults
    if x1_noise_stds:
        _x1_noise_stds.update(x1_noise_stds)

    n_a = dataset_size // 3
    n_b = dataset_size // 3
    n_c = dataset_size - n_a - n_b
    current_meaningful = {0} # X1 (observed at index 0) is always fundamental

    X1_true = np.random.randn(dataset_size)
    # Initialize all observed features, X_observed_all[:,0] will be overwritten
    X_observed_all = np.random.randn(dataset_size, n_features)
    Y_all = np.zeros(dataset_size)

    # Pop A
    idx_a_end = n_a
    X_observed_all[:idx_a_end, 0] = X1_true[:idx_a_end] + np.random.normal(0, _x1_noise_stds['A'], n_a)
    y_pop_a = 2 * X1_true[:idx_a_end]
    if n_features > 1: # Use X2 (feature at index 1)
        y_pop_a += X_observed_all[:idx_a_end, 1]
        current_meaningful.add(1)
    Y_all[:idx_a_end] = y_pop_a + np.random.normal(0, noise_scale_y, n_a)

    # Pop B
    idx_b_start = n_a
    idx_b_end = n_a + n_b
    X_observed_all[idx_b_start:idx_b_end, 0] = X1_true[idx_b_start:idx_b_end] + np.random.normal(0, _x1_noise_stds['B'], n_b)
    y_pop_b = 2 * X1_true[idx_b_start:idx_b_end]
    if n_features > 1: # Use X2
        y_pop_b += X_observed_all[idx_b_start:idx_b_end, 1]
        current_meaningful.add(1)
    Y_all[idx_b_start:idx_b_end] = y_pop_b + np.random.normal(0, noise_scale_y, n_b)

    # Pop C
    idx_c_start = n_a + n_b
    X_observed_all[idx_c_start:, 0] = X1_true[idx_c_start:] + np.random.normal(0, _x1_noise_stds['C'], n_c)
    y_pop_c = 2 * X1_true[idx_c_start:]
    if n_features > 1: # Use X2
        y_pop_c += X_observed_all[idx_c_start:, 1]
        current_meaningful.add(1)
    if n_features > 2: # Use X3 (feature at index 2) additionally for Pop C
        y_pop_c += X_observed_all[idx_c_start:, 2]
        current_meaningful.add(2)
    Y_all[idx_c_start:] = y_pop_c + np.random.normal(0, noise_scale_y, n_c)

    meaningful_indices = np.array(sorted(list(set(current_meaningful)))) # Use set to avoid duplicates then sort
    pop_data = [
        {
            'pop_id': 'A',
            'X_raw' : X_observed_all[:idx_a_end, :],
            'Y_raw' : Y_all[:idx_a_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'B',
            'X_raw' : X_observed_all[idx_b_start:idx_b_end, :],
            'Y_raw' : Y_all[idx_b_start:idx_b_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'C',
            'X_raw' : X_observed_all[idx_c_start:, :],
            'Y_raw' : Y_all[idx_c_start:],
            'meaningful_indices': meaningful_indices
        }
    ]
    return pop_data

def generate_baseline_failure_5_irrelevant_differently_distributed_features(
    dataset_size: int = 10000,
    n_features: int = 5,
    noise_scale_y: float = 1.0,
    noise_feat_dist_means: dict = None,  # e.g. {'A': 0, 'B': 5, 'C': -2}
    corr_strength: float = 0.2, 
    seed: int = None
):
    """
    Scenario 5: Three Relevant, One Irrelevant. 
    Y ~ 2*X0 + 1.5*X2 + 1.0*X3 + noise. 
    X1 is irrelevant but shifts by population.
    """
    if n_features < 4:
        raise ValueError("Scenario 5 now requires at least 4 features (3 relevants + 1 irrelevant).")
    if seed is not None:
        np.random.seed(seed)

    # defaults for the irrelevant feature
    _noise_feat_dist_means = {'A': 0.0, 'B': 5.0, 'C': -2.0}
    if noise_feat_dist_means:
        _noise_feat_dist_means.update(noise_feat_dist_means)

    # indices: 0,2,3 relevant; 1 irrelevant
    idx_relevant1 = 0
    idx_noise_dist = 1
    idx_relevant2 = 2
    idx_relevant3 = 3
    current_meaningful = {idx_relevant1, idx_relevant2, idx_relevant3}

    n_a = dataset_size // 3
    n_b = dataset_size // 3
    n_c = dataset_size - n_a - n_b

    X_all = np.random.randn(dataset_size, n_features)
    Y_all = np.zeros(dataset_size)

    # Y-model using three relevants
    def model_y(X_slice):
        return (
            2.0 * X_slice[:, idx_relevant1]
            + 1.5 * X_slice[:, idx_relevant2]
            + 1.0 * X_slice[:, idx_relevant3]
            + np.random.normal(0, noise_scale_y, X_slice.shape[0])
        )

    # assign irrelevant feature by population and compute Y
    # Pop A
    end_a = n_a
    X_all[:end_a, idx_noise_dist] = np.random.normal(_noise_feat_dist_means['A'], 1, n_a)
    Y_all[:end_a] = model_y(X_all[:end_a, :])

    # Pop B
    start_b, end_b = end_a, end_a + n_b
    X_all[start_b:end_b, idx_noise_dist] = np.random.normal(_noise_feat_dist_means['B'], 1, n_b)
    Y_all[start_b:end_b] = model_y(X_all[start_b:end_b, :])

    # Pop C
    start_c = end_b
    X_all[start_c:, idx_noise_dist] = np.random.normal(_noise_feat_dist_means['C'], 1, n_c)
    Y_all[start_c:] = model_y(X_all[start_c:, :])

    meaningful_indices = np.array(sorted(current_meaningful))
    pop_data = [
        {'pop_id': 'A', 'X_raw': X_all[:end_a, :], 'Y_raw': Y_all[:end_a], 'meaningful_indices': meaningful_indices},
        {'pop_id': 'B', 'X_raw': X_all[start_b:end_b, :], 'Y_raw': Y_all[start_b:end_b], 'meaningful_indices': meaningful_indices},
        {'pop_id': 'C', 'X_raw': X_all[start_c:, :], 'Y_raw': Y_all[start_c:], 'meaningful_indices': meaningful_indices},
    ]
    return pop_data

def generate_baseline_failure_6_hierarchical_importance(dataset_size=10000, n_features=20, noise_scale=0.1, seed=None):
    """
    Scenario 6: Hierarchical Feature Importance.
    Pop A (50%): Y ~ 2*X1 + X2 + epsilon
    Pop B (30%): Y ~ X1 + X3 + 3*(X4*X5) + epsilon (interaction term!)
    Pop C (20%): Y ~ X1 + X6 + X7*(X7>0) + epsilon (conditional effect)
    
    Key challenge: X4 and X5 have weak marginal effects, but their interaction
    is critical. X7 only matters when positive.
    """
    if seed is not None:
        np.random.seed(seed)

    n_a = int(0.50 * dataset_size)
    n_b = int(0.30 * dataset_size)
    n_c = dataset_size - n_a - n_b

    X_all = np.random.randn(dataset_size, n_features)
    Y_all = np.zeros(dataset_size)

    # Pop A - simple linear relationship
    idx_a_end = n_a
    X_pop_a = X_all[:idx_a_end, :]
    Y_all[:idx_a_end] = 2 * X_pop_a[:, 0] + X_pop_a[:, 1] + np.random.normal(0, noise_scale, n_a)

    # Pop B - with interaction
    idx_b_start = n_a
    idx_b_end = n_a + n_b
    X_pop_b = X_all[idx_b_start:idx_b_end, :]
    # X4*X5 interaction term with limited main effects
    Y_all[idx_b_start:idx_b_end] = (X_pop_b[:, 0] + X_pop_b[:, 2] + 
                                     0.2 * X_pop_b[:, 3] +  # Small main effect
                                     0.2 * X_pop_b[:, 4] +  # Small main effect
                                     3 * X_pop_b[:, 3] * X_pop_b[:, 4] +  # Strong interaction
                                     np.random.normal(0, noise_scale, n_b))

    # Pop C - with conditional effect
    idx_c_start = n_a + n_b
    X_pop_c = X_all[idx_c_start:, :]
    # X7 only matters when positive
    Y_all[idx_c_start:] = (X_pop_c[:, 0] + X_pop_c[:, 5] + 
                           2 * X_pop_c[:, 6] * (X_pop_c[:, 6] > 0) + 
                           np.random.normal(0, noise_scale, n_c))

    # All meaningful indices
    meaningful_indices = np.array([0, 1, 2, 3, 4, 5, 6])

    pop_data = [
        {
            'pop_id': 'A',
            'X_raw': X_pop_a,
            'Y_raw': Y_all[:idx_a_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'B',
            'X_raw': X_pop_b,
            'Y_raw': Y_all[idx_b_start:idx_b_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'C',
            'X_raw': X_pop_c,
            'Y_raw': Y_all[idx_c_start:],
            'meaningful_indices': meaningful_indices
        }
    ]
    return pop_data

def generate_baseline_failure_7_dilution(dataset_size=10000, n_features=20, noise_scale=0.1, corr_strength=0.4, seed=None):
    """
    Scenario 7: Effect Dilution with Correlations.
    Each population has important features that are correlated with noise features.
    
    Pop A (40%): Y ~ 2*X1 + X2 + epsilon
                 X1 correlated with X10-X12 (noise variables)
    
    Pop B (40%): Y ~ X3 + 2*X4 + epsilon
                 X3 correlated with X13-X15 (noise variables)
    
    Pop C (20%): Y ~ 3*X5 + epsilon
                 X5 strongly correlated with X16-X19 (noise variables)
    
    Key challenge: Lasso may select noise variables that are correlated with
    the true variables, especially in Pop C where dilution is strongest.
    """
    if seed is not None:
        np.random.seed(seed)

    n_a = int(0.40 * dataset_size)
    n_b = int(0.40 * dataset_size)
    n_c = dataset_size - n_a - n_b

    # Start with random features
    X_all = np.random.randn(dataset_size, n_features)
    Y_all = np.zeros(dataset_size)

    # Pop A
    idx_a_end = n_a
    # Create correlations between X1 and noise vars X10-X12
    for i in range(10, 13):
        X_all[:idx_a_end, i] = corr_strength * X_all[:idx_a_end, 0] + (1-corr_strength) * X_all[:idx_a_end, i]
    Y_all[:idx_a_end] = 2 * X_all[:idx_a_end, 0] + X_all[:idx_a_end, 1] + np.random.normal(0, noise_scale, n_a)

    # Pop B
    idx_b_start = n_a
    idx_b_end = n_a + n_b
    # Create correlations between X3 and noise vars X13-X15
    for i in range(13, 16):
        X_all[idx_b_start:idx_b_end, i] = corr_strength * X_all[idx_b_start:idx_b_end, 2] + (1-corr_strength) * X_all[idx_b_start:idx_b_end, i]
    Y_all[idx_b_start:idx_b_end] = X_all[idx_b_start:idx_b_end, 2] + 2 * X_all[idx_b_start:idx_b_end, 3] + np.random.normal(0, noise_scale, n_b)

    # Pop C
    idx_c_start = n_a + n_b
    # Create stronger correlations between X5 and noise vars X16-X19
    stronger_corr = min(0.8, corr_strength * 1.5)  # Increase correlation strength
    for i in range(16, 20):
        X_all[idx_c_start:, i] = stronger_corr * X_all[idx_c_start:, 4] + (1-stronger_corr) * X_all[idx_c_start:, i]
    Y_all[idx_c_start:] = 3 * X_all[idx_c_start:, 4] + np.random.normal(0, noise_scale, n_c)

    meaningful_indices = np.array([0, 1, 2, 3, 4])
    
    pop_data = [
        {
            'pop_id': 'A',
            'X_raw': X_all[:idx_a_end, :],
            'Y_raw': Y_all[:idx_a_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'B',
            'X_raw': X_all[idx_b_start:idx_b_end, :],
            'Y_raw': Y_all[idx_b_start:idx_b_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'C',
            'X_raw': X_all[idx_c_start:, :],
            'Y_raw': Y_all[idx_c_start:],
            'meaningful_indices': meaningful_indices
        }
    ]
    return pop_data

def generate_baseline_failure_8_signal_to_noise_evolution(dataset_size=10000, n_features=20, noise_scale=0.1, seed=None):
    """
    Scenario 8: Signal-to-Noise Evolution.
    Feature importance changes based on a contextual variable Z.
    
    Pop A (40%): Z ~ Uniform(0, 0.33)
                 Y ~ (3*Z)*X1 + (1-Z)*X2 + epsilon
    
    Pop B (40%): Z ~ Uniform(0.33, 0.67)
                 Y ~ (2-2*Z)*X1 + (3*Z-1)*X3 + epsilon
    
    Pop C (20%): Z ~ Uniform(0.67, 1.0)
                 Y ~ (1-Z)*X3 + (3*Z-2)*X4 + epsilon
                 
    Key challenge: As Z increases, importance shifts from X1→X2→X3→X4.
    Methods that don't account for this evolution will miss important features.
    """
    if seed is not None:
        np.random.seed(seed)

    n_a = int(0.40 * dataset_size)
    n_b = int(0.40 * dataset_size)
    n_c = dataset_size - n_a - n_b

    X_all = np.random.randn(dataset_size, n_features)
    Y_all = np.zeros(dataset_size)
    
    # Pop A: Z in [0, 0.33] - X1 importance decreases, X2 increases
    idx_a_end = n_a
    Z_a = np.random.uniform(0, 0.33, n_a)
    Y_all[:idx_a_end] = (3*Z_a)*X_all[:idx_a_end, 0] + (1-Z_a)*X_all[:idx_a_end, 1] + np.random.normal(0, noise_scale, n_a)
    
    # Pop B: Z in [0.33, 0.67] - X1 continues decreasing, X3 increases
    idx_b_start = n_a
    idx_b_end = n_a + n_b
    Z_b = np.random.uniform(0.33, 0.67, n_b)
    Y_all[idx_b_start:idx_b_end] = (2-2*Z_b)*X_all[idx_b_start:idx_b_end, 0] + (3*Z_b-1)*X_all[idx_b_start:idx_b_end, 2] + np.random.normal(0, noise_scale, n_b)
    
    # Pop C: Z in [0.67, 1.0] - X3 decreases, X4 increases
    idx_c_start = n_a + n_b
    Z_c = np.random.uniform(0.67, 1.0, n_c)
    Y_all[idx_c_start:] = (1-Z_c)*X_all[idx_c_start:, 2] + (3*Z_c-2)*X_all[idx_c_start:, 3] + np.random.normal(0, noise_scale, n_c)

    meaningful_indices = np.array([0, 1, 2, 3])
    
    pop_data = [
        {
            'pop_id': 'A',
            'X_raw': X_all[:idx_a_end, :],
            'Y_raw': Y_all[:idx_a_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'B',
            'X_raw': X_all[idx_b_start:idx_b_end, :],
            'Y_raw': Y_all[idx_b_start:idx_b_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'C',
            'X_raw': X_all[idx_c_start:, :],
            'Y_raw': Y_all[idx_c_start:],
            'meaningful_indices': meaningful_indices
        }
    ]
    return pop_data

def generate_baseline_failure_9_multiscale_relevance(dataset_size=10000, n_features=20, noise_scale=0.1, seed=None):
    """
    Scenario 9: Multiscale Relevance.
    Features have effects at different scales or granularities.
    
    Pop A (50%): Y ~ 2*X1 + small_scale_pattern(X2) + epsilon
                 where small_scale_pattern(x) = sin(10*x)
    
    Pop B (30%): Y ~ medium_scale_pattern(X1) + 2*X3 + epsilon
                 where medium_scale_pattern(x) = sin(3*x)
    
    Pop C (20%): Y ~ large_scale_pattern(X1) + large_scale_pattern(X4) + epsilon
                 where large_scale_pattern(x) = sin(x)
    
    Key challenge: Different frequencies of patterns make features contribute
    differently across populations. Linear methods may miss the multiscale nature.
    """
    if seed is not None:
        np.random.seed(seed)

    n_a = int(0.50 * dataset_size)
    n_b = int(0.30 * dataset_size)
    n_c = dataset_size - n_a - n_b

    X_all = np.random.randn(dataset_size, n_features)
    Y_all = np.zeros(dataset_size)
    
    # Pop A: Small scale pattern (high frequency) on X2
    idx_a_end = n_a
    Y_all[:idx_a_end] = 2*X_all[:idx_a_end, 0] + np.sin(10*X_all[:idx_a_end, 1]) + np.random.normal(0, noise_scale, n_a)
    
    # Pop B: Medium scale pattern on X1
    idx_b_start = n_a
    idx_b_end = n_a + n_b
    Y_all[idx_b_start:idx_b_end] = np.sin(3*X_all[idx_b_start:idx_b_end, 0]) + 2*X_all[idx_b_start:idx_b_end, 2] + np.random.normal(0, noise_scale, n_b)
    
    # Pop C: Large scale patterns on X1 and X4
    idx_c_start = n_a + n_b
    Y_all[idx_c_start:] = np.sin(X_all[idx_c_start:, 0]) + np.sin(X_all[idx_c_start:, 3]) + np.random.normal(0, noise_scale, n_c)

    meaningful_indices = np.array([0, 1, 2, 3])
    
    pop_data = [
        {
            'pop_id': 'A',
            'X_raw': X_all[:idx_a_end, :],
            'Y_raw': Y_all[:idx_a_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'B',
            'X_raw': X_all[idx_b_start:idx_b_end, :],
            'Y_raw': Y_all[idx_b_start:idx_b_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'C',
            'X_raw': X_all[idx_c_start:, :],
            'Y_raw': Y_all[idx_c_start:],
            'meaningful_indices': meaningful_indices
        }
    ]
    return pop_data

def generate_baseline_failure_10_threshold_effects(dataset_size=10000, n_features=20, noise_scale=0.1, seed=None):
    """
    Scenario 10: Threshold Effects with Redundancy.
    Features have effects only beyond certain thresholds with some redundancy.
    
    Pop A (40%): Y ~ X1 + X2*(X2>0.5) + epsilon
                 
    Pop B (40%): Y ~ X1 + X3*(X3>1.0) + epsilon
                 X4 redundant with X3 (correlated)
    
    Pop C (20%): Y ~ X1 + X5*(X5>1.5) + epsilon
                 X6, X7 redundant with X5 (correlated)
    
    Key challenge: Threshold effects create non-linearity, while
    redundant features create selection ambiguity.
    """
    if seed is not None:
        np.random.seed(seed)

    n_a = int(0.40 * dataset_size)
    n_b = int(0.40 * dataset_size)
    n_c = dataset_size - n_a - n_b

    X_all = np.random.randn(dataset_size, n_features)
    Y_all = np.zeros(dataset_size)
    
    # Pop A: Simple threshold effect
    idx_a_end = n_a
    Y_all[:idx_a_end] = X_all[:idx_a_end, 0] + X_all[:idx_a_end, 1] * (X_all[:idx_a_end, 1] > 0.5) + np.random.normal(0, noise_scale, n_a)
    
    # Pop B: Higher threshold with one redundant feature
    idx_b_start = n_a
    idx_b_end = n_a + n_b
    # X4 is redundant with X3
    X_all[idx_b_start:idx_b_end, 3] = 0.7 * X_all[idx_b_start:idx_b_end, 2] + 0.3 * X_all[idx_b_start:idx_b_end, 3]
    Y_all[idx_b_start:idx_b_end] = X_all[idx_b_start:idx_b_end, 0] + X_all[idx_b_start:idx_b_end, 2] * (X_all[idx_b_start:idx_b_end, 2] > 1.0) + np.random.normal(0, noise_scale, n_b)
    
    # Pop C: Highest threshold with two redundant features
    idx_c_start = n_a + n_b
    # X6, X7 are redundant with X5
    X_all[idx_c_start:, 5] = 0.8 * X_all[idx_c_start:, 4] + 0.2 * X_all[idx_c_start:, 5]
    X_all[idx_c_start:, 6] = 0.6 * X_all[idx_c_start:, 4] + 0.4 * X_all[idx_c_start:, 6]
    Y_all[idx_c_start:] = X_all[idx_c_start:, 0] + X_all[idx_c_start:, 4] * (X_all[idx_c_start:, 4] > 1.5) + np.random.normal(0, noise_scale, n_c)

    meaningful_indices = np.array([0, 1, 2, 4])  # Note: redundant features aren't "meaningful"
    
    pop_data = [
        {
            'pop_id': 'A',
            'X_raw': X_all[:idx_a_end, :],
            'Y_raw': Y_all[:idx_a_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'B',
            'X_raw': X_all[idx_b_start:idx_b_end, :],
            'Y_raw': Y_all[idx_b_start:idx_b_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'C',
            'X_raw': X_all[idx_c_start:, :],
            'Y_raw': Y_all[idx_c_start:],
            'meaningful_indices': meaningful_indices
        }
    ]
    return pop_data

def get_pop_data_baseline_failures(
    pop_configs: list,
    dataset_size: int = 10000,
    n_features: int = 5,
    noise_scale: float = 0.1,
    corr_strength: float = 0.0,
    estimator_type: str = 'plugin',
    device: str = 'cpu',
    base_model_type: str = 'rf',
    seed: int = None
):
    """
    Generate data for multiple populations based on the provided configurations.
    Each configuration should be a dictionary with the following keys:
    - 'pop_id': Unique identifier for the population
    - 'dataset_type': Type of dataset to generate (e.g., 'baseline_failure_1', 'baseline_failure_2', etc.)
    -'X_std': Standardized X
    - 'Y_std': Standardized Y
    'E_Yx_std'
    'term1_std'
    'meaningful_indices': List of indices of meaningful features
    """

    baseline_type = pop_configs[0]['dataset_type']
    
    if baseline_type == 'baseline_failure_1':
        pop_data = generate_baseline_failure_1_heterogeneous_importance(
            dataset_size=dataset_size,
            n_features=n_features,
            noise_scale=noise_scale,
            corr_strength=corr_strength,
            seed=seed
        )
    elif baseline_type == 'baseline_failure_2':
        pop_data = generate_baseline_failure_2_opposing_effects(
            dataset_size=dataset_size,
            n_features=n_features,
            noise_scale=noise_scale,
            seed=seed
        )
    elif baseline_type == 'baseline_failure_3':
        pop_data = generate_baseline_failure_3_non_linearity_subgroup(
            dataset_size=dataset_size,
            n_features=n_features,
            noise_scale=noise_scale,
            seed=seed
        )
    elif baseline_type == 'baseline_failure_4':
        pop_data = generate_baseline_failure_4_different_noise_structures_features(
            dataset_size=dataset_size,
            n_features=n_features,
            noise_scale_y=noise_scale,
            x1_noise_stds=pop_configs[0].get('x1_noise_stds', None),
            seed=seed
        )
    elif baseline_type == 'baseline_failure_5':
        pop_data = generate_baseline_failure_5_irrelevant_differently_distributed_features(
            dataset_size=dataset_size,
            n_features=n_features,
            noise_scale_y=noise_scale,
            noise_feat_dist_means=pop_configs[0].get('noise_feat_dist_means', None),
            seed=seed
        )
    elif baseline_type == 'baseline_failure_6':
        pop_data = generate_baseline_failure_6_hierarchical_importance(
            dataset_size=dataset_size,
            n_features=n_features,
            noise_scale=noise_scale,
            seed=seed
        )
    elif baseline_type == 'baseline_failure_7':
        pop_data = generate_baseline_failure_7_dilution(
            dataset_size=dataset_size,
            n_features=n_features,
            noise_scale=noise_scale,
            corr_strength=corr_strength,
            seed=seed
        )
    elif baseline_type == 'baseline_failure_8':
        pop_data = generate_baseline_failure_8_signal_to_noise_evolution(
            dataset_size=dataset_size,
            n_features=n_features,
            noise_scale=noise_scale,
            seed=seed
        )
    elif baseline_type == 'baseline_failure_9':
        pop_data = generate_baseline_failure_9_multiscale_relevance(
            dataset_size=dataset_size,
            n_features=n_features,
            noise_scale=noise_scale,
            seed=seed
        )
    elif baseline_type == 'baseline_failure_10':
        pop_data = generate_baseline_failure_10_threshold_effects(
            dataset_size=dataset_size,
            n_features=n_features,
            noise_scale=noise_scale,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown dataset type: {baseline_type}")
    
    final_pop_data = [] 
    for pop in pop_data:
        # --- Standardize Data ---
        X_std_np, Y_std_np, _, _, Y_mean, Y_std = standardize_data(pop['X_raw'], pop['Y_raw'])
        print(f"Population {pop['pop_id']}: Precomputing E[Y|X] ({estimator_type}/{base_model_type})...")
        # --- Precompute E[Y|X] using ORIGINAL Y scale ---
        try:
            if estimator_type == "plugin":
                E_Yx_orig_np = plugin_estimator_conditional_mean(pop['X_raw'], pop['Y_raw'],
                                                                 base_model_type, n_folds=N_FOLDS)
            elif estimator_type == "if":
                E_Yx_orig_np = IF_estimator_conditional_mean(pop['X_raw'], pop['Y_raw'], base_model_type, n_folds=N_FOLDS)
            else:
                raise ValueError("estimator_type must be 'plugin' or 'if'")
        except Exception as e:
            print(f"ERROR: Failed to precompute E[Y|X] for pop {pop['pop_id']}: {e}")
            continue

        # --- Standardize the E[Y|X] estimate ---
        E_Yx_std_np = (E_Yx_orig_np - Y_mean) / Y_std

        # --- Calculate Term 1 based on the STANDARDIZED estimate ---
        term1_std = np.mean(E_Yx_std_np ** 2)
        print(f"Population {pop['pop_id']}: Precomputed Term1_std = {term1_std:.4f}")

        # --- Convert to Tensors ---
        X_std_torch = torch.tensor(X_std_np, dtype=torch.float32).to(device)
        Y_std_torch = torch.tensor(Y_std_np, dtype=torch.float32).to(device)
        E_Yx_std_torch = torch.tensor(E_Yx_std_np, dtype=torch.float32).to(device)

        final_pop_data.append({
            'pop_id': pop['pop_id'],
            'X_std': X_std_torch,
            'Y_std': Y_std_torch,
            'E_Yx_std': E_Yx_std_torch,
            'term1_std': term1_std,
            'meaningful_indices': pop['meaningful_indices'],
            'X_raw': pop['X_raw'],
            'Y_raw': pop['Y_raw']
        })
    return final_pop_data
        
