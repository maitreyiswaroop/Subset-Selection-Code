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
    n_features: int = 5,
    noise_scale: float = 0.1,
    corr_strength: float = 0.2, 
    seed: int = None
):
    """
    Scenario 1: Heterogeneous Feature Importance.
    Pools three internal populations (A, B, C).
    Pop A (Large, 45%): Y ~ 2*X1 + 0.5*X2 + epsilon
    Pop B (Large, 45%): Y ~ 2*X1 + 0.6*X2 + epsilon
    Pop C (Small but Critical, 10%): Y ~ 0.1*X1 + 3*X3 + epsilon
    X4, X5 are correlated with X1 and X2.
    """
    if n_features < 3:
        raise ValueError("Scenario 1 requires at least 3 features (X1, X2, X3).")
    if seed is not None:
        np.random.seed(seed)

    n_a = int(0.45 * dataset_size)
    n_b = int(0.45 * dataset_size)
    n_c = dataset_size - n_a - n_b

    # X4, X5 are correlated with X1, X2
    # X_all = np.random.randn(dataset_size, n_features)
    # Y_all = np.zeros(dataset_size)

    # Generate correlated features
    X_all = np.random.randn(dataset_size, n_features)
    # # Correlate X1 and X2 with X4
    # if n_features > 3:
    #     X_all[:, 3] = X_all[:, 0] + corr_strength * X_all[:, 1]
    # # Correlate X1 and X2 with X5
    # if n_features > 4:
    #     X_all[:, 4] = X_all[:, 0] + corr_strength * X_all[:, 1]
    Y_all = np.zeros(dataset_size)


    # Pop A
    idx_a_end = n_a
    X_pop_a = X_all[:idx_a_end, :]
    # Create X4 as a linear combination of X1 and X2 with added noise for better correlation
    if n_features > 3:
        X_pop_a[:, 3] = corr_strength * X_pop_a[:, 0] + corr_strength * X_pop_a[:, 1] + (1- 2*corr_strength) * np.random.normal(0, 0.01, n_a)
    # Create X5 as a linear combination of X1 and X2 with added noise for better correlation
    if n_features > 4:
        X_pop_a[:, 4] = corr_strength * X_pop_a[:, 0] + corr_strength * X_pop_a[:, 1] + (1- 2*corr_strength) * np.random.normal(0, 0.01, n_a)
    Y_all[:idx_a_end] = 2 * X_pop_a[:, 0] + 1.0 * X_pop_a[:, 1] + np.random.normal(0, noise_scale, n_a)

    # Pop B
    idx_b_start = n_a
    idx_b_end = n_a + n_b
    X_pop_b = X_all[idx_b_start:idx_b_end, :]
    # correlate X1 and X2 with X4
    if n_features > 3:
        X_pop_b[:, 3] = corr_strength * X_pop_b[:, 0] + corr_strength * X_pop_b[:, 1] + (1- 2*corr_strength) * np.random.normal(0, 0.01, n_b)
    # correlate X1 and X2 with X5
    if n_features > 4:
        X_pop_b[:, 4] = corr_strength * X_pop_b[:, 0] + corr_strength * X_pop_b[:, 1] + (1- 2*corr_strength) * np.random.normal(0, 0.01, n_b)
    Y_all[idx_b_start:idx_b_end] = 1.0 * X_pop_b[:, 0] + 2.0 * X_pop_b[:, 1] + np.random.normal(0, noise_scale, n_b)

    # Pop C
    idx_c_start = n_a + n_b
    X_pop_c = X_all[idx_c_start:, :]
    Y_all[idx_c_start:] = 0.1 * X_pop_c[:, 0] + 0.5* X_pop_c[:, 1] + 3.0 * X_pop_c[:, 2] + np.random.normal(0, noise_scale, n_c)

    meaningful_indices = np.array(sorted(list(set([0, 1, 2])))) # X1, X2, X3

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
    noise_feat_dist_means: dict = None, # e.g. {'A': 0, 'B': 5, 'C': -2} for the irrelevant X_noise_dist
    corr_strength: float = 0.2, 
    seed: int = None
):
    """
    Scenario 5: Irrelevant but Differently Distributed Features. Y ~ 2*X_relevant1 + 1.5*X_relevant2 + eps_y
    X_noise_dist (X_idx_noise_dist) has different means in Pop A, B, C, but is irrelevant to Y.
    Pop A (33%): X_noise_dist mean m_A
    Pop B (33%): X_noise_dist mean m_B
    Pop C (34%): X_noise_dist mean m_C
    X_idx_relevant1 (0) and X_idx_relevant2 (2) are the true predictors.
    """
    if n_features < 3: # Needs X_relevant1 (0), X_noise_dist (1), X_relevant2 (2)
        raise ValueError("Scenario 5 requires at least 3 features for this setup.")
    if seed is not None:
        np.random.seed(seed)

    _noise_feat_dist_means = {'A': 0.0, 'B': 5.0, 'C': -2.0} # Defaults for the irrelevant feature
    if noise_feat_dist_means:
        _noise_feat_dist_means.update(noise_feat_dist_means)

    idx_relevant1 = 0
    idx_noise_dist = 1 # This feature is irrelevant to Y but has different distributions
    idx_relevant2 = 2
    current_meaningful = {idx_relevant1, idx_relevant2}

    n_a = dataset_size // 3
    n_b = dataset_size // 3
    n_c = dataset_size - n_a - n_b

    X_all = np.random.randn(dataset_size, n_features)
    Y_all = np.zeros(dataset_size)

    # Define Y relationship (same for all pops, based on relevant features)
    def model_y(X_slice_pop):
        return (2 * X_slice_pop[:, idx_relevant1] +
                1.5 * X_slice_pop[:, idx_relevant2] +
                np.random.normal(0, noise_scale_y, X_slice_pop.shape[0]))

    # Pop A
    idx_a_end = n_a
    X_all[:idx_a_end, idx_noise_dist] = np.random.normal(_noise_feat_dist_means['A'], 1, n_a)
    Y_all[:idx_a_end] = model_y(X_all[:idx_a_end, :])

    # Pop B
    idx_b_start = n_a
    idx_b_end = n_a + n_b
    X_all[idx_b_start:idx_b_end, idx_noise_dist] = np.random.normal(_noise_feat_dist_means['B'], 1, n_b)
    Y_all[idx_b_start:idx_b_end] = model_y(X_all[idx_b_start:idx_b_end, :])

    # Pop C
    idx_c_start = n_a + n_b
    X_all[idx_c_start:, idx_noise_dist] = np.random.normal(_noise_feat_dist_means['C'], 1, n_c)
    Y_all[idx_c_start:] = model_y(X_all[idx_c_start:, :])

    meaningful_indices = np.array(sorted(list(current_meaningful)))
    pop_data = [
        {
            'pop_id': 'A',
            'X_raw' : X_all[:idx_a_end, :],
            'Y_raw' : Y_all[:idx_a_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'B',
            'X_raw' : X_all[idx_b_start:idx_b_end, :],
            'Y_raw' : Y_all[idx_b_start:idx_b_end],
            'meaningful_indices': meaningful_indices
        },
        {
            'pop_id': 'C',
            'X_raw' : X_all[idx_c_start:, :],
            'Y_raw' : Y_all[idx_c_start:],
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
        
