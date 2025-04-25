# if_vs_plugin.py: compares IF vs plugin estimator for E[E[Y|X]^2] - E[E[Y|S]^2]
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
import os
import argparse
from scipy.special import softmax
from sklearn.neighbors import BallTree
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- Configuration ---
# ALPHA_MAX = 10.0 # Default Max Variance
# ALPHA_MAX_STR = str(ALPHA_MAX).replace('.', '') if ALPHA_MAX < 1 else str(int(ALPHA_MAX)) # Not used dynamically anymore

# --- Estimator Functions (Unchanged unless noted) ---

def plugin_estimator(X, Y, estimator_type="rf"):
    """
    Plugin estimator for E[Y|X] using either random forest or kernel regression.
    Returns a prediction function that, when given new X, returns estimated E[Y|X].
    """
    if estimator_type == "rf":
        model = RandomForestRegressor(n_estimators=100,
                                      min_samples_leaf=5,
                                      n_jobs=-1,
                                      random_state=42)
    else:
        model = KernelRidge(kernel='rbf')

    model.fit(X, Y)
    return model.predict

def plugin_estimator_squared_conditional(X, Y, estimator_type="rf", n_folds=5):
    """
    Plugin estimator for E[E[Y|X]^2] with optional K-fold cross-validation.

    Returns the direct estimate (a scalar value) of E[E[Y|X]^2] without returning a model.

    Parameters:
    -----------
    X : numpy.ndarray
        Input features, shape (n_samples, n_features)
    Y : numpy.ndarray
        Target values, shape (n_samples,)
    estimator_type : str, optional
        Type of base estimator to use ('rf' for Random Forest, 'krr' for Kernel Ridge)
    n_folds : int, optional
        Number of folds for cross-validation. If 1, uses no CV.

    Returns:
    --------
    float
        Estimate of E[E[Y|X]^2]
    """
    # No cross-validation
    if n_folds == 1:
        if estimator_type == "rf":
            model = RandomForestRegressor(n_estimators=100,
                                          min_samples_leaf=5,
                                          n_jobs=-1,
                                          random_state=42)
        else:
            model = KernelRidge(kernel='rbf')

        model.fit(X, Y)
        mu_X = model.predict(X)
        plugin_estimate = np.mean(mu_X ** 2)
        return plugin_estimate

    # K-fold cross-validation
    else:
        n_samples = X.shape[0]
        mu_X_all = np.zeros(n_samples)

        # Create the KFold object
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Iterate through folds
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train = Y[train_idx]

            # Choose and fit the model on the training data
            if estimator_type == "rf":
                model = RandomForestRegressor(n_estimators=100,
                                              min_samples_leaf=5,
                                              n_jobs=-1,
                                              random_state=42)
            else:
                model = KernelRidge(kernel='rbf')

            model.fit(X_train, Y_train)

            # Make predictions on test data
            mu_X_all[test_idx] = model.predict(X_test)

        # Compute E[mu(X)^2] using out-of-fold predictions
        plugin_estimate = np.mean(mu_X_all ** 2)
        return plugin_estimate

def IF_estimator_squared_conditional(X, Y, estimator_type="rf", n_folds=5):
    """
    Influence function-based estimator for E[E[Y|X]^2] with optional K-fold cross-validation.

    When n_folds=1 (default), uses the original implementation fitting on the entire dataset.
    When n_folds>1, implements K-fold cross-validation to compute out-of-fold predictions
    for the influence function correction, which can reduce bias.

    Parameters:
    -----------
    X : numpy.ndarray
        Input features, shape (n_samples, n_features)
    Y : numpy.ndarray
        Target values, shape (n_samples,)
    estimator_type : str, optional
        Type of base estimator to use ('rf' for Random Forest, 'krr' for Kernel Ridge)
    n_folds : int, optional
        Number of folds for cross-validation. If 1, uses the original implementation.

    Returns:
    --------
    float
        The bias-corrected estimate of E[E[Y|X]^2]
    """
    n_samples = X.shape[0]

    # Original implementation (no CV)
    if n_folds == 1:
        # Choose the model and parameters
        if estimator_type == "rf":
            model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42)
        else:
            model = KernelRidge(kernel='rbf')

        # Fit the model on the full dataset to estimate mu(X) = E[Y|X]
        model.fit(X, Y)

        # Compute the estimated conditional mean for all observations
        mu_X = model.predict(X)

        # Plugin estimator: estimate of E[mu(X)^2]
        plugin_estimate = np.mean(mu_X ** 2)

        # Compute residuals for the correction term
        residuals = Y - mu_X

        # Influence function correction term: 2 * E[(Y - mu(X)) * mu(X)]
        correction_term = 2 * np.mean(residuals * mu_X)

        # IF-corrected estimate
        if_estimate = plugin_estimate + correction_term

        return if_estimate

    # K-fold CV implementation
    else:
        # Create arrays to store predictions and residuals
        mu_X_all = np.zeros(n_samples)
        plugin_terms = np.zeros(n_folds)
        correction_terms = np.zeros(n_folds)

        # Create the KFold object
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Iterate through folds
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # Choose and fit the model on the training data
            if estimator_type == "rf":
                model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42)
            else:
                model = KernelRidge(kernel='rbf')

            model.fit(X_train, Y_train)

            # Make predictions on test data
            mu_X_test = model.predict(X_test)

            # Store predictions for later use
            mu_X_all[test_idx] = mu_X_test

            # Compute plugin term for this fold
            plugin_terms[fold_idx] = np.mean(mu_X_test ** 2)

            # Compute residuals and correction term for this fold
            residuals_test = Y_test - mu_X_test
            correction_terms[fold_idx] = 2 * np.mean(residuals_test * mu_X_test)

        # Compute the final estimate as the average across folds
        plugin_estimate = np.mean(plugin_terms)
        correction_term = np.mean(correction_terms)
        if_estimate = plugin_estimate + correction_term

        return if_estimate

def closed_form_objective(alpha, A):
    """
    Calculates the true objective T = E[V[E[Y|X]|S]] = sum(A_i^2 * var_i / (1 + var_i)).

    Parameters:
    -----------
    alpha : numpy.ndarray
        Vector of noise alpha (alpha_i in the formula if alpha represents variance).
    A : numpy.ndarray
        Vector of true coefficients.

    Returns:
    --------
    float
        The value of the objective functional T.
    """
    # Ensure alpha are non-negative before calculation
    safe_alpha = np.maximum(alpha, 0)
    return np.sum((A**2 * safe_alpha) / (1 + safe_alpha))

def estimate_conditional_with_knn(
    X_batch, S_batch, E_Y_given_X_batch, alpha, k=50,
    CLAMP_MIN=1e-4, bw_scale=1.0
):
    """Estimates E[ E_Y_given_X | S=s ] using KNN on variance-scaled features."""
    # alpha: vector of noise alpha for S features
    # scale features by 1 / sqrt(variance)
    var_safe = np.maximum(alpha, CLAMP_MIN) * bw_scale
    inv_sqrt_var = 1.0 / np.sqrt(var_safe)
    X_scaled = X_batch * inv_sqrt_var   # (n_train, d)
    S_scaled = S_batch * inv_sqrt_var   # (n_test,  d)

    tree = BallTree(X_scaled, leaf_size=40)
    dist, ind = tree.query(S_scaled, k=k)      # (n_test, k)
    # Gaussian weights
    logW = -0.5 * dist**2                     # (n_test, k)
    W = softmax(logW, axis=1)                 # (n_test, k)
    # gather E_Y for neighbors and do weighted average
    neighbor_preds = E_Y_given_X_batch[ind]    # (n_test, k)
    return (W * neighbor_preds).sum(axis=1)    # (n_test,)

def estimate_conditional_keops(
    X: Tensor,           # (n_train,d)
    S: Tensor,           # (n_test, d)
    E_Y_X: Tensor,       # (n_train,) - Estimated E[Y|X] values on training data
    alpha: Tensor,   # (d,) - Vector of noise alpha for S features
    clamp_min=1e-4
):
    """Estimates E[ E_Y_X | S=s ] using KeOps (or BallTree fallback) on variance-scaled features."""
    # alpha: vector of noise alpha for S features
    # 1) clamp & per‐dim inv sqrt of variance
    var_clamped = torch.clamp(alpha, min=clamp_min) # Use clamp_min for variance
    inv_sqrt_var_t = torch.rsqrt(var_clamped)[None, None, :] # (1,1,d) = 1/sqrt(variance)
    # 2) scale into Mahalanobis space
    Xs = X[None, :, :] * inv_sqrt_var_t             # (1,n_train,d)
    Ss = S[:, None, :] * inv_sqrt_var_t             # (n_test,1,d)

    n_train_curr = X.shape[0] # Use current X batch size

    # --- Fallback to BallTree logic (unchanged math, uses sqrt(variance)) ---
    if n_train_curr > 10000:
        # using the ball tree method for large datasets;
        # Ensure tensors are on CPU before converting to numpy if they might be on GPU
        X_scaled = Xs.squeeze(0).cpu().numpy()
        S_scaled = Ss.squeeze(1).cpu().numpy()
        k = min(10000, X_scaled.shape[0])
        tree = BallTree(X_scaled, leaf_size=40)
        dist, ind = tree.query(S_scaled, k=k)    # (n_test, k)
        # Gaussian weights
        logW_np = -0.5 * dist**2
        # Convert logW back to a torch tensor
        logW = torch.from_numpy(logW_np).to(X.device) # Use .to(X.device) for consistency
        W = torch.softmax(logW, axis=1)                     # (n_test, k)
        # gather E_Y for neighbors and do weighted average
        neighbor_preds = E_Y_X[ind]                   # (n_test, k)
        return (W * neighbor_preds).sum(axis=1)       # (n_test,)
    # --- KeOps / PyTorch logic (unchanged math, uses sqrt(variance)) ---
    else:
        # 3) pairwise squared distances
        D2 = torch.sum((Ss - Xs)**2, dim=-1)       # (n_test,n_train)
        # 4) Gaussian weights
        logW = -0.5 * D2                            # (n_test,n_train)
        W = torch.softmax(logW, axis=1)               # (n_test,n_train)
        # 5) weighted sum
        return (W @ E_Y_X) # (n_test,)

# def estimate_conditional_keops(
#     X: Tensor,           # (n_train,d) - Training features (original)
#     S: Tensor,           # (n_test, d) - Test features (noisy)
#     E_Y_X: Tensor,       # (n_train,) - Estimated E[Y|X] values for training data
#     alpha: Tensor,   # (d,) - Vector of noise alpha linking X to S
#     clamp_min=1e-4,
#     k_neighbors=10000    # Max number of neighbors to consider (like in original code)
# ):
#     """
#     Estimates E[ E_Y_X | S=s ] using PyTorch kernel regression (Nadaraya-Watson)
#     on variance-scaled features. Uses top-k neighbors for large datasets.
#     DIFFERENTIABLE w.r.t. alpha.

#     Args:
#         X: Training features.
#         S: Test features.
#         E_Y_X: Target values associated with X.
#         alpha: Noise alpha linking X to S.
#         clamp_min: Minimum value for variance clamping.
#         k_neighbors: Max number of neighbors to use for the weighted sum.

#     Returns:
#         Tensor: Estimated values of E[ E_Y_X | S=s ] for each s in S.
#     """
#     n_train = X.shape[0]
#     n_test = S.shape[0]
#     dev = X.device # Ensure all tensors are on the same device

#     # --- 1. Scaling (remains the same) ---
#     var_clamped = torch.clamp(alpha, min=clamp_min)
#     inv_sqrt_var_t = torch.rsqrt(var_clamped)[None, :] # Shape (1, d)
#     # Apply scaling - ensure inputs are float
#     X_scaled = X.float() * inv_sqrt_var_t # Shape (n_train, d)
#     S_scaled = S.float() * inv_sqrt_var_t # Shape (n_test, d)

#     # --- 2. Find K-Nearest Neighbors and Distances using PyTorch ---
#     # Determine the actual number of neighbors (k) to use
#     # k = min(k_neighbors, n_train) # Original code had k=min(10000, X_scaled.shape[0])
#     k = min(k_neighbors if k_neighbors > 0 else n_train, n_train) # Allow k_neighbors=-1 for all

#     # Calculate pairwise distances (efficiently)
#     # torch.cdist computes p-norm distance, p=2 is Euclidean
#     # Input shapes: (n_test, d), (n_train, d) -> Output shape: (n_test, n_train)
#     dists_sq = torch.cdist(S_scaled, X_scaled, p=2).pow(2) # Get squared Euclidean distances

#     # Find the k nearest neighbors (smallest distances) for each test point
#     # Using topk on negative squared distances is like finding k smallest
#     # Alternatively, use torch.sort (might be clearer)
#     # topk_dists_sq, topk_indices = torch.topk(-dists_sq, k=k, dim=1)
#     # topk_dists_sq = -topk_dists_sq # Make distances positive again
#     # --- OR use sort ---
#     sorted_dists_sq, sorted_indices = torch.sort(dists_sq, dim=1)
#     topk_dists_sq = sorted_dists_sq[:, :k]     # Shape: (n_test, k)
#     topk_indices = sorted_indices[:, :k]       # Shape: (n_test, k)

#     # --- 3. Calculate Weights for Neighbors ---
#     # Gaussian kernel based on squared distances
#     # Ensure distances are non-negative before log for stability if needed, though sq dists are fine
#     logW = -0.5 * topk_dists_sq          # Shape: (n_test, k)

#     # Normalize weights using softmax (operates on the k neighbors for each test point)
#     W = torch.softmax(logW, dim=1)       # Shape: (n_test, k)

#     # --- 4. Weighted Sum ---
#     # Gather the E_Y_X values for the neighbors
#     # Need to index E_Y_X using topk_indices. E_Y_X has shape (n_train,)
#     # We can use torch.gather or direct indexing if careful with shapes
#     # E_Y_X is (n_train,), W is (n_test, k), topk_indices is (n_test, k)
#     # Expand E_Y_X to match the gather operation if needed, or use advanced indexing
#     # Let's try direct indexing (simpler):
#     # For each test point i, we need E_Y_X[topk_indices[i, :]]
#     try:
#          # Newer PyTorch versions support direct tensor indexing like this
#          neighbor_preds = E_Y_X[topk_indices] # Shape: (n_test, k)
#     except IndexError:
#          # Fallback for older versions or edge cases - using gather
#          # E_Y_X needs to be (n_train, 1) or similar for gather along dim 0
#          E_Y_X_col = E_Y_X.unsqueeze(-1) # Shape: (n_train, 1)
#          # Indices need to be expanded to match the output shape dim except for the gather dim
#          indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, 1) # Shape: (n_test, k, 1)
#          neighbor_preds = torch.gather(E_Y_X_col.expand(n_train, 1), 0, indices_expanded.expand(-1,-1,1)).squeeze(-1) # Check dims carefully

#     # Calculate the weighted sum
#     output = (W * neighbor_preds).sum(axis=1) # Shape: (n_test,)

#     return output

# --- Unchanged functions ---
def estimate_conditional_kernel_oof(
    X_batch, S_batch, E_Y_X, alpha, n_folds=5, **kw
):
    # Pass alpha to the underlying knn function
    oof = np.zeros(len(S_batch))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for tr_idx, te_idx in kf.split(S_batch):
        oof[te_idx] = estimate_conditional_with_knn(
            X_batch, S_batch[te_idx], E_Y_X, alpha, **kw
        )
    return oof

def estimate_conditional_expectation_numpy(
    X_batch,       # should be your full training X of shape (n_train, d)
    S_batch,       # the noisy obs of shape        (n_test,  d)
    E_Y_given_X_batch,  # shape (n_train,)
    alpha,         # shape (d,) - Vector of noise alpha
    bw_scale=1.0,  # global bandwidth multiplier
    CLAMP_MIN=1e-4,
    max_chunk_train=5000,
    max_chunk_test=5000):
    """Estimates E[ E_Y_given_X | S=s ] using Numpy chunking on variance-scaled features."""
    n_train, d = X_batch.shape
    n_test     = S_batch.shape[0]

    # 1) clamp & scale variance
    var_safe   = np.maximum(alpha, CLAMP_MIN) * bw_scale   # (d,)
    inv_sqrt_var_np = (1.0 / np.sqrt(var_safe))[None, None, :]  # (1,1,d) = 1/sqrt(variance)

    numerator   = np.zeros(n_test, dtype=float)
    denominator = np.zeros(n_test, dtype=float)

    for j0 in range(0, n_test, max_chunk_test):
        j1        = min(n_test, j0 + max_chunk_test)
        S_chunk   = S_batch[j0:j1]                         # (cj, d)
        S_exp     = S_chunk[None,:,:] * inv_sqrt_var_np    # (1, cj, d)

        num_c = np.zeros(j1-j0)
        den_c = np.zeros(j1-j0)

        for i0 in range(0, n_train, max_chunk_train):
            i1      = min(n_train, i0 + max_chunk_train)
            X_chunk = X_batch[i0:i1]                       # (ci, d)
            E_chunk = E_Y_given_X_batch[i0:i1]             # (ci,)

            X_exp   = X_chunk[:,None,:] * inv_sqrt_var_np  # (ci,1,d)
            D2      = np.sum((X_exp - S_exp)**2, axis=2)   # (ci,cj)
            D2      = np.clip(D2, None, 10000)

            # stable normalization via softmax over axis=0 (the train axis)
            logW    = -0.5 * D2                            # (ci, cj)
            W       = softmax(logW, axis=0)                # (ci, cj)

            # accumulate
            num_c  += W.T.dot(E_chunk)                     # (cj,)
            den_c  += W.sum(axis=0)                        # (cj,)

        numerator  [j0:j1] = num_c
        denominator[j0:j1] = den_c + 1e-12

    return numerator / denominator

def plugin_estimator_conditional_mean_Kfold(X, Y, estimator_type="rf", n_folds=5):
    """
    Compute out-of-fold plugin estimates for E[Y|X] using K-fold CV.
    """
    n_samples, n_features = X.shape
    out_preds = np.zeros(n_samples)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train = Y[train_idx]
        if estimator_type == "rf":
            model = RandomForestRegressor(n_estimators=100,
                                          min_samples_leaf=5,
                                          n_jobs=-1,
                                          random_state=42)
        else:
            model = KernelRidge(kernel='rbf')
        model.fit(X_train, Y_train)
        mu_test = model.predict(X_test)
        out_preds[test_idx] = mu_test
    return out_preds

def IF_estimator_conditional_mean_Kfold(X, Y, estimator_type="rf", n_folds=5, k_neighbors=1000):
    """Computes out-of-fold IF-corrected estimates for E[Y|X]"""
    n_samples, n_features = X.shape
    out_preds = np.zeros(n_samples)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    bandwidth = 0.1 * np.sqrt(n_features) # Bandwidth for KNN correction step
    if n_samples < k_neighbors:
        k_neighbors = n_samples

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train = Y[train_idx]

        # fit your base model
        if estimator_type == "rf":
            model = RandomForestRegressor(n_estimators=100,
                                          min_samples_leaf=5,
                                          n_jobs=-1,
                                          random_state=42)
        else:
            model = KernelRidge(kernel='rbf')
        model.fit(X_train, Y_train)

        mu_test = model.predict(X_test)
        mu_train = model.predict(X_train)
        residuals_train = Y_train - mu_train

        # build BallTree on scaled X_train for correction term KNN
        scale = 1.0 / bandwidth # Scaling for KNN correction (different from S generation)
        tree = BallTree(X_train * scale, leaf_size=40)

        # query k nearest neighbors for all X_test at once
        dist, ind = tree.query(X_test * scale, k=k_neighbors)  # (n_test, k)
        logW = -0.5 * (dist**2)  # (n_test, k)
        W = softmax(logW, axis=1)

        # compute corrections vectorized over neighbors
        corrections = np.sum(W * residuals_train[ind], axis=1)  # (n_test,)

        out_preds[test_idx] = mu_test + corrections

    return out_preds
# --- Main execution ---
def main():
    parser = argparse.ArgumentParser(description="Compare IF vs Plugin Estimators (alpha = variance)")
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 17, 123, 456, 789],
                        help='List of random seeds')
    # Note: alpha-max now refers to MAXIMUM VARIANCE
    parser.add_argument('--alpha-max-list', nargs='*', type=float, # Changed nargs to *
                        default=[0.1, 0.5, 1.0, 5.0, 10.0], # Default list of max alpha
                        help='List of maximum values for noise variance (alpha)')
    parser.add_argument('--sample-size', type=int, default=5000,
                        help='Subsample size for estimating term1/term2/objective')
    args = parser.parse_args()

    seeds = args.seeds
    alpha_max_list = args.alpha_max_list
    n = args.sample_size
    eps_err = 1e-9 # Epsilon for division by zero in error calculation

    print(f"Running with seeds: {seeds}")
    print(f"Max variances (alpha_max): {alpha_max_list}")
    print(f"Sample size: {n}")

    # storage: for each alpha_max, across seeds, storing the errors
    stats = {
        a: {
            'term_1_wrt_true': {
                'plugin': [], 'if': []
            },
            'term_2_wrt_true': {
                'plugin': [], 'if': [], 'if_plugin': [], 'if_if': []
            },
            'term_2_wrt_if': {
                'if_plugin': [], 'if_if': []
            },
            'objective_wrt_true': {
                'plugin': [], 'if': [], 'if_plugin': [], 'if_if': []
            },
            'objective_wrt_if': {
                'if_plugin': [], 'if_if': []
            },
        }
        for a in alpha_max_list
    }

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed_idx+1}/{len(seeds)} ({seed}) ---")
        np.random.seed(seed)
        torch.manual_seed(seed) # Also seed torch if using GPU later potentially

        # generate large data once per seed
        N_large = 10**6
        d = 10 # Number of features
        X = np.random.normal(0, 1, (N_large, d))
        A = np.random.uniform(-1, 1, d)
        eps_noise = np.random.normal(0, 1, N_large) * np.sqrt(0.1) # Small noise scale
        Y = X.dot(A) + eps_noise

        for alpha_max in alpha_max_list:
            alpha = np.random.uniform(0, alpha_max, d)
            print(f"  Alpha Max (Max Variance): {alpha_max}; Max alpha: {np.max(alpha):.4f}; Min alpha: {np.min(alpha):.4f}")
            # Generate noise alpha for this run
            # Ensure variance alpha is non-negative

            # --- Calculate True Values ---
            # alpha here is VARIANCE
            true_term1 = np.sum(A**2)                             # E[E[Y|X]^2]
            true_term2 = np.sum(A**2 / (1 + alpha))           # E[E[Y|S]^2]
            true_obj   = np.sum((A**2 * alpha) / (1 + alpha)) # E[V[E[Y|X]|S]] = Term1 - Term2

            # --- Subsample Data ---
            idx_sub = np.random.choice(N_large, n, replace=False)
            X_sub = X[idx_sub]
            Y_sub = Y[idx_sub]

            # --- Generate S (Noisy Features) ---
            # Ensure alpha are positive for generating noise
            alpha_safe = np.maximum(alpha, 1e-12) # Add small epsilon for stability
            noise = np.random.multivariate_normal(
                    mean=np.zeros(d),
                    cov=np.diag(alpha_safe), # Use variance directly
                    size=n)
            S_sub = X_sub + noise

            # --- Convert relevant data to Tensors ---
            # Pass alpha (not std dev) to torch functions
            X_sub_t = torch.from_numpy(X_sub).float()
            S_sub_t = torch.from_numpy(S_sub).float()
            alpha_t = torch.from_numpy(alpha).float()


            # --- Estimate Terms ---
            # Plugin estimates
            print("    Estimating Plugin...")
            p1 = plugin_estimator_squared_conditional(X_sub, Y_sub, "rf", n_folds=5) # Est E[E[Y|X]^2]
            p2 = plugin_estimator_squared_conditional(S_sub, Y_sub, "rf", n_folds=5) # Est E[E[Y|S]^2]
            pobj = p1 - p2

            # IF-based estimates (Original IF for comparison)
            print("    Estimating IF (standard)...")
            if1 = IF_estimator_squared_conditional(X_sub, Y_sub, "rf", n_folds=5) # IF Est E[E[Y|X]^2]
            if2 = IF_estimator_squared_conditional(S_sub, Y_sub, "rf", n_folds=5) # IF Est E[E[Y|S]^2]
            ifobj = if1 - if2

            # IF-IF (Term 2 using IF E[Y|X] -> Kernel -> E[E[Y|S]^2])
            print("    Estimating IF-IF...")
            E_Y_X_if = IF_estimator_conditional_mean_Kfold(X_sub, Y_sub, "rf", n_folds=10)
            E_Y_X_if_t = torch.from_numpy(E_Y_X_if).float()
            # estimate_conditional_keops expects alpha
            E_Y_S_if = estimate_conditional_keops(X_sub_t, S_sub_t, E_Y_X_if_t, alpha_t).numpy()
            if2k = np.mean(E_Y_S_if**2) # E[ (E[ IF(E[Y|X]) | S ])^2 ]
            ifobjk = if1 - if2k # Objective using IF term 1 and IF-IF term 2

            # IF-Plugin (Term 2 using Plugin E[Y|X] -> Kernel -> E[E[Y|S]^2])
            print("    Estimating IF-Plugin...")
            E_Y_X_plugin = plugin_estimator_conditional_mean_Kfold(X_sub, Y_sub, "rf", n_folds=10)
            E_Y_X_plugin_t = torch.from_numpy(E_Y_X_plugin).float()
            # estimate_conditional_keops expects alpha
            E_Y_S_plugin = estimate_conditional_keops(X_sub_t, S_sub_t, E_Y_X_plugin_t, alpha_t).numpy()
            if2k_plugin = np.mean(E_Y_S_plugin**2) # E[ (E[ Plugin(E[Y|X]) | S ])^2 ]
            ifobjk_plugin = if1 - if2k_plugin # Objective using IF term 1 and IF-Plugin term 2

            # --- Store Percentage Errors ---
            # Ensure denominators are not zero
            true_term1_denom = abs(true_term1) + eps_err
            true_term2_denom = abs(true_term2) + eps_err
            true_obj_denom   = abs(true_obj) + eps_err
            if2_denom        = abs(if2) + eps_err
            ifobj_denom      = abs(ifobj) + eps_err

            stats[alpha_max]['term_1_wrt_true']['plugin'].append(abs(p1 - true_term1) / true_term1_denom * 100)
            stats[alpha_max]['term_1_wrt_true']['if'].append(abs(if1 - true_term1) / true_term1_denom * 100)

            stats[alpha_max]['term_2_wrt_true']['plugin'].append(abs(p2 - true_term2) / true_term2_denom * 100)
            stats[alpha_max]['term_2_wrt_true']['if'].append(abs(if2 - true_term2) / true_term2_denom * 100)
            stats[alpha_max]['term_2_wrt_true']['if_plugin'].append(abs(if2k_plugin - true_term2) / true_term2_denom * 100)
            stats[alpha_max]['term_2_wrt_true']['if_if'].append(abs(if2k - true_term2) / true_term2_denom * 100)

            stats[alpha_max]['objective_wrt_true']['plugin'].append(abs(pobj - true_obj) / true_obj_denom * 100)
            stats[alpha_max]['objective_wrt_true']['if'].append(abs(ifobj - true_obj) / true_obj_denom * 100)
            stats[alpha_max]['objective_wrt_true']['if_plugin'].append(abs(ifobjk_plugin - true_obj) / true_obj_denom * 100)
            stats[alpha_max]['objective_wrt_true']['if_if'].append(abs(ifobjk - true_obj) / true_obj_denom * 100)

            # Comparisons relative to the standard IF estimate
            stats[alpha_max]['objective_wrt_if']['if_plugin'].append(abs(ifobjk_plugin - ifobj) / ifobj_denom * 100)
            stats[alpha_max]['objective_wrt_if']['if_if'].append(abs(ifobjk - ifobj) / ifobj_denom * 100)

            stats[alpha_max]['term_2_wrt_if']['if_plugin'].append(abs(if2k_plugin - if2) / if2_denom * 100)
            stats[alpha_max]['term_2_wrt_if']['if_if'].append(abs(if2k - if2) / if2_denom * 100)

            print(f"    True Obj: {true_obj:.4f}, Plugin Obj: {pobj:.4f}, IF Obj: {ifobj:.4f}, IF-Plugin Obj: {ifobjk_plugin:.4f}, IF-IF Obj: {ifobjk:.4f}")

    # --- Aggregation and Plotting ---
    print("\n--- Aggregating and Plotting Results ---")
    alphas = alpha_max_list
    # Use simple integer indices 1..N on the X‑axis
    x_positions = np.arange(1, len(alphas) + 1)

    def mean_std(metric, method):
        # metric ∈ {'term_1_wrt_true','term_2_wrt_true', ...}
        # method ∈ {'plugin','if','if_plugin','if_if'}
        # Handle cases where an alpha_max might not have results if script interrupted
        data_for_alphas = [stats[a][metric][method] for a in alphas if a in stats and method in stats[a][metric]]
        if not data_for_alphas or not all(data_for_alphas): # Check if list is empty or contains empty lists
             print(f"Warning: No data found for metric='{metric}', method='{method}'. Skipping.")
             return np.full(len(alphas), np.nan), np.full(len(alphas), np.nan) # Return NaNs

        # Ensure all lists have the same length (number of seeds)
        num_seeds = len(data_for_alphas[0])
        if not all(len(lst) == num_seeds for lst in data_for_alphas):
            print(f"Warning: Inconsistent number of seed results for metric='{metric}', method='{method}'. Check for errors during runs.")
            # Pad with NaN or handle as appropriate - here we might just error out or return NaNs
            # For simplicity, let's try to proceed, but np.array might fail or give weird results
            # A safer approach is to filter/align data first. Let's just calculate based on available data.
            min_len = min(len(lst) for lst in data_for_alphas)
            arr = np.array([lst[:min_len] for lst in data_for_alphas])
        else:
             arr = np.array(data_for_alphas) # shape (len(alphas), len(seeds))


        # arr = np.array([stats[a][metric][method] for a in alphas])
        m = np.mean(arr, axis=1)
        s = np.std(arr, axis=1)
        return m, s

    # Plotting setup (same as before)
    metrics = [
        'term_1_wrt_true',
        'term_2_wrt_true',
        'term_2_wrt_if',
        'objective_wrt_true',
        'objective_wrt_if'
    ]
    method_map = {
        'term_1_wrt_true': ['plugin', 'if'],
        'term_2_wrt_true': ['plugin', 'if', 'if_plugin', 'if_if'],
        'term_2_wrt_if': ['if_plugin', 'if_if'],
        'objective_wrt_true': ['plugin', 'if', 'if_plugin', 'if_if'],
        'objective_wrt_if': ['if_plugin', 'if_if']
    }
    styles = {
        'plugin':     {'fmt': 'o-',  'label': 'Plugin'},
        'if':         {'fmt': 's-',  'label': 'IF'},
        'if_plugin':  {'fmt': '^--', 'label': 'IF-Plugin'},
        'if_if':      {'fmt': 'x--', 'label': 'IF-IF'}
    }

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), sharex=True)
    if len(metrics) == 1: # Handle case of single metric
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for method in method_map[metric]:
            m, s = mean_std(metric, method)
            if np.isnan(m).all(): # Skip plotting if no data
                continue
            st = styles[method]
            # Use descriptive label combining metric and method variant
            label = f"{st['label']}"
            if '_wrt_' in metric:
                label += f" (vs {metric.split('_wrt_')[1]})"

            ax.errorbar(
                x_positions, m, yerr=s,
                fmt=st['fmt'], capsize=5,
                label=label, # Use generated label
                alpha=0.8
            )
        # Improve titles
        title_parts = metric.replace('_wrt_', ' Error wrt ').replace('_', ' ').title()
        ax.set_title(title_parts)
        ax.set_ylabel('Percentage Error (%)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_yscale('log')
        # replace float‐alpha ticks with integer positions
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(i) for i in x_positions])

    # Common X-axis label
    axes[-1].set_xlabel('Index of $\\alpha_{max}$')

    fig.suptitle('Comparison of Estimators vs Max Alpha', fontsize=16)
    # subtitle listing actual alpha_max values
    subtitle = 'Alpha_max list: ' + ', '.join(map(str, alphas))
    # place subtitle at bottom center
    fig.text(0.5, 0.01, subtitle, ha='center', va='bottom', fontsize=10)

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    plot_filename = 'if_vs_plugin_variance_corrected.png'
    fig.savefig(plot_filename)
    print(f"\nPlot saved to {plot_filename}")
    plt.close(fig)

if __name__ == "__main__":
    main()