# estimators.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree
from sklearn.base import clone # Import clone
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import xgboost as xgb # Import XGBoost

N_FOLDS = 5
EPS = 1e-8 # Define EPS if not defined elsewhere

# =============================================================================
# K-fold based estimators for conditional means and squared functionals
# =============================================================================

def plugin_estimator_conditional_mean(X, Y, estimator_type="rf", n_folds=N_FOLDS,
                                      seed=42):
    """
    Compute out-of-fold plugin predictions for E[Y|X] using K-fold CV.
    Supports 'rf', 'krr', and 'xgb' estimator types.
    """
    n_samples = X.shape[0]
    out_preds = np.zeros(n_samples)

    # Define model based on type
    if estimator_type == "rf":
        model_base = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    elif estimator_type == "krr":
        model_base = KernelRidge(kernel='rbf', alpha=0.1) # Example parameters
    elif estimator_type == "xgb":
        model_base = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=seed, n_jobs=-1, tree_method='hist') # Added XGBoost
    else:
        raise ValueError(f"Unsupported estimator_type: {estimator_type}. Choose 'rf', 'krr', or 'xgb'.")

    if n_folds <= 1:
        model = clone(model_base) # Use clone for fresh model
        model.fit(X, Y)
        return model.predict(X)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train = Y[train_idx]

            model = clone(model_base) # Use clone for fresh model per fold
            model.fit(X_train, Y_train)
            out_preds[test_idx] = model.predict(X_test)
        return out_preds

def plugin_estimator_squared_conditional(X, Y, estimator_type="rf", n_folds=N_FOLDS,
                                         seed=42):
    """
    Compute the plugin estimator for E[E[Y|X]^2] using K-fold CV.
    Returns a scalar computed out-of-fold.
    Supports 'rf', 'krr', and 'xgb' estimator types.
    """
    n_samples = X.shape[0]

    # Define model based on type
    if estimator_type == "rf":
        model_base = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    elif estimator_type == "krr":
        model_base = KernelRidge(kernel='rbf', alpha=0.1)
    elif estimator_type == "xgb":
        model_base = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=seed, n_jobs=-1, tree_method='hist') # Added XGBoost
    else:
        raise ValueError(f"Unsupported estimator_type: {estimator_type}. Choose 'rf', 'krr', or 'xgb'.")

    if n_folds <= 1:
        model = clone(model_base)
        model.fit(X, Y)
        mu_X = model.predict(X)
        return np.mean(mu_X ** 2)
    else:
        mu_X_all = np.zeros(n_samples)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train = Y[train_idx]

            model = clone(model_base) # Use clone for fresh model per fold
            model.fit(X_train, Y_train)
            mu_X_all[test_idx] = model.predict(X_test)
        return np.mean(mu_X_all ** 2)


def IF_estimator_conditional_mean(X, Y, estimator_type="rf",
                                  n_folds=5,
                                  k_neighbors_factor=0.1, # k as a fraction of n_samples
                                  min_k_neighbors=10,     # Minimum k value
                                  bandwidth_factor=0.1,
                                  seed=42):  # Factor for bandwidth heuristic
    """
    Computes the Influence Function (IF) based estimator for the conditional mean E[Y|X].

    Uses K-Fold cross-validation to mitigate bias from using the same data for
    model fitting and residual calculation. Adjusts k dynamically based on fold size.
    Supports 'rf', 'krr', and 'xgb' estimator types.

    Args:
        X (np.ndarray): Input features (n_samples, n_features).
        Y (np.ndarray): Outcome variable (n_samples,).
        estimator_type (str, optional): Base model type ('rf', 'krr', 'xgb'). Defaults to "rf".
        n_folds (int, optional): Number of folds for cross-validation.
                                 Set to <= 1 to disable CV. Defaults to 5.
        k_neighbors_factor (float, optional): Factor to determine default k
                                              (k = n_samples * factor). Defaults to 0.1.
        min_k_neighbors (int, optional): Minimum value for k neighbors. Defaults to 10.
        bandwidth_factor (float, optional): Factor for bandwidth heuristic
                                           (bw = factor * sqrt(n_features)). Defaults to 0.1.


    Returns:
        np.ndarray: Out-of-fold predictions for E[Y|X] (n_samples,).
                    Returns plugin predictions if CV is disabled or fails.
    """
    n_samples, n_features = X.shape
    out_preds = np.zeros(n_samples)
    bandwidth = bandwidth_factor * np.sqrt(n_features)
    if bandwidth < EPS:
        print(f"Warning: Calculated bandwidth is very small ({bandwidth}). Setting to EPS.")
        bandwidth = EPS
    default_k = max(min_k_neighbors, int(n_samples * k_neighbors_factor))

    # Define model base outside the loop/if conditions
    if estimator_type == "rf":
        model_base = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    elif estimator_type == "krr":
        model_base = KernelRidge(kernel='rbf', alpha=0.1)
    elif estimator_type == "xgb":
        model_base = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=seed, n_jobs=-1, tree_method='hist') # Added XGBoost
    else:
        raise ValueError(f"Unsupported estimator_type: {estimator_type}. Choose 'rf', 'krr', or 'xgb'.")

    # --- Case 1: No Cross-Validation (n_folds <= 1) ---
    if n_folds <= 1:
        print("Warning: Running IF estimator without cross-validation (n_folds <= 1).")
        k_actual = min(default_k, n_samples - 1 if n_samples > 1 else 1)
        if k_actual < 1:
             print("Error: Cannot perform k-NN correction with k < 1.")
             try:
                 model_fallback = clone(model_base)
                 model_fallback.fit(X, Y)
                 return model_fallback.predict(X)
             except Exception as e_plugin:
                 print(f"Error during fallback plugin calculation: {e_plugin}")
                 return np.full(n_samples, np.nan)

        # Fit base model
        model = None # Initialize
        try:
            model = clone(model_base)
            model.fit(X, Y)
            mu_X = model.predict(X)
            residuals = Y - mu_X
        except Exception as e_fit:
            print(f"Error fitting base model (no CV): {e_fit}")
            return np.full(n_samples, np.nan)

        # Perform k-NN correction
        try:
            scale = 1.0 / bandwidth
            tree = BallTree(X * scale, leaf_size=40)
            dist, ind = tree.query(X * scale, k=k_actual)
            W = np.exp(-0.5 * (dist**2))
            W_sum = W.sum(axis=1, keepdims=True)
            W_sum = np.where(W_sum < EPS, EPS, W_sum)
            W /= W_sum
            corrections = np.sum(W * residuals[ind], axis=1)
            out_preds = mu_X + corrections
        except Exception as e_corr:
             print(f"Error during k-NN correction calculation (no CV): {e_corr}")
             out_preds = mu_X # Fallback to plugin

    # --- Case 2: K-Fold Cross-Validation ---
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_num = 0
        for train_idx, test_idx in kf.split(X):
            fold_num += 1
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train = Y[train_idx]
            n_train = X_train.shape[0]

            if n_train < max(2, min_k_neighbors):
                 print(f"Warning: Training fold {fold_num} size ({n_train}) too small. Skipping IF correction, using fallback.")
                 try:
                      model_fold = clone(model_base)
                      model_fold.fit(X_train, Y_train)
                      out_preds[test_idx] = model_fold.predict(X_test)
                 except Exception as e_fallback:
                      print(f"  Error during fallback plugin calculation for fold {fold_num}: {e_fallback}")
                      out_preds[test_idx] = np.nan
                 continue

            k_fold = max(1, min(default_k, n_train - 1))
            model = None
            try:
                model = clone(model_base) # Use clone for fresh model per fold
                model.fit(X_train, Y_train)
                mu_test = model.predict(X_test)
                mu_train = model.predict(X_train)
                residuals_train = Y_train - mu_train

                scale = 1.0 / bandwidth
                tree = BallTree(X_train * scale, leaf_size=40)
                dist, ind = tree.query(X_test * scale, k=k_fold)
                W = np.exp(-0.5 * (dist**2))
                W_sum = W.sum(axis=1, keepdims=True)
                W_sum = np.where(W_sum < EPS, EPS, W_sum)
                W /= W_sum
                corrections = np.sum(W * residuals_train[ind], axis=1)
                out_preds[test_idx] = mu_test + corrections

            except ValueError as ve:
                 if "k must be less than or equal to the number of training points" in str(ve) or "k exceeds number of points" in str(ve):
                     print(f"Error during k-NN query in fold {fold_num} (k={k_fold}, n_train={n_train}): {ve}")
                     if model is not None:
                         try: out_preds[test_idx] = model.predict(X_test)
                         except: out_preds[test_idx] = np.nan
                     else: out_preds[test_idx] = np.nan
                 else:
                     print(f"ValueError during fold {fold_num} processing: {ve}")
                     out_preds[test_idx] = np.nan
            except Exception as e:
                 print(f"Error during fold {fold_num} processing: {e}")
                 if model is not None:
                     try: out_preds[test_idx] = model.predict(X_test)
                     except: out_preds[test_idx] = np.nan
                 else:
                     out_preds[test_idx] = np.nan

    return out_preds

def IF_estimator_squared_conditional(X, Y, estimator_type="rf", n_folds=N_FOLDS, seed=42):
    """
    Compute the IF-based estimator for E[E[Y|X]^2] using K-fold CV.
    Supports 'rf', 'krr', and 'xgb' estimator types.
    """
    if isinstance(X, Tensor): X = X.detach().cpu().numpy()
    if isinstance(Y, Tensor): Y = Y.detach().cpu().numpy()

    # Define model based on type
    if estimator_type == "rf":
        model_base = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    elif estimator_type == "krr":
        model_base = KernelRidge(kernel='rbf', alpha=0.1)
    elif estimator_type == "xgb":
        model_base = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=seed, n_jobs=-1, tree_method='hist') # Added XGBoost
    else:
        raise ValueError(f"Unsupported estimator_type: {estimator_type}. Choose 'rf', 'krr', or 'xgb'.")

    if n_folds <= 1:
        model = clone(model_base)
        model.fit(X, Y)
        mu_X = model.predict(X)
        plugin_estimate = np.mean(mu_X ** 2)
        residuals = Y - mu_X
        correction_term = 2 * np.mean(residuals * mu_X)
        return plugin_estimate + correction_term
    else:
        plugin_terms = []
        correction_terms = []
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train = Y[train_idx]

            model = clone(model_base) # Use clone for fresh model per fold
            model.fit(X_train, Y_train)
            mu_X_test = model.predict(X_test)
            plugin_terms.append(np.mean(mu_X_test ** 2))
            residuals_test = Y[test_idx] - mu_X_test
            correction_terms.append(2 * np.mean(residuals_test * mu_X_test))

        # Handle potential NaNs if some folds failed? For now, assume they succeed.
        plugin_estimate = np.mean(plugin_terms) if plugin_terms else np.nan
        correction_term = np.mean(correction_terms) if correction_terms else np.nan

        if np.isnan(plugin_estimate) or np.isnan(correction_term):
            return np.nan
        else:
            return plugin_estimate + correction_term

# =============================================================================
# Kernel reweighting function (unchanged)
# =============================================================================
def estimate_conditional_expectation_knn(
        X_ref: torch.Tensor,       # Reference features [n_ref, d]
        S_query: torch.Tensor,     # Query features S(alpha) [n_query, d]
        E_Y_X_ref: torch.Tensor,   # Reference E[Y|X] values [n_ref]
        alpha: torch.Tensor,       # Noise parameters [d]
        k: int = 1000,             # Number of neighbors
        clamp_min: float = 1e-5,   # Min value for alpha and squared distances
        clamp_max_dist: float = 1e6 # Max value for squared distances
        ) -> torch.Tensor:
    """
    Differentiable kNN kernel-weighted estimate of E[Y|S_query] using references.
      W_ij ~ exp( -(1/2)* || (S_i - X_j) / sqrt(alpha) ||^2 )
      E[Y|S_i] = sum_{j in kNN(S_i)} W_ij * E_Y_X_ref[j].

    Ensures tensors are on the same device as S_query.
    """
    device = S_query.device
    n_ref = X_ref.shape[0]
    n_query = S_query.shape[0]

    # Move reference data to the correct device if necessary
    X_ref = X_ref.to(device)
    E_Y_X_ref = E_Y_X_ref.to(device)
    alpha = alpha.to(device) # Ensure alpha is also on the right device

    # 1) Clamp alpha and compute inverse sqrt variance
    alpha_safe = torch.clamp(alpha, min=clamp_min)       # (d,)
    inv_sqrt_alpha = torch.rsqrt(alpha_safe)             # (d,) -> 1/sqrt(alpha)

    # 2) Scale features into Mahalanobis space based on alpha
    Xs_scaled = X_ref * inv_sqrt_alpha                   # (n_ref, d)
    Ss_scaled = S_query * inv_sqrt_alpha                 # (n_query, d)

    # 3) Compute pairwise squared distances in scaled space
    #    D2_ij = || Ss_scaled_i - Xs_scaled_j ||^2
    #    Using cdist is generally efficient and stable
    D2 = torch.cdist(Ss_scaled, Xs_scaled, p=2).pow(2)     # (n_query, n_ref)

    # Clamp distances to avoid potential numerical issues (optional but safe)
    D2 = torch.clamp(D2, min=clamp_min, max=clamp_max_dist) # (n_query, n_ref)

    # 4) Find k-nearest neighbors for each query point S_i based on scaled distance
    actual_k = min(k, n_ref)
    if actual_k < 1:
        print(f"Warning: actual_k={actual_k} < 1 in kernel estimation. Returning mean.")
        # Return mean of reference E[Y|X] as fallback
        return torch.full((n_query,), E_Y_X_ref.mean(), device=device, dtype=S_query.dtype)

    # topk finds the k smallest distances and their indices
    # Use torch.no_grad() for idx finding if not backpropping through indices (usually safe)
    with torch.no_grad():
         # D2_knn: distances to k nearest neighbors (n_query, k)
         # knn_indices: indices of these neighbors in X_ref (n_query, k)
        D2_knn, knn_indices = torch.topk(D2, actual_k, dim=1, largest=False)

    # Important: Re-select distances using indices *within* the computation graph
    # if gradients through D2 are needed for alpha (which they are).
    # Gather the distances corresponding to the selected indices.
    # This ensures the gradient path for D2 -> alpha is maintained.
    D2_knn_grad = D2.gather(1, knn_indices) # (n_query, k)

    # 5) Calculate weights using softmax over the k neighbors
    #    logW = -0.5 * D2_knn (use the version with grad)
    logW = -0.5 * D2_knn_grad                            # (n_query, k)
    W = torch.softmax(logW, dim=1)                       # (n_query, k), rows sum to 1

    # 6) Gather the E[Y|X] values for the k neighbors
    #    knn_indices shape: (n_query, k)
    #    E_Y_X_ref shape: (n_ref,) -> Need to index E_Y_X_ref using knn_indices
    #    Use gather or direct indexing
    E_Y_X_neighbors = E_Y_X_ref[knn_indices]             # (n_query, k)

    # 7) Compute weighted average
    E_Y_S_estimate = (W * E_Y_X_neighbors).sum(dim=1)    # (n_query,)

    return E_Y_S_estimate


def estimate_conditional_kernel_oof(
    X_batch: torch.Tensor,
    S_batch: torch.Tensor,
    E_Y_X: torch.Tensor,
    alpha: torch.Tensor,
    n_folds: int = 5,
    clamp_min: float = 1e-4,
    clamp_max: float = 1e6,
    k: int = 100,
    seed: int = 42
) -> torch.Tensor:
    """
    Out-of-fold kNN-kernel estimates for E[Y|S].
    """
    n_test = S_batch.size(0)
    oof = torch.zeros(n_test, device=X_batch.device)

    # Ensure k is not larger than the smallest possible training fold size
    min_train_size = max(1, int(X_batch.shape[0] * (1 - 1/n_folds)) if n_folds > 1 else X_batch.shape[0])
    actual_k = min(k, min_train_size)
    if actual_k < 1:
        print("Warning: k adjusted to 0 in estimate_conditional_kernel_oof. Returning zeros.")
        return oof # Or handle differently

    if n_folds <= 1:
        oof = estimate_conditional_expectation(
            X_batch, S_batch, E_Y_X, alpha, k=actual_k, clamp_min=clamp_min, clamp_max=clamp_max
        )
        return oof
    else:
        # Note: This OOF implementation for the kernel estimator is slightly different
        # from the plugin/IF OOF. Here, for each test fold of S_batch, it uses the
        # *entire* X_batch and E_Y_X as the "training" set for the kernel weighting.
        # This might be intended, but differs from typical CV where the model/reference
        # data is also split. If true OOF is needed, X_batch and E_Y_X should also be split.
        # Assuming current implementation is intended:
        kf  = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for _, te_idx in kf.split(S_batch): # We only need test indices for S_batch
            oof[te_idx] = estimate_conditional_expectation(
                X_batch, S_batch[te_idx], E_Y_X, alpha, k=actual_k, clamp_min=clamp_min, clamp_max=clamp_max
            )
        return oof

def estimate_conditional_keops(
    X: Tensor,           # (n_train,d)
    S: Tensor,           # (n_test, d)
    E_Y_X: Tensor,       # (n_train,) - Estimated E[Y|X] values on training data
    alpha: Tensor,   # (d,) - Vector of noise alpha for S features
    clamp_min=1e-4
):
    """Estimates E[ E_Y_X | S=s ] using KeOps (or BallTree fallback) on variance-scaled features."""
    # Ensure E_Y_X is 1D
    if E_Y_X.ndim != 1:
        if E_Y_X.ndim == 2 and E_Y_X.shape[1] == 1:
            E_Y_X = E_Y_X.squeeze(1)
        else:
            raise ValueError(f"E_Y_X must be a 1D tensor, but got shape {E_Y_X.shape}")

    # alpha: vector of noise alpha for S features
    # 1) clamp & per‐dim inv sqrt of variance
    alpha_clamped = torch.clamp(alpha, min=clamp_min) # Use clamp_min for variance
    inv_sqrt_var_t = torch.rsqrt(alpha_clamped)[None, None, :] # (1,1,d) = 1/sqrt(variance)
    # 2) scale into Mahalanobis space
    Xs = X[None, :, :] * inv_sqrt_var_t             # (1,n_train,d)
    Ss = S[:, None, :] * inv_sqrt_var_t             # (n_test,1,d)

    n_train_curr = X.shape[0] # Use current X batch size

    # --- Fallback to BallTree logic (unchanged math, uses sqrt(variance)) ---
    # Increased threshold for fallback, KeOps might be slow for larger n_train
    if n_train_curr > 5000:
        # using the ball tree method for large datasets;
        # Ensure tensors are on CPU before converting to numpy if they might be on GPU
        X_scaled = Xs.squeeze(0).cpu().numpy()
        S_scaled = Ss.squeeze(1).cpu().numpy()
        # k for BallTree fallback
        k_bt = min(1000, X_scaled.shape[0] - 1 if X_scaled.shape[0] > 1 else 1)
        if k_bt < 1: return torch.zeros(S.shape[0], device=S.device) # Handle edge case

        tree = BallTree(X_scaled, leaf_size=40)
        dist, ind = tree.query(S_scaled, k=k_bt)    # (n_test, k)
        # Gaussian weights
        logW_np = -0.5 * dist**2
        # Convert logW back to a torch tensor
        logW = torch.from_numpy(logW_np).to(X.device) # Use .to(X.device) for consistency
        W = torch.softmax(logW, dim=1)                     # (n_test, k)
        # gather E_Y for neighbors and do weighted average
        # Ensure E_Y_X is on CPU if needed for indexing with numpy array 'ind'
        E_Y_X_np = E_Y_X.cpu().numpy() if E_Y_X.is_cuda else E_Y_X.numpy()
        neighbor_preds_np = E_Y_X_np[ind]                   # (n_test, k)
        neighbor_preds = torch.from_numpy(neighbor_preds_np).to(X.device) # Back to device
        return (W * neighbor_preds).sum(dim=1)       # (n_test,)
    else:
        # 3) pairwise squared distances
        D2 = torch.cdist(Ss.squeeze(1), Xs.squeeze(0), p=2).pow(2) # (n_test, n_train)
        # D2 = torch.sum((Ss - Xs)**2, dim=-1)       # Manual calculation (n_test,n_train)

        # 4) Gaussian weights
        logW = -0.5 * D2                            # (n_test,n_train)
        W = torch.softmax(logW, dim=1)               # (n_test,n_train)
        # 5) weighted sum
        return (W @ E_Y_X) # (n_test,)

def test_estimator(seeds, alpha_lists, X, Y, save_path=None):
    """
    Compares different estimators for the objective E[E[Y|X]^2] - E[E[Y|S]^2].

    Estimators for Term 2 (E[E[Y|S]^2]):
    - Plugin: plugin_estimator_squared_conditional(S, Y)
    - IF: IF_estimator_squared_conditional(S, Y)
    - IF-Plugin: Kernel(Plugin E[Y|X]) -> mean square
    - IF-IF: Kernel(IF E[Y|X]) -> mean square

    Term 1 (E[E[Y|X]^2]) is estimated using IF.

    Args:
        seeds (list): List of random seeds.
        alpha_lists (list): List of alpha vectors (or scalars if uniform noise).
                            Each element corresponds to one setting of alphas.
        X (Tensor or ndarray): Features.
        Y (Tensor or ndarray): Outcomes.
        save_path (str, optional): Path to save the comparison plot. Defaults to None.
    """
    # Sort the list of alphas by the maximum alpha value in each list/scalar
    alpha_lists = sorted(alpha_lists, key=lambda x: np.max(x) if isinstance(x, (np.ndarray, list)) else x)
    n = X.shape[0]  # Sample size
    eps_err = 1e-9 # Epsilon for division by zero in error calculation

    print(f"Running with seeds: {seeds}")
    print(f"Number of alpha settings: {len(alpha_lists)}")
    print(f"Sample size: {n}")

    # Storage: indexed by alpha setting index
    stats = {
        i: {
            'term_2_wrt_if': {'if_plugin': [], 'if_if': []}, # Compare Kernel Term2 estimates to IF Term2
            'objective_wrt_if': {'if_plugin': [], 'if_if': []} # Compare Obj(Kernel T2) to Obj(IF T2)
            # Add comparisons to Plugin if desired
            # 'term_2_wrt_plugin': {'if': [], 'if_plugin': [], 'if_if': []},
            # 'objective_wrt_plugin': {'if': [], 'if_plugin': [], 'if_if': []},
        }
        for i in range(len(alpha_lists))
    }

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed_idx+1}/{len(seeds)} ({seed}) ---")
        np.random.seed(seed)
        torch.manual_seed(seed) # Also seed torch

        # Ensure data are NumPy arrays for sklearn estimators
        if isinstance(X, Tensor): X_np = X.detach().cpu().numpy()
        else: X_np = np.array(X) # Ensure it's numpy
        if isinstance(Y, Tensor): Y_np = Y.detach().cpu().numpy()
        else: Y_np = np.array(Y) # Ensure it's numpy

        # --- Estimate Term 1 (using IF, assumed more stable/accurate) ---
        # This is constant for all alphas within a seed run
        print("    Estimating Term 1 (IF)...")
        if1 = IF_estimator_squared_conditional(X_np, Y_np, "rf", n_folds=N_FOLDS)
        if np.isnan(if1):
             print("    Term 1 (IF) calculation failed. Skipping seed.")
             continue

        for i, alpha_setting in enumerate(alpha_lists):
            # Handle both scalar alpha and vector alpha
            if isinstance(alpha_setting, (np.ndarray, list)):
                alpha = np.array(alpha_setting)
                alpha_max_str = f"{np.max(alpha):.4f}"
                alpha_min_str = f"{np.min(alpha):.4f}"
            else: # Assume scalar
                alpha = np.full(X_np.shape[1], alpha_setting) # Create vector
                alpha_max_str = f"{alpha_setting:.4f}"
                alpha_min_str = alpha_max_str

            print(f"\tAlpha setting {i}: Max={alpha_max_str}, Min={alpha_min_str}")

            # --- Generate S (Noisy Features) ---
            alpha_safe = np.maximum(alpha, 1e-12) # Ensure positivity
            noise = np.random.multivariate_normal(
                    mean=np.zeros(X_np.shape[1]),
                    cov=np.diag(alpha_safe),
                    size=n)
            S_np = X_np + noise

            # --- Convert relevant data to Tensors for Kernel estimator ---
            X_t = torch.from_numpy(X_np).float()
            S_t = torch.from_numpy(S_np).float()
            alpha_t = torch.from_numpy(alpha).float()

            # --- Estimate Term 2 variants ---
            print("      Estimating Term 2 variants...")
            try:
                # T2 Plugin: plugin_estimator_squared_conditional(S, Y)
                p2 = plugin_estimator_squared_conditional(S_np, Y_np, "rf", n_folds=N_FOLDS)

                # T2 IF: IF_estimator_squared_conditional(S, Y)
                if2 = IF_estimator_squared_conditional(S_np, Y_np, "rf", n_folds=N_FOLDS)

                # T2 IF-IF: Kernel(IF E[Y|X]) -> mean square
                E_Y_X_if = IF_estimator_conditional_mean(X_np, Y_np, "rf", n_folds=N_FOLDS)
                E_Y_X_if_t = torch.from_numpy(E_Y_X_if).float().to(X_t.device) # Move to same device
                E_Y_S_if = estimate_conditional_keops(X_t.to(E_Y_X_if_t.device), S_t.to(E_Y_X_if_t.device), E_Y_X_if_t, alpha_t.to(E_Y_X_if_t.device)).cpu().numpy()
                if2k = np.mean(E_Y_S_if**2)

                # T2 IF-Plugin: Kernel(Plugin E[Y|X]) -> mean square
                E_Y_X_plugin = plugin_estimator_conditional_mean(X_np, Y_np, "rf", n_folds=N_FOLDS)
                E_Y_X_plugin_t = torch.from_numpy(E_Y_X_plugin).float().to(X_t.device) # Move to same device
                E_Y_S_plugin = estimate_conditional_keops(X_t.to(E_Y_X_plugin_t.device), S_t.to(E_Y_X_plugin_t.device), E_Y_X_plugin_t, alpha_t.to(E_Y_X_plugin_t.device)).cpu().numpy()
                if2k_plugin = np.mean(E_Y_S_plugin**2)

            except Exception as e:
                 print(f"      Error during Term 2 estimation for alpha setting {i}: {e}")
                 # Store NaNs or skip this alpha setting for this seed
                 if i in stats: # Check if index exists
                     stats[i]['term_2_wrt_if']['if_plugin'].append(np.nan)
                     stats[i]['term_2_wrt_if']['if_if'].append(np.nan)
                     stats[i]['objective_wrt_if']['if_plugin'].append(np.nan)
                     stats[i]['objective_wrt_if']['if_if'].append(np.nan)
                 continue # Skip to next alpha

            # --- Calculate Objectives ---
            # Ensure Term 2 estimates are valid numbers before calculating objectives
            if np.isnan(if2) or np.isnan(if2k) or np.isnan(if2k_plugin):
                 print(f"      Skipping objective calculation due to NaN in Term 2 estimates.")
                 if i in stats:
                     stats[i]['term_2_wrt_if']['if_plugin'].append(np.nan if np.isnan(if2k_plugin) else abs(if2k_plugin - if2) / (abs(if2) + eps_err) * 100)
                     stats[i]['term_2_wrt_if']['if_if'].append(np.nan if np.isnan(if2k) else abs(if2k - if2) / (abs(if2) + eps_err) * 100)
                     stats[i]['objective_wrt_if']['if_plugin'].append(np.nan)
                     stats[i]['objective_wrt_if']['if_if'].append(np.nan)
                 continue

            ifobj = if1 - if2
            ifobjk = if1 - if2k
            ifobjk_plugin = if1 - if2k_plugin

            # --- Store Percentage Errors ---
            if2_denom = abs(if2) + eps_err
            ifobj_denom = abs(ifobj) + eps_err

            stats[i]['term_2_wrt_if']['if_plugin'].append(abs(if2k_plugin - if2) / if2_denom * 100)
            stats[i]['term_2_wrt_if']['if_if'].append(abs(if2k - if2) / if2_denom * 100)
            stats[i]['objective_wrt_if']['if_plugin'].append(abs(ifobjk_plugin - ifobj) / ifobj_denom * 100)
            stats[i]['objective_wrt_if']['if_if'].append(abs(ifobjk - ifobj) / ifobj_denom * 100)

            print(f"\t\tIF Obj: {ifobj:.4f}, IF-Plugin Obj: {ifobjk_plugin:.4f}, IF-IF Obj: {ifobjk:.4f}")
            print(f"\t\tIF T2: {if2:.4f}, IF-Plugin T2: {if2k_plugin:.4f}, IF-IF T2: {if2k:.4f}")


    # --- Aggregation and Plotting ---
    print("\n--- Aggregating and Plotting Results ---")
    # Use simple integer indices 1..N on the X‑axis
    x_positions = np.arange(1, len(alpha_lists) + 1)

    def mean_std(metric, method):
        # Handle cases where an alpha_max might not have results if script interrupted
        # Also handle potential NaNs stored during runs
        data_for_alphas = [stats[a][metric][method] for a in range(len(alpha_lists)) if a in stats and method in stats[a][metric]]

        if not data_for_alphas:
             print(f"Warning: No data found for metric='{metric}', method='{method}'. Skipping.")
             return np.full(len(alpha_lists), np.nan), np.full(len(alpha_lists), np.nan)

        # Calculate mean/std ignoring NaNs and handling potentially ragged lists if seeds failed
        means = []
        stds = []
        for alpha_data in data_for_alphas:
            valid_data = [d for d in alpha_data if not np.isnan(d)]
            if not valid_data:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(np.mean(valid_data))
                stds.append(np.std(valid_data))

        # Pad with NaNs if some alpha settings were skipped entirely
        if len(means) < len(alpha_lists):
             padded_means = np.full(len(alpha_lists), np.nan)
             padded_stds = np.full(len(alpha_lists), np.nan)
             # This assumes data_for_alphas corresponds to the first len(means) indices
             padded_means[:len(means)] = means
             padded_stds[:len(stds)] = stds
             return padded_means, padded_stds
        else:
             return np.array(means), np.array(stds)


    # Plotting setup
    metrics = ['term_2_wrt_if', 'objective_wrt_if'] # Focus on comparisons wrt IF
    method_map = {
        'term_2_wrt_if': ['if_plugin', 'if_if'],
        'objective_wrt_if': ['if_plugin', 'if_if']
    }
    styles = {
        'if_plugin':  {'fmt': '^--', 'label': 'Kernel(Plugin E[Y|X])'},
        'if_if':      {'fmt': 'x--', 'label': 'Kernel(IF E[Y|X])'}
    }

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), sharex=True)
    if len(metrics) == 1: axes = [axes] # Handle single metric case

    for ax, metric in zip(axes, metrics):
        for method in method_map[metric]:
            m, s = mean_std(metric, method)
            if np.isnan(m).all(): continue # Skip plotting if no data

            st = styles[method]
            label = f"{st['label']} (vs IF)" # Clarify baseline is IF

            ax.errorbar(x_positions, m, yerr=s, fmt=st['fmt'], capsize=5, label=label, alpha=0.8)

        title_parts = metric.replace('_wrt_if', ' Error wrt IF').replace('_', ' ').title()
        ax.set_title(title_parts)
        ax.set_ylabel('Percentage Error (%)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_yscale('log')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(i) for i in x_positions])

    axes[-1].set_xlabel('Index of Alpha Setting')
    fig.suptitle('Comparison of Kernel-based Estimators vs IF Estimator', fontsize=16)

    # Create subtitle with alpha setting details (showing max value)
    alpha_max_values = [f"{np.max(a):.2f}" if isinstance(a, (np.ndarray, list)) else f"{a:.2f}" for a in alpha_lists]
    subtitle = 'Alpha Setting Index -> Max Alpha Value:\n' + ', '.join([f"{i+1}:{val}" for i, val in enumerate(alpha_max_values)])
    fig.text(0.5, 0.01, subtitle, ha='center', va='bottom', fontsize=8) # Smaller font for subtitle

    fig.tight_layout(rect=[0, 0.05, 1, 0.96]) # Adjust rect for subtitle
    plot_filename = save_path or 'kernel_vs_if_comparison.png'
    fig.savefig(plot_filename, dpi=300)
    print(f"\nPlot saved to {plot_filename}")
    plt.close(fig)
