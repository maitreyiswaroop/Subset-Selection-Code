import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree
import torch
from torch import Tensor
import matplotlib.pyplot as plt

N_FOLDS = 5

# =============================================================================
# K-fold based estimators for conditional means and squared functionals
# =============================================================================

def plugin_estimator_conditional_mean(X, Y, estimator_type="rf", n_folds=N_FOLDS):
    """
    Compute out-of-fold plugin predictions for E[Y|X] using K-fold CV.
    """
    n_samples = X.shape[0]
    out_preds = np.zeros(n_samples)
    if n_folds == 1:
        if estimator_type == "rf":
            model = RandomForestRegressor(n_estimators=100,
                                          min_samples_leaf=5,
                                          n_jobs=-1,
                                          random_state=42)
        else:
            model = KernelRidge(kernel='rbf')
        model.fit(X, Y)
        return model.predict(X)
    else:
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
            out_preds[test_idx] = model.predict(X_test)
        return out_preds

def plugin_estimator_squared_conditional(X, Y, estimator_type="rf", n_folds=N_FOLDS):
    """
    Compute the plugin estimator for E[E[Y|X]^2] using K-fold CV.
    Returns a scalar computed out-of-fold.
    """
    n_samples = X.shape[0]
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
        return np.mean(mu_X ** 2)
    else:
        mu_X_all = np.zeros(n_samples)
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
            mu_X_all[test_idx] = model.predict(X_test)
        return np.mean(mu_X_all ** 2)
    

def IF_estimator_conditional_mean(X, Y, estimator_type="rf", 
                                  n_folds=5, k_neighbors=1000):
    n_samples, n_features = X.shape
    out_preds = np.zeros(n_samples)
    bandwidth = 0.1 * np.sqrt(n_features)
    if n_samples < k_neighbors:
        k_neighbors = n_samples

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
        residuals = Y - mu_X

        # build BallTree on scaled X
        scale = 1.0 / bandwidth
        tree = BallTree(X * scale, leaf_size=40)

        # query k nearest neighbors for all X at once
        dist, ind = tree.query(X * scale, k=k_neighbors)
        W = np.exp(-0.5 * (dist**2))                           # Gaussian kernel
        W /= W.sum(axis=1, keepdims=True)                     # normalize
        # compute corrections vectorized over neighbors 
        corrections = np.sum(W * residuals[ind], axis=1)      # (n_samples,)
        out_preds = mu_X + corrections
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
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

            # build BallTree on scaled X_train
            scale = 1.0 / bandwidth
            tree = BallTree(X_train * scale, leaf_size=40)

            # query k nearest neighbors for all X_test at once
            dist, ind = tree.query(X_test * scale, k=k_neighbors)  # (n_test, k)
            W = np.exp(-0.5 * (dist**2))                           # Gaussian kernel
            W /= W.sum(axis=1, keepdims=True)                     # normalize

            # compute corrections vectorized over neighbors
            corrections = np.sum(W * residuals_train[ind], axis=1)  # (n_test,)

            out_preds[test_idx] = mu_test + corrections

    return out_preds

def IF_estimator_squared_conditional(X, Y, estimator_type="rf", n_folds=N_FOLDS):
    """
    Compute the IF-based estimator for E[E[Y|X]^2] using K-fold CV.
    """
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
        residuals = Y - mu_X
        correction_term = 2 * np.mean(residuals * mu_X)
        return plugin_estimate + correction_term
    else:
        plugin_terms = []
        correction_terms = []
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
            mu_X_test = model.predict(X_test)
            plugin_terms.append(np.mean(mu_X_test ** 2))
            residuals_test = Y[test_idx] - mu_X_test
            correction_terms.append(2 * np.mean(residuals_test * mu_X_test))
        plugin_estimate = np.mean(plugin_terms)
        correction_term = np.mean(correction_terms)
        return plugin_estimate + correction_term
    
# =============================================================================
# Kernel reweighting function (unchanged)
# =============================================================================
def estimate_conditional_expectation(
        X_batch,             # Tensor[n_train, d]
        S_batch,             # Tensor[n_test,  d]
        E_Y_given_X_batch,   # Tensor[n_train]
        alpha,               # Tensor[d]
        clamp_min=1e-5,
        clamp_max=1e6,
        k=1000):
    """
    Differentiable kernel-weighted estimate of E[Y|S]:
      W_ij ~ exp( -(1/2)* || (S_i - X_j) / sqrt(alpha) ||^2 )
      E[Y|S_i] = \sum_j W_ij * E_Y_given_X_batch[j].
    """
    # 1) clamp and form per-dim inv sqrt variance
    alpha_safe = torch.clamp(alpha, min=clamp_min)       # (d,)
    inv_sqrt   = torch.rsqrt(alpha_safe)                # (d,)

    # 2) scale features
    Xs = X_batch * inv_sqrt                              # (n_train,d)
    Ss = S_batch * inv_sqrt                              # (n_test, d)

    # 3) pairwise squared distances
    #    D = ||Ss[:,None,:] - Xs[None,:,:]||_2  -> (n_test,n_train)
    D2 = torch.cdist(Ss, Xs, p=2).pow(2)                  # (n_test, n_train)
    # clamp distances to avoid numerical issues
    D2 = torch.clamp(D2, min=clamp_min, max=clamp_max)  # (n_test, n_train)
    with torch.no_grad():
        _, idx = D2.topk(k, dim=1, largest=False)
    D2_knn = D2.gather(1, idx)           
    # 4) softmax weights over training axis
    logW = -0.5 * D2_knn                                 # (n_test, n_train)
    W    = torch.softmax(logW, dim=1)                     # rows sum to 1

    # 5) weighted average of precomputed E[Y|X]
    return W.matmul(E_Y_given_X_batch[idx])              # (n_test,)

def estimate_conditional_kernel_oof(
    X_batch: torch.Tensor,
    S_batch: torch.Tensor,
    E_Y_X: torch.Tensor,
    alpha: torch.Tensor,
    n_folds: int = 5,
    clamp_min: float = 1e-4,
    clamp_max: float = 1e6,
    k: int = 100
) -> torch.Tensor:
    """
    Out-of-fold kNN-kernel estimates for E[Y|S].
    """
    n_test = S_batch.size(0)
    oof = torch.zeros(n_test, device=X_batch.device)
    if n_folds == 1:
        oof = estimate_conditional_expectation(
            X_batch, S_batch, E_Y_X, alpha, k=k, clamp_min=clamp_min, clamp_max=clamp_max
        )
        return oof
    else:
        kf  = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        for tr_idx, te_idx in kf.split(S_batch):
            oof[te_idx] = estimate_conditional_expectation(
                X_batch, S_batch[te_idx], E_Y_X, alpha, k=k, clamp_min=clamp_min, clamp_max=clamp_max
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
    # alpha: vector of noise alpha for S features
    # 1) clamp & per‐dim inv sqrt of variance
    alpha_clamped = torch.clamp(alpha, min=clamp_min) # Use clamp_min for variance
    inv_sqrt_var_t = torch.rsqrt(alpha_clamped)[None, None, :] # (1,1,d) = 1/sqrt(variance)
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
    else:
        # 3) pairwise squared distances
        D2 = torch.sum((Ss - Xs)**2, dim=-1)       # (n_test,n_train)
        # 4) Gaussian weights
        logW = -0.5 * D2                            # (n_test,n_train)
        W = torch.softmax(logW, axis=1)               # (n_test,n_train)
        # 5) weighted sum
        return (W @ E_Y_X) # (n_test,)
    
def test_estimator(seeds, alpha_lists, X, Y, save_path=None):
    """
    alpha_lists will be of the shape num_alphas x num_features.
    """
    # sort the list of alphas by the maximum alpha value in each list
    alpha_lists = sorted(alpha_lists, key=lambda x: np.max(x))
    n = X.shape[0]  # Sample size
    eps_err = 1e-9 # Epsilon for division by zero in error calculation

    print(f"Running with seeds: {seeds}")
    print(f"Max alpha: {np.max(alpha_lists):.4f}, Min alpha: {np.min(alpha_lists):.4f}")
    print(f"Sample size: {n}")

    # storage: for each alpha_max, across seeds, storing the errors
    stats = {
        a: {
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
        for a in len(alpha_lists)
    }

    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- Seed {seed_idx+1}/{len(seeds)} ({seed}) ---")
        np.random.seed(seed)
        torch.manual_seed(seed) # Also seed torch if using GPU later potentially

        for i, alpha in enumerate(alpha_lists):
            print(f"\tMax alpha: {np.max(alpha):.4f}; Min alpha: {np.min(alpha):.4f}")       
            # --- Subsample Data ---
            X_sub = X
            Y_sub = Y

            # --- Generate S (Noisy Features) ---
            # Ensure alpha are positive for generating noise
            alpha_safe = np.maximum(alpha, 1e-12) # Add small epsilon for stability
            noise = np.random.multivariate_normal(
                    mean=np.zeros(X_sub.shape[1]), # Mean is zero vector
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
            E_Y_X_if = IF_estimator_conditional_mean(X_sub, Y_sub, "rf", n_folds=10)
            E_Y_X_if_t = torch.from_numpy(E_Y_X_if).float()
            # estimate_conditional_keops expects alpha
            E_Y_S_if = estimate_conditional_keops(X_sub_t, S_sub_t, E_Y_X_if_t, alpha_t).numpy()
            if2k = np.mean(E_Y_S_if**2) # E[ (E[ IF(E[Y|X]) | S ])^2 ]
            ifobjk = if1 - if2k # Objective using IF term 1 and IF-IF term 2

            # IF-Plugin (Term 2 using Plugin E[Y|X] -> Kernel -> E[E[Y|S]^2])
            print("    Estimating IF-Plugin...")
            E_Y_X_plugin = plugin_estimator_conditional_mean(X_sub, Y_sub, "rf", n_folds=10)
            E_Y_X_plugin_t = torch.from_numpy(E_Y_X_plugin).float()
            # estimate_conditional_keops expects alpha
            E_Y_S_plugin = estimate_conditional_keops(X_sub_t, S_sub_t, E_Y_X_plugin_t, alpha_t).numpy()
            if2k_plugin = np.mean(E_Y_S_plugin**2) # E[ (E[ Plugin(E[Y|X]) | S ])^2 ]
            ifobjk_plugin = if1 - if2k_plugin # Objective using IF term 1 and IF-Plugin term 2

            # --- Store Percentage Errors ---
            # Ensure denominators are not zero
            if2_denom        = abs(if2) + eps_err
            ifobj_denom      = abs(ifobj) + eps_err

            # Comparisons relative to the standard IF estimate
            stats[i]['objective_wrt_if']['if_plugin'].append(abs(ifobjk_plugin - ifobj) / ifobj_denom * 100)
            stats[i]['objective_wrt_if']['if_if'].append(abs(ifobjk - ifobj) / ifobj_denom * 100)

            stats[i]['term_2_wrt_if']['if_plugin'].append(abs(if2k_plugin - if2) / if2_denom * 100)
            stats[i]['term_2_wrt_if']['if_if'].append(abs(if2k - if2) / if2_denom * 100)

            print(f"\t\tPlugin Obj: {pobj:.4f}, IF Obj: {ifobj:.4f}, IF-Plugin Obj: {ifobjk_plugin:.4f}, IF-IF Obj: {ifobjk:.4f}")

    # --- Aggregation and Plotting ---
    print("\n--- Aggregating and Plotting Results ---")
    alphas = alpha_lists
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
    # plot_filename = 'if_vs_plugin_variance_corrected.png'
    fig.savefig(save_path or 'if_vs_plugin_variance_corrected.png', dpi=300)
    print(f"\nPlot saved to {save_path or 'if_vs_plugin_variance_corrected.png'}")
    plt.close(fig)