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

ALPHA_MAX = 1.0 
ALPHA_MAX_STR = str(ALPHA_MAX).replace('.', '') if ALPHA_MAX < 1 else str(int(ALPHA_MAX))
# --- Previously defined functions ---

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

def closed_form(alpha, A):
    """
    sum(A_i^2 * alpha / (1 + alpha))
    """
    return np.sum((A**2 * alpha) / (1 + alpha))

def estimate_conditional_with_knn(
    X_batch, S_batch, E_Y_given_X_batch, alpha, k=50, 
    CLAMP_MIN=1e-4, bw_scale=1.0
):
    # scale features by alpha
    alpha_safe   = np.maximum(alpha, CLAMP_MIN)*bw_scale
    # alpha_safe = alpha
    inv_sqrt_var = 1.0/np.sqrt(alpha_safe)
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
    E_Y_X: Tensor,         # (n_train,)
    alpha: Tensor,       # (d,)
    clamp_min=1e-4
):
    # 1) clamp & per‐dim inv sqrt
    a = torch.clamp(alpha, min=clamp_min)
    inv = torch.rsqrt(a)[None, None, :]             # (1,1,d)
    # 2) scale into Mahalanobis space
    Xs = X[None, :, :] * inv                        # (1,n_train,d)
    Ss = S[:, None, :] * inv                        # (n_test,1,d)
    if X.shape[0] > 10000:
        # using the ball tree method for large datasets;
        X_scaled = Xs.squeeze(0).numpy() 
        S_scaled = Ss.squeeze(1).numpy()
        k = min(10000, X_scaled.shape[0])
        tree = BallTree(X_scaled, leaf_size=40)
        dist, ind = tree.query(S_scaled, k=k)    # (n_test, k)
        # Gaussian weights
        logW_np = -0.5 * dist**2   
        logW = torch.from_numpy(logW_np).to(X.device)                       # (n_test, k)
        W = torch.softmax(logW, axis=1)                     # (n_test, k)
        # gather E_Y for neighbors and do weighted average
        neighbor_preds = E_Y_X[ind]                   # (n_test, k)
        return (W * neighbor_preds).sum(axis=1)       # (n_test,)
    else:
        # 3) pairwise squared distances
        D2 = torch.sum((Ss - Xs)**2, dim=-1)       # (n_test,n_train)
        # 4) Gaussian weights
        # W_unnorm = torch.exp(-0.5 * D2)            # (n_test,n_train)
        # W = W_unnorm / W_unnorm.sum(dim=1, keepdim=True)
        logW = -0.5 * D2                            # (n_test,n_train)
        W = torch.softmax(logW, axis=1)               # (n_test,n_train)
        # 5) weighted sum
        return (W @ E_Y_X)  

def estimate_conditional_kernel_oof(
    X_batch, S_batch, E_Y_X, alpha, n_folds=5, **kw
):
    oof = np.zeros(len(S_batch))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for tr_idx, te_idx in kf.split(S_batch):
        # use all X_batch (or optionally X_batch[tr_idx] only) 
        # to estimate for S_batch[te_idx]
        oof[te_idx] = estimate_conditional_with_knn(
            X_batch, S_batch[te_idx], E_Y_X, alpha, **kw
        )
    return oof

def estimate_conditional_expectation_numpy(
    X_batch,       # should be your full training X of shape (n_train, d)
    S_batch,       # the noisy obs of shape        (n_test,  d)
    E_Y_given_X_batch,  # shape (n_train,)
    alpha,         # shape (d,)
    bw_scale=1.0,  # global bandwidth multiplier
    CLAMP_MIN=1e-4,
    max_chunk_train=5000,
    max_chunk_test=5000,
):
    n_train, d = X_batch.shape
    n_test     = S_batch.shape[0]

    # 1) clamp & scale alpha
    alpha_safe   = np.maximum(alpha, CLAMP_MIN)*bw_scale   # (d,)
    # print("alpha_safe: ", alpha_safe)
    inv_sqrt_var = 1.0 / np.sqrt(alpha_safe)[None, None, :]  # (1,1,d)

    numerator   = np.zeros(n_test, dtype=float)
    denominator = np.zeros(n_test, dtype=float)

    for j0 in range(0, n_test, max_chunk_test):
        j1        = min(n_test, j0 + max_chunk_test)
        S_chunk   = S_batch[j0:j1]                         # (cj, d)
        S_exp     = S_chunk[None,:,:] * inv_sqrt_var      # (1, cj, d)

        num_c = np.zeros(j1-j0)
        den_c = np.zeros(j1-j0)

        for i0 in range(0, n_train, max_chunk_train):
            i1      = min(n_train, i0 + max_chunk_train)
            X_chunk = X_batch[i0:i1]                       # (ci, d)
            E_chunk = E_Y_given_X_batch[i0:i1]             # (ci,)

            X_exp   = X_chunk[:,None,:] * inv_sqrt_var     # (ci,1,d)
            D2      = np.sum((X_exp - S_exp)**2, axis=2)   # (ci,cj)
            D2      = np.clip(D2, None, 10000)

            # stable normalization via softmax over axis=0 (the train axis)
            logW    = -0.5 * D2                            # (ci, cj)
            W       = softmax(logW, axis=0)               # (ci, cj)
            # without softmax         
            # W       = np.exp(-0.5 * D2)                   # (ci, cj)
            # W       = W / W.sum(axis=0, keepdims=True)

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

def IF_estimator_conditional_mean_Kfold(X, Y, estimator_type="rf", n_folds=5, k_neighbors=100):
    n_samples, n_features = X.shape
    out_preds = np.zeros(n_samples)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    bandwidth = 0.1 * np.sqrt(n_features)

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
        logW = -0.5 * (dist**2)  # (n_test, k)
        W = softmax(logW, axis=1)
        # W = np.exp(-0.5 * (dist**2))                           # Gaussian kernel
        # W /= W.sum(axis=1, keepdims=True)                     # normalize

        # compute corrections vectorized over neighbors
        corrections = np.sum(W * residuals_train[ind], axis=1)  # (n_test,)

        out_preds[test_idx] = mu_test + corrections

    return out_preds

def main():
    parser = argparse.ArgumentParser(description="Compare IF vs Plugin Estimators")
    parser.add_argument('--X-distribution', type=str, default='normal', choices=['normal', 'uniform', 'bernoulli'], help='Distribution type for X features') 
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    seed = args.seed
    # --- Testing and plotting code ---
    np.random.seed(seed)
    # 1. Generate a very large dataset (≈1e6 points)
    N_large = 10**6
    # X_large, Y_large = make_regression(n_samples=N_large, n_features=10, noise=1.0, random_state=seed)
    X_large = np.random.normal(0, 1, size=(N_large, 10))
    # Define A as a random vector
    A = np.random.uniform(-1, 1, size=10)
    # Generate Y = AX + eps, where eps ~ N(0, 1)
    eps = np.random.normal(0, 1, size=N_large)
    Y_large = X_large @ A + eps
    alpha = np.random.uniform(0.0, ALPHA_MAX, size=X_large.shape[1])
    print("\tAlpha values: ", alpha)
    print("\tA values: ", A)

    # scale X to zero mean and unit variance
    X_large = X_large - np.mean(X_large, axis=0)
    X_large = X_large / np.std(X_large, axis=0)
    cov_matrix = np.diag(alpha)
    S_alpha = np.random.multivariate_normal(
        mean=np.zeros(X_large.shape[1]),
        cov=cov_matrix,
        size=N_large) + X_large  

    true_functional = closed_form(alpha, A)
    true_term_1 = np.sum(A**2)
    true_term_2 = np.sum(A**2) - true_functional
    # print(f"True functional value (E[E[Y|X]^2] - E[E[Y|S(alpha)]^2]): {true_functional}")
    print(f"\nTrue functional objective: {true_functional}")
    print("True term 1: ", true_term_1)
    print("True term 2: ", true_term_2)

    # 3. Loop over different sample sizes and compute estimates.
    sample_sizes = [1000, 10000, 25000, 50000]#, 100000]#, 500000, 1000000]
    plugin_estimates_term1 = []
    plugin_estimates_term2 = []
    plugin_estimates_objective = []
    IF_estimates_term1 = []
    IF_estimates_term2 = []
    IF_estimates_objective = []

    IF_estimates_term1_kernel = []
    IF_estimates_term2_kernel = []
    IF_estimates_objective_kernel = []

    IF_estimates_term2_plugin = []
    IF_estimates_objective_plugin = []

    for n in sample_sizes:
        print('-'*50)
        print(f"\tSample size: {n}")
        X_sub = X_large[:n]
        Y_sub = Y_large[:n]
        S_sub = S_alpha[:n]
        
        # TERM 1
        # Plugin estimator for sub-sample: fit a model and compute mean(mu(X)^2)
        plugin_est = plugin_estimator_squared_conditional(X_sub, Y_sub, estimator_type="rf")
        plugin_estimates_term1.append(plugin_est)
        
        # IF-based estimator on the sub-sample
        if_est = IF_estimator_squared_conditional(X_sub, Y_sub, estimator_type="rf")
        IF_estimates_term1.append(if_est)
        print(f"\tTerm 1: {n}, Plugin estimate: {plugin_est}, IF estimate: {if_est}")

        # IF-based estimator using kernel method
        # term 1 is same as above
        IF_estimates_term1_kernel.append(if_est)

        # TERM 2
        # Plugin estimator for sub-sample: fit a model and compute mean(mu(S_alpha)^2)
        plugin_est2 = plugin_estimator_squared_conditional(S_sub, Y_sub, estimator_type="rf")
        plugin_estimates_term2.append(plugin_est2)

        # IF-based estimator on the sub-sample for S_alpha
        if_est2 = IF_estimator_squared_conditional(S_sub, Y_sub, estimator_type="rf")
        IF_estimates_term2.append(if_est2)

        # term 2 using kernel method
        E_Y_X = IF_estimator_conditional_mean_Kfold(X_sub, Y_sub, estimator_type="rf", n_folds=10)
        # E_Y_S_alpha = estimate_conditional_expectation_numpy(X_sub, S_sub, E_Y_X, alpha)
        # E_Y_S_alpha = estimate_conditional_kernel_oof(X_sub, S_sub, E_Y_X, alpha, n_folds=10)
        # E_Y_S_alpha = estimate_conditional_with_knn(
        #     X_sub, S_sub, E_Y_X, alpha, k=50, bw_scale=1.0
        # )
        E_Y_S_alpha = estimate_conditional_keops(
            Tensor(X_sub), Tensor(S_sub), Tensor(E_Y_X), Tensor(alpha)
        ).numpy()
        IF_estimates_term2_kernel.append(np.mean(E_Y_S_alpha ** 2))

        # # term 2 using plugin and kernel method
        E_Y_X = plugin_estimator_conditional_mean_Kfold(X_sub, Y_sub, estimator_type="rf", n_folds=10)
        # E_Y_S_alpha = estimate_conditional_expectation_numpy(X_sub, S_sub, E_Y_X, alpha)
        # E_Y_S_alpha = estimate_conditional_kernel_oof(X_sub, S_sub, E_Y_X, alpha, n_folds=10)
        # E_Y_S_alpha = estimate_conditional_with_knn(
        #     X_sub, S_sub, E_Y_X, alpha, k=50, bw_scale=1.0
        # )
        E_Y_S_alpha = estimate_conditional_keops(
            Tensor(X_sub), Tensor(S_sub), Tensor(E_Y_X), Tensor(alpha)
        ).numpy()
        IF_estimates_term2_plugin.append(np.mean(E_Y_S_alpha ** 2))
        print(f"\tTerm 2: {n}, Plugin estimate: {plugin_est2}, IF estimate: {if_est2}, IF-Kernel: {IF_estimates_term2_kernel[-1]}, IF-Plugin: {IF_estimates_term2_plugin[-1]}")

        plugin_objective = plugin_est - plugin_est2
        plugin_estimates_objective.append(plugin_objective)
        IF_objective = if_est - if_est2
        IF_estimates_objective.append(IF_objective)
        IF_objective_kernel = IF_estimates_term1_kernel[-1] - IF_estimates_term2_kernel[-1]
        IF_estimates_objective_kernel.append(IF_objective_kernel)
        IF_estimates_objective_plugin.append(if_est - IF_estimates_term2_plugin[-1])
        print(f"\tObjective: Plugin: {plugin_objective}, IF: {IF_objective}, IF-Kernel: {IF_objective_kernel}, IF-Plugin: {IF_estimates_objective_plugin[-1]}")

    # 4. Plot the estimates vs sample size
    plt.figure(figsize=(10, 15))
    plt.suptitle(f"IF vs Plugin Estimators Comparison (Seed: {seed}, Alpha Max: {ALPHA_MAX})", fontsize=16)
    # Plot for Term 1
    plt.subplot(3, 1, 1)
    plt.axhline(y=true_term_1, color='black', linestyle='--', label='True Term 1 (1e6)')
    plt.plot(sample_sizes, plugin_estimates_term1, marker='o', label='Plugin Estimate (Term 1)')
    plt.plot(sample_sizes, IF_estimates_term1, marker='s', label='IF-based Estimate (Term 1)')
    plt.plot(sample_sizes, IF_estimates_term1_kernel, marker='^', label='IF-based Kernel Estimate (Term 1)')
    plt.xscale('log')
    plt.xlabel("Sample Size (log scale)")
    plt.ylabel("Estimate of Term 1")
    plt.title("Term 1: Plugin vs. IF-based Estimators vs. True")
    plt.legend()
    plt.grid(True)

    # Plot for Term 2
    plt.subplot(3, 1, 2)
    plt.axhline(y=true_term_2, color='black', linestyle='--', label='True Term 2 (1e6)')
    plt.plot(sample_sizes, plugin_estimates_term2, marker='o', label='Plugin Estimate (Term 2)')
    plt.plot(sample_sizes, IF_estimates_term2, marker='s', label='IF-based Estimate (Term 2)')
    plt.plot(sample_sizes, IF_estimates_term2_kernel, marker='^', label='IF-based Kernel Estimate (Term 2)')
    plt.plot(sample_sizes, IF_estimates_term2_plugin, marker='v', label='IF-based Plugin Estimate (Term 2)')
    plt.xscale('log')
    plt.xlabel("Sample Size (log scale)")
    plt.ylabel("Estimate of Term 2")
    plt.title("Term 2: Plugin vs. IF-based Estimators vs. True")
    plt.legend()
    plt.grid(True)

    # Plot for Objective
    plt.subplot(3, 1, 3)
    plt.axhline(y=true_functional, color='black', linestyle='--', label='True Objective')
    plt.plot(sample_sizes, plugin_estimates_objective, marker='o', label='Plugin Objective Estimate')
    plt.plot(sample_sizes, IF_estimates_objective, marker='s', label='IF-based Objective Estimate')
    plt.plot(sample_sizes, IF_estimates_objective_kernel, marker='^', label='IF-based Kernel Objective Estimate')
    plt.plot(sample_sizes, IF_estimates_objective_plugin, marker='v', label='IF-based Plugin Objective Estimate')
    plt.xscale('log')
    plt.xlabel("Sample Size (log scale)")
    plt.ylabel("Objective Estimate")
    plt.title("Objective: Plugin vs. IF-based Estimators vs. True Functional Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"./results/if_vs_plugin/if_vs_plugin_comparison_seed_{seed}_{ALPHA_MAX_STR}.png")
    plt.close()

    # clear everything post experiment for memory management
    del X_large, Y_large, S_alpha
    print("Experiment completed and results saved.")

if __name__ == "__main__":
    main()