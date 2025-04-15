# if_vs_plugin.py: compares IF vs plugin estimator for E[E[Y|X]^2] - E[E[Y|S]^2] 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
import os
import argparse

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

def estimate_conditional_expectation_numpy(X_batch, S_batch, E_Y_given_X_batch, alpha, 
                                                      CLAMP_MIN=1e-4, max_chunk_train=5000, max_chunk_test=5000):
    """
    Estimate E[E[Y|X]|S] via a Gaussian kernel method for continuous data using NumPy,
    processing both X_batch and S_batch in blocks to control memory usage.
    
    Instead of computing the full (n_train x n_test) kernel matrix, we divide the training
    (X_batch) into chunks and the test set (S_batch) into chunks. For each block,
    we compute the kernel values and accumulate two measures:
      - Numerator: sum_{i in block} kernel(i,j)*E_Y_given_X_batch[i]
      - Denom: sum_{i in block} kernel(i,j)
    The final output for each test sample j is then given by: numerator[j] / denominator[j].
    
    Parameters:
        X_batch : np.ndarray
            Input training features, shape (n_samples, n_features).
        S_batch : np.ndarray
            Test (or smoothed) features, shape (n_samples, n_features). 
            (Often X_batch and S_batch have the same number of rows.)
        E_Y_given_X_batch : np.ndarray
            Precomputed estimates for each row in X_batch, shape (n_samples,).
        alpha : np.ndarray
            Smoothing parameters for each feature, shape (n_features,).
        CLAMP_MIN : float, optional
            Minimum allowed value for alpha (default is 1e-4).
        max_chunk_train : int, optional
            Maximum number of training samples to process in one block.
        max_chunk_test : int, optional
            Maximum number of test samples to process in one block.
    
    Returns:
        E_Y_given_S : np.ndarray
            Estimated conditional expectation from S, shape (n_samples,).
    """
    n_train = X_batch.shape[0]
    n_test = S_batch.shape[0]
    
    # Precompute the sqrt factor using the clamped alpha.
    alpha_clamped = np.maximum(alpha, CLAMP_MIN)  # shape: (n_features,)
    # We'll use broadcasting so that sqrt_factor has shape (1,1,n_features)
    sqrt_factor = np.sqrt(1.0 / (alpha_clamped[np.newaxis, np.newaxis, :] + 1e-2))
    
    # Initialize arrays to accumulate numerator and denominator for each test sample.
    numerator = np.zeros(n_test, dtype=np.float64)
    denominator = np.zeros(n_test, dtype=np.float64)
    
    # Process the test samples in chunks.
    for j_start in range(0, n_test, max_chunk_test):
        j_end = min(n_test, j_start + max_chunk_test)
        # S-test block: shape (chunk_test, n_features)
        S_chunk = S_batch[j_start:j_end]
        # Expand S_chunk: shape (1, chunk_test, n_features)
        S_chunk_expanded = S_chunk[np.newaxis, :, :]
        
        # For this S_chunk, accumulate contributions from all training samples in blocks.
        # Initialize temporary arrays for this S_chunk.
        num_chunk = np.zeros(j_end - j_start, dtype=np.float64)
        den_chunk = np.zeros(j_end - j_start, dtype=np.float64)
        
        for i_start in range(0, n_train, max_chunk_train):
            i_end = min(n_train, i_start + max_chunk_train)
            # Training block: shape (chunk_train, n_features)
            X_chunk = X_batch[i_start:i_end]
            # Corresponding E_Y_given_X for this training block
            E_chunk = E_Y_given_X_batch[i_start:i_end]
            # Expand X_chunk: shape (chunk_train, 1, n_features)
            X_chunk_expanded = X_chunk[:, np.newaxis, :]
            
            # Compute differences: shape (chunk_train, chunk_test, n_features)
            diff = X_chunk_expanded - S_chunk_expanded
            # Scale the differences
            scaled_diff = diff * sqrt_factor  # broadcasted multiplication
            # Compute squared distances over features: shape (chunk_train, chunk_test)
            squared_distances = np.sum(scaled_diff**2, axis=2)
            # Clip distances to avoid numerical issues
            clamped_sum = np.clip(squared_distances, None, 100)
            # Compute the Gaussian kernel: shape (chunk_train, chunk_test)
            kernel_block = np.exp(-0.5 * clamped_sum)
            
            # Instead of pre-normalizing columns, accumulate raw sums.
            # For each test sample in the block, accumulate:
            # numerator += sum_{i in train_block} kernel(i,j)*E_chunk[i]
            # denominator += sum_{i in train_block} kernel(i,j)
            num_chunk += np.dot(kernel_block.T, E_chunk)
            den_chunk += np.sum(kernel_block, axis=0)
        
        # Final normalized estimates for the test chunk.
        # Avoid division by zero with a small epsilon.
        epsilon = 1e-8
        numerator[j_start:j_end] = num_chunk
        denominator[j_start:j_end] = den_chunk + epsilon
    
    # Compute output
    E_Y_given_S = numerator / denominator
    return E_Y_given_S

def IF_estimator_conditional_mean_Kfold(X, Y, estimator_type="rf", n_folds=5):
    """
    Compute out-of-fold IF-corrected estimates for E[Y|X] using K-fold CV.
    For each fold, predictions are corrected using a kernel-weighted average of training residuals.
    """
    n_samples, n_features = X.shape
    out_preds = np.zeros(n_samples)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    bandwidth = 0.1 * np.sqrt(n_features)  # heuristic for kernel bandwidth
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
        mu_train = model.predict(X_train)
        mu_test = model.predict(X_test)
        residuals_train = Y_train - mu_train
        for i, x_test in enumerate(X_test):
            dists = np.sum((X_train - x_test)**2, axis=1)
            weights = np.exp(-dists / (2 * bandwidth**2))
            weights = weights / (np.sum(weights) + 1e-8)
            correction = np.sum(weights * residuals_train)
            out_preds[test_idx[i]] = mu_test[i] + correction
    return out_preds

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

def main():
    parser = argparse.ArgumentParser(description="Compare IF vs Plugin Estimators")
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
    alpha = np.random.uniform(0.1, 1.0, size=X_large.shape[1])
    print("\tAlpha values: ", alpha)
    print("\tA values: ", A)
    cov_matrix = np.diag(alpha)
    S_alpha = np.random.multivariate_normal(
        mean=np.zeros(X_large.shape[1]),
        cov=cov_matrix,
        size=N_large) + X_large  

    true_functional = closed_form(alpha, A)
    true_term_1 = np.sum(A**2)
    true_term_2 = np.sum(A**2) - true_functional
    print(f"True functional value (E[E[Y|X]^2] - E[E[Y|S(alpha)]^2]): {true_functional}")
    print("True term 1: ", true_term_1)
    print("True term 2: ", true_term_2)

    # 3. Loop over different sample sizes and compute estimates.
    sample_sizes = [1000, 10000, 50000, 100000, 500000]#, 1000000]
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
        E_Y_X = IF_estimator_conditional_mean_Kfold(X_sub, Y_sub, estimator_type="rf", n_folds=5)
        E_Y_S_alpha = estimate_conditional_expectation_numpy(X_sub, S_sub, E_Y_X, alpha)
        IF_estimates_term2_kernel.append(np.mean(E_Y_S_alpha ** 2))

        # term 2 using plugin and kernel method
        E_Y_X = plugin_estimator_conditional_mean_Kfold(X_sub, Y_sub, estimator_type="rf", n_folds=5)
        E_Y_S_alpha = estimate_conditional_expectation_numpy(X_sub, S_sub, E_Y_X, alpha)
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
    plt.xscale('log')
    plt.xlabel("Sample Size (log scale)")
    plt.ylabel("Objective Estimate")
    plt.title("Objective: Plugin vs. IF-based Estimators vs. True Functional Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"./results/if_vs_plugin/if_vs_plugin_comparison_seed_{seed}.png")
    plt.close()

    # clear everything post experiment for memory management
    del X_large, Y_large, S_alpha
    print("Experiment completed and results saved.")

if __name__ == "__main__":
    main()