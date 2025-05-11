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

def main():
    parser = argparse.ArgumentParser(description="Compare IF vs Plugin Estimators")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    # --- Testing and plotting code ---
    seed = args.seed
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

    # # 2. Compute the plugin estimate for the large sample.
    # # We use RandomForestRegressor as in our functions.
    # full_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42)
    # full_model.fit(X_large, Y_large)
    # mu_large = full_model.predict(X_large)
    # plugin_large_estimate_term1 = np.mean(mu_large ** 2)

    # full_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42)
    # full_model.fit(S_alpha, Y_large)
    # gamma_large = full_model.predict(S_alpha)
    # plugin_large_estimate_term2 = np.mean(gamma_large ** 2)

    # print("Plugin estimate on the large sample (1e6 points):\n\tTerm 1: ", plugin_large_estimate_term1)
    # print("\tTerm 2: ", plugin_large_estimate_term2)

    # print("\tObjective: ", plugin_large_estimate_term1 - plugin_large_estimate_term2)
    # 3. Loop over different sample sizes and compute estimates.
    sample_sizes = [1000, 10000, 50000, 100000, 500000, 1000000]
    plugin_estimates_term1 = []
    plugin_estimates_term2 = []
    plugin_estimates_objective = []
    IF_estimates_term1 = []
    IF_estimates_term2 = []
    IF_estimates_objective = []

    for n in sample_sizes:
        print(f"\tSample size: {n}")
        X_sub = X_large[:n]
        Y_sub = Y_large[:n]
        S_sub = S_alpha[:n]
        
        # Plugin estimator for sub-sample: fit a model and compute mean(mu(X)^2)
        plugin_est = plugin_estimator_squared_conditional(X_sub, Y_sub, estimator_type="rf")
        plugin_estimates_term1.append(plugin_est)
        
        # IF-based estimator on the sub-sample
        if_est = IF_estimator_squared_conditional(X_sub, Y_sub, estimator_type="rf")
        IF_estimates_term1.append(if_est)
        print(f"\tSample size: {n}, Plugin estimate: {plugin_est}, IF estimate: {if_est}")

        # Plugin estimator for sub-sample: fit a model and compute mean(mu(S_alpha)^2)
        plugin_est2 = plugin_estimator_squared_conditional(S_sub, Y_sub, estimator_type="rf")
        plugin_estimates_term2.append(plugin_est2)

        # IF-based estimator on the sub-sample for S_alpha
        if_est2 = IF_estimator_squared_conditional(S_sub, Y_sub, estimator_type="rf")
        IF_estimates_term2.append(if_est2)
        print(f"\t             {n}, Plugin estimate (S_alpha): {plugin_est2}, IF estimate (S_alpha): {if_est2}")
        plugin_objective = plugin_est - plugin_est2
        plugin_estimates_objective.append(plugin_objective)
        IF_objective = if_est - if_est2
        IF_estimates_objective.append(IF_objective)
        print(f"\t             {n}, Plugin estimate (Obj): {plugin_objective}, IF estimate (Obj): {IF_objective}")


    # 4. Plot the estimates vs sample size
    plt.figure(figsize=(10, 15))

    # Plot for Term 1
    plt.subplot(3, 1, 1)
    plt.axhline(y=true_term_1, color='black', linestyle='--', label='True Term 1 (1e6)')
    plt.plot(sample_sizes, plugin_estimates_term1, marker='o', label='Plugin Estimate (Term 1)')
    plt.plot(sample_sizes, IF_estimates_term1, marker='s', label='IF-based Estimate (Term 1)')
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
    plt.xscale('log')
    plt.xlabel("Sample Size (log scale)")
    plt.ylabel("Objective Estimate")
    plt.title("Objective: Plugin vs. IF-based Estimators vs. True Functional Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"./results/if_vs_plugin/if_vs_plugin_comparison_seed_{seed}.png")
    plt.close()

if __name__ == "__main__":
    main()