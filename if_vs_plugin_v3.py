# if_vs_plugin_v3.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from tqdm import tqdm  # for progress bars

def plugin_estimator(X, Y, estimator_type="linear"):
    """
    Plugin estimator for E[Y|X] using either linear regression or kernel regression.
    Returns a prediction function that, when given new X, returns estimated E[Y|X].
    """
    if estimator_type == "linear":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif estimator_type == "rf":
        model = RandomForestRegressor(n_estimators=100, 
                                      min_samples_leaf=5,
                                      n_jobs=-1,
                                      random_state=42)
    else:
        model = KernelRidge(kernel='rbf')
    
    model.fit(X, Y)
    return model.predict

def generate_smoothed_features(X, alphas):
    n_samples, n_features = X.shape
    smoothed_features = {}
    
    # Generate S(alpha) for each alpha
    for i, alpha in enumerate(alphas):
        # Create diagonal covariance matrix with alpha values
        cov_matrix = np.diag(alpha)
        
        # Generate noise ~ N(0, Cov)
        noise = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=cov_matrix,
            size=n_samples
        )
        
        # S(alpha) = X + noise
        smoothed_features[i] = X + noise
    
    return smoothed_features

def IF_estimator_squared_conditional(X, Y, estimator_type="linear"):
    """
    True influence function-based estimator for E[E[Y|X]^2] using linear model.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input features, shape (n_samples, n_features)
    Y : numpy.ndarray
        Target values, shape (n_samples,)
    estimator_type : str, optional
        Type of base estimator to use ('linear', 'rf', or 'krr')
        
    Returns:
    --------
    tuple
        (if_estimate, plugin_estimate) where if_estimate is the bias-corrected 
        estimate of E[E[Y|X]^2]
    """
    # For linear model, we can use a more efficient approach
    if estimator_type == "linear":
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif estimator_type == "rf":
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
    
    # IF-corrected estimate:
    if_estimate = plugin_estimate + correction_term
    
    return if_estimate, plugin_estimate

def compute_diff_estimates(X, Y, smoothed_features, alphas, estimator_type="rf"):
    """
    Compute E[E[Y|X]^2] - E[E[Y|S(alpha)]^2] for both plugin and IF estimators,
    averaging over multiple alpha values.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input features, shape (n_samples, n_features)
    Y : numpy.ndarray
        Target values, shape (n_samples,)
    smoothed_features : dict
        Dictionary mapping each alpha to its corresponding smoothed features
    alphas : list or numpy.ndarray
        List of alpha values used for smoothing
    estimator_type : str, optional
        Type of base estimator to use ('rf' for Random Forest, 'krr' for Kernel Ridge)
        
    Returns:
    --------
    tuple
        (if_diff_estimate, plugin_diff_estimate) containing the difference estimates
        using IF and plugin methods respectively
    """
    # Compute E[E[Y|X]^2]
    if_estimate_x, plugin_estimate_x = IF_estimator_squared_conditional(X, Y, estimator_type)
    
    # Compute average E[E[Y|S(alpha)]^2] for all alphas
    if_estimates_s = []
    plugin_estimates_s = []
    
    for i, alpha in enumerate(alphas):
        S_alpha = smoothed_features[i]  
        if_est, plugin_est = IF_estimator_squared_conditional(S_alpha, Y, estimator_type)
        if_estimates_s.append(if_est)
        plugin_estimates_s.append(plugin_est)
    
    # Average over all alphas
    if_estimate_s_avg = np.mean(if_estimates_s)
    plugin_estimate_s_avg = np.mean(plugin_estimates_s)
    
    # Compute differences
    # if_diff = if_estimate_x - if_estimate_s_avg
    if_diff = if_estimate_x - plugin_estimate_s_avg # changed since plugin performs better than IF for second term
    plugin_diff = plugin_estimate_x - plugin_estimate_s_avg
    
    return if_diff, plugin_diff

def closed_form(alpha, A):
    """
    sum(A_i^2 * alpha / (1 + alpha))
    """
    return np.sum((A**2 * alpha) / (1 + alpha))

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Generate a very large dataset (≈1e6 points)
    print("Generating large dataset...")
    N_large = 10**6  # Reduced from 10^7 for computational efficiency
    # Draw X from N(0, 1)
    X_large = np.random.normal(0, 1, size=(N_large, 10))
    # Define A as a random vector
    A = np.random.uniform(-1, 1, size=10)
    # Generate Y = AX + eps, where eps ~ N(0, 1)
    eps = np.random.normal(0, 1, size=N_large)
    Y_large = X_large @ A + eps
    # prev:
    # X_large, Y_large = make_regression(n_samples=N_large, n_features=10, noise=1.0, random_state=10)
    
    # 2. Define alphas for smoothing
    alphas = [np.random.uniform(0.1, 2.0, size=X_large.shape[1]) for _ in range(20)]
    # print(f"alphas: {alphas}")
    # 3. For the large dataset, precompute smoothed features for each alpha
    print("Precomputing smoothed features for the large dataset...")
    smoothed_features_large = generate_smoothed_features(X_large, alphas)
    
    # 4. Compute the "true" difference estimate using the closed form
    print("Computing 'true' difference estimate using the closed form...")

    true_functional = np.mean([closed_form(alpha, A) for alpha in alphas])
    print(f"True functional value (E[E[Y|X]^2] - E[E[Y|S(alpha)]^2]): {true_functional}")
    # print(f"'True' difference (large sample, N={subset_size}):")
    # print(f"  Plugin estimate: {true_plugin_diff}")
    # print(f"  IF estimate:     {true_if_diff}")
    
    # 5. Loop over different sample sizes and compute estimates
    sample_sizes = [100, 1000, 10000, 50000, 100000]
    plugin_diff_estimates = []
    if_diff_estimates = []
    
    print("Computing estimates for various sample sizes...")
    for n in tqdm(sample_sizes):
        X_sub = X_large[:n]
        Y_sub = Y_large[:n]
        
        # Use precomputed smoothed features for this subset
        smoothed_features_sub = {i: smoothed_features_large[i][:n] for i in range(len(alphas))}
        
        # Compute difference estimates
        if_diff, plugin_diff = compute_diff_estimates(
            X_sub, Y_sub, smoothed_features_sub, range(len(alphas)), estimator_type="rf"
        )
        
        plugin_diff_estimates.append(plugin_diff)
        if_diff_estimates.append(if_diff)
        
        print(f"  Sample size: {n}, Plugin diff: {plugin_diff}, IF diff: {if_diff}")
    
    # 6. Plot the estimates vs sample size
    plt.figure(figsize=(12, 8))
    # plt.axhline(y=true_plugin_diff, color='black', linestyle='--', 
    #            label=f'Large Sample Plugin Diff Estimate (N={subset_size})')
    # plt.axhline(y=true_if_diff, color='grey', linestyle='-.', 
    #            label=f'Large Sample IF Diff Estimate (N={subset_size})')
    plt.axhline(y=true_functional, color='black', linestyle='--',
                label='True Functional Value (E[E[Y|X]^2] - E[E[Y|S(alpha)]^2])')
    
    plt.plot(sample_sizes, plugin_diff_estimates, marker='o', color='blue', 
            label='Plugin Diff Estimate')
    plt.plot(sample_sizes, if_diff_estimates, marker='s', color='red', 
            label='IF-based Diff Estimate')
    
    plt.xscale('log')
    plt.xlabel("Sample Size (log scale)")
    plt.ylabel("Estimate of E[E[Y|X]^2] - E[E[Y|S(alpha)]^2]")
    plt.title("Plugin vs. IF-based Diff Estimators vs. Sample Size")
    plt.legend()
    plt.grid(True)
    
    # Add relative error subplot
    plt.figure(figsize=(12, 8))
    rel_error_plugin = np.abs(np.array(plugin_diff_estimates) - true_functional) / np.abs(true_functional)
    rel_error_if = np.abs(np.array(if_diff_estimates) - true_functional) / np.abs(true_functional)
    
    plt.plot(sample_sizes, rel_error_plugin, marker='o', color='blue', 
            label='Plugin Relative Error')
    plt.plot(sample_sizes, rel_error_if, marker='s', color='red', 
            label='IF-based Relative Error')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Sample Size (log scale)")
    plt.ylabel("Relative Error (log scale)")
    plt.title("Relative Error of Plugin vs. IF-based Diff Estimators")
    plt.legend()
    plt.grid(True)
    
    plt.show()

if __name__ == "__main__":
    main()