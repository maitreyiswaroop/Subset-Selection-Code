# if_vs_plugin.py: compares IF vs plugin estimator for E[E[Y|X]^2] 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge

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

def IF_estimator_squared_conditional(X, Y, estimator_type="rf"):
    """
    True influence function-based estimator for E[E[Y|X]^2] without using K-fold cross-validation.
    
    This estimator fits a model to estimate mu(X) = E[Y|X] on the entire dataset,
    computes the plugin estimate as:
      plugin_estimate = mean(mu(X)^2),
    and then applies the influence function correction:
      correction_term = 2 * mean((Y - mu(X)) * mu(X)).
    
    The final IF-corrected estimate is:
      IF_estimate = plugin_estimate + correction_term.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input features, shape (n_samples, n_features)
    Y : numpy.ndarray
        Target values, shape (n_samples,)
    estimator_type : str, optional
        Type of base estimator to use ('rf' for Random Forest, 'krr' for Kernel Ridge)
        
    Returns:
    --------
    float
        The bias-corrected estimate of E[E[Y|X]^2]
    """
    # Choose the model and parameters
    if estimator_type == "rf":
        model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42)
    else:
        model = KernelRidge(kernel='rbf')
    
    # Fit the model on the full dataset to estimate mu(X) = E[Y|X]
    model.fit(X, Y)
    
    # Compute the estimated conditional mean for all observations.
    mu_X = model.predict(X)
    
    # Plugin estimator: estimate of E[mu(X)^2]
    plugin_estimate = np.mean(mu_X ** 2)
    
    # Compute residuals for the correction term
    residuals = Y - mu_X
    
    # Influence function correction term: 2 * E[(Y - mu(X)) * mu(X)]
    correction_term = 2 * np.mean(residuals * mu_X)
    
    # IF-corrected estimate:
    if_estimate = plugin_estimate + correction_term
    
    return if_estimate

# --- Testing and plotting code ---

# 1. Generate a very large dataset (≈1e6 points)
N_large = 10**7
X_large, Y_large = make_regression(n_samples=N_large, n_features=10, noise=1.0, random_state=10)

# 2. Compute the plugin estimate for the large sample.
# We use RandomForestRegressor as in our functions.
full_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42)
full_model.fit(X_large, Y_large)
mu_large = full_model.predict(X_large)
plugin_large_estimate = np.mean(mu_large ** 2)

print("Plugin estimate on the large sample (1e6 points):", plugin_large_estimate)

# 3. Loop over different sample sizes and compute estimates.
sample_sizes = [100, 1000, 10000, 50000, 100000, 500000]
plugin_estimates = []
IF_estimates = []

for n in sample_sizes:
    X_sub = X_large[:n]
    Y_sub = Y_large[:n]
    
    # Plugin estimator for sub-sample: fit a model and compute mean(mu(X)^2)
    sub_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42)
    sub_model.fit(X_sub, Y_sub)
    mu_sub = sub_model.predict(X_sub)
    plugin_est = np.mean(mu_sub ** 2)
    plugin_estimates.append(plugin_est)
    
    # IF-based estimator on the sub-sample
    if_est = IF_estimator_squared_conditional(X_sub, Y_sub, estimator_type="rf")
    IF_estimates.append(if_est)
    print(f"\tSample size: {n}, Plugin estimate: {plugin_est}, IF estimate: {if_est}")

# 4. Plot the estimates vs sample size
plt.figure(figsize=(10, 6))
plt.axhline(y=plugin_large_estimate, color='black', linestyle='--', label='Large Sample Plugin Estimate (1e6)')
plt.plot(sample_sizes, plugin_estimates, marker='o', label='Plugin Estimate')
plt.plot(sample_sizes, IF_estimates, marker='s', label='IF-based Estimate')
plt.xscale('log')
plt.xlabel("Sample Size (log scale)")
plt.ylabel("Estimate of E[E[Y|X]^2]")
plt.title("Plugin vs. IF-based Estimators vs. Sample Size")
plt.legend()
plt.grid(True)
plt.show()