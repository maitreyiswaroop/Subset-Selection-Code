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
seed = 42
np.random.seed(seed)
# 1. Generate a very large dataset (≈1e6 points)
N_large = 10**6
X_large, Y_large = make_regression(n_samples=N_large, n_features=10, noise=1.0, random_state=seed)
alpha = np.random.uniform(0.1, 1.0, size=X_large.shape[1])
cov_matrix = np.diag(alpha)
S_alpha = np.random.multivariate_normal(
    mean=np.zeros(X_large.shape[1]),
    cov=cov_matrix,
    size=N_large) + X_large  


# 2. Compute the plugin estimate for the large sample.
# We use RandomForestRegressor as in our functions.
full_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42)
full_model.fit(X_large, Y_large)
mu_large = full_model.predict(X_large)
plugin_large_estimate_term1 = np.mean(mu_large ** 2)

full_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42)
full_model.fit(S_alpha, Y_large)
gamma_large = full_model.predict(S_alpha)
plugin_large_estimate_term2 = np.mean(gamma_large ** 2)

print("Plugin estimate on the large sample (1e6 points):\n\tTerm 1: ", plugin_large_estimate_term1)
print("\tTerm 2: ", plugin_large_estimate_term2)

print("\tObjective: ", plugin_large_estimate_term1 - plugin_large_estimate_term2)
# 3. Loop over different sample sizes and compute estimates.
sample_sizes = [100, 1000, 10000, 50000, 100000, 500000]
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
    sub_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42)
    sub_model.fit(X_sub, Y_sub)
    mu_sub = sub_model.predict(X_sub)
    plugin_est = np.mean(mu_sub ** 2)
    plugin_estimates_term1.append(plugin_est)
    
    # IF-based estimator on the sub-sample
    if_est = IF_estimator_squared_conditional(X_sub, Y_sub, estimator_type="rf")
    IF_estimates_term1.append(if_est)
    print(f"\tSample size: {n}, Plugin estimate: {plugin_est}, IF estimate: {if_est}")

    # Plugin estimator for sub-sample: fit a model and compute mean(mu(S_alpha)^2)
    sub_model = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42)
    sub_model.fit(S_sub, Y_sub)
    gamma_sub = sub_model.predict(S_sub)
    plugin_est2 = np.mean(gamma_sub ** 2)
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
plt.figure(figsize=(15, 12))

# Plot for Term 1
plt.subplot(3, 1, 1)
plt.axhline(y=plugin_large_estimate_term1, color='black', linestyle='--', label='Large Sample Plugin Estimate (1e6)')
plt.plot(sample_sizes, plugin_estimates_term1, marker='o', label='Plugin Estimate (Term 1)')
plt.plot(sample_sizes, IF_estimates_term1, marker='s', label='IF-based Estimate (Term 1)')
plt.xscale('log')
plt.xlabel("Sample Size (log scale)")
plt.ylabel("Estimate of Term 1")
plt.title("Term 1: Plugin vs. IF-based Estimators vs. Sample Size")
plt.legend()
plt.grid(True)

# Plot for Term 2
plt.subplot(3, 1, 2)
plt.axhline(y=plugin_large_estimate_term2, color='black', linestyle='--', label='Large Sample Plugin Estimate (1e6)')
plt.plot(sample_sizes, plugin_estimates_term2, marker='o', label='Plugin Estimate (Term 2)')
plt.plot(sample_sizes, IF_estimates_term2, marker='s', label='IF-based Estimate (Term 2)')
plt.xscale('log')
plt.xlabel("Sample Size (log scale)")
plt.ylabel("Estimate of Term 2")
plt.title("Term 2: Plugin vs. IF-based Estimators vs. Sample Size")
plt.legend()
plt.grid(True)

# Plot for Objective
plt.subplot(3, 1, 3)
plt.axhline(y=plugin_large_estimate_term1 - plugin_large_estimate_term2, color='black', linestyle='--', label='Large Sample Plugin Objective (1e6)')
plt.plot(sample_sizes, plugin_estimates_objective, marker='o', label='Plugin Objective Estimate')
plt.plot(sample_sizes, IF_estimates_objective, marker='s', label='IF-based Objective Estimate')
plt.xscale('log')
plt.xlabel("Sample Size (log scale)")
plt.ylabel("Objective Estimate")
plt.title("Objective: Plugin vs. IF-based Estimators vs. Sample Size")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"./if_vs_plugin/if_vs_plugin_comparison_seed_{seed}.png")
plt.close()