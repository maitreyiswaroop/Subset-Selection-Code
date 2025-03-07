import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_data(m1, m, n_samples, task = "regression", noise_scale=0.0, seed=None):
    assert m1 <= m, "m1 should be less than or equal to m"
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(0)
    # X ~ U[0,1]
    X = np.random.uniform(0, 1, (n_samples, m1))
    
    # random linear map A
    A = np.random.randn(m1)
    
    AX = X.dot(A)
    # AX = (AX - np.mean(AX)) / np.std(AX)
    
    # Y based on AX
    if task == "regression":
        Y = AX + noise_scale * np.random.randn(n_samples)
    elif task == "classification": # TODO: Check this
        AX = (AX - np.mean(AX)) / np.std(AX)
        Y = (AX > 0.5).astype(int)
    
    # new X with m dimensions
    new_X = np.zeros((n_samples, m))
    
    # copy m1 dimensions from original X
    new_X[:, :m1] = X
    
    # fill remaining dimensions with random values from U[0,1]
    new_X[:, m1:] = np.random.uniform(0, 1, (n_samples, m - m1))
    # shuffle the indices of the columns
    indices = np.random.permutation(m)
    new_X = new_X[:, indices]
    
    return new_X, Y, A, indices

# sanity check for closed-form solution
def sanity_check(m1=3, m=5, n_samples=10000, noise_scale=0.1, seed=None):
    # Generate data with known parameters
    X, Y, A, indices = generate_data(m1, m, n_samples, noise_scale=noise_scale, seed=seed)
    
    # Create true alpha (1 for original m1 features, 0 otherwise)
    true_alpha = np.zeros(m)
    original_indices = indices[:m1]  # First m1 indices after permutation
    true_alpha[original_indices] = 1
    
    # Create S(alpha) = alpha*X + (1-alpha)*Z
    Z = np.random.uniform(0, 1, X.shape)
    S = true_alpha * X + (1 - true_alpha) * Z
    
    # Closed-form solution
    closed_form = (1/12) * np.sum([(A[i]**2 * (1 - true_alpha[indices[i]])**2) / 
                                  (true_alpha[indices[i]]**2 + (1 - true_alpha[indices[i]])**2) 
                                  for i in range(m1)])
    
    print(f"Closed-form solution: {closed_form:.8f}")
    return X, Y, S, true_alpha, A, closed_form

# plugin estimator
def plugin_estimator(X, Y, S, model=LinearRegression()):
    """Plugin estimator using specified regression model"""
    # Split data to avoid overfitting
    X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(X, Y, S, test_size=0.3, random_state=42)
    
    # Fit models
    mu_model = model.fit(X_train, Y_train)
    gamma_model = model.fit(S_train, Y_train)
    
    # Get predictions
    mu = mu_model.predict(X_test) # mu = E[Y|X]
    gamma = gamma_model.predict(S_test) # gamma = E[Y|S]
    
    # Compute terms
    term1 = np.mean(mu**2)
    term2 = np.mean(gamma**2)
    return term1 - term2

# influence function based estimator
def if_estimator(X, Y, S, model=LinearRegression()):
    """IF-based estimator using specified regression model"""
    # Split data
    X_train, X_test, Y_train, Y_test, S_train, S_test = \
        train_test_split(X, Y, S, test_size=0.3, random_state=42)
    
    # Fit models
    mu_model = model.fit(X_train, Y_train) # mu = E[Y|X]
    gamma_model = model.fit(S_train, Y_train) # gamma = E[Y|S]
    
    # Get predictions
    mu = mu_model.predict(X_test)
    gamma = gamma_model.predict(S_test)
    
    # Compute IF correction terms
    plugin_estimate = np.mean(mu**2) - np.mean(gamma**2)
    correction = 2*(np.mean(mu*(Y_test - mu)) - np.mean(gamma*(Y_test - gamma)))
    
    return plugin_estimate + correction

def closed_form_estimator(A, alpha):
    """Closed-form solution using true parameters"""
    # Map alpha values to original feature indices
    # alpha_mapped = alpha[original_indices]
    alpha_mapped = alpha
    return (1/12) * np.sum([(a**2 * (1 - alpha_i)**2) / (alpha_i**2 + (1 - alpha_i)**2) 
                          for a, alpha_i in zip(A, alpha_mapped)])


def run_convergence_study(m1=3, m=12, 
                         n_samples_list=[500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 500000, 1000000],
                         n_alphas=10,
                         noise_scale=0.1,
                         data_seed=42):
    np.random.seed(data_seed)
    alpha_values = np.random.uniform(0, 1, (n_alphas, m))
    
    results = {
        'n_samples': [], 'alpha_id': [], 
        'plugin': [], 'if': [], 'closed_form': []
    }
    
    X_full, Y_full, A, indices = generate_data(m1, m, max(n_samples_list), 
                                              noise_scale=noise_scale, seed=data_seed)
    original_indices = indices[:m1]
    
    for n_samples in n_samples_list:
        # Use a random subset of the full dataset
        rand_indices = np.random.choice(max(n_samples_list), n_samples, replace=False)
        X = X_full[rand_indices]
        Y = Y_full[rand_indices]
        
        for alpha_idx, alpha_val in enumerate(alpha_values):
            alpha = alpha_val
            S_alpha = alpha * X + (1 - alpha) * np.random.uniform(0, 1, X.shape)
            

            X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(X, Y, S_alpha, test_size=0.3, random_state=42)
            mu_model = LinearRegression().fit(X_train, Y_train)
            gamma_model = LinearRegression().fit(S_train, Y_train)
            mu = mu_model.predict(X_test)
            gamma = gamma_model.predict(S_test)

            # plugin = plugin_estimator(X, Y, S_alpha)
            plugin = np.mean(mu**2) - np.mean(gamma**2)

            correction = 2*(np.mean(mu*(Y_test - mu)) - np.mean(gamma*(Y_test - gamma)))
            # if_est = if_estimator(X, Y, S_alpha)
            if_est = plugin + correction
            cf = closed_form_estimator(A, alpha)
            
            results['n_samples'].append(n_samples)
            results['alpha_id'].append(alpha_idx)
            results['plugin'].append(plugin)
            results['if'].append(if_est)
            results['closed_form'].append(cf)
    
    return results, alpha_values

def plot_convergence_results(results, alpha_values, save_path=None):
    sns.set_style("whitegrid")
    n_alphas = len(alpha_values)
    
    # Plot individual alpha convergence plots
    n_cols = 3
    n_rows = (n_alphas + n_cols - 1) // n_cols
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes1 = axes1.flatten()
    
    for alpha_idx in range(n_alphas):
        ax = axes1[alpha_idx]
        mask = np.array(results['alpha_id']) == alpha_idx
        n_samples = np.array(results['n_samples'])[mask]
        cf_val = np.array(results['closed_form'])[mask][0]
        
        ax.plot(n_samples, np.array(results['plugin'])[mask], 
               'o-', label='Plugin')
        ax.plot(n_samples, np.array(results['if'])[mask], 
               's--', label='IF')
        ax.axhline(y=cf_val, color='k', linestyle=':', label='Closed Form')
        
        ax.set_xscale('log')
        ax.set_xlabel('Dataset Size')
        ax.set_ylabel('Estimator Value')
        ax.set_title(f'α_{alpha_idx}')
        ax.legend()
    
    # Remove empty subplots
    for idx in range(n_alphas, len(axes1)):
        fig1.delaxes(axes1[idx])
    
    plt.tight_layout()
    
    # Average error plot
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    unique_n_samples = np.unique(results['n_samples'])
    avg_plugin_error = []
    avg_if_error = []
    
    for n in unique_n_samples:
        mask = np.array(results['n_samples']) == n
        plugin_errors = np.abs(np.array(results['plugin'])[mask] - 
                             np.array(results['closed_form'])[mask])
        if_errors = np.abs(np.array(results['if'])[mask] - 
                          np.array(results['closed_form'])[mask])
        
        avg_plugin_error.append(np.mean(plugin_errors))
        avg_if_error.append(np.mean(if_errors))
    
    ax2.loglog(unique_n_samples, avg_plugin_error, 'o-', label='Plugin')
    ax2.loglog(unique_n_samples, avg_if_error, 's--', label='IF')
    ax2.set_xlabel('Dataset Size')
    ax2.set_ylabel('Average Absolute Error')
    ax2.set_title('Average Error vs Dataset Size')
    ax2.legend()
    
    plt.tight_layout()
    if save_path is not None:
        fig1.savefig(save_path + '_alpha_convergence.png')
        fig2.savefig(save_path + '_average_error.png')
    else:
        plt.show()
    plt.close()

# Run simulation
results, alpha_values = run_convergence_study()
plot_convergence_results(results, alpha_values, save_path='results/estimators/convergence_study')