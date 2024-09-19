import numpy as np

def generate_data(m, n_samples, k):
    """
    Generate data drawn iid as X, Y ~ P consisting of m covariates X and an outcome Y.
    The marginal distribution of each coordinate is X_j ~ U[0,1].
    Also draw Z uniformly from [0,1] in each coordinate, independently.
    Enforce an l0 constraint on alpha to have ||alpha||_0 <= k.

    Parameters:
    m (int): Number of covariates.
    n_samples (int): Number of samples.
    k (int): l0 constraint on alpha.

    Returns:
    X (np.ndarray): Covariates matrix of shape (n_samples, m).
    Y (np.ndarray): Outcome vector of shape (n_samples,).
    S_alpha (np.ndarray): Transformed covariates matrix of shape (n_samples, m).
    alpha (np.ndarray): Weights vector of shape (m,).
    """
    # Draw X and Z uniformly from [0,1]
    X = np.random.uniform(0, 1, (n_samples, m))
    Z = np.random.uniform(0, 1, (n_samples, m))

    # Generate alpha with l0 constraint
    alpha = np.zeros(m)
    non_zero_indices = np.random.choice(m, k, replace=False)
    alpha[non_zero_indices] = np.random.uniform(0, 1, k)

    # Compute S(alpha)
    S_alpha = alpha * X + (1 - alpha) * Z

    # Generate outcome Y (for simplicity, let's assume Y is a linear combination of X)
    X_to_Y = np.random.uniform(0, 1, m)
    Y = np.dot(X, X_to_Y) + np.random.normal(0, 0.1, n_samples)

    return X, Y, S_alpha, alpha, X_to_Y