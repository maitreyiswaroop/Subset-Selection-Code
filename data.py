import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

def generate_data(m: int = 10, n_samples: int=10000, 
                  custom_X: np.ndarray=None, # hack yway to have own data with some covariance structure
                  custom_alpha: np.ndarray=None,
                  epsilon:int=0.001, k: int=7, seed: int=None, save_dir: str=None)->tuple:
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
    if seed:
        np.random.seed(seed)
    else:
        # draw a random seed
        seed = np.random.randint(0, 1000)
        np.random.seed(seed)
    # Draw X and Z uniformly from [0,1]
    if custom_X is None:
        X = np.random.uniform(0, 1, (n_samples, m))
    else:
        X = custom_X
    Z = np.random.uniform(0, 1, (n_samples, m))

    # Generate alpha with l0 constraint
    if custom_alpha is not None:
        alpha = custom_alpha
        k = np.count_nonzero(alpha)
    else:
        alpha = np.zeros(m)
        non_zero_indices = np.random.choice(m, k, replace=False)
        alpha[non_zero_indices] = np.random.uniform(0, 1, k)

    # Compute S(alpha)
    S_alpha = alpha * X + (1 - alpha) * Z

    # Generate outcome Y (for simplicity, let's assume Y is a linear combination of X)
    X_to_Y = np.random.uniform(0, 1, m)
    Y = np.dot(X, X_to_Y) + np.random.normal(0, epsilon, n_samples)

    if save_dir:
        with open(save_dir + 'X.pkl', 'wb') as f:
            pickle.dump(X, f)
        with open(save_dir + 'Y.pkl', 'wb') as f:
            pickle.dump(Y, f)
        with open(save_dir + 'S_alpha.pkl', 'wb') as f:
            pickle.dump(S_alpha, f)
        with open(save_dir + 'alpha.pkl', 'wb') as f:
            pickle.dump(alpha, f)
        with open(save_dir + 'X_to_Y.pkl', 'wb') as f:
            pickle.dump(X_to_Y, f)
        # saving a basic description of the dataset
        with open(save_dir + 'description.txt', 'w') as f:
            f.write(f"Dataset with {n_samples} samples and {m} covariates.\n")
            f.write(f"l0 constraint on alpha: {k}.\n")
            f.write(f"epsilon: {epsilon}.\n")
            f.write(f"seed: {seed}.\n")

    return X, Y, S_alpha, alpha, X_to_Y

def load_dataset(dataset_dir: str)->tuple:
    """
    Load the dataset from the given directory.

    Parameters:
    dataset_dir (str): Directory where the dataset is stored.

    Returns:
    X (np.ndarray): Covariates matrix of shape (n_samples, m).
    Y (np.ndarray): Outcome vector of shape (n_samples,).
    """
    with open(dataset_dir + 'X.pkl', 'rb') as f:
        X = pickle.load(f)
    with open(dataset_dir + 'Y.pkl', 'rb') as f:
        Y = pickle.load(f)
    with open(dataset_dir + 'S_alpha.pkl', 'rb') as f:
        S_alpha = pickle.load(f)
    with open(dataset_dir + 'alpha.pkl', 'rb') as f:
        alpha = pickle.load(f)
    with open(dataset_dir + 'X_to_Y.pkl', 'rb') as f:
        X_to_Y = pickle.load(f)

    return X, Y, S_alpha, alpha, X_to_Y

def plot_data(X, Y, Y_pred=None, title=None, separate=False, save_dir=None):
    """
    Plot the data.

    Parameters:
    X (np.ndarray): Covariates matrix of shape (n_samples, m).
    Y (np.ndarray): Outcome vector of shape (n_samples,).
    Y_pred (np.ndarray): Predicted outcome vector of shape (n_samples,).
    """
    if separate:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.scatter(X, Y, label='True', c='blue')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('True Data')
        ax1.legend()

        if Y_pred is not None:
            ax2.scatter(X, Y_pred, label='Predicted', c='red')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title('Predicted Data')
            ax2.legend()
    else:
        plt.scatter(X, Y, label='True', c = 'blue')
        if Y_pred is not None:
            plt.scatter(X, Y_pred, label='Predicted', c = 'red')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Data')
        plt.legend()

    if save_dir:
        plt.savefig(save_dir + 'data.png')
    else:
        plt.show()
    plt.close()