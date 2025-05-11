import matplotlib.pyplot as plt
import numpy as np
# from data import generate_data_continuous
from resnets import create_resnet_datasets

# plot Y vs X[i] for i in meaningful_indices
def plot_data(X, Y, meaningful_indices, save_dir=None):
    """
    Plot the data.
    
    Parameters:
    X (np.ndarray): Covariates matrix of shape (n_samples, m).
    Y (np.ndarray): Outcome vector of shape (n_samples,).
    A (np.ndarray): True coefficients vector of shape (m,).
    meaningful_indices (list): List of indices of meaningful features.
    """
    fig, axes = plt.subplots(1, len(meaningful_indices), figsize=(12, 6))
    # sort X,Y by X
    for i, idx in enumerate(meaningful_indices):
        axes[i].scatter(X[:, idx], Y, label=f'Feature {idx}', c='blue')
        axes[i].set_xlabel(f'X[{idx}]')
        axes[i].set_ylabel('Y')
        axes[i].set_title(f'Y vs X[{idx}]')
        axes[i].legend()
    
    if save_dir:
        plt.savefig(save_dir + 'resnet_data.png')
    else:
        plt.show()
    plt.close()

def main():
    # Generate data
    # new_X, Y, A, meaningful_indices = generate_data_continuous(
    #     pop_id=0, m1=4, m=20, 
    #     dataset_type="resnet", 
    #     dataset_size=1000,
    #     noise_scale=0.1, 
    #     seed=17, 
    #     common_meaningful_indices=[0,1]
    # )
    X_meaningful,Y = create_resnet_datasets(
            n = 1000,
            x_dist="normal",
            x_params=(0, 1),
            noise=0.1,
            x_dim=4,input_dim=4,
            hidden_dims=[10, 20, 80, 20, 10],
            num_blocks=5,
            use_conv=False,
            num_classes=None,
            seed=17 + 1*50,
            save = False,
            save_path='./results',
            visualise=True,
            activation='tanh',
            initialisation="kaiming")
    
    # Plot data
    # new_X = X_meaningful.reshape(-1, 4)
    # A = np.random.uniform(0.1, 2.0, size=new_X.shape[1])
    # meaningful_indices = [0, 1, 2, 3]
    # Y = Y.reshape(-1, 1)
    # # Plot the data
    # plot_data(new_X, Y, A, meaningful_indices, save_dir='./results/')

if __name__ == "__main__":
    main()