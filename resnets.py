import torch
import torch.nn as nn
import torch.nn.init as init
import math
import os
from visualisers import visualise_dataset
import numpy as np
import matplotlib.pyplot as plt

class CustomResNetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation, use_conv=False):
        super(CustomResNetBlock, self).__init__()
        self.use_conv = use_conv
        
        if use_conv:
            self.block = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim),
                activation,
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_dim)
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1),
                nn.BatchNorm2d(out_dim)
            ) if in_dim != out_dim else nn.Identity()
        else:
            self.block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                activation,
                nn.Linear(out_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
            self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
        self.activation = activation

    def forward(self, x):
        return self.activation(self.block(x) + self.shortcut(x))

class CustomResNet(nn.Module):
    def __init__(self, input_dim, 
                 hidden_dim, output_dim=1, 
                 num_blocks=5, 
                 use_conv=False, 
                 conv_params = None,
                 seed=None, activation='relu'):
        super(CustomResNet, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        
        # Set activation
        self.activation = (nn.ReLU() if activation == 'relu' else 
                         nn.Tanh() if activation == 'tanh' else 
                         nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU())
        
        self.use_conv = use_conv
        if use_conv:
            side_length = int(math.sqrt(input_dim))
            self.input_reshape = nn.Linear(input_dim, side_length * side_length)
            self.input_layer = nn.Sequential(
                nn.Conv2d(1, conv_params, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                self.activation
            )
        else:
            print("input dim: ", input_dim); print("hidden dim: ", hidden_dim)
            if type(hidden_dim) == list:
                hidden_dim = hidden_dim[0]
            self.input_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation
            )
        
        # Build residual blocks
        self.blocks = nn.ModuleList([
            CustomResNetBlock(hidden_dim, hidden_dim, self.activation, use_conv)
            for _ in range(num_blocks)
        ])
        
        # Output layer
        if use_conv:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def init_xavier(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
    
    def init_kaiming(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def init_he(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
    
    def init_normal(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def init_zeros(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.zeros_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def init_weights(self, init_type='xavier'):
        if init_type == 'xavier':
            self.apply(self.init_xavier)
        elif init_type == 'kaiming':
            self.apply(self.init_kaiming)
        elif init_type == 'he':
            self.apply(self.init_he)
        elif init_type == 'normal':
            self.apply(self.init_normal)
        elif init_type == 'zeros':
            self.apply(self.init_zeros)
    
    def forward(self, x):
        if self.use_conv:
            # Reshape for conv
            batch_size = x.size(0)
            side_length = int(math.sqrt(x.size(1)))
            x = self.input_reshape(x)
            x = x.view(batch_size, 1, side_length, side_length)
        
        # Input layer
        x = self.input_layer(x)
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        if self.use_conv:
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
        x = self.output_layer(x)
        
        return x

def create_dataset(n, x_dim, j, alpha):
    dataset = []
    for _ in range(n):
        if alpha == 0:
            x = torch.zeros(x_dim)
        else:
            x = torch.full((x_dim,), alpha)
        x[j] = torch.rand(1).item()
        dataset.append(x)
    # print(dataset)
    return dataset

def plot_resnet_output(model, n, x_dim, alphas, j, save_path):
    plt.figure(figsize=(10, 6))
    for i,alpha in enumerate(alphas):
        dataset = create_dataset(n, x_dim, j, alpha)
        inputs = torch.stack(dataset).to(torch.float32)
        # print(inputs.shape)
        # print(inputs.type())
        with torch.no_grad():
            outputs = model(inputs).detach().numpy()
            outputs = (outputs - np.min(outputs)) / (np.max(outputs) - np.min(outputs))
            # print(f"OUTPUT SHAPE:{outputs.shape}")
        plt.scatter(inputs[:, j].numpy(), outputs, label=f'alpha={alpha}', alpha=0.5)
        # plt.plot(inputs[:, j].numpy(), outputs, label=f'alpha={alpha}', alpha=0.5, marker='o')
        # if i==0:
        #     break
    plt.xlabel(f'Value of x[{j}]')
    plt.ylabel('Model Output')
    plt.title('ResNet Model Output vs Input Feature')
    plt.legend()
    plt.savefig(f'{save_path}/resnet_output_{j}.png')


def return_resnet_dataset(n: int, x_dist: str='uniform', x_params: tuple=(0, 1), noise: float=0.001, x_dim: int=1,
                          input_dim: int=1, hidden_dims: tuple=(64, 80, 96, 128, 160, 192, 224, 256, 224, 192, 160, 128, 96, 80, 64, 48, 32, 16, 8, 4), num_blocks: int=5, use_conv: bool=False,
                            num_classes: int=None, seed: int=17, visualise: bool=False, save_path: str=None, 
                            activation: str='relu', initialisation = 'random')->tuple:
    """
    Function to create a dataset using outputs from a ResNet model
    Args:
        n: int: number of samples
        x_dist: str: distribution to sample the input data from
        x_params: tuple: parameters for the input data distribution
        noise: float: noise to add to the output
        x_dim: int: dimension of the input data
        input_dim: int: dimension of the input data
        hidden_dims: list: hidden dimensions for the ResNet model
        num_blocks: int: number of blocks in the ResNet model
        use_conv: bool: use of convolutional layers in the ResNet model
        num_classes: int: number of output classes
        seed: int: random seed for reproducibility
        visualise: bool: whether to visualise the dataset
        save_path: str: path to save the datasets
    Returns:
        tuple: x, y
    """
    print('Creating ResNet dataset')
    print(hidden_dims)
    torch.manual_seed(seed)
    # generating the input data
    if x_dist == 'uniform':
        x = torch.rand(n, x_dim)*(x_params[1]-x_params[0])+x_params[0]
    elif x_dist == 'normal':
        x = torch.randn(n, x_dim)*x_params[1]+x_params[0]
    else:
        x = torch.rand(n, x_dim)*(x_params[1]-x_params[0])+x_params[0]

    # defining the ResNet model
    model = CustomResNet(input_dim=input_dim, 
                         hidden_dim=hidden_dims, 
                         output_dim=1, 
                         num_blocks=num_blocks, 
                         use_conv=use_conv, 
                         conv_params = None,
                         seed=seed, 
                         activation=activation)
    if initialisation == 'random':
        print('Initialising the model parameters randomly')
        # initialising the model parameters randomly
        model.apply(model.init_weights)
    elif initialisation == 'xavier':
        print('Initialising the model parameters using Xavier initialisation')
        # initialising the model parameters using Xavier initialisation
        model.apply(model.init_xavier)
    elif initialisation == 'he':
        print('Initialising the model parameters using He initialisation')
        # initialising the model parameters using He initialisation
        model.apply(model.init_he)
    elif initialisation == 'kaiming':
        print('Initialising the model parameters using Kaiming initialisation')
        # initialising the model parameters using Kaiming initialisation
        model.apply(model.init_kaiming)
    else:
        print('Initialising the model parameters randomly')
        model.apply(model.init_weights)
    model.eval()
    with torch.no_grad():
        # passing the input data through the ResNet model
        y = model(x)
        # scale y to 01
        y = (y-torch.min(y))/(torch.max(y)-torch.min(y))
        print("Shapes of x and y:")
        print(x.shape, y.shape)
        print(x.type(), y.type())

        for i in range(x_dim):
            if visualise:
                save_file = os.path.join(save_path, 'plots', 'denoised_fns', f'denoised_f_{i}_x.png')
                visualise_dataset(x[:, i], y, f'denoised_f_{i}(x)', show=False, save_path=save_file)
        
        # adding noise to the output
        y += noise * torch.randn_like(y)

        for i in range(x_dim):
            if visualise:
                save_file = os.path.join(save_path, 'plots', 'fns', f'f_{i}_x.png')
                visualise_dataset(x[:, i], y, f'f_{i}(x)', show=False, save_path=save_file)

        if save_path is not None:
            # saving the model
            torch.save(model.state_dict(), os.path.join(save_path, 'resnet_model.pt'))
            # saving a model summary
            with open(os.path.join(save_path, 'resnet_model_summary.txt'), 'w') as file:
                file.write(str(model))
        if visualise:
            n = 1000
            alphas = [0, 0.25, 0.5, 0.75, 1.0]
            for j in range(x_dim):
                print(f'Plotting the output of the ResNet model for x[{j}]')
                plot_resnet_output(model, n, x_dim, alphas, j, save_path=os.path.join(save_path, 'plots', 'outputs'))
        return x, y


def create_resnet_datasets(n: int, x_dist: str='uniform', 
                           x_params: tuple=(0, 1), 
                           noise: float=0.01, x_dim: int=1,
                          input_dim: int=1, hidden_dims: list=[10, 20, 30, 40, 50], 
                          num_blocks: int=5, use_conv: bool=False,
                          num_classes: int=None, seed: int=17, save: bool=True, save_path: str='data/',
                          visualise: bool=False, activation: str='relu', initialisation = 'random')->tuple:
    """
    Function to create a dataset using outputs from a ResNet model
    Args:
        n: int: number of samples
        x_dist: str: distribution to sample the input data from
        x_params: tuple: parameters for the input data distribution
        noise: float: noise to add to the output
        x_dim: int: dimension of the input data
        input_dim: int: dimension of the input data
        hidden_dims: list: hidden dimensions for the ResNet model
        num_blocks: int: number of blocks in the ResNet model
        use_conv: bool: use of convolutional layers in the ResNet model
        num_classes: int: number of output classes
        seed: int: random seed for reproducibility
        save: bool: whether to save the datasets
        save_path: str: path to save the datasets
        visualise: bool: whether to visualise the dataset
    Returns:
        tuple: x, y
    """
    if save:
        # Create all required directories
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'plots/fns'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'plots/denoised_fns'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'plots/outputs'), exist_ok=True)
    x, y = return_resnet_dataset(n, x_dist, x_params, noise, x_dim, input_dim, hidden_dims, num_blocks, use_conv, num_classes, seed, visualise, save_path, activation, initialisation)
    x = (x - x.mean())/x.std()
    y = (y - y.mean())/y.std()
    x = x.type(torch.float32)
    y = y.type(torch.float32)
    if save:
        torch.save(x, os.path.join(save_path, 'x.pt'))
        torch.save(y, os.path.join(save_path, 'y.pt'))
    return x, y