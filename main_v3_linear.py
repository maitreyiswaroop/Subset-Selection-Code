import numpy as np
from sklearn.ensemble import RandomForestRegressor
# linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
import pickle
import matplotlib.pyplot as plt
from data import generate_data, load_dataset, plot_data
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Conditional Density Estimation')
parser.add_argument('--dim_X', type=int, default=2, help='Dimension of X')
parser.add_argument('--k', type=int, default=1, help='Dimension of Y')
parser.add_argument('--size', type=int, default=10000, help='Size of the dataset')
parser.add_argument('--epsilon', type=float, default=0.01, help='Noise level')
parser.add_argument('--dataset_dir', type=str, default=None, help='Directory to load the dataset')
parser.add_argument('--custom_X', type=str, default=None, help='Custom X')
args = parser.parse_args()


class PlugInConditionalExpectationEstimator:
    def __init__(self, n_estimators=100, random_state=None):
        self.model = LinearRegression()

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

class PlugInConditionalVarianceEstimator:
    def __init__(self, n_estimators=100, random_state=None):
        self.conditional_expectation_estimator = LinearRegression()
        self.variance_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)        

    def fit(self, X, Y):
        self.conditional_expectation_estimator.fit(X, Y)
        # conditional expectation
        conditional_expectation = self.conditional_expectation_estimator.predict(X)
        # residuals
        residuals = Y - conditional_expectation
        # Fit a new model to the squared residuals
        self.variance_model.fit(X, residuals**2)

    def predict_variance(self, X):
        return self.variance_model.predict(X)

def closed_form_solution(alpha, A):
    print(f'alpha = {alpha}')
    print(f'A = {A}')
    t1 = np.sum(A**2)/3
    # 1/2 * sum_{i\neq j} A_i A_j
    # print(f'checking t1 {np.sum([A[i]*A[j] for i in range(len(A)) for j in range(len(A)) if i != j])/4}')
    t1 += np.sum([A[i]*A[j] for i in range(len(A)) for j in range(len(A)) if i != j])/4

    t2 = 0
    for i in range(len(A)):
        for j in range(len(A)):
            temp = A[i]*A[j]
            if i != j:
                temp/= 4.0
            else:
                if alpha[i] > 0:
                    temp*= (1.0-2.0*alpha[i]+5.0*alpha[i]**2)/(12.0*alpha[i]**2)
                    # print(f'temp = {temp}')
                else:
                    temp/= 4.0
            t2 += temp
    
    return t1 - t2

def E_S_variance(alpha):
    term1 = 0.0
    term2 = 0.0
    term3 = 0.0
    if alpha<0.5:
        term1 = alpha**3 /(48.0*(1-alpha)**3)
        term2 = alpha**2 *(1.0-2.0*alpha)/(12.0*(1-alpha)**3)
        term3 = alpha**3 /(48.0*(1.0-alpha)**3)
    elif alpha == 0.5:
        term1 = 1/48.0
        term2 = 0.0
        term3 = 1/48.0
    else:
        term1 = (1.0-alpha)/(48.0*alpha)
        term2 = (2.0*alpha-1)/(12.0*alpha)
        term3 = -1.0/48.0 + 1.0/(48.0*alpha)

    return term1 + term2 + term3

def closed_form_solution2(alpha, A):
    ans = 0.0
    for i,a in enumerate(A):
        if alpha[i]==0:
            ans+=a**2 / 12.0
        else:
            # sum+= (a*(1.0-alpha[i]))**2
            # sum+= ((1.0-alpha[i])*alpha[i])* a**2
            ans+= a**2 * E_S_variance(alpha[i])* (1.0 - alpha[i])**2 / alpha[i]**2
    return ans

def IF_based_estimate(X, Y, S_alpha):
    # plugin estimator for E[Y|X]
    plugin_mu_x_y = LinearRegression()
    plugin_mu_x_y.fit(X, Y)

    # t1 = plugin_mu_x_y.predict(X)**2 + 2*plugin_mu_x_y.predict(X)(Y - plugin_mu_x_y.predict(X))
    t1 = plugin_mu_x_y.predict(X)**2 + 2*plugin_mu_x_y.predict(X)*(Y - plugin_mu_x_y.predict(X))
    t1 = t1.mean()

    plugin_mu_s_y = LinearRegression()
    plugin_mu_s_y.fit(S_alpha, Y)

    t2 = plugin_mu_s_y.predict(S_alpha)**2 + 2*plugin_mu_s_y.predict(S_alpha)*(Y - plugin_mu_s_y.predict(S_alpha))
    t2 = t2.mean()

    return t1 - t2
    
def test_variance_across_S_alpha(X,S_alpha, alpha, save_dir=None):
    # in our expression for Var[X|S(alpha)], we found that the variance of X|S(alpha) is independent of S(alpha)
    # here we try to empirically verify this 
    # we draw paired samples of X and S_alpha
    # &bin them by dociles of S_alpha
    # and then compute the variance of X in each bin
    n_bins = 10
    fig, axes = plt.subplots(len(alpha), 1, figsize=(10, 5 * len(alpha)))
    for i in range(len(alpha)):
        bin_indices = np.digitize(S_alpha[:, i], np.linspace(0, 1, n_bins))
        variances = []
        for j in range(n_bins):
            variances.append(np.var(X[bin_indices == j, i]))
        axes[i].plot(variances)
        axes[i].set_title(f'Variance of X dimension {i} across S_alpha with alpha[{i}] = {alpha[i]}')
        axes[i].set_xlabel('Bin number')
        axes[i].set_ylabel('Variance of X')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'variance_across_S_alpha_dimensions.png'))
    plt.close()

    return variances


def main():
    for run_no in range(2, 10):
        results_dir = f'./results/run_{run_no}/'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results_dict = {'seed': [], 'dataset_size': [], 'alpha': [], 'A': [], 'closed_form_solution_1': [], 'closed_form_solution_2': [], 'plugin_target_1': [], 'plugin_target_2': [], 'IF_target': [], 'IF_error': [], 'plugin_error': [], 'plugin_error_2': []}
        np.random.seed(17)
        # custom_alpha = np.array([0.0, 0.81532217])
        dataset_size = 1000000

        # custom_alpha = np.ones(args.dim_X)
        if args.dataset_dir:
            X, Y, S_alpha, alpha, X_to_Y = load_dataset(args.dataset_dir)
        else:
            X, Y, S_alpha, alpha, X_to_Y = generate_data(m=run_no, n_samples=dataset_size, epsilon=args.epsilon, 
                                                        k=run_no//2 +1, save_dir=None, custom_X=args.custom_X, seed=0)

        if args.dim_X == 1:
            X = X.reshape(-1, 1)
        
        # if args.k == 1:
        #     Y = Y.reshape(-1, 1)
        closed_form_solution_value_1 = closed_form_solution(alpha, X_to_Y)
        print(f'\tClosed form solution1: {closed_form_solution_value_1}')
        closed_form_solution_value_2 = closed_form_solution2(alpha, X_to_Y)
        print(f'\tClosed form solution 2: {closed_form_solution_value_2}')
        for seed in range(1, 1000, 100):
            # if seed > 300:
            #     break
            print('='*10)
            print(f'Seed: {seed}')
            np.random.seed(seed + 1)
            n_samples = seed*1000

            selected_indices = np.random.choice(X.shape[0], n_samples, replace=False)

            X_train, Y_train, S_alpha_train = X[selected_indices], Y[selected_indices], S_alpha[selected_indices]
            # print(f'Closed form solution 2: {closed_form_solution2(alpha, X_to_Y)/np.sqrt(X_train.shape[0])}')
            # plugin estimators
            # the target functional is \mathbb{E}_{S}[\mathbb{V}_{X}[\mathbb{E}[Y|X]|S(\alpha)]]
            # plugin estimate of E[Y|X] : Method 1
            # linear regression
            plugin_mu = LinearRegression()
            plugin_mu.fit(X_train, Y_train)

            # plugin estimate of the conditional variance of plugin_mu given S(alpha)
            plugin_cv_estimator = PlugInConditionalVarianceEstimator()
            plugin_cv_estimator.fit(S_alpha_train, plugin_mu.predict(X_train))

            # plugin estimate of the target functional
            plugin_target_1 = plugin_cv_estimator.predict_variance(S_alpha_train).mean()

            print(f'\tPlugin estimate of the target functional via conditional variance: {plugin_target_1}')

            # plugin estimate of the target functional regression on squar4ed expectations: Method 2
            plugin_mu_x_y = LinearRegression()
            plugin_mu_x_y.fit(X_train, Y_train)
            # print the coefficients
            print(f'\t\tLinear regression Coefficients: {plugin_mu_x_y.coef_}')
            plugin_mu_s_y = LinearRegression()
            plugin_mu_s_y.fit(S_alpha_train, Y_train)

            t1 = plugin_mu_x_y.predict(X_train)**2 
            t1 = t1.mean()

            t2 = plugin_mu_s_y.predict(S_alpha_train)**2
            t2 = t2.mean()

            plugin_target_2 = t1 - t2
            print(f'\tPlugin estimate of the target functional via squared expectations: {plugin_target_2}')

            # IF based estimate
            IF_target = IF_based_estimate(X_train, Y_train, S_alpha_train)
            print(f'\tIF based estimate of the target functional: {IF_target}')

            results_dict['seed'].append(seed)
            results_dict['alpha'].append(alpha)
            results_dict['A'].append(X_to_Y)
            results_dict['closed_form_solution_1'].append(closed_form_solution_value_1)
            results_dict['closed_form_solution_2'].append(closed_form_solution_value_2)
            results_dict['plugin_target_1'].append(plugin_target_1)
            results_dict['plugin_target_2'].append(plugin_target_2)
            results_dict['IF_target'].append(IF_target)
            results_dict['dataset_size'].append(X_train.shape[0])
            # error for IF based estimate
            results_dict['IF_error'].append(closed_form_solution_value_2-IF_target)
            # print(f'IF error: {results_dict["IF_error"]}')
            # error for plugin based estimate
            results_dict['plugin_error'].append(closed_form_solution_value_2- plugin_target_1)
            results_dict['plugin_error_2'].append(closed_form_solution_value_2-plugin_target_2)

        print('all okay till here')
        results_dict = pd.DataFrame(results_dict)

        print('all okay till here2')
        # variance across S_alpha - verification
        # test_variance_across_S_alpha(X, S_alpha, alpha, save_dir=results_dir)

        # Check for NaN or infinite values
        if results_dict['dataset_size'].isnull().any() or results_dict['IF_error'].isnull().any():
            print("NaN values found in data")
        if np.isinf(results_dict['dataset_size']).any() or np.isinf(results_dict['IF_error']).any():
            print("Infinite values found in data")
        # print('all okay till here3')
        # error plots wrt to dataset size
        plt.figure()
        # print('all okay till here4')
        plt.plot(results_dict['dataset_size'], results_dict['plugin_error'], label='Plugin 1: conditional variance')
        plt.plot(results_dict['dataset_size'], results_dict['plugin_error_2'], label='Plugin 2')
        plt.plot(results_dict['dataset_size'], results_dict['IF_error'], label='IF', linestyle='--')
        plt.xlabel('Dataset size')
        plt.ylabel('Error')
        plt.legend()
        plt.savefig(f'{results_dir}error_vs_dataset_size.png')
        plt.close()

        # plot the vlaues 
        plt.figure()
        # print('all okay till here4')
        plt.plot(results_dict['dataset_size'], results_dict['closed_form_solution_2'], label='Closed form solution')
        plt.plot(results_dict['dataset_size'], results_dict['plugin_target_1'], label='Plugin 1: conditional variance')
        plt.plot(results_dict['dataset_size'], results_dict['plugin_target_2'], label='Plugin 2')
        plt.plot(results_dict['dataset_size'], results_dict['IF_target'], label='IF', linestyle='--')
        plt.xlabel('Dataset size')
        plt.ylabel('Target functional')
        plt.legend()
        plt.savefig(f'{results_dir}target_functional_convergence.png')
        plt.close()

        results_dict.to_csv(f'{results_dir}results_4_linear.csv', index=False)


        # save the generation params to a .txt file
        with open(f'{results_dir}generation_params.txt', 'w') as f:
            f.write(f'alpha: {alpha}\n')
            f.write(f'A: {X_to_Y}\n')
            f.write(f'closed_form_solution_1: {closed_form_solution_value_1}\n')
            f.write(f'closed_form_solution_2: {closed_form_solution_value_2}\n')
            f.write(f'k: {run_no}\n')
            f.write(f'seed: 0\n')
            f.write(f'epsilon: {args.epsilon}\n')
            f.write(f'dataset_size: {dataset_size}\n')

        # # plot closed form vs plugin and closed form vs IF
        # plt.figure()
        # plt.plot(results_dict['closed_form_solution_2'], results_dict['plugin_target_1'], label='Plugin 1')
        # plt.plot(results_dict['closed_form_solution_2'], results_dict['IF_target'], label='IF')
        # plt.xlabel('Closed form solution')
        # plt.ylabel('Estimate')
        # plt.legend()
        # plt.savefig(f'{results_dir}closed_form_vs_plugin_and_IF.png')
        # plt.close()

        # # plot closed form vs dataset size
        # plt.figure()
        # plt.plot(results_dict['closed_form_solution_1'], results_dict['dataset_size'], 'o')
        # plt.xlabel('Closed form solution')
        # plt.ylabel('Dataset size')
        # plt.savefig(f'{results_dir}closed_form_vs_dataset_size.png')
        # plt.close()

if __name__ == '__main__':
    main()