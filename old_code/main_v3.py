import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

class PlugInConditionalVarianceEstimator:
    def __init__(self, n_estimators=100, random_state=None):
        self.conditional_expectation_estimator = PlugInConditionalExpectationEstimator(n_estimators=n_estimators, random_state=random_state)
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
    print(f'checking t1 {np.sum([A[i]*A[j] for i in range(len(A)) for j in range(len(A)) if i != j])/4}')
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

def closed_form_solution2(alpha, A):
    ans = 0.0
    for i in range(len(A)):
        temp = A[i]**2
        # print(f'A[i] = {A[i]}')
        if alpha[i] > 0:
            temp*= ((1.0-alpha[i])/alpha[i])**2 
            # temp /=(alpha[i]**2)
        temp/= 12.0
        ans += temp
        # print(f'ans = {ans}')
    return ans

def IF_based_estimate(X, Y, S_alpha):
    # plugin estimator for E[Y|X]
    plugin_mu_x_y = PlugInConditionalExpectationEstimator()
    plugin_mu_x_y.fit(X, Y)

    # t1 = plugin_mu_x_y.predict(X)**2 + 2*plugin_mu_x_y.predict(X)(Y - plugin_mu_x_y.predict(X))
    t1 = plugin_mu_x_y.predict(X)**2 + 2*plugin_mu_x_y.predict(X)*(Y - plugin_mu_x_y.predict(X))
    t1 = t1.mean()

    plugin_mu_s_y = PlugInConditionalExpectationEstimator()
    plugin_mu_s_y.fit(S_alpha, Y)

    t2 = plugin_mu_s_y.predict(S_alpha)**2 + 2*plugin_mu_s_y.predict(S_alpha)*(Y - plugin_mu_s_y.predict(S_alpha))
    t2 = t2.mean()

    return t1 - t2

def main():
    results_dict = {'seed': [], 'dataset_size': [], 'alpha': [], 'A': [], 'closed_form_solution_1': [], 'closed_form_solution_2': [], 'plugin_target_1': [], 'plugin_target_2': [], 'IF_target': []}
    np.random.seed(0)
    custom_alpha = np.array([0.0, 0.81532217])
    dataset_size = 1000000

    # custom_alpha = np.ones(args.dim_X)
    if args.dataset_dir:
        X, Y, S_alpha, alpha, X_to_Y = load_dataset(args.dataset_dir)
    else:
        X, Y, S_alpha, alpha, X_to_Y = generate_data(m=args.dim_X, n_samples=dataset_size, epsilon=args.epsilon, 
                                                    k=args.k, save_dir=None, custom_X=args.custom_X, seed=0, custom_alpha=custom_alpha)

    if args.dim_X == 1:
        X = X.reshape(-1, 1)
    
    # if args.k == 1:
    #     Y = Y.reshape(-1, 1)

    for seed in range(10):
        np.random.seed(seed)
        n_samples = 10000+ seed*10000

        selected_indices = np.random.choice(X.shape[0], n_samples, replace=False)

        X_train, Y_train, S_alpha_train = X[selected_indices], Y[selected_indices], S_alpha[selected_indices]

        closed_form_solution_value_1 = closed_form_solution(alpha, X_to_Y)
        print(f'\tClosed form solution1: {closed_form_solution_value_1}')
        closed_form_solution_value_2 = closed_form_solution2(alpha, X_to_Y)
        print(f'\tClosed form solution 2: {closed_form_solution_value_2}')
        # print(f'Closed form solution 2: {closed_form_solution2(alpha, X_to_Y)/np.sqrt(X_train.shape[0])}')
        # plugin estimators
        # the target functional is \mathbb{E}_{S}[\mathbb{V}_{X}[\mathbb{E}[Y|X]|S(\alpha)]]
        # plugin estimate of E[Y|X] : Method 1
        plugin_mu = PlugInConditionalExpectationEstimator()
        plugin_mu.fit(X_train, Y_train)

        # plugin estimate of the conditional variance of plugin_mu given S(alpha)
        plugin_cv_estimator = PlugInConditionalVarianceEstimator()
        plugin_cv_estimator.fit(S_alpha_train, plugin_mu.predict(X_train))

        # plugin estimate of the target functional
        plugin_target_1 = plugin_cv_estimator.predict_variance(S_alpha_train).mean()

        print(f'\tPlugin estimate of the target functional via conditional variance: {plugin_target_1}')

        # plugin estimate of the target functional regression on squar4ed expectations: Method 2
        plugin_mu_x_y = PlugInConditionalExpectationEstimator()
        plugin_mu_x_y.fit(X_train, Y_train)
        plugin_mu_s_y = PlugInConditionalExpectationEstimator()
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

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv('results_4.csv', index=False)

    # plot closed form vs plugin and closed form vs IF
    plt.figure()
    plt.plot(results_df['closed_form_solution_1'], results_df['plugin_target_1'], 'o', label='Plugin 1')
    plt.plot(results_df['closed_form_solution_1'], results_df['IF_target'], 'o', label='IF')
    plt.xlabel('Closed form solution')
    plt.ylabel('Estimate')
    plt.legend()
    plt.savefig('closed_form_vs_plugin_and_IF.png')
    plt.close()

    # plot closed form vs dataset size
    plt.figure()
    plt.plot(results_df['closed_form_solution_1'], results_df['dataset_size'], 'o')
    plt.xlabel('Closed form solution')
    plt.ylabel('Dataset size')
    plt.savefig('closed_form_vs_dataset_size.png')
    plt.close()

if __name__ == '__main__':
    main()