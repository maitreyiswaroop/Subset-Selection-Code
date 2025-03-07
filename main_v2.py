import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
import pickle
import matplotlib.pyplot as plt
from data import generate_data, load_dataset, plot_data
import argparse

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
    t1 += np.sum([A[i]*A[j] for i in range(len(A)) for j in range(len(A)) if i != j])/4

    t2 = 0
    for i in range(len(A)):
        for j in range(len(A)):
            temp = A[i]*A[j]
            if i != j:
                temp*= 1/4
            else:
                if alpha[i] > 0:
                    temp*= (1-2*alpha[i]+5*alpha[i]**2)/(12*alpha[i]**2)
                    print(f'temp = {temp}')
                else:
                    temp*= 1/4
            t2 += temp
    
    return t1 - t2

def closed_form_solution2(alpha, A):
    ans = 0.0
    for i in range(len(A)):
        temp = A[i]**2
        print(f'A[i] = {A[i]}')
        print(f'alpha[i] = {alpha[i]}')
        if alpha[i] > 0:
            print("alpha[i] > 0")
            temp*= ((1.0-alpha[i])/alpha[i])**2 
            # temp /=(alpha[i]**2)
        # temp/= 12.0
        ans += temp
        print(f'ans = {ans}')
    return ans/12.0

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
    # custom_alpha = np.ones(args.dim_X)
    if args.dataset_dir:
        X, Y, S_alpha, alpha, X_to_Y = load_dataset(args.dataset_dir)
    else:
        # find the largest directory number
        data_dir = './datasets/dataset_1/'
        if not os.path.exists('./datasets/dataset_1/'):
            os.makedirs('./datasets/dataset_1/')
        else:
            dirs = [int(d.split('_')[-1]) for d in os.listdir('./datasets/') if d.startswith('dataset_')]
            if len(dirs) == 0:
                dir_num = 1
            else:
                dir_num = max(dirs) + 1
            os.makedirs(f'./datasets/dataset_{dir_num}/')
            data_dir = f'./datasets/dataset_{dir_num}/'
        X, Y, S_alpha, alpha, X_to_Y = generate_data(m=args.dim_X, n_samples=args.size, epsilon=args.epsilon, 
                                                     k=args.k, save_dir=data_dir, custom_X=args.custom_X)

    if args.dim_X == 1:
        X = X.reshape(-1, 1)
    
    # if args.k == 1:
    #     Y = Y.reshape(-1, 1)

    # partition the data into training and test sets
    X_train, X_test, Y_train, Y_test, S_alpha_train, S_alpha_test = train_test_split(X, Y, S_alpha, test_size=0.2)

    # print(f'Closed form solution: {closed_form_solution(alpha, X_to_Y)}')

    print(f'Closed form solution 2: {closed_form_solution2(alpha, X_to_Y)}')
    # print(f'Closed form solution 2: {closed_form_solution2(alpha, X_to_Y)/np.sqrt(X_train.shape[0])}')
    # plugin estimators
    # the target functional is \mathbb{E}_{S}[\mathbb{V}_{X}[\mathbb{E}[Y|X]|S(\alpha)]]
    # plugin estimate of E[Y|X]
    plugin_mu = PlugInConditionalExpectationEstimator()
    plugin_mu.fit(X_train, Y_train)

    # plugin estimate of the conditional variance of plugin_mu given S(alpha)
    plugin_cv_estimator = PlugInConditionalVarianceEstimator()
    plugin_cv_estimator.fit(S_alpha_train, plugin_mu.predict(X_train))

    # plugin estimate of the target functional
    plugin_target = plugin_cv_estimator.predict_variance(S_alpha_test).mean()

    # try using the other formulae

    print(f'Plugin estimate of the target functional: {plugin_target}')

    # IF based estimate
    IF_target = IF_based_estimate(X_train, Y_train, S_alpha_train)
    print(f'IF based estimate of the target functional: {IF_target}')


if __name__ == '__main__':
    main()