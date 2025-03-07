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


class PlugInConditionalVarianceEstimator:
    def __init__(self, n_estimators=100, random_state=None):
        self.conditional_expectation_estimator = LinearRegression()
        self.variance_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.fitted = False

    def fit(self, X, y):
        # Convert inputs to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
            
        self.conditional_expectation_estimator.fit(X, y)
        # conditional expectation
        conditional_expectation = self.conditional_expectation_estimator.predict(X)
        # residuals
        residuals = y - conditional_expectation
        # Fit variance model to squared residuals
        self.variance_model.fit(X, residuals**2)
        
        self.fitted = True
        return self

    def predict_variance(self, X):
        if not self.fitted:
            raise RuntimeError("Estimator must be fitted before making predictions")
            
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
            
        variance_predictions = self.variance_model.predict(X)
        
        # Ensure non-negative variance predictions
        return np.maximum(variance_predictions, 0)

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