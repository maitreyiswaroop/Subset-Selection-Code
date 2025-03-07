import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
import pickle
import matplotlib.pyplot as plt
from data import generate_data, load_dataset, plot_data
import argparse

parser = argparse.ArgumentParser(description='Dataset Viewer')
parser.add_argument('--dataset_no', type=int, default=1, help='Dataset number')
args = parser.parse_args()

def main():
    dataset_dir = 'datasets/dataset_' + str(args.dataset_no) + '/'
    X, Y, S_alpha, alpha, X_to_Y = load_dataset(dataset_dir)

    print(f'X shape: {X.shape}')
    print(f'Y shape: {Y.shape}')
    print(f'S_alpha shape: {S_alpha.shape}')
    print(f'alpha: {alpha}')
    print(f'X_to_Y shape: {X_to_Y.shape}')

if __name__ == '__main__':
    args = parser.parse_args()
    main()

