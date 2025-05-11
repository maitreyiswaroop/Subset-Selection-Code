# data_bike.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_bike_sharing_dataset(
    n_samples=None,
    n_noise_features=4,
    test_size=None,
    random_state=42,
    file_path="bike_sharing_dataset/hour.csv",  # Use the hour dataset
    scale_continuous=False,
    target_log_transform=False,
    clip_continuous=True,
    target_normalize=True
):
    """
    Load the Bike Sharing dataset (hour) from a local file and prepare it for variable selection.
    """
    try:
        print(f"Loading Bike Sharing dataset from local file: {file_path}")
        data = pd.read_csv(file_path)
        print(f"Successfully loaded data with {len(data)} rows")
    except FileNotFoundError:
        print(f"File not found at {file_path}. Please check the path.")
        return None
    
    # If we need to limit the sample size
    if n_samples is not None and n_samples < len(data):
        data = data.sample(n=n_samples, random_state=random_state)
    
    # Define features for the hour dataset:
    # We ignore dteday, yr, casual, and registered.
    cat_features = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
    cont_features = ['temp', 'atemp', 'hum', 'windspeed']
    target = 'cnt'
    
    # Check if the expected columns exist
    expected_columns = set(cat_features + cont_features + [target])
    missing_columns = expected_columns - set(data.columns)
    if missing_columns:
        print(f"Warning: The following expected columns are missing from the dataset: {missing_columns}")
    
    X = data[cat_features + cont_features].copy()
    y = data[target].values

    if target_normalize:
        target_scaler = MinMaxScaler()
        y = y.reshape(-1, 1)
        y = target_scaler.fit_transform(y)
        y = y.flatten()
    else:
        target_scaler = None  # In case you need it later to inverse-transform
    
    # Optionally transform the target (e.g., log transformation to reduce skew)
    if target_log_transform:
        y = np.log(y + 1e-3)
    
    # Create mapping of original features to their corresponding indices
    feature_groups = []
    all_feature_names = []
    
    cat_encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_cat = cat_encoder.fit_transform(X[cat_features])
    
    # Process categorical features: assign each categorical feature to a group
    cat_feature_names = []
    for i, feature in enumerate(cat_features):
        try:
            # Use categories (skipping the first due to drop='first')
            categories = cat_encoder.categories_[i][1:]
            encoded_names = [f"{feature}_{val}" for val in categories]
        except Exception:
            n_categories = X_cat.shape[1] // len(cat_features)
            encoded_names = [f"{feature}_{j+1}" for j in range(n_categories)]
        cat_feature_names.extend(encoded_names)
        
        start_idx = len(all_feature_names)
        end_idx = start_idx + len(encoded_names)
        feature_groups.append(list(range(start_idx, end_idx)))
        all_feature_names.extend(encoded_names)
    
    # Process continuous features
    X_cont = X[cont_features].values
    
    if clip_continuous:
        lower = np.percentile(X_cont, 1, axis=0)
        upper = np.percentile(X_cont, 99, axis=0)
        X_cont = np.clip(X_cont, lower, upper)
    
    if scale_continuous:
        scaler = StandardScaler()
        X_cont = scaler.fit_transform(X_cont)
    
    # # Each continuous feature is its own group
    # for i, feature in enumerate(cont_features):
    #     feature_groups.append([len(all_feature_names) + i])
    #     all_feature_names.append(feature)
    start_idx = len(all_feature_names)
    for i, feature in enumerate(cont_features):
        feature_groups.append([start_idx + i])
        all_feature_names.append(feature)

    
    # Combine categorical and continuous features
    X_real = np.hstack([X_cat, X_cont])
    
    # Record which groups are meaningful (all of the real features)
    meaningful_groups = list(range(len(feature_groups)))
    print(f"\t\t  Meaningful groups: {meaningful_groups}")
    print(f"\t\t  Feature groups: {feature_groups}")
    print(f"\t\t  All feature names: {all_feature_names}")
    
    # Generate noise categorical features with similar distributions
    np.random.seed(random_state)
    X_noise = np.empty((X.shape[0], 0))
    noise_feature_groups = []
    
    for i in range(n_noise_features):
        cat_idx = np.random.randint(0, len(cat_features))
        n_categories = len(cat_encoder.categories_[cat_idx])
        random_cats = np.random.randint(0, n_categories, size=X.shape[0])
        noise_cat_matrix = np.zeros((X.shape[0], n_categories - 1))
        for j in range(X.shape[0]):
            if random_cats[j] > 0:
                noise_cat_matrix[j, random_cats[j] - 1] = 1
        
        start_idx = X_real.shape[1] + X_noise.shape[1]
        end_idx = start_idx + noise_cat_matrix.shape[1]
        noise_feature_groups.append(list(range(start_idx, end_idx)))
        
        noise_feature_names = [f"noise_{i}_{j}" for j in range(n_categories - 1)]
        all_feature_names.extend(noise_feature_names)
        
        X_noise = np.hstack([X_noise, noise_cat_matrix])
    
    # print shapes
    print(f"\t\t  X_real shape: {X_real.shape}")
    print(f"\t\t  X_noise shape: {X_noise.shape}")
    X_combined = np.hstack([X_real, X_noise])
    print(f"\t\t  X_combined shape: {X_combined.shape}")
    feature_groups.extend(noise_feature_groups)
    
    if test_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=test_size, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = X_combined, None, y, None

    # printing ranges of values
    print(f"\t\t  X_train: min = {X_train.min()}, max = {X_train.max()}, mean = {X_train.mean()}")
    print(f"\t\t  y_train: min = {y_train.min()}, max = {y_train.max()}, mean = {y_train.mean()}")
    if X_test is not None:
        print(f"\t\t  X_test: min = {X_test.min()}, max = {X_test.max()}, mean = {X_test.mean()}")
        print(f"\t\t  y_test: min = {y_test.min()}, max = {y_test.max()}, mean = {y_test.mean()}")

    print(f"\t\t  X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    print('-'*50)
    print(f"\t\t  Meaningful groups: {meaningful_groups}")
    print(f"\t\t  Feature groups: {feature_groups}")
    print(f"\t\t  All feature names: {all_feature_names}")
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_groups': feature_groups,
        'meaningful_groups': meaningful_groups,
        'all_feature_names': all_feature_names
    }

def prepare_data_for_variable_selection(data_dict):
    """
    Prepare the data dictionary from load_bike_sharing_dataset
    for use with the gradient descent variable selection algorithm.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary returned by load_bike_sharing_dataset.
    
    Returns:
    --------
    X : torch.Tensor
        Features tensor.
    Y : torch.Tensor
        Target tensor.
    feature_group_sizes : list
        List containing the size of each feature group.
    meaningful_indices : numpy array
        Indices of meaningful features.
    """
    import torch
    
    X = torch.tensor(data_dict['X_train'], dtype=torch.float32)
    Y = torch.tensor(data_dict['y_train'], dtype=torch.float32)
    
    # Get the size of each feature group
    feature_group_sizes = [len(group) for group in data_dict['feature_groups']]
    
    # Get the indices of the meaningful groups
    meaningful_indices = np.array(data_dict['meaningful_groups'])
    
    return X, Y, feature_group_sizes, meaningful_indices

# Example usage:
if __name__ == "__main__":
    # Load the dataset
    bike_data = load_bike_sharing_dataset(n_samples=1000, n_noise_features=4)
    
    # Print some information
    print(f"\t Number of samples: {bike_data['X_train'].shape[0]}")
    print(f"\t Number of features: {bike_data['X_train'].shape[1]}")
    print(f"\t Number of feature groups: {len(bike_data['feature_groups'])}")
    print(f"\t Meaningful feature groups: {bike_data['meaningful_groups']}")
    
    for i, group in enumerate(bike_data['feature_groups']):
        is_meaningful = i in bike_data['meaningful_groups']
        status = "Meaningful" if is_meaningful else "Noise"
        print(f"\t Group {i} ({status}): {len(group)} columns, indices {group}")
        
        # Print feature names for this group
        names = [bike_data['all_feature_names'][idx] for idx in group]
        print(f"\t    Features: {names}")