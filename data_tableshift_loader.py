# data_tableshift_loader.py
import numpy as np
import pandas as pd
try:
    from tableshift import get_dataset
except ImportError:
    print("TableShift library not found. Please install it: pip install tableshift")
    # You might also need to set up its environment if it has complex dependencies,
    # refer to the TableShift GitHub README.
    get_dataset = None # Define to prevent NameError if import fails early

from sklearn.preprocessing import OneHotEncoder    # << add this import
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_tableshift_data_as_populations(
    tableshift_string_identifier: str,
    cache_dir: str = "./tableshift_cache",
    verbose: bool = True,
    preprocessing: str = 'zero_fill',              # << new arg
    onehot_kwargs: dict = None                     # << optional args for OneHotEncoder
):
    """
    Loads data for a given TableShift dataset and structures it as a list of
    populations, where each population corresponds to a unique domain.

    The data from 'train', 'validation', 'id_test', 'ood_validation', 
    and 'ood_test' splits are combined to form the populations.

    Args:
        tableshift_string_identifier (str): The TableShift name for the dataset
            (e.g., "acsincome", "acsunemployment", "acsfoodstamps").
        cache_dir (str): Path to directory where TableShift should cache datasets.
        verbose (bool): Whether to print information during loading.

    Returns:
        list: A list of dictionaries, where each dictionary represents a domain (population)
              and contains:
              - 'pop_id': The domain identifier (as a string).
              - 'X_raw': NumPy array of features for that domain.
              - 'Y_raw': NumPy array of outcomes for that domain (as a 1D array).
              - 'meaningful_indices': NumPy array of all feature indices 
                                      (np.arange(n_features)).
    """
    if get_dataset is None:
        print("TableShift is not available. Cannot load data.")
        return []
        
    if verbose:
        print(f"Loading TableShift dataset: {tableshift_string_identifier}...")

    try:
        # Using a cache_dir is highly recommended for TableShift
        dset = get_dataset(tableshift_string_identifier, cache_dir=cache_dir)
    except Exception as e:
        print(f"Error loading dataset {tableshift_string_identifier} from TableShift: {e}")
        print("Please ensure you have the TableShift environment set up correctly and "
              "any necessary permissions/downloads for this dataset are handled "
              "(see TableShift documentation).")
        return []

    # Standard splits defined by TableShift
    splits_to_load = ['train', 'validation', 'id_test', 'ood_validation', 'ood_test']
    
    all_X_list = []
    all_y_list = []
    all_domain_list = [] # To store domain indicators

    for split in splits_to_load:
        try:
            if verbose:
                print(f"  Attempting to load split: {split}...")
            # The get_pandas method returns X, y, domain_indicator, group_indicator
            X_pd, y_pd, domain_pd, _ = dset.get_pandas(split)
            
            if X_pd is not None and not X_pd.empty:
                all_X_list.append(X_pd)
                all_y_list.append(y_pd)
                all_domain_list.append(domain_pd) # domain_indicator is crucial
                if verbose:
                    print(f"    Successfully loaded split: {split} with {len(X_pd)} samples.")
            elif verbose:
                print(f"    Split {split} is empty or None.")
        except KeyError:
            if verbose:
                print(f"    Split {split} not found for dataset {tableshift_string_identifier}. Skipping.")
        except Exception as e:
            if verbose:
                print(f"    Error loading split {split} for {tableshift_string_identifier}: {e}. Skipping.")
            
    if not all_X_list:
        if verbose:
            print(f"No data loaded for dataset {tableshift_string_identifier}. Returning empty list.")
        return []

    # Concatenate all data from the loaded splits
    combined_X_pd = pd.concat(all_X_list, ignore_index=True)
    combined_y_pd = pd.concat(all_y_list, ignore_index=True)
    combined_domain_pd = pd.concat(all_domain_list, ignore_index=True)

    # Convert features to NumPy.
    # TableShift might return data with mixed types (numeric, categorical).
    # Forcing to numeric here. Non-numeric columns will become NaN, then filled with 0.
    # This might not be optimal for all models/features and may require
    # more sophisticated preprocessing or featurization in your main pipeline
    # if your estimators cannot handle this format.
    # --- FEATURE PREPROCESSING ---
    if preprocessing == 'zero_fill':
        # existing behavior: coerce non-numeric to NaN then fill with 0
        X_pd_num = combined_X_pd.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_np = X_pd_num.values

    elif preprocessing == 'drop_non_numeric':
        # drop any column that isn't already numeric
        X_pd_num = combined_X_pd.select_dtypes(include=['number'])
        X_np = X_pd_num.values

    elif preprocessing == 'onehot':
        # numeric + one-hot encode all object/category columns
        num_df = combined_X_pd.select_dtypes(include=['number'])
        cat_df = combined_X_pd.select_dtypes(include=['object','category'])
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore',
                                **(onehot_kwargs or {}))
        if cat_df.shape[1] > 0:
            cat_arr = encoder.fit_transform(cat_df)
            X_np = np.hstack([num_df.values, cat_arr])
        else:
            X_np = num_df.values

    else:
        raise ValueError(f"Unknown preprocessing option: {preprocessing}")


    y_np = combined_y_pd.values.ravel() # Ensure y is a 1D array
    domain_np = combined_domain_pd.values.ravel() # Domain indicators as a 1D array

    if verbose:
        print(f"  Total samples loaded across all splits: {X_np.shape[0]}, Features: {X_np.shape[1]}")

    unique_domains = np.unique(domain_np)
    if verbose:
        print(f"  Found {len(unique_domains)} unique domains (populations): {unique_domains}")

    populations_data = []
    for domain_id in unique_domains:
        domain_mask = (domain_np == domain_id)
        X_domain = X_np[domain_mask]
        y_domain = y_np[domain_mask]

        if X_domain.shape[0] == 0:
            if verbose:
                print(f"    Skipping domain '{domain_id}' as it has 0 samples after processing.")
            continue
            
        if verbose:
            print(f"    Processing domain '{domain_id}': {X_domain.shape[0]} samples.")

        populations_data.append({
            'pop_id': str(domain_id), # Use string for pop_id for consistency
            'X_raw': X_domain,
            'Y_raw': y_domain,
            # For real datasets, 'meaningful_indices' is unknown.
            # We set it to all features. Your selection method will determine relevance.
            'meaningful_indices': np.arange(X_domain.shape[1]) 
        })
        
    if verbose:
        print(f"Finished processing {tableshift_string_identifier}. Generated {len(populations_data)} populations.")
    return populations_data

# --- Wrapper functions for the specific datasets you requested ---

def generate_income_data_from_tableshift(cache_dir: str = "./tableshift_cache", verbose: bool = True):
    """Loads the 'acsincome' TableShift dataset."""
    return get_tableshift_data_as_populations("acsincome", cache_dir=cache_dir, verbose=verbose)

def generate_unemployment_data_from_tableshift(cache_dir: str = "./tableshift_cache", verbose: bool = True):
    """Loads the 'acsunemployment' TableShift dataset."""
    return get_tableshift_data_as_populations("acsunemployment", cache_dir=cache_dir, verbose=verbose)

def generate_foodstamps_data_from_tableshift(cache_dir: str = "./tableshift_cache", verbose: bool = True):
    """Loads the 'acsfoodstamps' TableShift dataset."""
    return get_tableshift_data_as_populations("acsfoodstamps", cache_dir=cache_dir, verbose=verbose)

def main():
    parser = argparse.ArgumentParser(
        description="Test TableShift loader and plot feature histograms"
    )
    parser.add_argument(
        "--dataset", type=str, default="acsincome",
        choices=["acsincome","acsunemployment","acsfoodstamps"],
        help="Which TableShift dataset to load"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="./tableshift_cache",
        help="Cache directory for TableShift"
    )
    parser.add_argument(
        "--preprocessing", type=str,
        choices=["zero_fill","drop_non_numeric","onehot"],
        default="zero_fill",
        help="How to preprocess features"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print loading details"
    )
    parser.add_argument(
        "--n-features", type=int, default=5,
        help="Number of features to plot (from index 0)"
    )
    parser.add_argument(
        "--out-dir", type=str, default="./results",
        help="Output directory for plots"
    )
    args = parser.parse_args()

    # Load populations
    pops = get_tableshift_data_as_populations(
        args.dataset,
        cache_dir=args.cache_dir,
        verbose=args.verbose,
        preprocessing=args.preprocessing
    )
    if not pops:
        print("No populations loaded. Exiting.")
        return

    # Pooled data
    X_list = [pop["X_raw"] for pop in pops]
    X_pooled = np.vstack(X_list)
    print(f"Pooled X shape: {X_pooled.shape}")
    print(f"Number of populations: {len(pops)}")

    for pop in pops:
        Xi = pop["X_raw"]
        print(f" Pop {pop['pop_id']}: shape={Xi.shape}, count={Xi.shape[0]} samples")

    # Determine how many features to plot
    d = X_pooled.shape[1]
    n_plot = min(args.n_features, d)

    # Create output folder
    out_dir = f"{args.out_dir}plots_{args.dataset}_{args.preprocessing}"
    os.makedirs(out_dir, exist_ok=True)

    # Plot histograms for each feature index
    for f_idx in range(n_plot):
        n_cols = len(pops) + 1
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), constrained_layout=True)

        # Pooled
        axes[0].hist(X_pooled[:, f_idx], bins=30, color="gray", alpha=0.7)
        axes[0].set_title(f"Pooled\nfeat {f_idx}")
        axes[0].set_ylabel("Count")

        # Each population
        for j, pop in enumerate(pops):
            Xi = pop["X_raw"][:, f_idx]
            axes[j+1].hist(Xi, bins=30, alpha=0.7)
            axes[j+1].set_title(f"Pop {pop['pop_id']}\nsize {Xi.shape[0]}")

        fig.suptitle(f"Feature {f_idx} Distribution ({args.dataset})", fontsize=14)
        fname = os.path.join(out_dir, f"feature_{f_idx}_hist.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved histogram for feature {f_idx} to {fname}")

if __name__ == "__main__":
    main()