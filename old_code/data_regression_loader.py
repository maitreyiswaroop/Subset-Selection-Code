# data_regression_loader.py
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer # For binning numeric features
import os
import argparse
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive environments (e.g., servers)
import matplotlib.pyplot as plt

# Attempt to import folktables
try:
    from folktables import ACSDataSource, ACSIncome
    FOLKTABLES_AVAILABLE = True
except ImportError:
    print("Warning: folktables library not found. ACS dataset loading will not be available. "
          "Install with: pip install folktables")
    FOLKTABLES_AVAILABLE = False
    ACSDataSource, ACSIncome = None, None # Define to prevent NameError if import fails early


def _preprocess_features(
    X_pd: pd.DataFrame,
    preprocessing_method: str = 'zero_fill',
    onehot_kwargs: dict = None,
    verbose: bool = True
):
    """Helper function to preprocess features into a NumPy array and get feature names."""
    if verbose:
        print(f"  Preprocessing features using method: '{preprocessing_method}'")
        print(f"    Original X input shape: {X_pd.shape}")

    X_np_processed = None
    processed_feature_names = []

    if preprocessing_method == 'zero_fill':
        # Coerce non-numeric to NaN then fill with 0
        X_pd_num = X_pd.apply(pd.to_numeric, errors='coerce').fillna(0)
        X_np_processed = X_pd_num.values
        processed_feature_names = X_pd_num.columns.tolist()
        if verbose:
            print(f"    Applied 'zero_fill'. Processed X shape: {X_np_processed.shape}")

    elif preprocessing_method == 'drop_non_numeric':
        # Keep only numeric columns
        X_pd_num = X_pd.select_dtypes(include=np.number)
        X_np_processed = X_pd_num.values
        processed_feature_names = X_pd_num.columns.tolist()
        if verbose:
            print(f"    Applied 'drop_non_numeric'. Processed X shape: {X_np_processed.shape}. Kept columns: {processed_feature_names}")

    elif preprocessing_method == 'onehot':
        numeric_cols = X_pd.select_dtypes(include=np.number).columns
        categorical_cols = X_pd.select_dtypes(include=['object', 'category']).columns

        numeric_df = X_pd[numeric_cols]
        X_numeric_np = numeric_df.values
        processed_feature_names.extend(numeric_cols.tolist())
        
        all_encoded_parts = [X_numeric_np]

        if not categorical_cols.empty:
            cat_df = X_pd[categorical_cols]
            encoder_params = {'sparse_output': False, 'handle_unknown': 'ignore'}
            if onehot_kwargs:
                encoder_params.update(onehot_kwargs)
            
            if verbose:
                print(f"    One-hot encoding categorical columns: {categorical_cols.tolist()}")

            encoder = OneHotEncoder(**encoder_params)
            cat_df_str = cat_df.astype(str) # Ensure string type for OHE consistency
            cat_np_encoded = encoder.fit_transform(cat_df_str)
            all_encoded_parts.append(cat_np_encoded)
            
            try:
                ohe_feature_names = encoder.get_feature_names_out(categorical_cols)
                processed_feature_names.extend(ohe_feature_names.tolist())
            except Exception as e:
                if verbose:
                    print(f"    Note: Could not get detailed feature names from OneHotEncoder ({e}). Using generic names for OHE features.")
                for i in range(cat_np_encoded.shape[1]):
                    base_name = categorical_cols[0] if len(categorical_cols) == 1 else "cat"
                    processed_feature_names.append(f"ohe_{base_name}_feat_{i}")
            
            if verbose:
                print(f"    Numeric part shape: {X_numeric_np.shape}, Categorical one-hot encoded part shape: {cat_np_encoded.shape}")
        else:
            if verbose:
                print("    No categorical columns explicitly typed as 'object' or 'category' found for one-hot encoding.")
        
        if len(all_encoded_parts) > 1 and all_encoded_parts[1].shape[1] > 0 : # check if cat_np_encoded actually has columns
            X_np_processed = np.hstack(all_encoded_parts)
        else:
            X_np_processed = all_encoded_parts[0] 

        if verbose:
            print(f"    Applied 'onehot'. Final processed X shape: {X_np_processed.shape}")
    else:
        raise ValueError(f"Unknown preprocessing_method: {preprocessing_method}")

    return X_np_processed, processed_feature_names


def get_custom_regression_data_as_populations(
    data_loader_func, 
    dataset_name: str,
    domain_column_name: str = None,
    domain_creation_method: str = 'categorical', 
    num_domains_for_binning: int = 3,
    min_samples_per_domain: int = 50,
    preprocessing_method: str = 'zero_fill',
    onehot_kwargs: dict = None,
    verbose: bool = True,
    loader_kwargs: dict = None 
):
    if verbose:
        print(f"Loading and processing dataset: {dataset_name}...")

    X_pd_full, y_pd_full, default_domain_col_from_loader = data_loader_func(**(loader_kwargs or {}))

    if domain_column_name is None: 
        domain_column_name = default_domain_col_from_loader
    
    domain_series_processed = None
    X_pd_for_features = X_pd_full.copy() 

    if domain_column_name and domain_column_name in X_pd_full.columns:
        if verbose:
            print(f"  Defining domains based on column: '{domain_column_name}' using method: '{domain_creation_method}'")
        
        original_domain_data = X_pd_full[domain_column_name]
        
        if domain_creation_method == 'categorical':
            domain_series_processed = original_domain_data.astype('category').cat.codes 
        elif domain_creation_method == 'custom_medinc': 
            if not pd.api.types.is_numeric_dtype(original_domain_data):
                raise ValueError("MedInc must be numeric for custom_medinc split.")
            bins = [-np.inf, 2.0, 5.0, np.inf]
            binned_codes = pd.cut(original_domain_data.values, bins=bins, labels=False, include_lowest=True, right=True)
            domain_series_processed = pd.Series(binned_codes, index=X_pd_full.index).astype(int)
            if verbose:
                print(f"    Applied custom MedInc thresholds. Domain IDs: {np.unique(domain_series_processed.values)}")
        elif domain_creation_method in ['bin_equal_width', 'bin_equal_freq']:
            if not pd.api.types.is_numeric_dtype(original_domain_data):
                raise ValueError(f"Column '{domain_column_name}' must be numeric for binning methods.")
            
            if original_domain_data.isnull().any():
                fill_value = original_domain_data.median()
                if verbose:
                    print(f"    Found NaNs in domain column '{domain_column_name}', filling with median ({fill_value}).")
                original_domain_data = original_domain_data.fillna(fill_value)

            strategy = 'uniform' if domain_creation_method == 'bin_equal_width' else 'quantile'
            discretizer = KBinsDiscretizer(n_bins=num_domains_for_binning, encode='ordinal', strategy=strategy, subsample=None, random_state=0)
            
            try:
                binned_data = discretizer.fit_transform(original_domain_data.values.reshape(-1, 1))
                domain_series_processed = pd.Series(binned_data.ravel().astype(int), index=X_pd_full.index)
            except ValueError as e:
                raise ValueError(f"Binning column '{domain_column_name}' failed. Original error: {e}")

            if verbose:
                print(f"    Binned '{domain_column_name}' into {num_domains_for_binning} bins. Resulting domain IDs: {np.unique(domain_series_processed.values)}")
        else:
            raise ValueError(f"Unknown domain_creation_method: {domain_creation_method}")
        
        if domain_column_name in X_pd_for_features.columns:
             # dropping moved after preprocessing so that OHE will see the domain column
             if verbose:
                print(f"    (postpone drop of '{domain_column_name}' until after preprocessing)")
    else:
        if verbose:
            print("  No domain column specified or found for splitting. Treating dataset as a single population (domain '0').")
        domain_series_processed = pd.Series(0, index=X_pd_full.index) 

    X_np_processed_full, final_feature_names = _preprocess_features(
        X_pd_for_features, preprocessing_method, onehot_kwargs, verbose
    )

    # — now remove any processed features that came from our domain column —
    if domain_column_name:
        to_keep = [n for n in final_feature_names
                   if not n.startswith(f"{domain_column_name}_")]
        keep_idx = [i for i,n in enumerate(final_feature_names) if n in to_keep]
        X_np_processed_full = X_np_processed_full[:, keep_idx]
        final_feature_names    = to_keep

    y_np_full = y_pd_full.values.ravel()

    unique_domain_ids = np.unique(domain_series_processed.values) 
    populations_data = []

    if verbose:
        print(f"  Total samples available for splitting: {X_np_processed_full.shape[0]}, Processed Features: {X_np_processed_full.shape[1]}")
        if final_feature_names:
            print(f"    Feature names after processing (first 30): {final_feature_names[:30]}{'...' if len(final_feature_names) > 30 else ''}")
        print(f"  Splitting into {len(unique_domain_ids)} potential domains based on IDs: {unique_domain_ids}")
        print(f"  Value counts for generated domain IDs (domain_series_processed):\n{pd.Series(domain_series_processed.values).value_counts().sort_index(ascending=True)}")

    for domain_id_val in unique_domain_ids:
        domain_mask = (domain_series_processed.values == domain_id_val)
        X_domain = X_np_processed_full[domain_mask]
        y_domain = y_np_full[domain_mask]
        if verbose:
            print(f"    For domain_id_val {domain_id_val}:")
            print(f"      Number of True values in mask: {np.sum(domain_mask)}")
            print(f"      X_domain shape: {X_domain.shape}, y_domain shape: {y_domain.shape}")
            if X_domain.shape[0] > 0 and X_domain.shape[1] > 0:
                # Print mean of first feature to see if it differs across domains
                print(f"      Mean of first feature in X_domain: {X_domain[:, 0].mean():.4f}")
        if X_domain.shape[0] < min_samples_per_domain:
            if verbose:
                print(f"    Skipping domain ID '{domain_id_val}' as it has {X_domain.shape[0]} samples (min required: {min_samples_per_domain}).")
            continue
            
        if verbose:
            print(f"    Domain ID '{domain_id_val}': {X_domain.shape[0]} samples, {X_domain.shape[1]} features.")

        populations_data.append({
            'pop_id': str(domain_id_val),
            'X_raw': X_domain, 
            'Y_raw': y_domain,
            'meaningful_indices': np.arange(X_domain.shape[1]), 
            'feature_names': final_feature_names 
        })
        
    if verbose:
        print(f"Finished processing {dataset_name}. Generated {len(populations_data)} populations.")
    return populations_data


# --- Specific Data Loader Functions ---

def _load_california_housing_data_internal():
    data = fetch_california_housing(as_frame=True)
    X_pd_full = data.frame.drop(columns=['MedHouseVal'])
    y_pd_full = data.frame['MedHouseVal']
    default_domain_col = 'MedInc' 
    return X_pd_full, y_pd_full, default_domain_col

def generate_california_housing_populations(**kwargs):
    # Set defaults specific to this dataset if not provided in kwargs
    kwargs.setdefault('domain_column_name', 'MedInc')
    kwargs.setdefault('domain_creation_method', 'bin_equal_freq')
    kwargs.setdefault('num_domains_for_binning', 3)
    return get_custom_regression_data_as_populations(
        _load_california_housing_data_internal, "CaliforniaHousing", **kwargs
    )

def _load_diabetes_regression_data_internal():
    data = load_diabetes(as_frame=True)
    X_pd_full = data.frame.drop(columns=['target'])
    y_pd_full = data.frame['target']
    default_domain_col = 'sex' 
    return X_pd_full, y_pd_full, default_domain_col

def generate_diabetes_regression_populations(**kwargs):
    kwargs.setdefault('domain_column_name', 'sex')
    kwargs.setdefault('domain_creation_method', 'categorical')
    return get_custom_regression_data_as_populations(
        _load_diabetes_regression_data_internal, "DiabetesRegression", **kwargs
    )

def _load_energy_efficiency_data_internal(file_path="ENB2012_data.xlsx", target_column="Y1"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Energy efficiency dataset ('{file_path}') not found. Download from UCI ML Repo."
        )
    df = pd.read_excel(file_path) 
    df.columns = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "Y1", "Y2"]
    if target_column not in ["Y1", "Y2"]:
        raise ValueError("target_column must be 'Y1' or 'Y2'")
    X_pd_full = df[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]]
    y_pd_full = df[target_column]
    default_domain_col = 'X6' 
    return X_pd_full, y_pd_full, default_domain_col

def generate_energy_efficiency_populations(file_path="ENB2012_data.xlsx", target_column="Y1", **kwargs):
    kwargs.setdefault('domain_column_name', 'X6')
    kwargs.setdefault('domain_creation_method', 'categorical')
    def loader_func(): 
        return _load_energy_efficiency_data_internal(file_path=file_path, target_column=target_column)
    return get_custom_regression_data_as_populations(
        loader_func, f"EnergyEfficiency(Target:{target_column})", **kwargs
    )

def _load_acs_income_regression_data_internal(
    acs_states=['CA'], acs_year='2018', verbose=True,
    folktables_data_dir: str = "./folktables_data_cache" # Added argument
):
    if not FOLKTABLES_AVAILABLE:
        raise ImportError("folktables library is required for ACS data but not found.")
    
    if verbose:
        print(f"  _load_acs_income_regression_data_internal received initial folktables_data_dir: {folktables_data_dir}")
        print(f"  Loading ACS data for states: {acs_states}, year: {acs_year} using folktables...")
        print(f"  Folktables data will be stored/looked for in: {os.path.abspath(folktables_data_dir)}")
    abs_folktables_data_dir = os.path.abspath(os.path.expanduser(folktables_data_dir))
    
    if verbose:
        print(f"  Resolved folktables data directory to (absolute): {abs_folktables_data_dir}")
    # Ensure the folktables data directory exists
    os.makedirs(folktables_data_dir, exist_ok=True)
    
    data_source = ACSDataSource(survey_year=acs_year, horizon='1-Year', survey='person', root_dir=folktables_data_dir) # Use root_dir
    acs_data_pd = data_source.get_data(states=acs_states, download=True)

    if acs_data_pd.empty:
        raise ValueError(f"No ACS data returned for states {acs_states}, year {acs_year}. Check folktables setup and data availability.")

    income_features = ACSIncome.features
    X_pd_full = acs_data_pd[income_features].copy() 

    y_pincp = acs_data_pd['PINCP']
    valid_income_mask = (y_pincp > 0)
    X_pd_full = X_pd_full.loc[valid_income_mask].reset_index(drop=True) # Reset index after filtering
    y_pd_full = pd.Series(np.log1p(y_pincp[valid_income_mask].values), name='log_PINCP').reset_index(drop=True)


    if X_pd_full.empty:
        raise ValueError("No data remaining after filtering for PINCP > 0.")

    categorical_acs_features = ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'SEX', 'RAC1P']
    if verbose:
        print(f"    Casting ACS features to 'category' for OHE: {categorical_acs_features}")
    for col in categorical_acs_features:
        if col in X_pd_full.columns:
            X_pd_full[col] = X_pd_full[col].astype('category')
    
    default_domain_col = None
    if 'ST' in acs_data_pd.columns: # ST is state FIPS code
         # Align 'ST' with the filtered X_pd_full before assigning
        state_domain_series = acs_data_pd.loc[valid_income_mask, 'ST'].astype('category').reset_index(drop=True)
        X_pd_full['STATE_DOMAIN'] = state_domain_series
        if len(acs_states) > 1 : # Only set as default if multiple states requested, otherwise let user choose
             default_domain_col = 'STATE_DOMAIN'

    if default_domain_col is None : # Fallback if not multiple states or ST somehow not available
        default_domain_col = 'RAC1P' 
        if 'RAC1P' not in X_pd_full.columns and verbose: 
            print(f"Warning: Default domain RAC1P not in features. Available: {X_pd_full.columns.tolist()}")
            default_domain_col = None 

    if verbose:
        print(f"  ACS data loaded: X shape {X_pd_full.shape}, y shape {y_pd_full.shape}")
        print(f"  Default domain column for ACS will be: {default_domain_col}")
        
    return X_pd_full, y_pd_full, default_domain_col

def generate_acs_income_regression_populations(
    acs_states_str: str = 'CA', 
    acs_year: str = '2018',
    folktables_data_dir: str = "./folktables_data_cache", # Added
    **kwargs 
):
    if not FOLKTABLES_AVAILABLE:
        print("Skipping ACS Income Regression: folktables library not available.")
        return []
        
    states_list = [s.strip().upper() for s in acs_states_str.split(',')]
    
    loader_kwargs_acs = {
        'acs_states': states_list, 
        'acs_year': acs_year,
        'folktables_data_dir': folktables_data_dir, # Pass this new arg
        'verbose': kwargs.get('verbose', True)
    }

    # Set ACS specific defaults if not overridden by user in kwargs
    # If multiple states, default to STATE_DOMAIN, otherwise RAC1P
    # The actual creation of STATE_DOMAIN and selection of default happens in _load_acs_income_regression_data_internal
    # So, we just need to ensure domain_column_name is not set here if it should use the loader's default
    if kwargs.get('domain_column_name') is None: # Only if user didn't specify one
        if len(states_list) > 1:
            kwargs['domain_column_name'] = 'STATE_DOMAIN'
        else:
            kwargs['domain_column_name'] = 'RAC1P' # Default for single state analysis
            
    kwargs.setdefault('domain_creation_method', 'categorical')
    # Default preprocessing to onehot for ACS as it has many coded categoricals
    kwargs.setdefault('preprocessing_method', 'onehot') 
        
    return get_custom_regression_data_as_populations(
        _load_acs_income_regression_data_internal, 
        f"ACSIncomeRegression(States:{'_'.join(states_list)}-Year:{acs_year})",
        loader_kwargs=loader_kwargs_acs,
        **kwargs
    )


def main():
    parser = argparse.ArgumentParser(description="Load and inspect custom regression datasets, with domain splitting.")
    parser.add_argument(
        "--dataset", type=str, default="california_housing",
        choices=["california_housing", "diabetes_regression", "energy_efficiency", "acs_income_reg"],
        help="Which regression dataset to load."
    )
    # Args for Energy Efficiency
    parser.add_argument(
        "--energy_file_path", type=str, default="ENB2012_data.xlsx",
        help="Path to the Energy Efficiency dataset Excel file (if --dataset=energy_efficiency)."
    )
    parser.add_argument(
        "--energy_target_column", type=str, default="Y1", choices=["Y1", "Y2"],
        help="Target column for Energy Efficiency dataset ('Y1' or 'Y2')."
    )
    # Args for ACS Income Regression
    parser.add_argument(
        "--acs_states", type=str, default="CA",
        help="Comma-separated list of state abbreviations for ACS data (e.g., 'CA,NY,TX')."
    )
    parser.add_argument(
        "--acs_year", type=str, default="2018",
        help="Survey year for ACS data (e.g., '2018')."
    )
    parser.add_argument( # New argument for folktables data directory
        "--folktables_data_dir", type=str, default="./folktables_data_cache",
        help="Directory to download/cache raw folktables ACS data."
    )
    # Domain creation args
    parser.add_argument(
        "--domain_column", type=str, default=None,
        help="Column name from the raw features to use for creating domains/populations. "
             "If None, dataset-specific defaults are used."
    )
    parser.add_argument(
        "--domain_method", type=str, default=None, 
        choices=['categorical', 'bin_equal_width', 'bin_equal_freq', 'custom_medinc'],
        help="Method to create domains from the domain_column. Dataset-specific defaults if None."
    )
    parser.add_argument(
        "--num_bins", type=int, default=3,
        help="Number of bins if using a binning domain_method."
    )
    parser.add_argument(
        "--min_samples_domain", type=int, default=50,
        help="Minimum samples required for a domain to be included."
    )
    # Preprocessing and plotting args
    parser.add_argument(
        "--preprocessing", type=str, default=None, # Default will be set per dataset if None
        choices=["zero_fill", "drop_non_numeric", "onehot"],
        help="Feature preprocessing method. If None, dataset-specific defaults are used (e.g. onehot for ACS)."
    )
    parser.add_argument(
        "--n_features_plot", type=int, default=5,
        help="Number of features to plot histograms for."
    )
    parser.add_argument(
        "--out_dir", type=str, default="./dataset_plots_regression",
        help="Output directory for plots."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    args = parser.parse_args()

    populations = []
    dataset_plot_name = args.dataset
    
    gen_kwargs = {
        "domain_column_name": args.domain_column,
        "domain_creation_method": args.domain_method, 
        "num_domains_for_binning": args.num_bins,
        "min_samples_per_domain": args.min_samples_domain,
        "preprocessing_method": args.preprocessing, # Will be overridden by dataset defaults if None
        "verbose": args.verbose
        # onehot_kwargs can be added here if needed, e.g. from another CLI arg
    }

    if args.dataset == "california_housing":
        # Apply dataset-specific defaults if user didn't provide them
        if gen_kwargs["domain_column_name"] is None: gen_kwargs["domain_column_name"] = 'MedInc'
        if gen_kwargs["domain_creation_method"] is None: gen_kwargs["domain_creation_method"] = 'bin_equal_freq' # or 'custom_medinc'
        if gen_kwargs["preprocessing_method"] is None: gen_kwargs["preprocessing_method"] = 'zero_fill'
        
        raw_X_pd_cali, _, _ = _load_california_housing_data_internal()
        os.makedirs(args.out_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        if 'MedInc' in raw_X_pd_cali.columns:
            plt.hist(raw_X_pd_cali['MedInc'], bins=30, color='blue', alpha=0.7)
            plt.title("MedInc Distribution (California Housing) - Before Domain Splitting")
            plt.xlabel("MedInc")
        else:
            plt.title("California Housing Data Loaded (MedInc not plotted)")
        plt.ylabel("Count")
        domain_dist_path = os.path.join(args.out_dir, f"{args.dataset}_MedInc_raw_distribution.png")
        plt.savefig(domain_dist_path); plt.close()
        if args.verbose: print(f"Saved MedInc raw distribution plot to: {domain_dist_path}")
        
        populations = generate_california_housing_populations(**gen_kwargs)

    elif args.dataset == "diabetes_regression":
        if gen_kwargs["domain_column_name"] is None: gen_kwargs["domain_column_name"] = 'sex'
        if gen_kwargs["domain_creation_method"] is None: gen_kwargs["domain_creation_method"] = 'categorical'
        if gen_kwargs["preprocessing_method"] is None: gen_kwargs["preprocessing_method"] = 'zero_fill'
        populations = generate_diabetes_regression_populations(**gen_kwargs)

    elif args.dataset == "energy_efficiency":
        if gen_kwargs["domain_column_name"] is None: gen_kwargs["domain_column_name"] = 'X6'
        if gen_kwargs["domain_creation_method"] is None: gen_kwargs["domain_creation_method"] = 'categorical'
        if gen_kwargs["preprocessing_method"] is None: gen_kwargs["preprocessing_method"] = 'zero_fill'
        dataset_plot_name = f"{args.dataset}_{args.energy_target_column}"
        populations = generate_energy_efficiency_populations(
            file_path=args.energy_file_path, 
            target_column=args.energy_target_column,
            **gen_kwargs
        )
    elif args.dataset == "acs_income_reg":
        if not FOLKTABLES_AVAILABLE:
            print("Cannot run ACS Income Regression: folktables library not installed/found.")
            return
        # ACS specific defaults are handled within its generator if gen_kwargs values are None
        # e.g., domain_column defaults to STATE_DOMAIN or RAC1P
        # e.g., preprocessing defaults to onehot
        if gen_kwargs["preprocessing_method"] is None: gen_kwargs["preprocessing_method"] = 'onehot'
        dataset_plot_name = f"{args.dataset}_{args.acs_states.replace(',', '_')}_{args.acs_year}"
        populations = generate_acs_income_regression_populations(
            acs_states_str=args.acs_states,
            acs_year=args.acs_year,
            folktables_data_dir=args.folktables_data_dir, # Pass the new arg
            **gen_kwargs
        )
    else:
        print(f"Dataset {args.dataset} selection not implemented in main example.")
        return

    if not populations:
        print("No populations were generated based on the criteria. Exiting.")
        return
    else: # --- ADD THIS BLOCK for debugging ---
        print(f"\n--- Verifying generated population data (first few populations) ---")
        for i, pop_dict in enumerate(populations[:min(3, len(populations))]): # Check first 3
            print(f"  Population pop_id='{pop_dict['pop_id']}':")
            print(f"    X_raw shape: {pop_dict['X_raw'].shape}")
            print(f"    Y_raw shape: {pop_dict['Y_raw'].shape}")
            if pop_dict['X_raw'].shape[0] > 0 and pop_dict['X_raw'].shape[1] > 0:
                print(f"    X_raw[0, 0] (first sample, first feature): {pop_dict['X_raw'][0, 0]}")
                print(f"    Mean of X_raw[:, 0] (first feature): {np.mean(pop_dict['X_raw'][:, 0]):.4f}")
            if pop_dict['Y_raw'].shape[0] > 0:
                print(f"    Mean of Y_raw: {np.mean(pop_dict['Y_raw']):.4f}")
        if len(populations) > 3:
            print("    ...")
    actual_domain_column_used = "N/A_or_default"
    actual_domain_method_used = "N/A_or_default"
    # Try to get actuals if possible, for better plot naming. This is a bit tricky with current structure.
    # For now, use command line args or "default" if CLI arg was None.
    actual_domain_column = args.domain_column if args.domain_column else "default"
    actual_domain_method = args.domain_method if args.domain_method else "default"


    print(f"\n--- Dataset Summary: {dataset_plot_name} ---")
    print(f"Number of populations generated: {len(populations)}")
    for i, pop in enumerate(populations):
        print(f"  Population '{pop['pop_id']}': X_raw (processed) shape {pop['X_raw'].shape}, Y_raw shape {pop['Y_raw'].shape}")
        if i == 0 and pop.get('feature_names'):
             print(f"    Processed feature names (first 30 or fewer): {pop['feature_names'][:30]}{'...' if len(pop['feature_names']) > 30 else ''}")


    # --- Plotting ---
    X_list_for_plot = [pop["X_raw"] for pop in populations if pop["X_raw"].shape[0] > 0]
    Y_list_for_plot = [pop["Y_raw"] for pop in populations if pop["Y_raw"].shape[0] > 0]
    
    if not X_list_for_plot or not Y_list_for_plot :
        print("No data available in any population for plotting.")
        return
        
    X_pooled = np.vstack(X_list_for_plot)
    Y_pooled = np.concatenate(Y_list_for_plot)
    
    plot_out_dir_name = f"plots_{dataset_plot_name}_proc_{args.preprocessing or 'default'}_dom_{actual_domain_column}_meth_{actual_domain_method}"
    plot_out_dir = os.path.join(args.out_dir, plot_out_dir_name)
    os.makedirs(plot_out_dir, exist_ok=True)
    print(f"\nSaving plots to: {plot_out_dir}")

    num_features_to_plot = min(args.n_features_plot, X_pooled.shape[1])
    
    base_feature_names = populations[0].get('feature_names', [f"feat_{i}" for i in range(X_pooled.shape[1])])

    for f_idx in range(num_features_to_plot):
        max_pops_to_plot = 4
        selected_pops_for_plot = populations[:min(len(populations), max_pops_to_plot)]
        num_subplots = len(selected_pops_for_plot) + 1 

        # Create a figure with multiple subplots
        fig_hist, axes_hist = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 4.5), constrained_layout=True)
        if num_subplots == 1: 
            axes_hist = [axes_hist]  # Convert to list if only one subplot
        elif not isinstance(axes_hist, np.ndarray) and not isinstance(axes_hist, list):
            axes_hist = [axes_hist]  # Ensure axes_hist is always iterable
        
        current_feature_name = base_feature_names[f_idx] if f_idx < len(base_feature_names) else f"feat_{f_idx}"

        # Plot pooled data in first subplot
        axes_hist[0].hist(X_pooled[:, f_idx], bins=30, color="gray", alpha=0.7, density=True)
        axes_hist[0].set_title(f"Pooled\n{current_feature_name}")
        axes_hist[0].set_ylabel("Density")

        # Plot each population in subsequent subplots
        for j, pop_data in enumerate(selected_pops_for_plot):
            ax_idx = j + 1
            if pop_data["X_raw"].shape[0] > 0 and f_idx < pop_data["X_raw"].shape[1]:
                Xi_feature = pop_data["X_raw"][:, f_idx]
                axes_hist[ax_idx].hist(Xi_feature, bins=30, alpha=0.7, density=True)
                axes_hist[ax_idx].set_title(f"Pop {pop_data['pop_id']} (n={Xi_feature.shape[0]})\n{current_feature_name}")
            else:
                axes_hist[ax_idx].set_title(f"Pop {pop_data['pop_id']}\nNo data for feat {f_idx}")
                axes_hist[ax_idx].text(0.5, 0.5, "N/A", ha='center', va='center')

        fig_hist.suptitle(f"Feature '{current_feature_name}' Distribution ({dataset_plot_name})", fontsize=16)
        safe_feature_name = "".join(c if c.isalnum() else "_" for c in current_feature_name)
        hist_fname = os.path.join(plot_out_dir, f"feature_{f_idx}_{safe_feature_name}_hist.png")
        try: fig_hist.savefig(hist_fname)
        except Exception as e: print(f"  Error saving hist plot {hist_fname}: {e}")
        finally: plt.close(fig_hist)
        if args.verbose: print(f"  Saved histogram for feature {f_idx} ('{current_feature_name}')")
        
    fig_scatter, axes_scatter = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 4.5), constrained_layout=True, sharey=True)
    if num_subplots == 1:
        axes_scatter = [axes_scatter]  # Convert to list if only one subplot
    elif not isinstance(axes_scatter, np.ndarray) and not isinstance(axes_scatter, list):
        axes_scatter = [axes_scatter]  # Ensure axes_scatter is always iterable

    # Plot pooled data in first subplot
    axes_scatter[0].scatter(X_pooled[:, f_idx], Y_pooled, s=5, alpha=0.3, color="gray", rasterized=True)
    axes_scatter[0].set_title(f"Pooled Y vs\n{current_feature_name}")
    axes_scatter[0].set_xlabel(current_feature_name)
    axes_scatter[0].set_ylabel("Y (Target)")

    # Plot each population in subsequent subplots
    for j, pop_data in enumerate(selected_pops_for_plot):
        ax_idx = j + 1
        if pop_data["X_raw"].shape[0] > 0 and f_idx < pop_data["X_raw"].shape[1] and pop_data["Y_raw"].shape[0] > 0:
            Xi = pop_data["X_raw"][:, f_idx]
            Yi = pop_data["Y_raw"]
            axes_scatter[ax_idx].scatter(Xi, Yi, s=5, alpha=0.3, rasterized=True)
            axes_scatter[ax_idx].set_title(f"Pop {pop_data['pop_id']} Y vs\n{current_feature_name}")
            axes_scatter[ax_idx].set_xlabel(current_feature_name)
        else:
            axes_scatter[ax_idx].set_title(f"Pop {pop_data['pop_id']}\nNo data for feat {f_idx}")
            axes_scatter[ax_idx].text(0.5, 0.5, "N/A", ha='center', va='center')

        fig_scatter.suptitle(f"Target vs Feature '{current_feature_name}' ({dataset_plot_name})", fontsize=16)
        scatter_fname = os.path.join(plot_out_dir, f"feature_{f_idx}_{safe_feature_name}_scatter.png")
        try: fig_scatter.savefig(scatter_fname, dpi=100) 
        except Exception as e: print(f"  Error saving scatter plot {scatter_fname}: {e}")
        finally: plt.close(fig_scatter)
        if args.verbose: print(f"  Saved scatter for feature {f_idx} ('{current_feature_name}')")

if __name__ == "__main__":
    main()