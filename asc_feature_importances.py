#!/usr/bin/env python3
"""
acs_feature_importance_extreme_memory.py

Ultra memory-efficient version for analyzing ACS data feature importance
across different population splits.
"""
import os
import numpy as np
import pandas as pd
import gc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
from scipy.spatial.distance import jensenshannon
from itertools import combinations

# Import data loading functions
from data_acs import generate_data_acs

def analyze_acs_feature_importance(states=["CA", "NY", "FL"], 
                                  year=2018, 
                                  target="PINCP", 
                                  output_dir="./acs_analysis",
                                  root_dir="data",
                                  max_samples=50000):  # Reduced sample size
    """Load ACS data, fit Random Forest, analyze feature importances"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading ACS data for states: {states}, year: {year}")
    print(f"Using data directory: {root_dir}")
    
    # Load the data
    X_all, Y_all, Xs, Ys, feature_cols, states_loaded = generate_data_acs(
        states=states,
        year=year,
        target=target,
        save_dir=None,
        root_dir=root_dir
    )
    
    print(f"Initial data loaded: X shape={X_all.shape}, y shape={Y_all.shape}")
    
    # First check for NaN in target variable
    nan_mask_y = ~np.isnan(Y_all)
    X_all = X_all[nan_mask_y]
    Y_all = Y_all[nan_mask_y]
    print(f"After removing NaN targets: X shape={X_all.shape}, y shape={Y_all.shape}")
    
    # Convert to DataFrame for easier column handling
    data_df = pd.DataFrame(X_all, columns=feature_cols)
    
    # Remove columns with more than 10% NaN values
    nan_percentages = data_df.isna().mean()
    cols_to_drop = nan_percentages[nan_percentages > 0.1].index.tolist()
    
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} columns with >10% NaN values: {cols_to_drop}")
        data_df = data_df.drop(columns=cols_to_drop)
    
    # Median impute remaining NaN values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    data_df = pd.DataFrame(
        imputer.fit_transform(data_df),
        columns=data_df.columns
    )
    
    print(f"After column filtering and imputation: {data_df.shape[1]} columns remain")
    
    # Add target back
    data_df[target] = Y_all
    
    # Convert back to numpy for model training
    X_all = data_df.drop(columns=[target]).values
    feature_cols = data_df.drop(columns=[target]).columns.tolist()
    
    # Subsample if still too large
    if len(X_all) > max_samples:
        print(f"Subsampling to {max_samples} samples...")
        indices = np.random.choice(len(X_all), max_samples, replace=False)
        X_all = X_all[indices]
        Y_all = Y_all[indices]
    
    print(f"Final dataset: X shape={X_all.shape}, y shape={Y_all.shape}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, Y_all, test_size=0.2, random_state=42
    )
    
    # Train a smaller Random Forest
    print("Training Random Forest model...")
    rf = RandomForestRegressor(n_estimators=20, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Cleanup
    del X_train
    gc.collect()
    
    # Use DataFrame only for critical operations
    data_df = pd.DataFrame(X_all, columns=feature_cols)
    data_df[target] = Y_all
    
    # Add state information
    state_mapping = dict(zip(STATE_FIPS.values(), STATE_FIPS.keys()))
    if 'STATEFIP' in data_df.columns:
        data_df['state'] = data_df['STATEFIP'].map(state_mapping)
    
    # Train a minimal Random Forest model
    print("Training minimal Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, Y_all, test_size=0.2, random_state=42
    )
    
    # Use an extremely simplified model to save memory
    rf = RandomForestRegressor(n_estimators=20, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Clean up immediately
    del X_train
    gc.collect()
    
    # Get score and feature importances
    test_score = rf.score(X_test, y_test)
    print(f"Random Forest R² score: {test_score:.4f}")
    
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Save feature importances
    csv_file = os.path.join(output_dir, "acs_feature_importances.csv")
    importance_df.to_csv(csv_file, index=False)
    print(f"Feature importances saved to {csv_file}")
    
    # Clean up full model
    del rf, X_test, importances
    gc.collect()
    
    # Process one population split at a time
    for split_type in ['sex', 'race', 'state']:
        process_single_split(data_df, split_type, feature_cols, target, output_dir)
        gc.collect()
    
    print("Analysis complete. Memory-efficient processing finished.")
    return

# Process a single population split to save memory
def process_single_split(data_df, split_type, feature_cols, target, output_dir):
    """Process a single population split to avoid keeping all in memory"""
    print(f"\nProcessing {split_type} split...")
    
    # Create population criteria based on split type
    if split_type == 'sex' and 'SEX' in data_df.columns:
        populations = {
            "Male": data_df[data_df['SEX'] == 1],
            "Female": data_df[data_df['SEX'] == 2]
        }
    elif split_type == 'race' and 'RAC1P' in data_df.columns:
        populations = {
            "White": data_df[data_df['RAC1P'] == 1],
            "Black": data_df[data_df['RAC1P'] == 2]
        }
    elif split_type == 'state':
        # Check various possible state column names
        state_col = None
        for col_name in ['STATEFIP', 'ST', 'state', 'STATE']:
            if col_name in data_df.columns:
                state_col = col_name
                break
                
        if state_col:
            print(f"Found state column: {state_col}")
            # Get the top 2 states by count
            top_states = data_df[state_col].value_counts().nlargest(2).index.tolist()
            populations = {
                f"State_{state}": data_df[data_df[state_col] == state]
                for state in top_states
            }
        else:
            print(f"Skipping {split_type}: no state column found in available columns")
            print(f"Available columns: {data_df.columns.tolist()}")
            return
    else:
        print(f"Skipping {split_type}: required columns not found")
        return
    
    # Subsample each population to 2000 records maximum
    for pop_name in populations:
        if len(populations[pop_name]) > 2000:
            populations[pop_name] = populations[pop_name].sample(2000, random_state=42)
    
    # Train models and get importances
    model_results = {}
    
    for pop_name, pop_df in populations.items():
        # Skip if too small
        if len(pop_df) < 500:
            continue
            
        # Extract features and target
        X = pop_df.drop([target], axis=1)
        y = pop_df[target]
        
        # Train minimal model
        model = xgb.XGBRegressor(
            n_estimators=20,  # Very small model
            learning_rate=0.1,
            max_depth=3,      # Reduced depth
            subsample=0.8     # Use only 80% of data per tree
        )
        
        # Force numeric-only columns
        X = X.select_dtypes(include=['number'])
        
        # Train
        model.fit(X, y)
        
        # Get importances
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Store results
        model_results[pop_name] = {"importances": importance_df}
        
        # Clean up immediately
        del model
        gc.collect()
    
    # Calculate divergence if we have enough populations
    if len(model_results) >= 2:
        # Simple divergence calculation
        pop_names = list(model_results.keys())
        
        # Get all features from both populations
        all_features = set()
        for pop in pop_names:
            all_features.update(model_results[pop]["importances"]["feature"])
        
        # Create vectors with zeros for missing features
        vectors = {}
        for pop in pop_names:
            imp_dict = dict(zip(
                model_results[pop]["importances"]["feature"],
                model_results[pop]["importances"]["importance"]
            ))
            
            # Create vector with all features
            vector = np.array([imp_dict.get(feat, 0.0) for feat in all_features])
            
            # Normalize
            sum_values = vector.sum()
            if sum_values > 0:
                vector = vector / sum_values
            
            vectors[pop] = vector
        
        # Calculate JS divergence
        divergence = jensenshannon(vectors[pop_names[0]], vectors[pop_names[1]])
        
        # Save result
        result = {
            "population_split": split_type,
            "outcome": target,
            "importance_divergence": divergence,
            "populations": pop_names
        }
        
        # Append to results file
        result_df = pd.DataFrame([result])
        result_path = os.path.join(output_dir, "acs_minimal_divergence.csv")
        
        # Append or create
        if os.path.exists(result_path):
            result_df.to_csv(result_path, mode='a', header=False, index=False)
        else:
            result_df.to_csv(result_path, index=False)
        
        print(f"  {split_type} divergence: {divergence:.4f}")
    
    # Clean up
    del populations, model_results
    gc.collect()

# State FIPS codes
STATE_FIPS = {"CA": 6, "NY": 36, "FL": 12}

if __name__ == "__main__":
    analyze_acs_feature_importance(max_samples=50000)  # Much smaller sample size