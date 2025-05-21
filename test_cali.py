#!/usr/bin/env python3
"""
California Housing Feature Importance Divergence Analysis

This script analyzes how much feature importances differ across various 
population splits in the California Housing dataset using XGBoost models.

It tries different population splitting strategies and outcome variables, 
ranking them by the divergence in important features across populations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_california_housing
import xgboost as xgb
from scipy.spatial.distance import jensenshannon
from itertools import combinations

def load_california_housing():
    """
    Load and prepare the California Housing dataset.
    
    Returns:
        DataFrame with the data
    """
    # Load the dataset
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()
    
    # Add the target variable
    df['median_house_value'] = housing.target
    
    # Create some additional features for analysis
    df['rooms_per_household'] = df['AveRooms'] / df['AveOccup']
    df['bedrooms_ratio'] = df['AveBedrms'] / df['AveRooms']
    df['log_population'] = np.log1p(df['Population'])
    
    return df

def create_population_splits(df):
    """
    Create different population splitting strategies.
    
    Returns:
        dict: Mapping of split name to dict of population criteria
    """
    # Create region splits based on latitude and longitude
    df_copy = df.copy()
    
    # Income splits
    df_copy['income_group'] = pd.qcut(df_copy['MedInc'], 3, labels=['Low', 'Medium', 'High'])
    
    # Housing age splits
    df_copy['age_group'] = pd.qcut(df_copy['HouseAge'], 3, labels=['New', 'Medium', 'Old'])
    
    # Urban density splits based on population
    df_copy['density_group'] = pd.qcut(df_copy['Population'], 3, labels=['Low', 'Medium', 'High'])
    
    # Coastal vs Inland (simplified by longitude)
    # California coast is roughly at longitude <= -122
    df_copy['coastal'] = (df_copy['Longitude'] <= -122).map({True: 'Coastal', False: 'Inland'})
    
    # Geographic regions (North/South and Coastal/Inland)
    # Divide California roughly at latitude 37
    df_copy['region'] = 'Other'
    df_copy.loc[(df_copy['Latitude'] >= 37) & (df_copy['Longitude'] <= -122), 'region'] = 'NorthCoast'
    df_copy.loc[(df_copy['Latitude'] >= 37) & (df_copy['Longitude'] > -122), 'region'] = 'NorthInland'
    df_copy.loc[(df_copy['Latitude'] < 37) & (df_copy['Longitude'] <= -122), 'region'] = 'SouthCoast'
    df_copy.loc[(df_copy['Latitude'] < 37) & (df_copy['Longitude'] > -122), 'region'] = 'SouthInland'
    
    # Room count splits
    df_copy['room_group'] = pd.qcut(df_copy['AveRooms'], 3, labels=['Few', 'Average', 'Many'])
    
    # Create population splits dictionary
    population_splits = {
        "income": {
            "Low": {"income_group": "Low"},
            "Medium": {"income_group": "Medium"},
            "High": {"income_group": "High"}
        },
        "house_age": {
            "New": {"age_group": "New"},
            "Medium": {"age_group": "Medium"},
            "Old": {"age_group": "Old"}
        },
        "urban_density": {
            "Low": {"density_group": "Low"},
            "Medium": {"density_group": "Medium"},
            "High": {"density_group": "High"}
        },
        "coastal": {
            "Coastal": {"coastal": "Coastal"},
            "Inland": {"coastal": "Inland"}
        },
        "region": {
            "NorthCoast": {"region": "NorthCoast"},
            "NorthInland": {"region": "NorthInland"},
            "SouthCoast": {"region": "SouthCoast"},
            "SouthInland": {"region": "SouthInland"}
        },
        "rooms": {
            "Few": {"room_group": "Few"},
            "Average": {"room_group": "Average"},
            "Many": {"room_group": "Many"}
        }
    }
    
    return population_splits, df_copy

def define_outcome_variables(df):
    """
    Define different outcome variables to test.
    
    Returns:
        dict: Mapping of outcome name to column name and task type
    """
    # Define outcomes
    outcomes = {
        "house_value": {"column": "median_house_value", "task": "regression"},
        "rooms_per_household": {"column": "rooms_per_household", "task": "regression"},
        "bedrooms_ratio": {"column": "bedrooms_ratio", "task": "regression"},
        "log_population": {"column": "log_population", "task": "regression"}
    }
    
    return outcomes

def split_data_by_population(df, population_criteria):
    """
    Split data into specified populations based on criteria.
    
    Args:
        df: DataFrame with the data
        population_criteria: Dict mapping population names to filter criteria
        
    Returns:
        dict: Mapping of population names to filtered DataFrames
    """
    population_dfs = {}
    
    for pop_name, criteria in population_criteria.items():
        # Create mask for this population based on criteria
        mask = pd.Series(True, index=df.index)
        for column, value in criteria.items():
            mask = mask & (df[column] == value)
        
        pop_df = df[mask]
        
        if pop_df.empty:
            print(f"WARNING: No data found for population {pop_name}")
        else:
            population_dfs[pop_name] = pop_df
    
    return population_dfs

def preprocess_population_data(population_dfs, feature_cols, outcome_col):
    """
    Prepare feature and target data for each population.
    
    Returns:
        dict: Population X and y data
    """
    pop_data = {}
    
    for pop_name, pop_df in population_dfs.items():
        # Skip empty populations
        if pop_df.empty:
            continue
            
        # Extract features and target
        X = pop_df[feature_cols].copy()
        y = pop_df[outcome_col].copy()
        
        # Handle NaN values
        imputer = SimpleImputer(strategy='median')
        X_imp = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Filter out any rows with NaN in target
        mask = ~y.isna()
        X_clean = X_imp[mask]
        y_clean = y[mask]
        
        if len(X_clean) > 0:
            pop_data[pop_name] = {
                "X": X_clean,
                "y": y_clean
            }
    
    return pop_data

def train_xgboost_models(pop_data, task_type):
    """
    Train XGBoost models for each population and extract feature importances.
    
    Args:
        pop_data: Dict with population X and y data
        task_type: 'classification' or 'regression'
        
    Returns:
        dict: XGBoost models and feature importances for each population
    """
    results = {}
    
    for pop_name, data in pop_data.items():
        X, y = data["X"], data["y"]
        
        # For California Housing, we'll only use regression
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        
        # Train model
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate performance metrics
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        performance = {"r2": r2, "mse": mse}
        
        # Get feature importances
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Store results
        results[pop_name] = {
            "model": model,
            "importances": importance_df,
            "performance": performance
        }
    
    return results

def calculate_importance_divergence(model_results):
    """
    Calculate divergence in feature importances between population pairs.
    
    Args:
        model_results: Dict with models and importances for each population
        
    Returns:
        float: Average divergence score across population pairs
    """
    if len(model_results) < 2:
        return 0.0  # Need at least two populations to compare
    
    # Create a unified importance vector for each population
    all_features = sorted(set(
        feature 
        for pop_results in model_results.values() 
        for feature in pop_results["importances"]["feature"]
    ))
    
    # Create normalized importance vectors for each population
    pop_importance_vectors = {}
    for pop_name, results in model_results.items():
        imp_df = results["importances"]
        imp_dict = dict(zip(imp_df["feature"], imp_df["importance"]))
        
        # Create vector with all features, filling in zeros for missing ones
        vector = np.array([imp_dict.get(feat, 0.0) for feat in all_features])
        
        # Normalize to sum to 1 for comparison
        total = vector.sum()
        if total > 0:
            vector = vector / total
        
        pop_importance_vectors[pop_name] = vector
    
    # Calculate Jensen-Shannon divergence between all pairs
    divergences = []
    for pop1, pop2 in combinations(pop_importance_vectors.keys(), 2):
        vec1 = pop_importance_vectors[pop1]
        vec2 = pop_importance_vectors[pop2]
        
        # Calculate Jensen-Shannon divergence
        js_div = jensenshannon(vec1, vec2)
        divergences.append(js_div)
    
    # Return average divergence across all pairs
    return np.mean(divergences) if divergences else 0.0

def analyze_top_features(model_results, top_n=10):
    """
    Analyze differences in top features across populations.
    
    Returns:
        dict: Analysis of top features
    """
    # Extract top N features for each population
    pop_top_features = {}
    for pop_name, results in model_results.items():
        imp_df = results["importances"]
        pop_top_features[pop_name] = imp_df.head(top_n)["feature"].tolist()
    
    # Calculate Jaccard similarity between top feature sets
    jaccard_scores = []
    for pop1, pop2 in combinations(pop_top_features.keys(), 2):
        set1 = set(pop_top_features[pop1])
        set2 = set(pop_top_features[pop2])
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        jaccard = intersection / union if union > 0 else 0
        jaccard_scores.append(jaccard)
    
    # Find features unique to specific populations
    all_top_features = set()
    for features in pop_top_features.values():
        all_top_features.update(features)
    
    unique_features = {}
    for pop_name, features in pop_top_features.items():
        features_set = set(features)
        other_pops_features = set()
        for other_pop, other_features in pop_top_features.items():
            if other_pop != pop_name:
                other_pops_features.update(other_features)
        
        unique = features_set - other_pops_features
        if unique:
            unique_features[pop_name] = list(unique)
    
    return {
        "top_features_by_population": pop_top_features,
        "avg_jaccard_similarity": np.mean(jaccard_scores) if jaccard_scores else 1.0,
        "unique_features": unique_features
    }

def run_analysis(seed=42):
    """
    Run the full analysis across all population splits and outcomes.
    
    Returns:
        DataFrame: Results ranked by divergence
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Load the data
    print("Loading California Housing dataset...")
    df = load_california_housing()
    
    # Create population splits
    print("Creating population splits...")
    population_splits, df = create_population_splits(df)
    
    # Define feature columns to use
    feature_cols = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
        'Population', 'AveOccup', 'Latitude', 'Longitude'
    ]
    
    # Define outcome variables
    outcomes = define_outcome_variables(df)
    
    # Store results
    results = []
    
    # Process each combination of population split and outcome
    total_combinations = len(population_splits) * len(outcomes)
    print(f"Analyzing {total_combinations} combinations of population splits and outcomes...")
    
    for i, (split_name, populations) in enumerate(population_splits.items(), 1):
        for outcome_name, outcome_info in outcomes.items():
            print(f"Processing {split_name} split with {outcome_name} outcome ({i}/{len(population_splits)})...")
            
            outcome_col = outcome_info["column"]
            task_type = outcome_info["task"]
            
            # Split data by population
            pop_dfs = split_data_by_population(df, populations)
            
            # Check if we have at least two populations with data
            if len(pop_dfs) < 2:
                print(f"Skipping {split_name}/{outcome_name}: not enough populations with data")
                continue
            
            # Preprocess data for each population
            pop_data = preprocess_population_data(pop_dfs, feature_cols, outcome_col)
            
            # Check if we have at least two populations with data after preprocessing
            if len(pop_data) < 2:
                print(f"Skipping {split_name}/{outcome_name}: not enough populations with data after preprocessing")
                continue
            
            # Train XGBoost models and get feature importances
            model_results = train_xgboost_models(pop_data, task_type)
            
            # Calculate divergence in feature importances
            divergence = calculate_importance_divergence(model_results)
            
            # Analyze top features
            feature_analysis = analyze_top_features(model_results)
            
            # Store results
            result = {
                "population_split": split_name,
                "outcome": outcome_name,
                "importance_divergence": divergence,
                "avg_jaccard_similarity": feature_analysis["avg_jaccard_similarity"],
                "num_populations": len(model_results),
                "model_results": model_results,
                "feature_analysis": feature_analysis
            }
            
            results.append(result)
    
    # Convert to DataFrame and rank by divergence
    results_df = pd.DataFrame([
        {
            "population_split": r["population_split"],
            "outcome": r["outcome"],
            "importance_divergence": r["importance_divergence"],
            "avg_jaccard_similarity": r["avg_jaccard_similarity"],
            "num_populations": r["num_populations"]
        }
        for r in results
    ])
    
    # Rank by divergence (higher = more different)
    ranked_df = results_df.sort_values("importance_divergence", ascending=False).reset_index(drop=True)
    
    # Store full results for further analysis
    full_results = {
        "ranked_df": ranked_df,
        "detailed_results": results
    }
    
    return full_results

def visualize_results(results, output_dir=None):
    """
    Create visualizations of the analysis results.
    
    Args:
        results: The analysis results
        output_dir: Directory to save outputs (uses current directory if None)
    """
    # Ensure output directory exists
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    ranked_df = results["ranked_df"]
    detailed_results = results["detailed_results"]
    
    # 1. Bar chart of divergence scores
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x="population_split", 
        y="importance_divergence", 
        hue="outcome", 
        data=ranked_df
    )
    plt.title("Feature Importance Divergence by Population Split and Outcome")
    plt.xlabel("Population Split")
    plt.ylabel("Importance Divergence Score")
    plt.xticks(rotation=45)
    plt.legend(title="Outcome", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "california_importance_divergence.png"), dpi=300)
    
    # 2. Heatmap of top features for the highest divergence combination
    if detailed_results:
        # Get the highest divergence result
        top_result = max(detailed_results, key=lambda x: x["importance_divergence"])
        
        # Create a heatmap of feature importance for each population
        top_model_results = top_result["model_results"]
        split_name = top_result["population_split"]
        outcome_name = top_result["outcome"]
        
        # Get all unique features across populations
        all_features = sorted(set(
            feature 
            for pop_results in top_model_results.values() 
            for feature in pop_results["importances"]["feature"]
        ))
        
        # Create importance matrix
        importance_matrix = []
        population_names = []
        
        for pop_name, results in top_model_results.items():
            imp_df = results["importances"]
            imp_dict = dict(zip(imp_df["feature"], imp_df["importance"]))
            
            # Create vector with all features, filling in zeros for missing ones
            vector = [imp_dict.get(feat, 0.0) for feat in all_features]
            importance_matrix.append(vector)
            population_names.append(pop_name)
        
        # Convert to array
        importance_array = np.array(importance_matrix)
        
        # Create heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            importance_array, 
            cmap="YlGnBu", 
            xticklabels=all_features, 
            yticklabels=population_names,
            cbar_kws={"label": "Feature Importance"}
        )
        plt.title(f"Feature Importance by Population for {split_name} / {outcome_name}")
        plt.xlabel("Features")
        plt.ylabel("Population")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"california_top_heatmap_{split_name}_{outcome_name}.png"), dpi=300)
    
    # 3. Plot performance metrics across populations for top divergence combinations
    if detailed_results and len(detailed_results) > 0:
        # Get top 3 highest divergence results
        top_3_results = sorted(detailed_results, key=lambda x: x["importance_divergence"], reverse=True)[:3]
        
        for result in top_3_results:
            split_name = result["population_split"]
            outcome_name = result["outcome"]
            model_results = result["model_results"]
            
            # Extract R² scores for each population
            r2_scores = {
                pop_name: results["performance"]["r2"] 
                for pop_name, results in model_results.items()
            }
            
            # Plot R² scores
            plt.figure(figsize=(10, 5))
            plt.bar(r2_scores.keys(), r2_scores.values())
            plt.title(f"R² Score by Population for {split_name} / {outcome_name}")
            plt.xlabel("Population")
            plt.ylabel("R² Score")
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"california_r2_scores_{split_name}_{outcome_name}.png"), dpi=300)
    
    # Display results
    print("Top population splits by feature importance divergence:")
    print(ranked_df.head(10))


def main():
    """Main execution function"""
    print("Starting California Housing Feature Importance Divergence Analysis...")
    
    # Get current directory for saving files
    current_dir = os.getcwd()
    print(f"Working directory: {current_dir}")
    
    # Create output directory
    output_dir = os.path.join(current_dir, "california_analysis_output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")
    
    # Run the analysis
    results = run_analysis(seed=42)
    
    # Visualize the results
    visualize_results(results, output_dir=output_dir)
    
    # Save results to CSV
    results["ranked_df"].to_csv(os.path.join(output_dir, "california_divergence_results.csv"), index=False)
    
    print(f"Analysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()