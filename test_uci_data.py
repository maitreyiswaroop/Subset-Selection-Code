import numpy as np
from data_uci_adult import get_uci_pop_data

# Test with default settings
print("Testing get_uci_pop_data with default settings...")
pop_data = get_uci_pop_data(
    populations=['Male', 'Female', 'WhiteMale', 'NonWhiteMale'],
    categorical_encoding='onehot',
    subsample=True,
    subsample_fraction=0.1
)

# Print population information
print(f"Number of populations: {len(pop_data)}")
for pop in pop_data:
    pop_id = pop['pop_id']
    X_shape = pop['X_raw'].shape
    Y_shape = pop['Y_raw'].shape
    print(f"Population {pop_id}: X shape {X_shape}, Y shape {Y_shape}")
    
    # Calculate basic statistics
    if X_shape[0] > 0:
        # For each numerical feature
        num_features = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
        for i, feat in enumerate(num_features):
            if i < X_shape[1]:  # Ensure feature exists
                values = pop['X_raw'][:, i]
                print(f"  {feat}: mean={values.mean():.2f}, min={values.min():.2f}, max={values.max():.2f}")
        
        # Classification target stats
        if 'Y_raw' in pop and pop['Y_raw'].size > 0:
            pos_rate = pop['Y_raw'].mean()
            print(f"  Target (>50K): {pos_rate:.2%}")

# Test with numerical features only
print("\nTesting with numerical features only...")
pop_data_num = get_uci_pop_data(
    populations=['Male', 'Female'],
    categorical_encoding=None,  # Numerical features only
    subsample=True,
    subsample_fraction=0.1
)

for pop in pop_data_num:
    print(f"Population {pop['pop_id']} (numeric only): X shape {pop['X_raw'].shape}")

# Test with label encoding
print("\nTesting with label encoding...")
pop_data_label = get_uci_pop_data(
    populations=['Young', 'Middle', 'Senior'],
    categorical_encoding='label',  # Label encoding
    subsample=True,
    subsample_fraction=0.1
)

for pop in pop_data_label:
    print(f"Population {pop['pop_id']} (label encoding): X shape {pop['X_raw'].shape}")