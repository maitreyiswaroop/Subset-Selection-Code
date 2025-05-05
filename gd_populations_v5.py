# gd_populations_v5.py: REINFORCE gradient for Term 2

"""
This script performs variable subset selection using gradient descent.
It precomputes the conditional expectation predictions E[Y|X] and the squared functional
term (i.e. E[E[Y|X]^2]) in a K-fold manner for enhanced robustness.

Version 5 uses the REINFORCE (Score Function) estimator for the gradient of Term 2
(E[E[Y|S]^2]) with respect to alpha, avoiding backpropagation through the
conditional expectation estimator E[Y|S].

The objective L(alpha) = (Term1 - Term2(alpha)) + Penalty(alpha) is MINIMIZED.
Minimizing (Term1 - T2) encourages small alpha for relevant features.
Minimizing Penalty (e.g., Reciprocal_L1) encourages large alpha.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree
import argparse
from torch.utils.data import DataLoader
import re
import time # For timing
import math # For plotting grid

# Assume these imports exist and work as in v4
from data import generate_data_continuous, generate_data_continuous_with_corr
# from tune_estimator import find_best_estimator # Not used directly in main script run
from estimators import * # Needs plugin_estimator_squared_conditional, plugin_estimator_conditional_mean, IF_estimator_squared_conditional, estimate_conditional_keops, IF_estimator_conditional_mean

# Global hyperparameters and clamping constants
CLAMP_MAX = 10.0
CLAMP_MIN = 1e-3 # Increased slightly for stability with division by alpha
EPS = 1e-8       # Small epsilon for division stability
FREEZE_THRESHOLD = 0.1  # Threshold below which alpha values are frozen
N_FOLDS = 5             # Number of K-folds for precomputation (if using IF estimators)

# =============================================================================
# Regularization penalty (Now truly a penalty to encourage large alpha)
# =============================================================================

def compute_penalty(alpha, penalty_type, penalty_lambda, epsilon=EPS):
    """
    Compute a penalty term P(alpha) designed to encourage large alpha values.
    We minimize L = (T1 - T2) + P(alpha).
    """
    # Clamp alpha within the function to ensure gradients are computed correctly
    # The parameter alpha itself is clamped after the optimizer step.
    alpha_clamped = torch.clamp(alpha, min=CLAMP_MIN, max=CLAMP_MAX)

    if penalty_type is None or penalty_lambda == 0:
        return torch.tensor(0.0, device=alpha.device, requires_grad=alpha.requires_grad)
    elif penalty_type == "Reciprocal_L1":
        # Minimizing this encourages large alpha
        return penalty_lambda * torch.sum(1.0 / (alpha_clamped + epsilon))
    elif penalty_type == "Neg_L1":
         # Minimizing this encourages small alpha (like traditional L1)
         # This might be used if the goal was sparsity in a different sense
         print("Warning: Using Neg_L1 penalty encourages small alpha.")
         return penalty_lambda * torch.sum(torch.abs(alpha_clamped))
    elif penalty_type == "Max_Dev":
         # Minimizing this encourages alpha -> 1.0
         max_val = torch.tensor(1.0, device=alpha.device)
         return penalty_lambda * torch.sum(torch.abs(max_val - alpha_clamped))
    elif penalty_type == "Quadratic_Barrier":
         # Minimizing this encourages large alpha
         return penalty_lambda * torch.sum((alpha_clamped + epsilon) ** (-2))
    elif penalty_type == "Exponential":
         # Minimizing this encourages large alpha
         return penalty_lambda * torch.sum(torch.exp(-alpha_clamped))
    elif penalty_type == "None":
        return torch.tensor(0.0, device=alpha.device, requires_grad=alpha.requires_grad)
    else:
        raise ValueError("Unknown penalty_type: " + str(penalty_type))

# =============================================================================
# Objective Value Computation (for tracking/selection, NOT for gradient)
# =============================================================================

def compute_objective_value(X, E_Y_given_X, term1,
                            alpha, penalty_lambda=0, penalty_type=None,
                            num_mc_samples=25, base_model_type="rf"): # Added base_model_type
    """
    Computes the value of the objective L = (T1 - T2) + P for tracking purposes.
    Uses Monte Carlo sampling for Term 2, estimating E[Y|S] with IF estimator.
    Does NOT compute gradients via autograd for T2.
    """
    X = X.to(alpha.device)
    E_Y_given_X = E_Y_given_X.to(alpha.device)
    # Use a detached, clamped version of alpha for value calculation
    alpha_val = alpha.detach().clone().clamp_(min=CLAMP_MIN, max=CLAMP_MAX)

    term1 = torch.tensor(term1, dtype=alpha_val.dtype, device=alpha_val.device)

    avg_term2 = 0.0
    # Estimate Term 2: E[ (E[Y|S])^2 ] using MC samples
    for _ in range(num_mc_samples):
        with torch.no_grad(): # Ensure no gradients are computed here
            # Sample noise DIFFERENTLY each time inside the loop
            epsilon = torch.randn_like(X)
            S_alpha = X + epsilon * torch.sqrt(alpha_val)

            # Estimate E[Y|S] for THIS noise sample
            # Using IF estimator for potentially better accuracy in value computation
            # Requires S_alpha and E_Y_given_X (precomputed on original X)
            # Convert to numpy if IF estimator requires it
            S_alpha_np = S_alpha.cpu().numpy()
            E_Y_given_X_np = E_Y_given_X.cpu().numpy() # Assuming E_Y_given_X is standardized
            # Need to decide which IF estimator to call - one that takes S and E[Y|X]?
            # Or one that takes S and Y? Let's assume IF_estimator_conditional_mean exists
            # and works on S and Y (standardized).
            # We need Y here. This function signature needs Y.
            # Let's revert to estimate_conditional_keops for simplicity here,
            # as compute_objective_value_if handles the direct IF call.
            E_Y_S = estimate_conditional_keops(X, S_alpha, E_Y_given_X, alpha_val)

            # Calculate term2 for THIS noise sample and accumulate
            term2_sample = E_Y_S.pow(2).mean()
            avg_term2 += term2_sample

    term2_value = avg_term2 / num_mc_samples

    # Compute penalty value using the detached alpha
    alpha_val.requires_grad_(False) # Explicitly ensure no grad needed
    penalty_value = compute_penalty(alpha_val, penalty_type, penalty_lambda)

    # Objective L = (T1 - T2) + P
    objective_val = term1 - term2_value + penalty_value
    return objective_val.item() # Return scalar Python number

def compute_objective_value_if(X, Y, term1_std, # Use standardized term1
                               alpha, penalty_lambda=0, penalty_type=None,
                               base_model_type="rf"):
    """
    Computes the value of the objective L = (T1_std - T2) + P for tracking purposes.
    Uses the IF estimator for Term 2.
    Does NOT compute gradients via autograd for T2.
    Assumes IF_estimator_squared_conditional works on standardized Y.
    """
    # Ensure inputs are on the correct device
    X = X.to(alpha.device)
    Y = Y.to(alpha.device) # Use the standardized Y passed in

    # Use a detached, clamped version of alpha for noise generation
    alpha_val = alpha.detach().clone().clamp_(min=CLAMP_MIN, max=CLAMP_MAX)

    # Ensure term1_std is a tensor on the correct device/dtype
    term1_std_tensor = torch.tensor(term1_std, dtype=alpha_val.dtype, device=alpha_val.device)

    # --- Calculate Term 2 using the IF estimator ---
    term2_value_tensor = torch.tensor(0.0, device=alpha.device) # Initialize
    try:
        with torch.no_grad(): # Ensure no gradients computed during IF estimation
            # Sample noise using the current alpha
            epsilon = torch.randn_like(X)
            S_alpha = X + epsilon * torch.sqrt(alpha_val)

            # Convert S and Y to numpy for the estimator if needed
            S_alpha_np = S_alpha.cpu().numpy()
            Y_np = Y.cpu().numpy() # Use standardized Y

            # Call the IF estimator for E[(E[Y|S])^2]
            term2_value_float = IF_estimator_squared_conditional(
                S_alpha_np, Y_np, estimator_type=base_model_type # Pass base model if needed
            )
            term2_value_tensor = torch.tensor(term2_value_float, dtype=alpha_val.dtype, device=alpha.device)

    except Exception as e:
        print(f"Warning: IF_estimator_squared_conditional failed: {e}")
        # Handle error: maybe return NaN or a large value?
        return float('nan')


    # --- Compute penalty value using the detached alpha ---
    alpha_val.requires_grad_(False) # Ensure no grad needed
    penalty_value = compute_penalty(alpha_val, penalty_type, penalty_lambda)

    # Objective L = (T1_std - T2) + P
    objective_val = term1_std_tensor - term2_value_tensor + penalty_value
    return objective_val.item() # Return scalar Python number


# =============================================================================
# Experiment runner for multi-population variable selection (v5: REINFORCE)
# =============================================================================

def get_pop_data(pop_configs, m1, m,
                  dataset_size=10000,
                  noise_scale=0.0,
                  corr_strength=0.0,
                  common_meaningful_indices=None,
                  estimator_type="plugin",
                  device="cpu",
                  base_model_type="rf",
                  batch_size=10000, # Currently assumes full batch in run_experiment
                  seed=None):
    """
    Generate datasets for each population based on the provided configurations.
    Precomputes E[Y|X] and Term 1.
    """
    k_common = max(1, m1 // 2)
    if common_meaningful_indices is None:
        common_meaningful_indices = np.arange(k_common)

    pop_data = []
    for pop_config in pop_configs:
        pop_id = pop_config['pop_id']
        dataset_type = pop_config['dataset_type']
        current_seed = seed + pop_id if seed is not None else None # Ensure different data per pop if seed is set

        if corr_strength > 0:
            new_X, Y, A, meaningful_indices = generate_data_continuous_with_corr(
                pop_id=pop_id, m1=m1, m=m,
                dataset_type=dataset_type,
                dataset_size=dataset_size,
                noise_scale=noise_scale,
                corr_strength=corr_strength,
                seed=current_seed,
                common_meaningful_indices=common_meaningful_indices
            )
        else:
            new_X, Y, A, meaningful_indices = generate_data_continuous(
                pop_id=pop_id, m1=m1, m=m,
                dataset_type=dataset_type,
                dataset_size=dataset_size,
                noise_scale=noise_scale,
                seed=current_seed,
                common_meaningful_indices=common_meaningful_indices
            )
        X_np = new_X
        Y_np = Y
        X = torch.tensor(X_np, dtype=torch.float32)
        # Standardize the features
        X_mean = X.mean(dim=0)
        X_std = X.std(dim=0)
        X = (X - X_mean) / (X_std + EPS)

        Y = torch.tensor(Y_np, dtype=torch.float32)
        Y_mean = Y.mean()
        Y_std = Y.std()
        Y_torch_std = (Y - Y_mean) / (Y_std + EPS)  # Standardized Y tensor

        print(f"Population {pop_id}: Precomputing E[Y|X] and Term 1 using {estimator_type}/{base_model_type}...")
        # Use original (non-standardized) data for estimators if they expect it
        if estimator_type == "plugin":
            # Note: These estimators might internally use KFold
            term1 = plugin_estimator_squared_conditional(X_np, Y_np, estimator_type=base_model_type)
            E_Y_given_X_np = plugin_estimator_conditional_mean(X_np, Y_np, estimator_type=base_model_type)
        elif estimator_type == "if":
            term1 = IF_estimator_squared_conditional(X_np, Y_np, estimator_type=base_model_type)
            # Still use plugin for the conditional mean estimate needed in Term 2 calculation
            E_Y_given_X_np = plugin_estimator_conditional_mean(X_np, Y_np, estimator_type=base_model_type)
        else:
            raise ValueError("estimator_type must be 'plugin' or 'if'")

        E_Y_given_X = torch.tensor(E_Y_given_X_np, dtype=torch.float32)
        # Standardize the precomputed E[Y|X] to match standardized Y
        E_Y_given_X_std = (E_Y_given_X - Y_mean) / (Y_std + EPS)

        print(f"Population {pop_id}: Term 1 (orig scale) = {term1:.4f}")

        pop_data.append({
            'pop_id': pop_id,
            'X': X.to(device),  # Standardized X
            'Y': Y_torch_std.to(device),  # Standardized Y
            'E_Y_given_X': E_Y_given_X_std.to(device),  # Standardized E[Y|X]
            'meaningful_indices': meaningful_indices,
            # 'term1': term1, # Keep original scale term1 if needed elsewhere?
            'term1_std': E_Y_given_X_std.pow(2).mean().item()  # Standardized term1
        })
        print(f"Population {pop_id}: Term 1 (standardized) = {pop_data[-1]['term1_std']:.4f}")
    return pop_data

def init_alpha(m, alpha_init="random", noise = 0.1, # Increased default noise slightly
               device="cpu"):
    """
    Initialize the alpha parameter based on the specified initialization method.
    """
    init_val = torch.ones(m, device=device) # Default base
    if re.match(r"random_(\d+(\.\d+)?)", alpha_init):
        try:
            k = float(alpha_init.split('_')[1])
            print(f"Initializing alpha randomly around {k}")
            init_val = k * torch.ones(m, device=device) + noise * torch.abs(torch.randn(m, device=device))
        except ValueError:
             raise ValueError(f"Invalid numeric value in alpha_init: {alpha_init}")
    elif alpha_init == "ones":
        print("Initializing alpha to ones + noise")
        init_val = torch.ones(m, device=device) + noise * torch.abs(torch.randn(m, device=device))
    elif alpha_init == "random":
        # Ensure initial values are positive and somewhat varied
        print("Initializing alpha randomly around 1")
        init_val = torch.ones(m, device=device) + noise * torch.abs(torch.randn(m, device=device))
    else:
        raise ValueError("alpha_init must be 'ones', 'random', or 'random_k.k'")

    # Ensure initial values are within clamp bounds
    init_val.clamp_(min=CLAMP_MIN, max=CLAMP_MAX)
    return torch.nn.Parameter(init_val)


def run_experiment_multi_population(pop_configs, m1, m,
                                    dataset_size=5000,
                                    budget=None,
                                    noise_scale=0.0,
                                    corr_strength=0.0,
                                    num_epochs=30,
                                    penalty_type=None, # Renamed from reg_type
                                    penalty_lambda=0,  # Renamed from reg_lambda
                                    learning_rate=0.001,
                                    batch_size=10000, # Full batch assumed for now
                                    optimizer_type='sgd', seed=None,
                                    alpha_init="random",
                                    early_stopping_patience=10, # Increased patience
                                    save_path='./results/multi_population/',
                                    estimator_type="plugin",  # "plugin" or "if"
                                    base_model_type="rf",     # "rf" or "krr"
                                    # looped=False, # Keep this option? Seems less relevant now
                                    param_freezing=True,
                                    N_grad_samples=50, # Number of MC samples for REINFORCE gradient
                                    use_baseline=True, # Use baseline for variance reduction
                                    objective_value_estimator='if', # 'if' or 'mc'
                                    verbose=False):
    """
    Main experiment runner using REINFORCE gradient estimation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # Ensure batch_size doesn't exceed dataset_size if full batch is used
    batch_size = min(batch_size, dataset_size)
    os.makedirs(save_path, exist_ok=True)

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        seed = np.random.randint(0, 10000)
    print(f"Using seed: {seed}")

    if budget is None:
        # Calculate budget based on common and population-specific features
        k_common = max(1, m1 // 2)
        k_pop_specific = m1 - k_common
        budget = k_common + len(pop_configs) * k_pop_specific
    print(f"Budget for variable selection: {budget}")

    # Define common meaningful indices (as in v2/v4)
    k_common = max(1, m1 // 2)
    common_meaningful_indices = np.arange(k_common)

    # Generate datasets for each population
    pop_data = get_pop_data(
        pop_configs=pop_configs,
        m1=m1, m=m,
        dataset_size=dataset_size,
        noise_scale=noise_scale,
        corr_strength=corr_strength,
        common_meaningful_indices=common_meaningful_indices,
        estimator_type=estimator_type, # For precomputing Term1 and E[Y|X]
        device=device,
        base_model_type=base_model_type,
        batch_size=batch_size, # Passed but not used if full batch assumed
        seed=seed
    )

    # Initialize alpha (the variable weight parameters)
    alpha = init_alpha(m, alpha_init=alpha_init, noise=0.1, device=device)
    print(f"Initialized alpha: {alpha.detach().cpu().numpy()}")

    # Setup optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam([alpha], lr=learning_rate)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD([alpha], lr=learning_rate, momentum=0.9, nesterov=True)
    else:
         raise ValueError("Unsupported optimizer_type. Choose 'adam' or 'sgd'.")

    # History tracking
    alpha_history = [alpha.detach().cpu().numpy().copy()]
    objective_history = []
    gradient_history = [] # Store computed total gradients
    term2_grad_history = [] # Store T2 gradient component
    penalty_grad_history = [] # Store Penalty gradient component

    best_objective_val = float('inf') # We are minimizing L
    best_alpha = alpha.detach().cpu().numpy().copy()
    early_stopping_counter = 0

    print(f"\nStarting optimization for {num_epochs} epochs...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        optimizer.zero_grad() # Clear gradients stored in optimizer state

        population_objective_values = []
        pop_forward_components = [] # Stores {X, Y, E_Y_given_X, term1_std}

        # --- Step 1: Forward Pass for Objective VALUE and Finding Winning Population ---
        # Calculate the objective value L = (T1_std - T2) + P for each population
        # using the current alpha to find the population with the maximum value.
        # This value is for tracking and selection, NOT for gradient calculation via autograd.
        current_alpha_val = alpha.detach().clone() # Use detached alpha for value calc
        obj_calc_start_time = time.time()
        for pop in pop_data:
            if objective_value_estimator == 'if':
                 obj_val = compute_objective_value_if(
                     pop['X'], pop['Y'], pop['term1_std'], # Use standardized term1
                     current_alpha_val,
                     penalty_lambda, penalty_type,
                     base_model_type=base_model_type
                 )
            elif objective_value_estimator == 'mc':
                 obj_val = compute_objective_value(
                     pop['X'], pop['E_Y_given_X'], pop['term1_std'], # Use standardized term1
                     current_alpha_val,
                     penalty_lambda, penalty_type,
                     num_mc_samples=N_grad_samples, # Use same MC samples as grad for consistency
                     base_model_type=base_model_type
                 )
            else:
                raise ValueError("objective_value_estimator must be 'if' or 'mc'")

            population_objective_values.append(obj_val)
            # Store components needed for gradient calculation later
            pop_forward_components.append({
                'X': pop['X'],
                'Y': pop['Y'], # Store Y as well
                'E_Y_given_X': pop['E_Y_given_X'],
                'term1_std': pop['term1_std'],
                'pop_id': pop['pop_id']
            })
        obj_calc_time = time.time() - obj_calc_start_time

        # --- Step 2: Determine Winning Population ---
        # We minimize L = max_pop (T1 - T2 + P). Find the population with the current max L.
        # Handle potential NaN values from IF estimator failure
        valid_obj_values = [v for v in population_objective_values if not math.isnan(v)]
        if not valid_obj_values:
             print(f"ERROR: All population objective values are NaN at epoch {epoch}. Stopping.")
             # Optionally return current best or raise error
             # For now, let's break the loop
             break
        current_robust_objective_value = max(valid_obj_values)

        # Find index corresponding to the max valid value
        winning_pop_index = -1
        for idx, val in enumerate(population_objective_values):
            if val == current_robust_objective_value:
                winning_pop_index = idx
                break

        if winning_pop_index == -1:
             print(f"ERROR: Could not determine winning population at epoch {epoch}. Stopping.")
             break # Should not happen if valid_obj_values is not empty

        winning_pop_data = pop_forward_components[winning_pop_index]
        winning_pop_id = winning_pop_data['pop_id']

        objective_history.append(current_robust_objective_value) # Track the robust objective value

        # --- Step 3: Calculate Gradient using REINFORCE for Term 2 of Winning Population ---
        grad_calc_start_time = time.time()
        X_win = winning_pop_data['X']
        E_Y_given_X_win = winning_pop_data['E_Y_given_X']
        N_win = X_win.shape[0]

        # Use the actual alpha parameter tensor for gradient calculation
        # Clamping is done internally where needed (e.g., division, penalty)
        current_alpha_param = alpha

        grad_term2_accum = torch.zeros_like(current_alpha_param)

        # Perform K (=N_grad_samples) Monte Carlo samples for the gradient estimate
        for k in range(N_grad_samples):
            # Use a detached, clamped alpha for noise generation and estimator call
            # This ensures the gradient doesn't flow through these operations wrongly
            current_alpha_clamped_detached = current_alpha_param.data.clamp(CLAMP_MIN, CLAMP_MAX)

            with torch.no_grad(): # Sample noise and estimate g(S) without tracking gradients here
                epsilon_k = torch.randn_like(X_win)
                S_alpha_k = X_win + epsilon_k * torch.sqrt(current_alpha_clamped_detached)

                # Estimate g(S) = E[Y|S] using KeOps estimator for gradient step
                # Pass the DETACHED alpha.
                g_hat_S_k = estimate_conditional_keops(X_win, S_alpha_k, E_Y_given_X_win, current_alpha_clamped_detached)
                g_hat_S_k_squared = g_hat_S_k.pow(2) # Shape [N_win]

            # Baseline for variance reduction
            if use_baseline:
                # Using batch mean of g(S)^2 as baseline
                baseline = g_hat_S_k_squared.mean()
            else:
                baseline = 0.0

            # Compute score function term: (epsilon^2 - 1) / (2 * alpha)
            # Use the clamped alpha value for stability in division
            score_term = (epsilon_k.pow(2) - 1.0) / (2.0 * current_alpha_clamped_detached + EPS) # Shape [N_win, m]

            # Accumulate gradient estimate for T2: E[ g(S)^2 * score ]
            # grad_T2_sample = E[ (g(S)^2 - baseline) * score ]
            term_to_average = (g_hat_S_k_squared - baseline).unsqueeze(1) * score_term # Shape [N_win, m]
            grad_term2_accum += term_to_average.mean(dim=0) # Average over batch dimension N_win

        # Final gradient estimate for T2 (averaged over MC samples)
        grad_term2_final = grad_term2_accum / N_grad_samples
        term2_grad_history.append(grad_term2_final.detach().cpu().numpy().copy())

        # --- Step 4: Calculate Gradient of Penalty Term P(alpha) ---
        # Use autograd for the penalty part, applied to the *parameter* alpha
        # Ensure alpha requires grad before this calculation
        current_alpha_param.requires_grad_(True)
        penalty_term_value = compute_penalty(current_alpha_param, penalty_type, penalty_lambda)

        if penalty_term_value.requires_grad:
             # Calculate gradients ONLY for the penalty term P(alpha)
             grad_penalty = torch.autograd.grad(penalty_term_value, current_alpha_param, retain_graph=False)[0]
        else:
             grad_penalty = torch.zeros_like(current_alpha_param)
        penalty_grad_history.append(grad_penalty.detach().cpu().numpy().copy())
        # Detach alpha again after grad calculation
        current_alpha_param.requires_grad_(False)


        # --- Step 5: Combine Gradients and Assign ---
        # We minimize L = (T1 - T2) + P
        # Gradient is grad(L) = -grad(T2) + grad(P)
        total_gradient = -grad_term2_final + grad_penalty

        # Assign the manually computed gradient to the .grad attribute
        alpha.grad = total_gradient

        gradient_history.append(total_gradient.detach().cpu().numpy().copy())
        grad_calc_time = time.time() - grad_calc_start_time

        # --- Step 6: Optional: Parameter Freezing & Gradient Clipping ---
        if param_freezing:
            with torch.no_grad():
                # Freeze gradients for alpha values that are already small
                frozen_mask = alpha.data < FREEZE_THRESHOLD
                if alpha.grad is not None:
                    alpha.grad[frozen_mask] = 0
                    # Also clear optimizer momentum for frozen parameters if using SGD
                    if optimizer_type.lower() == 'sgd':
                        state = optimizer.state.get(alpha) # Check if state exists
                        if state and 'momentum_buffer' in state:
                           buf = state['momentum_buffer']
                           buf[frozen_mask] = 0

        # Clip the final computed gradient
        if alpha.grad is not None:
            grad_norm_before_clip = torch.linalg.norm(alpha.grad).item()
            torch.nn.utils.clip_grad_norm_([alpha], max_norm=10.0)
            grad_norm_after_clip = torch.linalg.norm(alpha.grad).item()
        else:
             print("Warning: No gradient computed for optimizer step.")
             grad_norm_before_clip = 0.0
             grad_norm_after_clip = 0.0


        # --- Step 7: Optimizer Step ---
        optimizer.step() # Updates alpha based on alpha.grad

        # --- Step 8: Clamp Alpha Parameter ---
        # Clamp the parameter itself after the optimizer step
        with torch.no_grad():
            alpha.data.clamp_(min=CLAMP_MIN, max=CLAMP_MAX)

        # --- History tracking and Logging ---
        alpha_history.append(alpha.detach().cpu().numpy().copy())
        epoch_time = time.time() - epoch_start_time

        if verbose or (epoch % 10 == 0):
            print(f"Epoch {epoch}/{num_epochs} | EpTime: {epoch_time:.2f}s (Obj: {obj_calc_time:.2f}s, Grad: {grad_calc_time:.2f}s) | WinPop: {winning_pop_id} | Robust Obj: {current_robust_objective_value:.4f}")
            if verbose:
                 print(f"  Alpha: {alpha.detach().cpu().numpy()}")
                 print(f"  Grad Norm (B/A Clip): {grad_norm_before_clip:.4f} / {grad_norm_after_clip:.4f}")


        # --- Early Stopping Logic ---
        # Stop if the objective hasn't improved significantly
        if current_robust_objective_value < best_objective_val - EPS: # Check for improvement
            best_objective_val = current_robust_objective_value
            best_alpha = alpha.detach().cpu().numpy().copy()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch} due to lack of improvement.")
            break

    # --- End of Training Loop ---
    total_time = time.time() - start_time
    print(f"\nOptimization finished in {total_time:.2f} seconds.")
    print(f"Best robust objective value achieved: {best_objective_val:.4f}")

    # (Optional) Save diagnostics, for example plotting the objective history
    if verbose:
        try:
            plt.figure(figsize=(12, 10))

            plt.subplot(3, 1, 1)
            valid_epochs = np.arange(len(objective_history))
            plt.plot(valid_epochs, objective_history, label="Robust Objective L = max(T1-T2+P)")
            plt.xlabel("Epoch")
            plt.ylabel("Objective Value")
            plt.title("Objective Value vs Epoch")
            plt.legend()
            plt.grid(True)

            plt.subplot(3, 1, 2)
            grad_norms = [np.linalg.norm(g) for g in gradient_history if g is not None]
            term2_grad_norms = [np.linalg.norm(g) for g in term2_grad_history if g is not None]
            penalty_grad_norms = [np.linalg.norm(g) for g in penalty_grad_history if g is not None]
            epochs_with_grad = np.arange(len(grad_norms))
            if grad_norms:
                 plt.plot(epochs_with_grad, grad_norms, label="Total Grad Norm ||-∇T2 + ∇P||")
            if term2_grad_norms:
                 plt.plot(epochs_with_grad, term2_grad_norms, label="Term 2 Grad Norm ||∇T2||", linestyle='--')
            if penalty_grad_norms:
                 plt.plot(epochs_with_grad, penalty_grad_norms, label="Penalty Grad Norm ||∇P||", linestyle=':')

            plt.xlabel("Epoch")
            plt.ylabel("Gradient Norm")
            plt.title("Gradient Norms vs Epoch")
            plt.legend()
            plt.grid(True)
            plt.yscale('log') # Use log scale for norms

            plt.subplot(3, 1, 3)
            alpha_hist_np = np.array(alpha_history)
            epochs_alpha = np.arange(alpha_hist_np.shape[0])
            num_alphas_to_plot = min(m, 10) # Plot up to 10 alphas
            indices_to_plot = np.linspace(0, m - 1, num_alphas_to_plot, dtype=int) # Select evenly spaced indices
            for i in indices_to_plot:
                plt.plot(epochs_alpha, alpha_hist_np[:, i], label=f'alpha_{i}')
            plt.xlabel("Epoch")
            plt.ylabel("Alpha Value")
            plt.title(f"Alpha Trajectories (Indices: {indices_to_plot})")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True)
            plt.yscale('log') # Log scale often useful for alpha


            plt.tight_layout()
            plt.savefig(os.path.join(save_path, "optimization_diagnostics.png"))
            plt.close()
            print(f"Diagnostic plots saved to {os.path.join(save_path, 'optimization_diagnostics.png')}")
        except Exception as e:
            print(f"Warning: Failed to generate diagnostic plots: {e}")

    # Return results dictionary
    # Ensure results are serializable (convert numpy arrays to lists)
    final_alpha_np = best_alpha # Use the best alpha found
    selected_indices = np.argsort(final_alpha_np)[:budget]

    # Make sure meaningful_indices are lists for JSON serialization
    meaningful_indices_list = []
    for pop in pop_data:
        mi = pop['meaningful_indices']
        if isinstance(mi, np.ndarray):
            meaningful_indices_list.append(mi.tolist())
        elif isinstance(mi, list):
            meaningful_indices_list.append(mi)
        else:
            meaningful_indices_list.append(list(mi)) # Attempt conversion

    return {
        'final_objective': best_objective_val if not math.isnan(best_objective_val) else None,
        'final_alpha': final_alpha_np.tolist(), # Best alpha
        'selected_indices': selected_indices.tolist(),
        'selected_alphas': final_alpha_np[selected_indices].tolist(),
        'objective_history': [o if not math.isnan(o) else None for o in objective_history],
        'alpha_history': [a.tolist() for a in alpha_history],
        # 'gradient_history': [g.tolist() if g is not None else None for g in gradient_history], # Can be large
        'populations': [pop['pop_id'] for pop in pop_data],
        'meaningful_indices': meaningful_indices_list,
        'total_time_seconds': total_time,
        'stopped_epoch': epoch
    }


# =============================================================================
# Objective Landscape Plotting Function
# =============================================================================

def plot_objective_landscape(final_alpha, pop_data, penalty_type, penalty_lambda,
                             base_model_type, save_path,
                             dims_to_plot=None, num_points=20, perturb_factor=2.0,
                             use_if_for_plot=True):
    """
    Plots the objective function landscape around the final alpha values
    by perturbing one dimension at a time.

    Args:
        final_alpha (np.ndarray): The optimized alpha vector.
        pop_data (list): List of population data dictionaries.
        penalty_type (str): Type of penalty used.
        penalty_lambda (float): Strength of the penalty.
        base_model_type (str): Base model type for IF estimator.
        save_path (str): Directory to save the plot.
        dims_to_plot (list, optional): List of alpha dimensions (indices) to plot.
                                      If None, plots all dimensions. Defaults to None.
        num_points (int, optional): Number of points to evaluate per dimension. Defaults to 20.
        perturb_factor (float, optional): Factor to determine perturbation range
                                          (e.g., 2.0 means perturb from alpha/2 to alpha*2).
                                          Defaults to 2.0.
        use_if_for_plot (bool, optional): Whether to use the potentially slow IF estimator
                                          for objective value calculation during plotting.
                                          Defaults to True.
    """
    print("\nGenerating objective landscape plots...")
    m = len(final_alpha)
    device = pop_data[0]['X'].device # Get device from data

    if dims_to_plot is None:
        dims_to_plot = list(range(m))

    num_plots = len(dims_to_plot)
    if num_plots == 0:
        print("No dimensions selected for plotting.")
        return

    # Determine grid size for subplots
    ncols = math.ceil(math.sqrt(num_plots))
    nrows = math.ceil(num_plots / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4), squeeze=False)
    axes_flat = axes.flatten() # Flatten axes array for easy iteration

    plot_count = 0
    for dim_idx in dims_to_plot:
        if dim_idx >= m:
            print(f"Warning: Dimension index {dim_idx} out of bounds (m={m}). Skipping.")
            continue

        ax = axes_flat[plot_count]
        current_alpha_val = final_alpha[dim_idx]

        # Define perturbation range, respecting clamp bounds
        alpha_low = max(CLAMP_MIN, current_alpha_val / perturb_factor)
        alpha_high = min(CLAMP_MAX, current_alpha_val * perturb_factor)
        # Ensure range is valid
        if alpha_low >= alpha_high:
             # If range is invalid (e.g., current_alpha_val is CLAMP_MIN), create a small range around it
             alpha_low = max(CLAMP_MIN, current_alpha_val * 0.9)
             alpha_high = min(CLAMP_MAX, current_alpha_val * 1.1)
             if alpha_low >= alpha_high: # Still invalid? Use fixed range
                  alpha_low = CLAMP_MIN
                  alpha_high = CLAMP_MIN * 10 # Arbitrary small range


        perturbed_alphas = np.linspace(alpha_low, alpha_high, num_points)
        objective_values = []

        print(f"  Plotting dimension {dim_idx} (Value: {current_alpha_val:.3f})...")
        alpha_tensor = torch.tensor(final_alpha, dtype=torch.float32, device=device)

        for pert_val in perturbed_alphas:
            alpha_tensor_perturbed = alpha_tensor.clone()
            alpha_tensor_perturbed[dim_idx] = pert_val

            # Calculate robust objective for this perturbed alpha
            pop_objectives_perturbed = []
            for pop in pop_data:
                if use_if_for_plot:
                    obj_val = compute_objective_value_if(
                        pop['X'], pop['Y'], pop['term1_std'],
                        alpha_tensor_perturbed, # Pass perturbed tensor
                        penalty_lambda, penalty_type,
                        base_model_type=base_model_type
                    )
                else:
                    # Use the faster MC version for plotting if specified
                    obj_val = compute_objective_value(
                         pop['X'], pop['E_Y_given_X'], pop['term1_std'],
                         alpha_tensor_perturbed,
                         penalty_lambda, penalty_type,
                         num_mc_samples=5, # Use fewer samples for speed in plotting
                         base_model_type=base_model_type
                    )
                pop_objectives_perturbed.append(obj_val)

            # Handle NaNs and find max
            valid_perturbed_objs = [v for v in pop_objectives_perturbed if not math.isnan(v)]
            if not valid_perturbed_objs:
                robust_obj_perturbed = float('nan')
            else:
                robust_obj_perturbed = max(valid_perturbed_objs)
            objective_values.append(robust_obj_perturbed)

        # Plotting for the current dimension
        ax.plot(perturbed_alphas, objective_values, marker='.', linestyle='-')
        # Mark the final alpha value
        ax.axvline(current_alpha_val, color='r', linestyle='--', label=f'Final α={current_alpha_val:.3f}')
        ax.set_xlabel(f'alpha_{dim_idx}')
        ax.set_ylabel('Robust Objective')
        ax.set_title(f'Objective vs alpha_{dim_idx}')
        ax.legend()
        ax.grid(True)
        plot_count += 1

    # Hide any unused subplots
    for i in range(plot_count, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    plot_filename = "objective_landscape.png"
    plt.savefig(os.path.join(save_path, plot_filename))
    plt.close(fig)
    print(f"Objective landscape plots saved to {os.path.join(save_path, plot_filename)}")


# =============================================================================
# Utility functions for JSON serialization and run numbering
# (Code identical to v4)
# =============================================================================

def convert_numpy_to_python(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        # Handle NaN and Inf
        if np.isnan(obj):
            return None # Or 'NaN' as string
        if np.isinf(obj):
            return None # Or 'Infinity' or '-Infinity'
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_to_python(item) for item in obj] # Recursively convert array elements
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    # Handle torch tensors if they sneak in
    elif torch.is_tensor(obj):
         return convert_numpy_to_python(obj.detach().cpu().numpy())
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    return obj

def get_latest_run_number(save_path):
    """
    Determine the latest run number in the save path directory.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path) # Create if it doesn't exist
        return 0
    existing_runs = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d))]
    run_numbers = []
    for d in existing_runs:
        match = re.match(r'run_(\d+)', d)
        if match:
            run_numbers.append(int(match.group(1)))

    if not run_numbers:
        return 0
    return max(run_numbers) + 1

def compute_population_stats(selected_indices, meaningful_indices_list):
    """
    Compute population-wise statistics for selected variables.
    """
    pop_stats = []
    percentages = [] # Store percentages for overall calculation

    # Ensure selected_indices is a set for efficient lookup
    selected_set = set(selected_indices)

    for i, meaningful in enumerate(meaningful_indices_list):
        meaningful_set = set(meaningful)
        common = selected_set.intersection(meaningful_set)
        count = len(common)
        total = len(meaningful_set)
        percentage = (count / total * 100) if total > 0 else 0.0 # Handle division by zero
        percentages.append(percentage)
        pop_stats.append({
            'population': i,
            'selected_relevant_count': count,
            'total_relevant': total,
            'percentage': percentage
        })

    # Calculate overall stats safely
    min_percentage = min(percentages) if percentages else 0.0
    max_percentage = max(percentages) if percentages else 0.0
    median_percentage = float(np.median(percentages)) if percentages else 0.0

    min_pop_index = np.argmin(percentages) if percentages else -1
    max_pop_index = np.argmax(percentages) if percentages else -1

    overall_stats = {
        'min_percentage': min_percentage,
        'max_percentage': max_percentage,
        'median_percentage': median_percentage,
        'min_population_details': pop_stats[min_pop_index] if min_pop_index != -1 else None,
        'max_population_details': pop_stats[max_pop_index] if max_pop_index != -1 else None,
    }

    return pop_stats, overall_stats


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-population variable selection using REINFORCE (v5)')
    # Data parameters
    parser.add_argument('--m1', type=int, default=4, help='Number of meaningful features per population')
    parser.add_argument('--m', type=int, default=20, help='Total number of features') # Reduced default for faster testing
    parser.add_argument('--dataset-size', type=int, default=5000) # Reduced default
    parser.add_argument('--noise-scale', type=float, default=0.1, help='Noise added to Y in data generation')
    parser.add_argument('--corr-strength', type=float, default=0.0, help='Correlation strength between features')
    parser.add_argument('--populations', nargs='+', default=['linear_regression', 'sinusoidal_regression'], help='Types of datasets for populations')

    # Optimization parameters
    parser.add_argument('--num-epochs', type=int, default=100) # Reduced default
    parser.add_argument('--penalty-type', type=str, default='Reciprocal_L1', choices=['Reciprocal_L1', 'Neg_L1', 'Max_Dev', 'Quadratic_Barrier', 'Exponential', 'None'], help='Type of penalty encouraging large alpha')
    parser.add_argument('--penalty-lambda', type=float, default=0.001, help='Strength of the penalty term')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate for optimizer')
    parser.add_argument('--batch-size', type=int, default=5000, help='Batch size (currently full batch assumed)')
    parser.add_argument('--optimizer-type', type=str, default='adam', choices=['sgd', 'adam'], help='Optimizer type')
    parser.add_argument('--alpha-init', type=str, default='random_1', help='Initialization method for alpha (e.g., random_1 for random around 1)')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--param-freezing', action='store_true', default=True, help='Enable freezing small alpha values during training')
    parser.add_argument('--no-param-freezing', action='store_false', dest='param_freezing', help='Disable parameter freezing')

    # REINFORCE parameters
    parser.add_argument('--N-grad-samples', type=int, default=25, help='Number of MC samples for REINFORCE gradient')
    parser.add_argument('--use-baseline', action='store_true', default=True, help='Use baseline subtraction for variance reduction')
    parser.add_argument('--no-baseline', action='store_false', dest='use_baseline', help='Disable baseline subtraction')

    # Estimation parameters
    parser.add_argument('--estimator-type', type=str, default='if', choices=['plugin', 'if'], help='Estimator type for Term 1 and E[Y|X] precomputation')
    parser.add_argument('--base-model-type', type=str, default='rf', choices=['rf', 'krr', 'xgb'], help='Base model for E[Y|X] estimation (Random Forest or Kernel Ridge)')
    parser.add_argument('--objective-value-estimator', type=str, default='if', choices=['if', 'mc'], help='Estimator for objective value tracking (IF or MC)')


    # Plotting parameters
    parser.add_argument('--plot-landscape', action='store_true', help='Generate objective landscape plots after optimization')
    parser.add_argument('--plot-dims', type=int, nargs='+', default=None, help='Specific dimensions (indices) to plot in landscape (default: all)')
    parser.add_argument('--plot-mc', action='store_true', help='Use faster MC estimator for landscape plotting instead of IF')


    # Other parameters
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--save-path', type=str, default='./results_v5/multi_population/', help='Base directory to save results')
    parser.add_argument('--verbose', action='store_true', help='Enable detailed logging and plotting')

    return parser.parse_args()

def main():
    args = parse_args()
    base_save_path = args.save_path
    # Create base path if it doesn't exist
    os.makedirs(base_save_path, exist_ok=True)

    run_no = get_latest_run_number(base_save_path)
    save_path = os.path.join(base_save_path, f'run_{run_no}/')
    os.makedirs(save_path, exist_ok=True)
    print(f"Results will be saved in: {save_path}")

    pop_configs = [
        {'pop_id': i, 'dataset_type': args.populations[i]}
        for i in range(len(args.populations))
    ]

    # Calculate budget based on common and population-specific features
    k_common = max(1, args.m1 // 2)
    k_pop_specific = args.m1 - k_common
    budget = k_common + len(pop_configs) * k_pop_specific
    # print(f"Calculated budget: {budget} (Common: {k_common}, PopSpecific: {k_pop_specific} x {len(pop_configs)})") # Printed in run_experiment

    # Save experiment parameters for reproducibility
    experiment_params = vars(args).copy() # Get dict from argparse namespace
    # experiment_params['budget'] = budget # Budget is now calculated and printed inside run_experiment
    experiment_params['pop_configs'] = pop_configs
    experiment_params['script_version'] = 'v5_reinforce_plot'

    print("\nRunning multi-population experiment (v5) with parameters:")
    # Use convert_numpy_to_python just in case, though args should be standard types
    serializable_params = convert_numpy_to_python(experiment_params)
    print(json.dumps(serializable_params, indent=4))
    with open(os.path.join(save_path, 'experiment_params.json'), 'w') as f:
        json.dump(serializable_params, f, indent=4)

    # --- Run the main experiment ---
    results = run_experiment_multi_population(
        pop_configs=pop_configs,
        m1=args.m1,
        m=args.m,
        dataset_size=args.dataset_size,
        budget=budget, # Pass calculated budget
        noise_scale=args.noise_scale,
        corr_strength=args.corr_strength,
        num_epochs=args.num_epochs,
        penalty_type=args.penalty_type,
        penalty_lambda=args.penalty_lambda,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        optimizer_type=args.optimizer_type,
        seed=args.seed,
        alpha_init=args.alpha_init,
        estimator_type=args.estimator_type,
        base_model_type=args.base_model_type,
        early_stopping_patience=args.patience,
        save_path=save_path, # Pass save_path for potential internal saving/plotting
        param_freezing=args.param_freezing,
        N_grad_samples=args.N_grad_samples,
        use_baseline=args.use_baseline,
        objective_value_estimator=args.objective_value_estimator,
        verbose=args.verbose
    )

    # --- Post-processing and Saving Results ---
    print("\n--- Experiment Finished ---")
    if results['final_objective'] is not None:
        print(f"Final Robust Objective (minimized): {results['final_objective']:.4f}")
    else:
        print("Final Robust Objective: NaN (Optimization failed)")

    final_alpha = np.array(results['final_alpha'])
    print(f"Final Alpha (best): {final_alpha}")

    selected_indices = results['selected_indices'] # Already calculated in run_experiment
    print(f"Selected Variables (indices, budget={budget}): {selected_indices}")
    print(f"Selected Variables (alpha values): {final_alpha[selected_indices]}")

    # Calculate overall recall across all meaningful indices
    all_meaningful_indices = set()
    meaningful_indices_list = results['meaningful_indices'] # Already converted to list in run_experiment
    for indices in meaningful_indices_list:
        all_meaningful_indices.update(indices)

    selected_set = set(selected_indices)
    intersection = selected_set.intersection(all_meaningful_indices)
    recall = len(intersection) / len(all_meaningful_indices) if len(all_meaningful_indices) > 0 else 0.0
    precision = len(intersection) / len(selected_set) if len(selected_set) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\nOverall Performance:")
    print(f"  Recall: {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")


    print("\nPopulation-wise Important Parameters:")
    populations = results.get('populations', [])
    # meaningful_indices_list already available
    for i, pop_id in enumerate(populations):
        print(f"  Population {pop_id} - Meaningful indices: {meaningful_indices_list[i]}")

    # Compute and print population-wise stats
    pop_stats, overall_stats = compute_population_stats(selected_indices, meaningful_indices_list)
    print("\nPopulation-wise statistics for selected variables:")
    stats_str = "Population-wise statistics for selected variables:\n"
    for stat in pop_stats:
        line = (f"  Population {stat['population']}: {stat['selected_relevant_count']} out of {stat['total_relevant']} "
                f"selected ({stat['percentage']:.2f}%)\n")
        print(line, end='')
        stats_str += line
    print("Overall statistics:")
    print(f"  Min Percentage: {overall_stats['min_percentage']:.2f}%")
    print(f"  Max Percentage: {overall_stats['max_percentage']:.2f}%")
    print(f"  Median Percentage: {overall_stats['median_percentage']:.2f}%")
    stats_str += f"\nOverall statistics: {json.dumps(convert_numpy_to_python(overall_stats), indent=2)}\n"
    stats_str += f"\nOverall Recall: {recall:.4f}\nOverall Precision: {precision:.4f}\nOverall F1 Score: {f1_score:.4f}\n"


    # Write the statistics to a text file
    stats_file_path = os.path.join(save_path, 'summary_stats.txt')
    with open(stats_file_path, 'w') as stats_file:
        stats_file.write(stats_str)
    print(f"\nSummary statistics saved to: {stats_file_path}")

    # Save main results dictionary
    results_to_save = results.copy()
    # Add overall metrics
    results_to_save['overall_recall'] = recall
    results_to_save['overall_precision'] = precision
    results_to_save['overall_f1_score'] = f1_score
    results_to_save['population_stats'] = pop_stats
    results_to_save['overall_stats'] = overall_stats

    # Remove potentially large history lists before saving JSON if needed
    # results_to_save.pop('alpha_history', None)
    # results_to_save.pop('objective_history', None)

    results_file_path = os.path.join(save_path, 'results.json')
    try:
        # Use the enhanced converter for saving
        serializable_results = convert_numpy_to_python(results_to_save)
        with open(results_file_path, 'w') as f:
            json.dump(serializable_results, f, indent=4, allow_nan=False) # Disallow NaN for strict JSON
        print(f"Full results dictionary saved to: {results_file_path}")
    except Exception as e:
        print(f"Error saving results JSON: {e}")
        # Try saving without potentially problematic fields or converting NaN/Inf
        results_to_save.pop('alpha_history', None)
        results_to_save.pop('objective_history', None)
        results_to_save.pop('gradient_history', None)
        try:
            # Use converter again on simplified dict
            serializable_results = convert_numpy_to_python(results_to_save)
            with open(results_file_path, 'w') as f:
                json.dump(serializable_results, f, indent=4, allow_nan=False)
            print(f"Simplified results dictionary saved to: {results_file_path}")
        except Exception as e2:
            print(f"Error saving simplified results JSON: {e2}")

    # --- Generate Objective Landscape Plot (if requested) ---
    if args.plot_landscape:
        # Need pop_data again, re-generate or pass from run_experiment?
        # For simplicity, let's re-fetch it here. This is inefficient but avoids
        # returning large data structures from run_experiment.
        # Alternatively, modify run_experiment to return pop_data if needed.
        print("\nRe-fetching population data for plotting...")
        plot_pop_data = get_pop_data(
            pop_configs=pop_configs, m1=args.m1, m=args.m,
            dataset_size=args.dataset_size, noise_scale=args.noise_scale,
            corr_strength=args.corr_strength,
            common_meaningful_indices=np.arange(max(1, args.m1 // 2)),
            estimator_type=args.estimator_type, device="cuda" if torch.cuda.is_available() else "cpu",
            base_model_type=args.base_model_type, batch_size=args.batch_size, seed=args.seed
        )

        plot_objective_landscape(
            final_alpha=final_alpha, # Use the best alpha found
            pop_data=plot_pop_data,
            penalty_type=args.penalty_type,
            penalty_lambda=args.penalty_lambda,
            base_model_type=args.base_model_type,
            save_path=save_path,
            dims_to_plot=args.plot_dims, # Pass user-specified dims or None
            use_if_for_plot=(not args.plot_mc) # Use IF unless --plot-mc is specified
        )


if __name__ == '__main__':
    main()