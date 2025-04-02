import argparse
import os
import pandas as pd
import numpy as np
import torch
from grad_desc_diagnostic_v2 import run_experiment_with_diagnostics

def tune_lambda(dataset_size, m1, m, reg_type, dataset_type, num_epochs,
                learning_rate, batch_size, optimizer_type, noise_scale, seed, save_path):
    # Define candidate lambda values (log scale for diversity)
    candidate_lambdas = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
    results_dict = {}
    best_obj = float('inf')
    best_lambda = None
    best_results = None

    print(f"Starting hyperparameter tuning for reg_type = {reg_type}")
    for lam in candidate_lambdas:
        print(f"Evaluating reg_lambda = {lam}")
        results = run_experiment_with_diagnostics(
            dataset_size=dataset_size,
            m1=m1,
            m=m,
            dataset_type=dataset_type,
            num_epochs=num_epochs,
            reg_type=reg_type,
            reg_lambda=lam,
            learning_rate=learning_rate,
            batch_size=batch_size,
            optimizer_type=optimizer_type,
            noise_scale=noise_scale,
            seed=seed,
            verbose=False,
            discrete=True,
            save_path=save_path  
        )
        results_dict[lam] = results
        print(f"  -> Final objective: {results['final_objective']:.4f}")
        print(f"  -> Final alpha: {results['final_alpha']}")
        if results['final_objective'] < best_obj:
            best_obj = results['final_objective']
            best_lambda = lam
            best_results = results

    print("Hyperparameter tuning completed.")
    print(f"Best reg_lambda: {best_lambda} with objective {best_obj:.4f}")
    return best_lambda, best_results, results_dict

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning for reg_lambda with various regularizer types"
    )
    parser.add_argument("--reg_type", type=str, required=True,
                        help="Regularizer type (e.g., 'Neg_L1', 'Max_Dev', 'Reciprocal_L1', 'Quadratic_Barrier', 'Exponential')")
    parser.add_argument("--dataset_size", type=int, default=1000, help="Dataset size")
    parser.add_argument("--m1", type=int, default=2, help="Value of m1")
    parser.add_argument("--m", type=int, default=6, help="Value of m")
    parser.add_argument("--seed", type=int, default=10, help="Random seed")
    parser.add_argument("--save_path", type=str, default="./results/hyperparam_tuning/",
                        help="Path where results will be saved")
    parser.add_argument("--dataset_type", type=str, default="quadratic_regression", help="Type of dataset")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--optimizer_type", type=str, default="adam", help="Optimizer type ('adam' or 'sgd')")
    parser.add_argument("--noise_scale", type=float, default=0.01, help="Noise scale")

    args = parser.parse_args()

    # Ensure save_path exists
    os.makedirs(args.save_path, exist_ok=True)

    best_lambda, best_results, all_results = tune_lambda(
        dataset_size=args.dataset_size,
        m1=args.m1,
        m=args.m,
        reg_type=args.reg_type,
        dataset_type=args.dataset_type,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        optimizer_type=args.optimizer_type,
        noise_scale=args.noise_scale,
        seed=args.seed,
        save_path=args.save_path
    )

    # Save the tuning results in a CSV file.
    tuning_data = []
    for lam, res in all_results.items():
        final_alpha = res['final_alpha']
        if isinstance(final_alpha, torch.Tensor):
            final_alpha = final_alpha.cpu().numpy().tolist()
        tuning_data.append({
            'reg_lambda': lam,
            'final_objective': res['final_objective'],
            'final_alpha': final_alpha  # Store multiple alpha values
        })
    
    df = pd.DataFrame(tuning_data)
    csv_file = os.path.join(args.save_path, f"tuning_results_{args.reg_type}.csv")
    df.to_csv(csv_file, index=False)
    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    main()