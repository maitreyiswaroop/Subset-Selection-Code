# todo: get rid of minibatch implementation of kernel method - do some other memory handling to prevent overflow and use entire dataset at a time
# todo: the find_best_estimator needs to be fixed - it should only compute the IF_based squared condtional as the target, and then compute E_Y_X using plugin or IF_based_method, or both, and use this for E_Y_S, and use that for term2
import argparse
import json
import numpy as np
from data import generate_data_continuous, generate_data_continuous_with_corr
from estimators import (
    IF_estimator_squared_conditional,
    plugin_estimator_squared_conditional,
    plugin_estimator_conditional_mean,
    IF_estimator_conditional_mean,
    estimate_conditional_kernel_oof
)
import matplotlib.pyplot as plt

def plot_comparison(
    results: list,
    save_dir: str = None,
    title: str = "Comparison of Estimators"
):
    """
    Plot the comparison of plugin and IF estimators based on the results.
    
    Parameters:
    -----------
    results : list
        List of dictionaries containing 'estimator', 'clamp_min', 'clamp_max', 'k', and 'error'.
    save_dir : str, optional
        Directory to save the plot. If None, the plot will not be saved.
    title : str, optional
        Title of the plot.
    """

    # Extract data for plotting
    clamp_mins = [res['clamp_min'] for res in results]
    clamp_maxs = [res['clamp_max'] for res in results]
    ks = [res['k'] for res in results]
    errors = [res['error'] for res in results]

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(clamp_mins, clamp_maxs, c=errors, cmap='viridis', s=100)
    
    # Add color bar
    plt.colorbar(scatter, label='Error')
    
    # Set labels and title
    plt.xlabel('Clamp Min')
    plt.ylabel('Clamp Max')
    plt.title(title)
    
    # Save or show the plot
    if save_dir:
        plt.savefig(f"{save_dir}/comparison_plot.png")
        print(f"Plot saved to {save_dir}/comparison_plot.png")
    else:
        plt.show()

def find_best_estimator(
    X: np.ndarray,
    Y: np.ndarray,
    n_folds: int = 5,
    clamp_mins: list = [0.01, 0.1, 1.0, 10.0],
    clamp_maxs: list = [1.0, 10.0, 100.0],
    ks:       list = [50, 100, 500, 1000],
    n_alphas: int  = 10,
    seed:     int  = 42,
    verbose:  bool = False, 
    save_dir: str = None  
) -> dict:
    """
    Grid‐search across (clamp_min, clamp_max, k) for the k‐NN‐kernel estimator
    and compare to the plugin estimator, using the IF‐corrected term2 as reference.
    Returns a dict with the best 'estimator' and its hyperparams.
    """
    np.random.seed(seed)
    # 1) draw test alphas log‐uniformly
    # alphas = np.exp(np.linspace(np.log(0.001), np.log(10.0), n_alphas))
    for alpha_max in [0.1, 1.0, 2.0, 5.0, 10.0, 10.0]:
        alphas = np.random.uniform(0.01, alpha_max, size=(n_alphas, X.shape[1])) + 0.01 * np.random.randn(n_alphas, X.shape[1])
        if verbose:
            print(f"Generated {n_alphas} alphas with max value {alpha_max}: {alphas}")
    # 2) precompute out‐of‐fold plugin conditional mean once
    E_Y_X_plugin = plugin_estimator_conditional_mean(X, Y, n_folds=n_folds)
    E_Y_X_IF = IF_estimator_conditional_mean(X, Y, n_folds=n_folds)

    # accumulator for plugin error
    plugin_err = 0.0
    # accumulator for each kernel config
    kernel_errs_plugin = {
        (cm, cM, k): 0.0
        for cm in clamp_mins
        for cM in clamp_maxs if cM > cm
        for k  in ks
    }

    kernel_errs_IF = {
        (cm, cM, k): 0.0
        for cm in clamp_mins
        for cM in clamp_maxs if cM > cm
        for k  in ks
    }

    # 3) loop over alphas
    for alpha in alphas:
        # simulate noisy S
        noise = np.random.randn(*X.shape) * np.sqrt(alpha)[None,:]
        S = X + noise

        # reference term2 via IF‐estimator
        true2   = IF_estimator_squared_conditional(S, Y,
                    estimator_type="rf", n_folds=n_folds)
        # plugin term2
        plug2   = plugin_estimator_squared_conditional(S, Y,
                    estimator_type="rf", n_folds=n_folds)
        plugin_err += abs(plug2 - true2)

        # evaluate every kernel setting
        for (cm, cM, k), acc in kernel_errs_plugin.items():
            preds_plugin = estimate_conditional_kernel_oof(
                X, S, E_Y_X_plugin, alpha,
                n_folds     = n_folds,
                clamp_min   = cm,
                clamp_max   = cM,
                k           = k
            )
            k2 = np.mean(preds_plugin**2)
            kernel_errs_plugin[(cm, cM, k)] = acc + abs(k2 - true2)

            preds_IF = estimate_conditional_kernel_oof(
                X, S, E_Y_X_IF, alpha,
                n_folds     = n_folds,
                clamp_min   = cm,
                clamp_max   = cM,
                k           = k
            )
            k2_IF = np.mean(preds_IF**2)
            kernel_errs_IF[(cm, cM, k)] += abs(k2_IF - true2)

    if verbose:
        print(f"Plugin estimator error: {plugin_err}")
        print(f"Best kernel plugin hyperparams: {best_kernel_hp_plugin}, error: {best_kernel_err_plugin}")
        print(f"Best kernel IF hyperparams: {best_kernel_hp_IF}, error: {best_kernel_err_IF}")
        if save_dir:
            plot_comparison(
                results=[
                    {'estimator': 'plugin', 'clamp_min': None, 'clamp_max': None, 'k': None, 'error': plugin_err},
                    {'estimator': 'kernel_plugin', 'clamp_min': best_kernel_hp_plugin[0], 
                     'clamp_max': best_kernel_hp_plugin[1], 'k': best_kernel_hp_plugin[2], 
                     'error': best_kernel_err_plugin},
                    {'estimator': 'kernel_IF', 'clamp_min': best_kernel_hp_IF[0], 
                     'clamp_max': best_kernel_hp_IF[1], 'k': best_kernel_hp_IF[2], 
                     'error': best_kernel_err_IF}
                ],
                save_dir=save_dir,
                title="Comparison of Plugin and Kernel Estimators"
            )

    # 4) average errors
    plugin_err /= n_alphas
    for hp in kernel_errs_plugin:
        kernel_errs_plugin[hp] /= n_alphas
        kernel_errs_IF[hp] /= n_alphas

    # 5) pick best
    best_kernel_hp_plugin, best_kernel_err_plugin = min(kernel_errs_plugin.items(), key=lambda x: x[1])
    best_kernel_hp_IF, best_kernel_err_IF = min(kernel_errs_IF.items(), key=lambda x: x[1])
    if best_kernel_err_plugin < best_kernel_err_IF:
        best_kernel_hp = best_kernel_hp_plugin
        return {
            'estimator': 'kernel_plugin',
            'clamp_min': best_kernel_hp[0],
            'clamp_max': best_kernel_hp[1],
            'k':         best_kernel_hp[2],
            'error':     float(best_kernel_err_plugin)
        }
    else:
        best_kernel_hp = best_kernel_hp_IF
        cm, cM, k = best_kernel_hp
        return {
            'estimator': 'kernel_IF',
            'clamp_min': cm,
            'clamp_max': cM,
            'k':         k,
            'error':     float(best_kernel_err_IF)
        }
    

def main():
    p = argparse.ArgumentParser(
        description="Grid‐search tune clamp_min, clamp_max, k for estimate_conditional_kernel_oof"
    )
    p.add_argument('--pop-id',      type=int,   default=0)
    p.add_argument('--m1',          type=int,   default=4)
    p.add_argument('--m',           type=int,   default=10)
    p.add_argument('--dataset-size',type=int,   default=10000)
    p.add_argument('--noise-scale', type=float, default=0.1)
    p.add_argument('--corr-strength',type=float,default=0.0)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--n-folds',     type=int,   default=5)
    p.add_argument('--clamp-mins',  type=float, nargs='+', default=[0.01,0.1,1.0,10.0])
    p.add_argument('--clamp-maxs',  type=float, nargs='+', default=[1.0,10.0,100.0])
    p.add_argument('--ks',          type=int,   nargs='+', default=[50,100,500,1000])
    args = p.parse_args()

    np.random.seed(args.seed)
    # 1) Generate one population
    X, Y, A, _ = ( generate_data_continuous_with_corr if args.corr_strength>0
                   else generate_data_continuous )(
        pop_id=args.pop_id,
        m1=args.m1,
        m=args.m,
        dataset_type='linear_regression',
        dataset_size=args.dataset_size,
        noise_scale=args.noise_scale,
        corr_strength=args.corr_strength,
        seed=args.seed
    )
    # 2) form S = X + noise*sqrt(alpha)
    alpha = np.random.uniform(0.01, 10.0, size=args.m) + 0.01*np.random.randn(args.m)
    noise = np.random.randn(*X.shape) * np.sqrt(alpha)[None,:]
    S = X + noise

    # 3) precompute out‐of‐fold IF‐term2 and E_Y_X
    IF_term2 = IF_estimator_squared_conditional(
        S, Y, estimator_type="rf", n_folds=args.n_folds
    )
    E_Y_X = plugin_estimator_conditional_mean(
        X, Y, estimator_type="rf", n_folds=args.n_folds
    )

    # 4) grid‐search over clamp_min, clamp_max, k
    best = None
    all_results = []
    for cm in args.clamp_mins:
        for cM in args.clamp_maxs:
            if cM <= cm: 
                continue
            for k in args.ks:
                preds = estimate_conditional_kernel_oof(
                    X, S, E_Y_X, alpha,
                    n_folds=args.n_folds,
                    clamp_min=cm,
                    clamp_max=cM,
                    k=k
                )
                term2 = np.mean(preds**2)
                err   = abs(term2 - IF_term2)
                rec = {'clamp_min':cm, 'clamp_max':cM, 'k':k,
                       'term2':term2, 'IF_term2':IF_term2, 'error':err}
                all_results.append(rec)
                if best is None or err < best['error']:
                    best = rec

    # 5) dump best & full grid
    out = {'best': best, 'grid': all_results}
    print("Best params:", best)
    with open('tune_params_results.json','w') as f:
        json.dump(out, f, indent=2)

if __name__ == '__main__':
    main()