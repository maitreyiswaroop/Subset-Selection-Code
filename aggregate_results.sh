#!/bin/bash
#SBATCH --job-name="agg_results"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/aggs/agg_results_%j.out  # CHANGE: Specify log directory
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/aggs/agg_results_%j.err   # CHANGE: Specify log directory
#SBATCH --partition=debug                  # CHANGE: Specify your cluster partition

# --- Environment Setup ---
echo "Setting up environment..."
# Activate your conda environment (adjust path if needed)
source /home/$USER/miniconda/etc/profile.d/conda.sh # CHANGE: Specify conda path
conda activate venv                       # CHANGE: Specify your environment name

# # quadratic_regression / interaction_regression
# path=/data/user_data/mswaroop/Subset-Selection-Code/results_v8/mc_plugin/baseline_failure_5/baseline_failure_5
# python aggregate_results.py $path --output $path/aggregated_results --run_numbers 17 14 22 23 24

# # quadratic_regression / interaction_regression
# path=/data/user_data/mswaroop/Subset-Selection-Code/results_v8/mc_plugin/baseline_failure_2/baseline_failure_2
# python aggregate_results.py $path --output $path/aggregated_results --run_numbers 0 1 2 3 4

# # path=/data/user_data/mswaroop/Subset-Selection-Code/results_v10/
# # python aggregate_results.py $path --output $path/aggregated_results --run_numbers 0
path=/data/user_data/mswaroop/Subset-Selection-Code/results_v10/final_may_14_ACS/no_lasso_tune/
python aggregate_results.py $path --output $path/aggregated_results --file_paths /data/user_data/mswaroop/Subset-Selection-Code/results_v10/final_may_14_ACS/no_lasso_tune/123/smooth_minmax/run_0/results_comparison_budget_7.csv /data/user_data/mswaroop/Subset-Selection-Code/results_v10/final_may_14_ACS/no_lasso_tune/456/smooth_minmax/run_0/results_comparison_budget_7.csv /data/user_data/mswaroop/Subset-Selection-Code/results_v10/final_may_14_ACS/no_lasso_tune/789/smooth_minmax/run_0/results_comparison_budget_7.csv /data/user_data/mswaroop/Subset-Selection-Code/results_v10/final_may_14_ACS/no_lasso_tune/101112/smooth_minmax/run_0/results_comparison_budget_7.csv /data/user_data/mswaroop/Subset-Selection-Code/results_v10/final_may_14_ACS/no_lasso_tune/131415/smooth_minmax/run_0/results_comparison_budget_7.csv

# path=/data/user_data/mswaroop/Subset-Selection-Code/results_v10/mc_plugin/final_may_14/
# python aggregate_results.py $path --output $path/aggregated_results --file_paths /data/user_data/mswaroop/Subset-Selection-Code/results_v10/mc_plugin/final_may_14/2/run_0/results_comparison_budget_5.csv /data/user_data/mswaroop/Subset-Selection-Code/results_v10/mc_plugin/final_may_14/26/run_0/results_comparison_budget_5.csv /data/user_data/mswaroop/Subset-Selection-Code/results_v10/mc_plugin/final_may_14/29/run_0/results_comparison_budget_5.csv 

path=/data/user_data/mswaroop/Subset-Selection-Code/results_v10/mc_plugin/final_may_14/
python aggregate_results.py $path --output $path/aggregated_results_10 --file_paths /data/user_data/mswaroop/Subset-Selection-Code/results_v10/mc_plugin/final_may_14/2/run_0/results_comparison_budget_10.csv /data/user_data/mswaroop/Subset-Selection-Code/results_v10/mc_plugin/final_may_14/26/run_0/results_comparison_budget_10.csv /data/user_data/mswaroop/Subset-Selection-Code/results_v10/mc_plugin/final_may_14/29/run_0/results_comparison_budget_10.csv 


# path=/data/user_data/mswaroop/Subset-Selection-Code/results_v8/mc_plugin/uci_age/tuning
# python aggregate_results.py $path --output $path/aggregated_results --run_numbers 17 --suffix _budget_5
# for i in 3
# do
#     path=/data/user_data/mswaroop/Subset-Selection-Code/results_v8/mc_plugin/baseline_failure_$i/baseline_failure_$i
#     python aggregate_results.py $path 0-4 --output $path/aggregated_results
#     # path=/data/user_data/mswaroop/Subset-Selection-Code/results_v8/kernel_if_like/baseline_failure_$i/baseline_failure_$i
#     # python aggregate_results.py $path 0-4 --output $path/aggregated_results
# done

# regressions=(piecewise_regression interaction_regression quadratic_regression)

# for i in "${regressions[@]}"
# do
#     path=/data/user_data/mswaroop/Subset-Selection-Code/results_v8/kernel_if_like/$i
#     python aggregate_results.py $path 0-4 --output $path/aggregated_results

#     path=/data/user_data/mswaroop/Subset-Selection-Code/results_v8/mc_plugin/$i
#     python aggregate_results.py $path 0-4 --output $path/aggregated_results
# done