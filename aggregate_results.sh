#!/bin/bash
#SBATCH --job-name="agg_results"
#SBATCH --output=/data/user_data/mswaroop/Subset-Selection-Code/logs/aggs/agg_results_%j.out  # CHANGE: Specify log directory
#SBATCH --error=/data/user_data/mswaroop/Subset-Selection-Code/logs/aggs/agg_results_%j.err   # CHANGE: Specify log directory
#SBATCH --partition=debug                  # CHANGE: Specify your cluster partition

# --- Environment Setup ---
# echo "Setting up environment..."
# # Activate your conda environment (adjust path if needed)
# source /home/$USER/miniconda/etc/profile.d/conda.sh # CHANGE: Specify conda path
# conda activate venv                       # CHANGE: Specify your environment name

# path=/Users/mswaroop/Desktop/Projects/Bryan/Subset_selection/Subset-Selection-Code/results_v8/mc_plugin/baseline_failure_5/baseline_failure_5/
# python3 aggregate_results.py $path --output $path/aggregated_results --run_numbers 8 10 12 13 16 --suffix _budget_5

path=/Users/mswaroop/Desktop/Projects/Bryan/Subset_selection/Subset-Selection-Code/results_v8/mc_plugin/baseline_failure_5/may15/baseline_failure_5/
python3 aggregate_results.py $path --output $path/aggregated_results_5_mlp --run_numbers 0 1 2 3 4 --suffix _budget_5_mlp

path=/Users/mswaroop/Desktop/Projects/Bryan/Subset_selection/Subset-Selection-Code/results_v8/mc_plugin/baseline_failure_5/may15/baseline_failure_5/
python3 aggregate_results.py $path --output $path/aggregated_results_2_mlp --run_numbers 0 1 2 3 4 --suffix _budget_2_mlp

path=/Users/mswaroop/Desktop/Projects/Bryan/Subset_selection/Subset-Selection-Code/results_v8/mc_plugin/baseline_failure_5/may15/baseline_failure_5/
python3 aggregate_results.py $path --output $path/aggregated_results_7_mlp --run_numbers 0 1 2 3 4 --suffix _budget_7_mlp

path=/Users/mswaroop/Desktop/Projects/Bryan/Subset_selection/Subset-Selection-Code/results_v8/mc_plugin/baseline_failure_5/may15/baseline_failure_5/
python3 aggregate_results.py $path --output $path/aggregated_results_10_mlp --run_numbers 0 1 2 3 4 --suffix _budget_10_mlp
# path=/data/user_data/mswaroop/Subset-Selection-Code/results_v8/mc_plugin/uci_age/tuning
# python aggregate_results.py $path --output $path/aggregated_results --run_numbers 9 6 5 --suffix _budget_5
# path=/Users/mswaroop/Desktop/Projects/Bryan/Subset_selection/Subset-Selection-Code/results_v8/mc_plugin/baseline_failure_8/baseline_failure_8
# python3 aggregate_results.py $path --output $path/aggregated_results --run_numbers 16 17 18 19 20 --suffix _budget_5
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