#!/bin/bash

# Loop to run if_vs_plugin for 5 seeds
for seed in {1..5}
do
    echo "Running if_vs_plugin with seed $seed"
    python3 if_vs_plugin.py --seed "$seed"
done