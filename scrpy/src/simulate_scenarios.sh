#!/bin/bash

for v in {0..80}
do
    python3 simulate_scr.py --config_file ../data/simulation/config/lowfrag_train_$v.csv --trap_layout ../data/simulation/trap_locations/lowfrag_covariate_grid_14_14.csv --K 10
done
