#! /bin/bash
#SBATCH --job-name=scr_est
#SBATCH --workdir=/nethome/agupta375/code/scrpy/src
#SBATCH --output=/nethome/agupta375/code/scrpy/src/log.out
#SBATCH --error=/nethome/agupta375/code/scrpy/src/log.err
#SBATCH --array=0-9
#SBATCH -n 1 # num tasks
#SBATCH -N 1 # on one node
#SBATCH -c 4 # cpus per task
#SBATCH --mem-per-cpu=500


python3 run_scr_estimation.py \
--config_file ../data/simulation/config/lowfrag_test_${1}.csv \
--ac_realization_no 0 \
--trap_layout_file ../data/simulation/trap_locations/lowfrag_covariate_grid_14_14.csv \
--K 10 \
--caphist_realization_no $SLURM_ARRAY_TASK_ID \
--traps_to_remove backward_greedy_budget_50_trap_removal_order.pkl \
--num_traps_to_remove 96