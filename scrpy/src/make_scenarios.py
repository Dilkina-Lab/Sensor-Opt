"""
Define parameters describing realizations of a population of individuals in a landscape.

The following parameters are written to a config file:
landscape_raster_file   -   covariate raster for landscape
raster_cell_size        -   multiplier for length of grid cell
N                       -   number of individuals in the landscape
n_ac_realizations       -   number of random realizations of N activity centers
activity_centers_seed   -   random seed for simulating activity center realizations
beta1                   -   coefficient for habitat suitability wrt raster covariate
alpha1                  -   coefficient related to home range size
alpha2                  -   coefficient related to resistance wrt raster covariate
alpha0                  -   baseline detection probability

A specified set of the above parameters can be perturbed/sampled from distributions to generate
multiple scenarios from the combinations of these parameter values.
"""

import csv
import itertools
import numpy as np

config_params = {}
config_params['landscape_raster_file'] = '../data/simulation/covariate_rasters/lowfrag_covariate.tif'
config_params['raster_cell_size'] = 0.1
config_params['n_ac_realizations'] = 1
config_params['activity_centers_seed'] = 5678
# config_params['trap_config'] = 'grid'
# config_params['n_traps_x'] = 14
# config_params['n_traps_y'] = 14
# config_params['K'] = 10
# config_params['capture_histories_seed'] = 5678
config_params['beta1'] = -3

params_to_perturb = ['N', 'alpha0', 'alpha1', 'alpha2']
base_vals = {}
base_vals['N'] = 100
base_vals['alpha0'] = 2
base_vals['alpha1'] = 3.369959
base_vals['alpha2'] = 2.25

perturbation = 0.1 # perturb value within set range of base value
nvals_per_param = 3
expanded_vals = dict.fromkeys(params_to_perturb)
for p in params_to_perturb:
    maxval = base_vals[p]*(1 + perturbation)
    minval = base_vals[p]*(1 - perturbation)
    if p == 'N':
        vals = sorted(np.random.randint(np.ceil(minval), np.floor(maxval), size=nvals_per_param))
    else:
        vals = sorted(np.random.uniform(minval, maxval, size=nvals_per_param))
    expanded_vals[p] = vals
param_combinations = list(itertools.product(*[expanded_vals[p] for p in params_to_perturb]))

counter = 0
for pc in param_combinations:
    config_file_name = '../data/simulation/config/lowfrag_train_%d.csv'%counter
    with open(config_file_name, 'w') as f:
        fwriter = csv.writer(f)
        for p in config_params.keys():
            fwriter.writerow([p, config_params[p]])
        for pi in range(len(params_to_perturb)):
            p = params_to_perturb[pi]
            fwriter.writerow([p, pc[pi]])
    counter += 1
