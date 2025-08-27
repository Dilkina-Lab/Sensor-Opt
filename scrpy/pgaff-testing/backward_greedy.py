import numpy as np
import os
import pandas as pd
import pyreadr
import sys
sys.path.append(os.path.abspath('../src/'))
import argparse
from functools import partial
import math
import multiprocessing as mp
from scipy import stats
import sys
from tqdm import tqdm
from datetime import datetime
import time
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("distributed").setLevel(logging.ERROR)

def compute_expected_n(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    """
    Computes expected number of unique individuals detected in a spatial capture-recapture study.
        ac_locs (2D array): Shape (num_activity_centers, 2) - x and y coordinates of activity centers.
        trap_locs (2D array): Shape (num_traps, 2) - x and y coordinates of all potential trap locations.
        g0 (float): Detection probability at the activity center.
        sigma (float): Scale parameter of the detection function.
        K (int): Number of sampling periods.
        density (float): Density of individuals per unit area.
        prob_cap (2D array): Shape (num_traps, num_activity_centers) - capture probability at each trap j of individuals with activity center l.
        trap_x (1D array): 1D array of activated trap locations.
    """
    alpha1 = (1/(2*sigma*sigma))
    prob_cap = ((g0)*np.exp(-alpha1*(distances**2)))      # Array of size (# traps, # poential activity centers)
    for t in range(len(trap_locs)): # for trap locations that are not selected, set all capture probs to 0
        if int(trap_x[t]) == 0:
            prob_cap[t,...] = np.zeros(ac_locs.shape[0])
    i_cap_hist = np.squeeze(np.zeros((len(trap_locs), 1)))  # Initialize empty capture history 
    p_empty_cap_hist = compute_cond_lik_ind(len(ac_locs), prob_cap, K, len(trap_locs), i_cap_hist)
    p_nonempty = 1 - p_empty_cap_hist
    expected_n = np.sum(p_nonempty*density)
    return expected_n

def compute_cond_lik_ind(num_activity_centers, est_prob_cap, K, num_traps, ind_cap_hist):
    """
    Computes the likelihood of an individual's capture history conditional on their activity center location. 
        est_prob_cap (2D array): Shape (num_traps, num_activity_centers) - capture probability at each trap j of individuals with activity center l.   
        num_activity_centers (int): Number of potential activity centers. 
        K (int): Number of sampling periods.
        num_traps (int): Number of potential trap locations.
        ind_cap_hist (1D array): Capture history of an individual.
    """
    broadcast_i_cap_hist = np.broadcast_to(ind_cap_hist[:, np.newaxis], (num_traps, num_activity_centers))
    probs = stats.binom.pmf(broadcast_i_cap_hist, K, est_prob_cap)
    zero_mask = probs == 0.0
    log_probs = np.log(probs, where=np.invert(zero_mask))
    log_probs[zero_mask] = -sys.maxsize - 1
    log_cond_lik_sums = np.sum(log_probs, axis=0)
    return np.exp(log_cond_lik_sums)

def compute_expected_n_across_scenarios(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    nscenarios = len(g0)
    e_n = np.zeros((nscenarios,1))
    for s in range(nscenarios):
        e_n[s,0] = compute_expected_n(ac_locs, trap_locs, g0[s], sigma[s], K, density[s], distances, trap_x)
    return(e_n)

def compute_expected_c(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    alpha1 = (1/(2*sigma*sigma))
    prob_cap = ((g0)*np.exp(-alpha1*(distances**2)))
    density_array = np.array(density).flatten()
    for t in range(len(trap_locs)):
        if int(trap_x[t]) == 0:
            prob_cap[t,...] = np.zeros(ac_locs.shape[0])
    broadcast_density = np.broadcast_to(density_array, (len(trap_locs), len(density_array)))
    expected_c = np.sum(prob_cap*broadcast_density)*K
    return expected_c

def compute_expected_c_across_scenarios(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    nscenarios = len(g0)
    e_c = np.zeros((nscenarios,1))
    for s in range(nscenarios):
        e_c[s,0] = compute_expected_c(ac_locs, trap_locs, g0[s], sigma[s], K, density[s], distances, trap_x)
    return(e_c)

# Read in True N file and construct dictionary of draw ID to true N
true_n = pd.read_csv('./full_grid_1km/True_N_per_draw.csv')
draw_to_trueN = dict(zip(true_n['Parameter_draw'], true_n['N']))

def backward_greedy(scenarios, trap_loc, centers, K, distances, draw, draw_to_trueN):
    trap_x = np.ones((len(trap_loc),))
    D = []
    g0 = []
    sigma = []
    density_prior = []

    E_n_curr = []
    E_c_curr = []
    E_r_curr = []
    RSE_curr = []
    RSE_hist = []

    for s in range(len(scenarios)):
        D.append(scenarios[s][0])
        g0.append(scenarios[s][1])
        sigma.append(scenarios[s][2])

        density_prior_file =  f'full_grid_1km/D_mod/Dmod_draw_{scenarios[s][3]}.csv'
        density_df = pd.read_csv(density_prior_file)
        density_df['D_mod'] = density_df['D_mod'] * 25
        density_prior.append(density_df['D_mod'].values.tolist())
        E_n_curr.append(compute_expected_n(centers, trap_loc, g0[s], sigma[s], K, density_prior[s], distances, trap_x))
        E_c_curr.append(compute_expected_c(centers, trap_loc, g0[s], sigma[s], K, density_prior[s], distances, trap_x))
        E_r_curr.append(E_c_curr[s] - E_n_curr[s])
        print(f"Scenario {s}: E[n]={E_n_curr[s]}, E[r]={E_r_curr[s]}")
        RSE_curr.append(1/np.sqrt(min([E_n_curr[s], E_r_curr[s]])))

    RSE_hist.append(np.mean(RSE_curr))    
    remove_hist = []
    activated_trap_hist = []
    rse_tracker = {}
    n_tracker = {}

    counter = 0
    max_iters = int(sum(trap_x) - 1)
    pbar = tqdm(total=max_iters, desc="Backward Greedy Progress")
    while sum(trap_x) > 2:
        pbar.update(1)
        counter += 1
        trap_indices = [i for i, x in enumerate(trap_x) if int(x) == 1]
        activated_trap_hist.append(trap_indices)
        trap_x_temp = np.copy(np.broadcast_to(trap_x, (len(trap_indices),trap_x.shape[0])))
        for pos in range(len(trap_indices)):
            trap_idx = trap_indices[pos]
            trap_x_temp[pos, trap_idx] = 0

        func1 = partial(compute_expected_n_across_scenarios, centers, trap_loc, g0, sigma, K, density_prior, distances)
        func2 = partial(compute_expected_c_across_scenarios, centers, trap_loc, g0, sigma, K, density_prior, distances)
        pool = mp.Pool(min(mp.cpu_count(), 10))
        E_n_per_scenario = np.squeeze(np.array(pool.map(func1, trap_x_temp)))
        E_c_per_scenario = np.squeeze(np.array(pool.map(func2, trap_x_temp)))
        pool.close()
        pool.join()

        E_r_per_scenario = E_c_per_scenario
        min_n_r_per_scenario = np.zeros(E_r_per_scenario.shape)
        RSE_per_scenario = np.zeros(E_r_per_scenario.shape)
        for t in range(len(trap_indices)):
            for s in range(len(scenarios)):
                E_r_per_scenario[t,s] -= E_n_per_scenario[t,s]
                # print if E_n or E_r is the minimum
                if E_n_per_scenario[t,s] > E_r_per_scenario[t,s]:
                    print(f"ALERT")
                min_n_r_per_scenario[t,s] = min(E_n_per_scenario[t,s], E_r_per_scenario[t,s])
                RSE_per_scenario[t,s] = 1/np.sqrt(min_n_r_per_scenario[t,s])
        RSE_temp = np.mean(RSE_per_scenario, axis=1).tolist()

        variances = np.var(RSE_per_scenario, axis=1)
        average_variance = variances.mean()
        print(f"Iteration {counter} average variance: {average_variance}")
        rse_tracker[counter] = {
            cam_id: RSE_temp[i] 
            for i, cam_id in enumerate(trap_indices)
        }

        min_change_idx = RSE_temp.index(min(RSE_temp)) 
        remove = trap_indices[min_change_idx]
        trap_x[remove] = 0

        # This block tracks true N, estimated N, and error for each scenario
        n_tracker[counter] = {}
        for i, param_draw in enumerate(draw):
            true_N = draw_to_trueN[param_draw]
            est_N = E_n_per_scenario[min_change_idx, i]
            diff_N = est_N - true_N
            n_tracker[counter][param_draw] = [true_N, est_N, diff_N]

        RSE_hist.append(RSE_temp[min_change_idx])
        remove_hist.append(remove)
        print(remove, RSE_temp[min_change_idx])
    
    remaining_traps = [i for i, x in enumerate(trap_x) if int(x) == 1]
    print(f"Final remaining traps: {remaining_traps}")

    # with open('secr/Backward Greedy/BG12/activated_trap_hist.txt', 'w') as f:
    #     for item in activated_trap_hist:
    #         f.write("%s\n" % item)
    # with open('secr/Backward Greedy/BG12/remove_hist.txt', 'w') as f:
    #     for item in remove_hist:
    #         f.write("%s\n" % item)
    # with open('secr/Backward Greedy/BG12/rse_hist.txt', 'w') as f:
    #     for item in RSE_hist:
    #         f.write("%s\n" % item)
    # with open('secr/Backward Greedy/BG12/rse_tracker.txt', 'w') as f:
    #     for key, value in rse_tracker.items():
    #         f.write(f"Iteration {key}: {value}\n")
    # with open('secr/Backward Greedy/BG12/n_tracker.txt', 'w') as f:
    #     for it, draw_dict in n_tracker.items():
    #         f.write(f"Iteration {it}:\n")
    #         for draw_id, vals in draw_dict.items():
    #             f.write(f"  Draw {draw_id}: TrueN={vals[0]}, EstN={vals[1]}, Diff={vals[2]}\n")
    # pbar.close()
    # return(remove_hist, RSE_hist, activated_trap_hist)

# Read in parameter draws
params = pd.read_csv('./full_grid_1km/param_values_for_each_draw150.csv')
params = params.rename(columns={'Unnamed: 0': 'index'})
# true_n_filtered = true_n[true_n['N'].between(30, 80)]['Parameter_draw'].values.tolist()
# params = params[params['index'].isin(true_n_filtered)]
params = params.iloc[:2, :]    # Only keep the first 50 draws for testing

D = params['D'].values
g0 = params['g0'].values
sigma = params['sigma'].values
K = 5
draw = params['index'].values.tolist()
print(f"Parameter draws evaluated over: {draw}")

ac_coords = pyreadr.read_r('./full_grid_1km/500m_mask.RDS')
ac_coords = ac_coords[None]
ac_coords_list = ac_coords[['x', 'y']].values.tolist()
ac_coords_list = np.array(ac_coords_list)

trap_coords = pd.read_csv('./full_grid_1km/1000m_trap_grid.csv')
trap_coords = trap_coords.drop(columns = ['Unnamed: 0'])
# exclude_trap_coords = pd.read_csv('./500m_data/traps_to_remove_50m.csv')
exclude_trap_coords = pd.read_csv('./full_grid_1km/traps_to_remove_1km.csv')
trap_coords = trap_coords[~trap_coords['Trap_index'].isin(exclude_trap_coords['Trap_index'])]
trap_coords_list = []
for i in range(trap_coords.shape[0]):
    trap_coords_list.append((trap_coords['x'].iloc[i], trap_coords['y'].iloc[i]))
trap_coords_list = np.array(trap_coords_list)
# limit to first 200
trap_coords_list = trap_coords_list[:200]
print(f"{trap_coords_list.shape} candidate trap locations")

# np.save('./secr/Backward Greedy/BG12/considered_trap_locs.npy', trap_coords_list)
trap_coords_list_df = pd.DataFrame(trap_coords_list, columns=['x', 'y'])
# trap_coords_list_df.to_csv('./secr/Backward Greedy/BG12/considered_trap_locs.csv', index=False)

traps = trap_coords_list[:, np.newaxis, :]  # Add a new axis to traps to make it 3D
centers = ac_coords_list[np.newaxis, :, :]  # Add a new axis to centers to make it 3D
differences = traps - centers
distances = np.linalg.norm(differences, axis=2)

scenarios = list(zip(*[D, g0, sigma, draw]))

start_date = datetime.now()
start_time = time.time()

backward_greedy(scenarios, trap_coords_list, ac_coords_list, K, distances, draw, draw_to_trueN)

# end_time = time.time()
# end_date = datetime.now()

# with open('secr/Backward Greedy/BG12/runtime.txt', 'w') as f:
#     f.write(f"Start time: {start_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
#     f.write(f"End time: {end_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
#     f.write(f"Start seconds: {start_time} seconds\n")
#     f.write(f"End seconds: {end_time} seconds\n")
#     f.write(f"Total runtime: {end_time - start_time} seconds\n")
# print(f"Total runtime: {end_time - start_time} seconds")

# # Generate Files for SECR Analysis
# trap_coords_list = np.load('secr/Backward Greedy/BG12/considered_trap_locs.npy')     # read in the traps considered in the algorithm run

# with open('secr/Backward Greedy/BG12/activated_trap_hist.txt', 'r') as f:
#     lines = f.readlines()

# # Find the first line with exactly 60 elements
# selected_line = None
# for line in lines:
#     # Clean and validate the line
#     cleaned = line.strip().replace('[', '').replace(']', '').replace(' ', '')
#     if cleaned:  # Skip empty lines
#         elements = cleaned.split(',')
#         if len(elements) == 55:
#             selected_line = cleaned
#             break  # Remove this line if you want the LAST occurrence instead

# if not selected_line:
#     raise ValueError("No line with 55 elements found in the file")

# # convert string to list of integers
# selected_traps = [int(x) for x in selected_line.split(',')]

# # # Convert selected_traps to a numpy array and sort it
# selected_traps = np.array(selected_traps)
# selected_traps = np.sort(selected_traps)

# # subset trap_coords_list to include ONLY the selected trap indices
# trap_coords_list_sub = trap_coords_list[selected_traps]
# trap_coords_list_sub_df = pd.DataFrame(trap_coords_list_sub, columns=['x', 'y'])

# # get all the potential trap coordinates
# trap_coords = pd.read_csv('500m_data/500m_trap_grid.csv')
# trap_coords = trap_coords.rename(columns={'X': 'x', 'Y': 'y'})
# trap_coords = trap_coords.drop(columns=['Unnamed: 0'])

# # get the x,y coords and index of the traps that were selected
# trap_coords_list_sub_df = trap_coords_list_sub_df.merge(trap_coords, on=['x', 'y'], how='left')
# trap_coords_list_sub_df.to_csv('secr/Backward Greedy/BG12/selected_traps.csv', index=False)
# trap_coords_list_sub_df

# # CORRECTED LOGIC: Find excluded trap IDs using set operations on Trap_index values
# # Get the Trap_index values for selected traps
# selected_trap_ids = set(trap_coords_list_sub_df['Trap_index'])

# # Get all Trap_index values from the full grid
# all_trap_ids = set(trap_coords['Trap_index'])

# # Excluded trap IDs are those in the full grid but NOT in selected_trap_ids
# excluded_trap_ids = sorted(all_trap_ids - selected_trap_ids)

# # convert to txt file
# with open('secr/Backward Greedy/BG12/BG12-excluded_traps.txt', 'w') as f:
#     for item in excluded_trap_ids:
#         f.write("%s\n" % item)

# print(f"Selected {len(selected_traps)} traps")
# print(f"Excluded {len(excluded_trap_ids)} traps")
# print(f"Total traps in full grid: {len(all_trap_ids)}")
# print(f"Verification: {len(selected_trap_ids) + len(excluded_trap_ids)} should equal {len(all_trap_ids)}")
