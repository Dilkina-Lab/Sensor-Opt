import numpy as np
import os
import pandas as pd
import pyreadr
import pickle
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

#################################################################################################################################
############                                        UTILITY FUNCTIONS                                                ############
#################################################################################################################################

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


#################################################################################################################################
############                                         GREEDY FUNCTIONS                                                ############
#################################################################################################################################
def backward_greedy(scenarios, trap_loc, centers, K, distances, draw, draw_to_trueN):
    print("Starting Backward Greedy Algorithm for RSE Minimization")
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

        density_prior_file =  f'full_grid_1km/9-5 data/D_mod/Dmod_draw_{scenarios[s][3]}.csv'
        density_df = pd.read_csv(density_prior_file)
        density_df['D_mod'] = density_df['D_mod'] * 25
        density_prior.append(density_df['D_mod'].values.tolist())
        E_n_curr.append(compute_expected_n(centers, trap_loc, g0[s], sigma[s], K, density_prior[s], distances, trap_x))
        E_c_curr.append(compute_expected_c(centers, trap_loc, g0[s], sigma[s], K, density_prior[s], distances, trap_x))
        E_r_curr.append(E_c_curr[s] - E_n_curr[s])
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
    return activated_trap_hist, RSE_hist, trap_x



def forward_greedy(scenarios, trap_loc, centers, K, distances, draw, draw_to_trueN, max_traps):
    print("Starting Forward Greedy Algorithm for RSE Minimization")
    trap_x = np.zeros((len(trap_loc),))  # start with zero traps
    D = []
    g0 = []
    sigma = []
    density_prior = []

    # Precompute density prior information for all scenarios
    for s in range(len(scenarios)):
        D.append(scenarios[s][0])
        g0.append(scenarios[s][1])
        sigma.append(scenarios[s][2])

        density_prior_file = f'full_grid_1km/9-5 data/D_mod/Dmod_draw_{scenarios[s][3]}.csv'
        density_df = pd.read_csv(density_prior_file)
        density_df['D_mod'] = density_df['D_mod'] * 25
        density_prior.append(density_df['D_mod'].values.tolist())

    RSE_hist = []
    add_hist = []
    activated_trap_hist = []
    rse_tracker = {}
    n_tracker = {}

    counter = 0
    pbar = tqdm(total=max_traps, desc="Forward Greedy Progress")
    while sum(trap_x) < max_traps:
        pbar.update(1)
        counter += 1

        trap_indices = [i for i, x in enumerate(trap_x) if int(x) == 0]  # inactive traps
        activated_trap_hist.append([i for i, x in enumerate(trap_x) if int(x) == 1])

        trap_x_temp = np.copy(np.broadcast_to(trap_x, (len(trap_indices), trap_x.shape[0])))
        for pos in range(len(trap_indices)):
            trap_idx = trap_indices[pos]
            trap_x_temp[pos, trap_idx] = 1  # try adding this trap

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
                E_r_per_scenario[t, s] -= E_n_per_scenario[t, s]
                min_n_r_per_scenario[t, s] = min(E_n_per_scenario[t, s], E_r_per_scenario[t, s])
                RSE_per_scenario[t, s] = 1 / np.sqrt(min_n_r_per_scenario[t, s])

        RSE_temp = np.mean(RSE_per_scenario, axis=1).tolist()
        variances = np.var(RSE_per_scenario, axis=1)
        average_variance = variances.mean()
        print(f"Iteration {counter} RSE candidates: {RSE_temp}")
        print(f"Iteration {counter} average variance: {average_variance}")

        rse_tracker[counter] = {
            cam_id: RSE_temp[i]
            for i, cam_id in enumerate(trap_indices)
        }

        # pick the trap whose *addition* gives minimum mean RSE
        min_change_idx = RSE_temp.index(min(RSE_temp))
        add = trap_indices[min_change_idx]
        trap_x[add] = 1

        # Track N estimates
        n_tracker[counter] = {}
        for i, param_draw in enumerate(draw):
            true_N = draw_to_trueN[param_draw]
            est_N = E_n_per_scenario[min_change_idx, i]
            diff_N = est_N - true_N
            n_tracker[counter][param_draw] = [true_N, est_N, diff_N]

        RSE_hist.append(RSE_temp[min_change_idx])
        add_hist.append(add)
        print(f"Added trap {add}, RSE={RSE_temp[min_change_idx]}")
        print(f"Selected traps so far: {add_hist}")

    selected_traps = [i for i, x in enumerate(trap_x) if int(x) == 1]
    print(f"Final selected traps: {selected_traps}")
    return add_hist, RSE_hist, trap_x


#################################################################################################################################
############                                        READ IN PARAMETERS                                               ############
#################################################################################################################################
# Read in True N file
true_n = pd.read_csv('./full_grid_1km/9-5 data/True_N_per_draw.csv')
draw_to_trueN = dict(zip(true_n['Parameter_draw'], true_n['N']))

# Read in parameter draws
params = pd.read_csv('./full_grid_1km/9-5 data/param_values_for_each_draw300_9-5-25.csv')
params = params.rename(columns={'Unnamed: 0': 'index'})
params_ids = list(range(1, 151))
params = params[params['index'].isin(params_ids)]

# Extract parameter values
D, g0, sigma = params['D'].values, params['g0'].values, params['sigma'].values
draw = params['index'].tolist()
draw_to_trueN = dict(zip(true_n['Parameter_draw'], true_n['N']))
K= 5                                        # Number of sampling periods
print(f"Parameter draws evaluated over: {draw}")

# Read in potential activity center locations
ac_coords = pyreadr.read_r('./full_grid_1km/9-5 data/500m_mask_8-14-25.RDS')
ac_coords = ac_coords[None]
ac_coords_list = ac_coords[['x', 'y']].values.tolist()
ac_coords_list = np.array(ac_coords_list)

# Read in potential trap locations
trap_coords = pd.read_csv('./full_grid_1km/9-5 data/sim_9-5-25_1kmgrid_300samps_1000m_trap_grid.csv')
trap_coords = trap_coords.drop(columns = ['Unnamed: 0'])
exclude_trap_coords = trap_coords[trap_coords['Deployed_trap_2024_binary']==0]
trap_coords = trap_coords[~trap_coords['Trap_index'].isin(exclude_trap_coords['Trap_index'])]
trap_coords_list = []
print(f"Total potential trap locations: {trap_coords.shape[0]}")
for i in range(trap_coords.shape[0]):
    trap_coords_list.append((trap_coords['x'].iloc[i], trap_coords['y'].iloc[i]))
trap_coords_list = np.array(trap_coords_list)
trap_coords_list_df = pd.DataFrame(trap_coords_list, columns=['x', 'y'])
trap_coords_list_df.to_csv('./secr/Greedy Removal/FGR2/considered_trap_locs.csv', index=False)


# Calculate euclidean distances from traps to activity centers
traps = trap_coords_list[:, np.newaxis, :]  # Add a new axis to traps to make it 3D
centers = ac_coords_list[np.newaxis, :, :]  # Add a new axis to centers to make it 3D
differences = traps - centers
distances = np.linalg.norm(differences, axis=2)

scenarios = list(zip(*[D, g0, sigma, draw]))

# log the starting times before running greedy algorithms
start_date = datetime.now()
start_time = time.time()


#################################################################################################################################
############                                         RUN GREEDY FUNCTIONS                                            ############
#################################################################################################################################

# Backward Greedy
# selected_traps, RSE_hist, trap_x = backward_greedy(scenarios, trap_coords_list, ac_coords_list, K, distances, draw, draw_to_trueN)

# Forward Greedy
selected_traps, RSE_hist, trap_x = forward_greedy(scenarios, trap_coords_list, ac_coords_list, K, distances, draw, draw_to_trueN, 68)


#################################################################################################################################
############                                           PROCESS RESULTS                                               ############
#################################################################################################################################
# Log the end times after running greedy algorithms
end_date = datetime.now()
end_time = time.time()

# export the total runtime to a text file
with open('secr/Greedy Removal/FGR2/runtime.txt', 'w') as f:
    # write the start time and date
    f.write(f"Start time: {start_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"End time: {end_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    # write the total runtime in seconds
    f.write(f"Start seconds: {start_time} seconds\n")
    f.write(f"End seconds: {end_time} seconds\n")
    f.write(f"Total runtime: {end_time - start_time} seconds\n")
print(f"Total runtime: {end_time - start_time} seconds")

# Save the results from greedy algorithms
np.save('./secr/Greedy Removal/FGR2/all_selected_traps.npy', selected_traps)
# Save the expected number of detections history as a txt file
# with open('./secr/Greedy Removal/FGR2/en_hist.txt', 'w') as f:
#     for en in en_hist:
#         f.write(f"{en}\n")
# same RSE history as txt file
with open('./secr/Greedy Removal/FGR2/rse_hist.txt', 'w') as f:
    for rse in RSE_hist:
        f.write(f"{rse}\n")
# Save the trap_x configuration
with open('./secr/Greedy Removal/FGR2/considered_trap_locs.pkl', 'wb') as f:
    pickle.dump(trap_x, f)
# Print the selected traps
print("Selected traps:", selected_traps)
# Print the expected number of detections history
# print("Expected number of detections history:", en_hist)


#################################################################################################################################
############                                      GENERATE FILES FOR SECR                                            ############
#################################################################################################################################
algorithm_type = 'forward'  # or 'forward'

base_dir = './secr/Greedy Removal/FGR2'
trap_coords_list = pd.read_csv(f'./secr/Greedy Removal/FGR2/considered_trap_locs.csv')

# All trap coordinates (full grid)
trap_coords = pd.read_csv('./full_grid_1km/9-5 data/1000m_trap_grid_8-14-25.csv')
trap_coords = trap_coords.rename(columns={'X': 'x', 'Y': 'y'})
trap_coords = trap_coords.drop(columns=['Unnamed: 0'])
all_trap_ids = set(trap_coords['Trap_index'])

# Make sure output directory exists
os.makedirs(base_dir, exist_ok=True)

if algorithm_type == 'forward':
    # Load selected traps from npy file for forward greedy
    selected_traps = np.load(f'{base_dir}/all_selected_traps.npy')
    selected_traps = np.array(selected_traps)

    for n_cams in [10, 20, 30, 40, 50, 60]:
        # Take the first n_cams traps (as in your pasted code)
        traps_subset = selected_traps[:n_cams]

        trap_coords_list_sub_df = trap_coords_list.iloc[traps_subset].copy()
        trap_coords_list_sub_df['Trap_index'] = trap_coords_list_sub_df.index + 1  # Match SECR spec

        trap_csv_path = f'{base_dir}/selected_traps_{n_cams}.csv'
        trap_coords_list_sub_df.to_csv(trap_csv_path, index=True)

        selected_trap_ids = set(trap_coords_list_sub_df['Trap_index'])
        excluded_trap_ids = sorted(all_trap_ids - selected_trap_ids)

        excluded_txt_path = f'{base_dir}/FGR2-excluded_traps-{n_cams}.txt'
        with open(excluded_txt_path, 'w') as f:
            for item in excluded_trap_ids:
                f.write("%s\n" % item)

        print(f"[{n_cams} cams] Selected {len(traps_subset)} traps, "
              f"Excluded {len(excluded_trap_ids)} traps, "
              f"Total traps in full grid: {len(all_trap_ids)}")
        print(f"   Verification: {len(selected_trap_ids) + len(excluded_trap_ids)} = {len(all_trap_ids)}")

elif algorithm_type == 'backward':
    # Assume backward_greedy() returned activated_trap_hist in addition to selected_traps etc.
    # activated_trap_hist contains list of active trap indices after each removal
    # You need to have this variable assigned from your run, e.g.:
    # selected_traps, RSE_hist, trap_x, activated_trap_hist, remove_hist = backward_greedy(...)

    for n_cams in [10, 20, 30, 40, 50, 60, 70]:
        chosen_set = None
        # Find iteration with exactly n_cams traps active
        for trap_set in selected_traps:
            if len(trap_set) == n_cams:
                chosen_set = trap_set
                break
        if chosen_set is None:
            print(f"No set with exactly {n_cams} traps found, skipping...")
            continue

        traps_subset = chosen_set

        trap_coords_list_sub_df = trap_coords_list.iloc[traps_subset].copy()
        trap_coords_list_sub_df['Trap_index'] = trap_coords_list_sub_df.index + 1  # Match SECR spec

        trap_csv_path = f'{base_dir}/selected_traps_{n_cams}.csv'
        trap_coords_list_sub_df.to_csv(trap_csv_path, index=True)

        selected_trap_ids = set(trap_coords_list_sub_df['Trap_index'])
        excluded_trap_ids = sorted(all_trap_ids - selected_trap_ids)

        excluded_txt_path = f'{base_dir}/FGR2-excluded_traps-{n_cams}.txt'
        with open(excluded_txt_path, 'w') as f:
            for item in excluded_trap_ids:
                f.write("%s\n" % item)

        print(f"[{n_cams} cams] Selected {len(traps_subset)} traps, "
              f"Excluded {len(excluded_trap_ids)} traps, "
              f"Total traps in full grid: {len(all_trap_ids)}")