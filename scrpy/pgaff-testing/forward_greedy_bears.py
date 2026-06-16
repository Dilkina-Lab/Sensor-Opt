import numpy as np
import os
import pandas as pd
import pyreadr
import sys
sys.path.append(os.path.abspath('../src/'))
import argparse
from functools import partial
import pickle
import math
import multiprocessing as mp
from scipy import stats
import sys
from tqdm import tqdm
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("distributed").setLevel(logging.ERROR)


#################################################################################################################################
############                                        UTILITY FUNCTIONS                                                ############
#################################################################################################################################

def compute_expected_n(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    """
    Computes expected number of unique individuals detected in a spatial capture-recapture study
    BASED ON THE FAST HEURISTIC from Efford/Boulanger.
        ac_locs (2D array): Shape (num_activity_centers, 2) - x and y coordinates of activity centers.
        trap_locs (2D array): Shape (num_traps, 2) - x and y coordinates of all potential trap locations.
        g0 (float): Detection probability at the activity center.
        sigma (float): Scale parameter of the detection function.
        K (int): Number of sampling periods.
        density (float): Density of animals
        prob_cap (2D array): Shape (num_traps, num_activity_centers) - capture probability at each trap j of individuals with activity center l.
        trap_x (1D array): 1D array of activated trap locations.
    """
    alpha1 = (1/(2*sigma*sigma))
    prob_cap = ((g0)*np.exp(-alpha1*(distances**2)))        # Array of size (# traps, # potential activity centers)
    for t in range(len(trap_locs)):                         # for trap locations that are not selected, set all capture probs to 0
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
        est_prob_cap (2D array): Shape (num_traps, num_activity_centers)
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
    """
    Computes expected number of unique individuals detected across multiple scenarios
    based on the fast heuristic from Efford/Boulanger.
    """
    nscenarios = len(g0)
    e_n = np.zeros((nscenarios,1))
    for s in range(nscenarios):
        e_n[s,0] = compute_expected_n(ac_locs, trap_locs, g0[s], sigma[s], K, density[s], distances, trap_x)
    return(e_n)

def compute_expected_c(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    """
    Computes expected number of captures in a spatial capture-recapture study
    BASED ON THE FAST HEURISTIC from Efford/Boulanger.
        ac_locs (2D array): Shape (num_activity_centers, 2)
        trap_locs (2D array): Shape (num_traps, 2)
        g0 (float): Detection probability at the activity center.
        sigma (float): Scale parameter of the detection function.
        K (int): Number of sampling periods.
        density (float): Density of animals
        trap_x (1D array): 1D array of activated trap locations.
    """
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
    """
    Computes expected number of captures across multiple scenarios based on the fast heuristic from Efford/Boulanger.
    """
    nscenarios = len(g0)
    e_c = np.zeros((nscenarios,1))
    for s in range(nscenarios):
        e_c[s,0] = compute_expected_c(ac_locs, trap_locs, g0[s], sigma[s], K, density[s], distances, trap_x)
    return(e_c)

#################################################################################################################################
############                                         GREEDY FUNCTIONS                                                ############
#################################################################################################################################

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

        density_prior_file = f'./Data/Bears/constant detection/full_grid_1km/grizz_2018/D_mod/Dmod_draw_{scenarios[s][3]}.csv'

        density_df = pd.read_csv(density_prior_file)
        density_df['D_mod'] = density_df['D_mod'] * 25      # scale up
        density_prior.append(density_df['D_mod'].values.tolist())

    RSE_hist = []
    add_hist = []
    rse_tracker = {}
    n_tracker = {}

    counter = 0
    pbar = tqdm(total=max_traps, desc="Forward Greedy Progress")
    while sum(trap_x) < max_traps:
        pbar.update(1)
        counter += 1

        trap_indices = [i for i, x in enumerate(trap_x) if int(x) == 0]

        trap_x_temp = np.copy(np.broadcast_to(trap_x, (len(trap_indices), trap_x.shape[0])))
        for pos in range(len(trap_indices)):
            trap_idx = trap_indices[pos]
            trap_x_temp[pos, trap_idx] = 1  # try adding this trap

        func1 = partial(compute_expected_n_across_scenarios, centers, trap_loc, g0, sigma, K, density_prior, distances)
        func2 = partial(compute_expected_c_across_scenarios, centers, trap_loc, g0, sigma, K, density_prior, distances)
        pool = mp.Pool(min(mp.cpu_count(), 10))
        E_n_per_scenario = np.array(pool.map(func1, trap_x_temp)).reshape(len(trap_indices), len(scenarios))
        E_c_per_scenario = np.array(pool.map(func2, trap_x_temp)).reshape(len(trap_indices), len(scenarios))
        pool.close()
        pool.join()

        E_r_per_scenario = E_c_per_scenario.copy()
        min_n_r_per_scenario = np.zeros(E_r_per_scenario.shape)
        RSE_per_scenario = np.zeros(E_r_per_scenario.shape)

        for t in range(len(trap_indices)):
            for s in range(len(scenarios)):
                E_r_per_scenario[t, s] -= E_n_per_scenario[t, s]
                min_n_r_per_scenario[t, s] = min(E_n_per_scenario[t, s], E_r_per_scenario[t, s])
                RSE_per_scenario[t, s] = 1 / np.sqrt(min_n_r_per_scenario[t, s])

        RSE_temp = np.mean(RSE_per_scenario, axis=1).tolist()

        rse_tracker[counter] = {
            cam_id: RSE_temp[i]
            for i, cam_id in enumerate(trap_indices)
        }

        min_change_idx = RSE_temp.index(min(RSE_temp))
        add = trap_indices[min_change_idx]
        trap_x[add] = 1

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
true_n = pd.read_csv('./Data/Bears/constant detection/full_grid_1km/grizz_2018/True_N_per_draw.csv')
draw_to_trueN = dict(zip(true_n['Parameter_draw'], true_n['N']))

# Read in parameter draws
params_full = pd.read_csv('./Data/Bears/constant detection/full_grid_1km/grizz_2018/param_values_for_each_draw300_2018.csv')
params_full = params_full.rename(columns={'Unnamed: 0': 'index'})

# SAA groups — all 20 groups will be run in sequence
sa_groups_dict = {
    "SA1":  [64, 74, 76, 78, 109, 125, 164, 204, 238, 250],
    "SA2":  [31, 43, 68, 79, 91, 98, 212, 222, 289, 298],
    "SA3":  [112, 114, 115, 125, 158, 176, 183, 225, 283, 287],
    "SA4":  [16, 25, 38, 60, 80, 145, 225, 234, 238, 275],
    "SA5":  [38, 57, 68, 85, 94, 105, 130, 193, 197, 282],
    "SA6":  [7, 10, 34, 47, 78, 91, 165, 193, 204, 282],
    "SA7":  [57, 67, 69, 74, 112, 125, 145, 197, 230, 231],
    "SA8":  [6, 26, 31, 77, 94, 120, 164, 197, 219, 222],
    "SA9":  [8, 18, 20, 69, 115, 166, 174, 180, 194, 238],
    "SA10": [11, 83, 158, 197, 212, 216, 222, 239, 257, 287],
    "SA11": [6, 64, 69, 98, 112, 153, 166, 195, 204, 296],
    "SA12": [10, 57, 58, 60, 130, 148, 176, 187, 193, 231],
    "SA13": [18, 57, 61, 78, 105, 130, 183, 227, 233, 287],
    "SA14": [10, 23, 69, 91, 94, 102, 145, 197, 219, 222],
    "SA15": [17, 58, 79, 91, 145, 227, 231, 238, 257, 289],
    "SA16": [18, 34, 69, 114, 174, 176, 181, 195, 197, 282],
    "SA17": [17, 57, 149, 193, 222, 230, 231, 233, 287, 298],
    "SA18": [47, 77, 93, 94, 148, 187, 197, 227, 234, 296],
    "SA19": [7, 77, 80, 98, 109, 204, 225, 238, 257, 287],
    "SA20": [34, 68, 80, 93, 105, 155, 197, 219, 282, 298],
}

K = 5
max_traps = 60          # stop after placing 60 cameras
n_cams_outputs = [20, 30, 40, 50, 60]  # output CSVs at these checkpoints

# Read in potentialAC locations
ac_coords = pyreadr.read_r('./Data/Bears/constant detection/full_grid_1km/grizz_2018/500m_mask_grizzly_2018.RDS')
ac_coords = ac_coords[None]
ac_coords_list = ac_coords[['x', 'y']].values.tolist()
ac_coords_list = np.array(ac_coords_list)

# Read in trap coordinates
trap_coords = pd.read_csv('./Data/Bears/constant detection/full_grid_1km/grizz_2018/2000m_trapping_grid.csv')
# trap_coords = trap_coords.drop(columns=['Unnamed: 0'])

# trap_coords = pd.read_csv('./Data/Bears/constant detection/full_grid_1km/grizz_2018/Traps_2018_SCM.csv')
# trap_coords['Trap_index'] = trap_coords['Trap_index'].astype(int)

all_trap_ids = set(trap_coords['Trap_index'])

trap_coords_list = []
for i in range(trap_coords.shape[0]):
    trap_coords_list.append((trap_coords['x'].iloc[i], trap_coords['y'].iloc[i]))
trap_coords_list = np.array(trap_coords_list)
print(f"{trap_coords_list.shape} candidate trap locations")

# Precompute distances between all traps and all activity centers
traps_3d = trap_coords_list[:, np.newaxis, :]
centers_3d = ac_coords_list[np.newaxis, :, :]
differences = traps_3d - centers_3d
distances = np.linalg.norm(differences, axis=2)

################################################################################################################################
###########                                    RUN GREEDY FOR ALL SAA GROUPS                                        ############
################################################################################################################################

sa_keys_to_run = list(sa_groups_dict.keys())

for sa_key in sa_keys_to_run:
    draw_ids = sa_groups_dict[sa_key]

    print("\n" + "#" * 80)
    print(f"Running {sa_key} with draws {draw_ids}")
    print("#" * 80)

    # Filter parameter draws for this SA group
    params = params_full[params_full['index'].isin(draw_ids)].copy()
    params = params.sort_values('index')

    D = params['D'].values
    g0 = params['g0'].values
    sigma = params['sigma'].values
    draw = params['index'].values.tolist()
    print(f"Parameter draws evaluated over ({sa_key}): {draw}")

    scenarios = list(zip(*[D, g0, sigma, draw]))

    start_date = datetime.now()
    start_time = time.time()

    selected_traps, RSE_hist, trap_x = forward_greedy(
        scenarios, trap_coords_list, ac_coords_list, K, distances,
        draw, draw_to_trueN, max_traps
    )

    end_date = datetime.now()
    end_time = time.time()

    # Output directory for this SA group
    base_dir = f'./secr/Forward Greedy/Bears2018_2km/{sa_key}'
    os.makedirs(base_dir, exist_ok=True)

    # Runtime log
    with open(os.path.join(base_dir, 'runtime.txt'), 'w') as f:
        f.write(f"Start time: {start_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {end_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Start seconds: {start_time} seconds\n")
        f.write(f"End seconds: {end_time} seconds\n")
        f.write(f"Total runtime: {end_time - start_time} seconds\n")
    print(f"[{sa_key}] Total runtime: {end_time - start_time} seconds")

    # Save selected traps array (main output)
    np.save(os.path.join(base_dir, 'all_selected_traps.npy'), selected_traps)

    # Save RSE history
    with open(os.path.join(base_dir, 'rse_hist.txt'), 'w') as f:
        for rse in RSE_hist:
            f.write(f"{rse}\n")

    # Save considered trap locations pickle
    with open(os.path.join(base_dir, 'considered_trap_locs.pkl'), 'wb') as f:
        pickle.dump(trap_x, f)

    print(f"[{sa_key}] Selected traps:", selected_traps)

    ###############################################################################################################################
    ###########                             GENERATE OUTPUT FILES AT n_cams CHECKPOINTS                            ############
    ###############################################################################################################################

    trap_coords_list_df = pd.DataFrame(trap_coords_list, columns=['x', 'y'])
    trap_coords_list_df.to_csv(os.path.join(base_dir, 'considered_trap_locs.csv'), index=False)

    # Reload full trap coords to get Trap_index for exclusion files
    # Full grid
    trap_coords_full = pd.read_csv('./Data/Bears/constant detection/full_grid_1km/grizz_2018/2000m_trapping_grid.csv')
    # trap_coords_full = trap_coords_full.rename(columns={'Unnamed: 0': 'row_num'})
    all_trap_ids = set(trap_coords_full['Trap_index'])
    # Prior deployment grid
    # trap_coords_full = pd.read_csv('./Data/Bears/constant detection/full_grid_1km/grizz_2018/Traps_2018_SCM.csv')
    # trap_coords_full['Trap_index'] = trap_coords_full['Trap_index'].astype(int)
    # all_trap_ids = set(trap_coords_full['Trap_index'])

    for n_cams in n_cams_outputs:
        traps_subset = selected_traps[:n_cams]
        selected_arr = np.array(traps_subset)

        trap_coords_list_sub_df = trap_coords_full.iloc[selected_arr].copy()

        trap_csv_path = os.path.join(base_dir, f'selected_traps_{n_cams}.csv')
        trap_coords_list_sub_df.to_csv(trap_csv_path, index=False)

        selected_trap_ids = set(trap_coords_list_sub_df['Trap_index'].astype(int))
        excluded_trap_ids = sorted(all_trap_ids - selected_trap_ids)

        excluded_txt_path = os.path.join(base_dir, f'{sa_key}-excluded_traps-{n_cams}.txt')
        with open(excluded_txt_path, 'w') as f:
            for item in excluded_trap_ids:
                f.write(f"{item}\n")

        print(f"[{sa_key}] [{n_cams} cams] Selected {len(traps_subset)} traps, "
            f"Excluded {len(excluded_trap_ids)} traps, "
            f"Total traps in full grid: {len(all_trap_ids)}")
        print(f"   Verification: {len(selected_trap_ids) + len(excluded_trap_ids)} = {len(all_trap_ids)}")