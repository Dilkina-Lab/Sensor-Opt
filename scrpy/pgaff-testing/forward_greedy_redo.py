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
    alpha1 = (1 / (2 * sigma * sigma))
    prob_cap = (g0 * np.exp(-alpha1 * (distances ** 2)))          # (n_traps, n_AC)
    for t in range(len(trap_locs)):
        if int(trap_x[t]) == 0:
            prob_cap[t, ...] = 0.0
    i_cap_hist = np.squeeze(np.zeros((len(trap_locs), 1)))
    p_empty_cap_hist = compute_cond_lik_ind(ac_locs.shape[0], prob_cap, K, len(trap_locs), i_cap_hist)
    p_nonempty = 1 - p_empty_cap_hist
    expected_n = np.sum(p_nonempty * density)
    return expected_n

def compute_cond_lik_ind(num_activity_centers, est_prob_cap, K, num_traps, ind_cap_hist):
    broadcast_i_cap_hist = np.broadcast_to(ind_cap_hist[:, np.newaxis], (num_traps, num_activity_centers))
    probs = stats.binom.pmf(broadcast_i_cap_hist, K, est_prob_cap)
    zero_mask = probs == 0.0
    log_probs = np.log(probs, where=~zero_mask)
    log_probs[zero_mask] = -sys.maxsize - 1
    log_cond_lik_sums = np.sum(log_probs, axis=0)
    return np.exp(log_cond_lik_sums)

def compute_expected_n_across_scenarios(ac_locs_list, trap_locs, g0, sigma, K, density_list, distances_list, trap_x):
    nscenarios = len(g0)
    e_n = np.zeros((nscenarios, 1))
    for s in range(nscenarios):
        e_n[s, 0] = compute_expected_n(ac_locs_list[s], trap_locs, g0[s], sigma[s], K,
                                       density_list[s], distances_list[s], trap_x)
    return e_n

def compute_expected_c(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    alpha1 = (1 / (2 * sigma * sigma))
    prob_cap = (g0 * np.exp(-alpha1 * (distances ** 2)))
    density_array = np.array(density).flatten()
    for t in range(len(trap_locs)):
        if int(trap_x[t]) == 0:
            prob_cap[t, ...] = 0.0
    broadcast_density = np.broadcast_to(density_array, (len(trap_locs), len(density_array)))
    expected_c = np.sum(prob_cap * broadcast_density) * K
    return expected_c

def compute_expected_c_across_scenarios(ac_locs_list, trap_locs, g0, sigma, K, density_list, distances_list, trap_x):
    nscenarios = len(g0)
    e_c = np.zeros((nscenarios, 1))
    for s in range(nscenarios):
        e_c[s, 0] = compute_expected_c(ac_locs_list[s], trap_locs, g0[s], sigma[s], K,
                                       density_list[s], distances_list[s], trap_x)
    return e_c

#################################################################################################################################
############                                         GREEDY FUNCTIONS                                                ############
#################################################################################################################################

def forward_greedy(scenarios, trap_loc, ac_locs_list, K, distances_list, draw, draw_to_trueN, max_traps):
    print("Starting Forward Greedy Algorithm for RSE Minimization")
    trap_x = np.zeros((len(trap_loc),))
    D = []
    g0 = []
    sigma = []
    density_prior = []

    # Precompute density prior for realized ACs only, per scenario
    for s in range(len(scenarios)):
        D.append(scenarios[s][0])
        g0.append(scenarios[s][1])
        sigma.append(scenarios[s][2])

        draw_id = scenarios[s][3]
        # realized ACs for this draw
        ac_df = pd.read_csv(
            f'/Users/hannahmurray/Documents/GitHub/Sensor-Opt/scrpy/pgaff-testing/'
            f'full_grid_1km/10-3 data (Marten)/Constant_detection/AC_locations/AC_draw_{draw_id}.csv'
        )
        # density at realized AC locations: assume D_mod has density on full mask, so subset by AC coords
        Dmod_df = pd.read_csv(
            f'full_grid_1km/10-3 data (Marten)/Constant_detection/D_mod/Dmod_draw_{draw_id}.csv'
        )
        Dmod_df['D_mod'] = Dmod_df['D_mod'] * 25  # same scaling

        # merge to get D_mod only at realized ACs (inner join on x,y)
        ac_with_D = ac_df.merge(Dmod_df, on=['x', 'y'], how='left')
        density_prior.append(ac_with_D['D_mod'].values.tolist())
        ac_locs_list[s][:] = ac_with_D[['x', 'y']].values  # ensure same order as density

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
            trap_x_temp[pos, trap_idx] = 1

        func1 = partial(compute_expected_n_across_scenarios, ac_locs_list, trap_loc,
                        g0, sigma, K, density_prior, distances_list)
        func2 = partial(compute_expected_c_across_scenarios, ac_locs_list, trap_loc,
                        g0, sigma, K, density_prior, distances_list)
        pool = mp.Pool(min(mp.cpu_count(), 10))
        E_n_per_scenario = np.squeeze(np.array(pool.map(func1, trap_x_temp)))
        E_c_per_scenario = np.squeeze(np.array(pool.map(func2, trap_x_temp)))
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

true_n = pd.read_csv('./full_grid_1km/10-3 data (Marten)/Constant_detection/True_N_per_draw.csv')
draw_to_trueN = dict(zip(true_n['Parameter_draw'], true_n['N']))

params = pd.read_csv('./full_grid_1km/10-3 data (Marten)/Constant_detection/param_values_for_each_draw300_marten.csv')
params = params.rename(columns={'Unnamed: 0': 'index'})
params = params[params['index'].isin([34, 68, 80, 93, 105, 155, 197, 219, 282, 298])]

D = params['D'].values
g0 = params['g0'].values
sigma = params['sigma'].values
K = 5
draw = params['index'].values.tolist()
print(f"Parameter draws evaluated over: {draw}")

# candidate trap locations (unchanged)
trap_coords = pd.read_csv('./full_grid_1km/10-3 data (Marten)/1000m_trap_grid_marten.csv')
trap_coords = trap_coords.drop(columns=['Unnamed: 0'])
trap_coords_list = trap_coords[['x', 'y']].values
trap_coords_list_df = pd.DataFrame(trap_coords_list, columns=['x', 'y'])

# realized ACs & distances per scenario
ac_locs_list = []
distances_list = []
for d in draw:
    ac_df = pd.read_csv(
        f'/Users/hannahmurray/Documents/GitHub/Sensor-Opt/scrpy/pgaff-testing/'
        f'full_grid_1km/10-3 data (Marten)/Constant_detection/AC_locations/AC_draw_{d}.csv'
    )
    ac_array = ac_df[['x', 'y']].values
    ac_locs_list.append(ac_array)

    traps_3d = trap_coords_list[:, np.newaxis, :]
    centers_3d = ac_array[np.newaxis, :, :]
    differences = traps_3d - centers_3d
    distances = np.linalg.norm(differences, axis=2)
    distances_list.append(distances)

scenarios = list(zip(*[D, g0, sigma, draw]))

start_date = datetime.now()
start_time = time.time()

################################################################################################################################
###########                                         RUN GREEDY FUNCTIONS                                            ############
################################################################################################################################

selected_traps, RSE_hist, trap_x = forward_greedy(
    scenarios, trap_coords_list, ac_locs_list, K, distances_list, draw, draw_to_trueN, 80
)

################################################################################################################################
###########                                           PROCESS RESULTS                                               ############
################################################################################################################################

end_date = datetime.now()
end_time = time.time()

base_dir = './secr/Forward Greedy/SA2_redo'
os.makedirs(base_dir, exist_ok=True)

with open(os.path.join(base_dir, 'runtime.txt'), 'w') as f:
    f.write(f"Start time: {start_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"End time: {end_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Start seconds: {start_time} seconds\n")
    f.write(f"End seconds: {end_time} seconds\n")
    f.write(f"Total runtime: {end_time - start_time} seconds\n")
print(f"Total runtime: {end_time - start_time} seconds")

np.save(os.path.join(base_dir, 'all_selected_traps.npy'), selected_traps)

with open(os.path.join(base_dir, 'rse_hist.txt'), 'w') as f:
    for rse in RSE_hist:
        f.write(f"{rse}\n")

with open(os.path.join(base_dir, 'considered_trap_locs.pkl'), 'wb') as f:
    pickle.dump(trap_x, f)
print("Selected traps:", selected_traps)

trap_coords_list_df.to_csv(os.path.join(base_dir, 'considered_trap_locs.csv'), index=False)

# Export selected traps / excluded traps per budget
trap_coords = trap_coords.rename(columns={'X': 'x', 'Y': 'y'}) if 'X' in trap_coords.columns else trap_coords
all_trap_ids = set(trap_coords['Trap_index'])

for n_cams in [10, 20, 30, 40, 50, 60, 70, 80]:
    traps_subset = selected_traps[:n_cams]
    trap_coords_list_sub_df = trap_coords[trap_coords['Trap_index'].isin(traps_subset)].copy()
    trap_csv_path = os.path.join(base_dir, f'selected_traps_{n_cams}.csv')
    trap_coords_list_sub_df.to_csv(trap_csv_path, index=False)

    selected_trap_ids = set(trap_coords_list_sub_df['Trap_index'])
    excluded_trap_ids = sorted(all_trap_ids - selected_trap_ids)

    excluded_txt_path = os.path.join(base_dir, f'SA2_redo-excluded_traps-{n_cams}.txt')
    with open(excluded_txt_path, 'w') as f:
        for item in excluded_trap_ids:
            f.write(f"{item}\n")

    print(f"[{n_cams} cams] Selected {len(traps_subset)} traps, "
          f"Excluded {len(excluded_trap_ids)} traps, "
          f"Total traps in full grid: {len(all_trap_ids)}")
    print(f"   Verification: {len(selected_trap_ids) + len(excluded_trap_ids)} = {len(all_trap_ids)}")
