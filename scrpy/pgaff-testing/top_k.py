import numpy as np
import os
import pandas as pd
import pyreadr
import sys
sys.path.append(os.path.abspath('../src/'))
from functools import partial
import multiprocessing as mp
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("distributed").setLevel(logging.ERROR)

from datetime import datetime
import time


#################################################################################################################################
############                                        UTILITY FUNCTIONS                                                ############
#################################################################################################################################


def compute_expected_n(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    alpha1 = (1 / (2 * sigma * sigma))
    prob_cap = ((g0) * np.exp(-alpha1 * (distances ** 2)))
    for t in range(len(trap_locs)):
        if int(trap_x[t]) == 0:
            prob_cap[t, ...] = np.zeros(ac_locs.shape[0])
    i_cap_hist = np.squeeze(np.zeros((len(trap_locs), 1)))
    p_empty_cap_hist = compute_cond_lik_ind(len(ac_locs), prob_cap, K, len(trap_locs), i_cap_hist)
    p_nonempty = 1 - p_empty_cap_hist
    expected_n = np.sum(p_nonempty * density)
    return expected_n


def compute_cond_lik_ind(num_activity_centers, est_prob_cap, K, num_traps, ind_cap_hist):
    broadcast_i_cap_hist = np.broadcast_to(ind_cap_hist[:, np.newaxis], (num_traps, num_activity_centers))
    probs = stats.binom.pmf(broadcast_i_cap_hist, K, est_prob_cap)
    zero_mask = probs == 0.0
    log_probs = np.log(probs, where=np.invert(zero_mask))
    log_probs[zero_mask] = -sys.maxsize - 1
    log_cond_lik_sums = np.sum(log_probs, axis=0)
    return np.exp(log_cond_lik_sums)


def compute_expected_n_across_scenarios(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    nscenarios = len(g0)
    e_n = np.zeros((nscenarios, 1))
    for s in range(nscenarios):
        e_n[s, 0] = compute_expected_n(ac_locs, trap_locs, g0[s], sigma[s], K, density[s], distances, trap_x)
    return e_n


def compute_expected_c(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    alpha1 = (1 / (2 * sigma * sigma))
    prob_cap = ((g0) * np.exp(-alpha1 * (distances ** 2)))
    density_array = np.array(density).flatten()
    for t in range(len(trap_locs)):
        if int(trap_x[t]) == 0:
            prob_cap[t, ...] = np.zeros(ac_locs.shape[0])
    broadcast_density = np.broadcast_to(density_array, (len(trap_locs), len(density_array)))
    expected_c = np.sum(prob_cap * broadcast_density) * K
    return expected_c


def compute_expected_c_across_scenarios(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    nscenarios = len(g0)
    e_c = np.zeros((nscenarios, 1))
    for s in range(nscenarios):
        e_c[s, 0] = compute_expected_c(ac_locs, trap_locs, g0[s], sigma[s], K, density[s], distances, trap_x)
    return e_c


def _load_density_prior_for_scenarios(scenarios, density_base='full_grid_1km/10-3 data (Marten)/Constant_detection/D_mod'):
    g0_list = []
    sigma_list = []
    density_prior = []
    for s in range(len(scenarios)):
        g0_list.append(scenarios[s][1])
        sigma_list.append(scenarios[s][2])
        density_prior_file = os.path.join(density_base, f'Dmod_draw_{scenarios[s][3]}.csv')
        density_df = pd.read_csv(density_prior_file)
        density_df['D_mod'] = density_df['D_mod'] * 25
        density_prior.append(density_df['D_mod'].values.tolist())
    return g0_list, sigma_list, density_prior


#################################################################################################################################
############                                     TOP-K SINGLE-SITE BASELINE                                          ###########
#################################################################################################################################


def topk_single_site_baseline(scenarios, trap_loc, centers, K, distances, k=70, out_dir=None,
                              trap_grid_csv='./full_grid_1km/10-3 data (Marten)/1000m_trap_grid_marten.csv',
                              density_base='full_grid_1km/10-3 data (Marten)/Constant_detection/D_mod'):
    print(f"Starting Top-{k} Single-Site Baseline on {len(scenarios)} scenarios")

    g0_list, sigma_list, density_prior = _load_density_prior_for_scenarios(scenarios, density_base=density_base)
    num_traps = len(trap_loc)

    trap_x_matrix = np.zeros((num_traps, num_traps))
    for i in range(num_traps):
        trap_x_matrix[i, i] = 1

    print(f"Evaluating {num_traps} single-site configurations in parallel...")

    func1 = partial(compute_expected_n_across_scenarios, centers, trap_loc,
                    g0_list, sigma_list, K, density_prior, distances)
    func2 = partial(compute_expected_c_across_scenarios, centers, trap_loc,
                    g0_list, sigma_list, K, density_prior, distances)

    pool = mp.Pool(min(mp.cpu_count(), 10))
    E_n_per_scenario = np.squeeze(np.array(pool.map(func1, trap_x_matrix)))
    E_c_per_scenario = np.squeeze(np.array(pool.map(func2, trap_x_matrix)))
    pool.close()
    pool.join()

    nscenarios = len(scenarios)
    E_r_per_scenario = E_c_per_scenario.copy()
    RSE_per_scenario = np.zeros(E_r_per_scenario.shape)

    for t in range(num_traps):
        for s in range(nscenarios):
            E_r_per_scenario[t, s] -= E_n_per_scenario[t, s]
            min_n_r = min(E_n_per_scenario[t, s], E_r_per_scenario[t, s])
            RSE_per_scenario[t, s] = 1 / np.sqrt(min_n_r) if min_n_r > 0 else np.inf

    RSE_single = np.mean(RSE_per_scenario, axis=1)
    ranked_indices = np.argsort(RSE_single)
    topk_indices = ranked_indices[:k].tolist()

    print(f"Top-{k} traps by individual expected RSE: {topk_indices}")

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f'topk_selected_traps_{k}.npy'), topk_indices)
        np.save(os.path.join(out_dir, 'topk_rse_single_scores.npy'), RSE_single)

        with open(os.path.join(out_dir, f'topk_rse_hist_{k}.txt'), 'w') as f:
            for idx in topk_indices:
                f.write(f"{RSE_single[idx]}\n")

        trap_coords = pd.read_csv(trap_grid_csv)
        trap_coords = trap_coords.rename(columns={'X': 'x', 'Y': 'y'})
        if 'Unnamed: 0' in trap_coords.columns:
            trap_coords = trap_coords.drop(columns=['Unnamed: 0'])
        all_trap_ids = set(trap_coords['Trap_index'])

        trap_coords_list_sub_df = trap_coords[trap_coords['Trap_index'].isin(topk_indices)].copy()
        trap_csv_path = os.path.join(out_dir, f'topk_selected_traps_{k}.csv')
        trap_coords_list_sub_df.to_csv(trap_csv_path, index=False)

        excluded_trap_ids = sorted(all_trap_ids - set(topk_indices))
        excluded_txt_path = os.path.join(out_dir, f'topk_excluded_traps_{k}.txt')
        with open(excluded_txt_path, 'w') as f:
            for item in excluded_trap_ids:
                f.write(f"{item}\n")

    return topk_indices, RSE_single


#################################################################################################################################
############                                     FULL-LAYOUT RSE EVALUATION                                         ###########
#################################################################################################################################


def evaluate_layout_rse(selected_traps, scenarios, trap_loc, centers, K, distances,
                        out_dir=None, prefix='layout_eval',
                        density_base='full_grid_1km/10-3 data (Marten)/Constant_detection/D_mod'):
    g0_list, sigma_list, density_prior = _load_density_prior_for_scenarios(scenarios, density_base=density_base)

    trap_x = np.zeros((len(trap_loc),))
    for idx in selected_traps:
        trap_x[idx] = 1

    e_n_per_scenario = np.squeeze(
        compute_expected_n_across_scenarios(
            centers, trap_loc, g0_list, sigma_list, K, density_prior, distances, trap_x
        )
    )
    e_c_per_scenario = np.squeeze(
        compute_expected_c_across_scenarios(
            centers, trap_loc, g0_list, sigma_list, K, density_prior, distances, trap_x
        )
    )

    e_r_per_scenario = e_c_per_scenario - e_n_per_scenario
    rse_per_scenario = np.zeros(len(scenarios))

    for s in range(len(scenarios)):
        min_n_r = min(e_n_per_scenario[s], e_r_per_scenario[s])
        rse_per_scenario[s] = 1 / np.sqrt(min_n_r) if min_n_r > 0 else np.inf

    mean_rse = float(np.mean(rse_per_scenario))

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f'{prefix}_rse_per_scenario.npy'), rse_per_scenario)
        np.save(os.path.join(out_dir, f'{prefix}_expected_n_per_scenario.npy'), e_n_per_scenario)
        np.save(os.path.join(out_dir, f'{prefix}_expected_c_per_scenario.npy'), e_c_per_scenario)
        with open(os.path.join(out_dir, f'{prefix}_mean_rse.txt'), 'w') as f:
            f.write(f"{mean_rse}\n")

    return mean_rse, rse_per_scenario, e_n_per_scenario, e_c_per_scenario


#################################################################################################################################
############                                        READ IN PARAMETERS                                               ############
#################################################################################################################################

BASE_DIR = '/Users/hannahmurray/Documents/GitHub/Sensor-Opt/scrpy/pgaff-testing'
TOPK_BASE_DIR = os.path.join(BASE_DIR, 'secr', 'Top K')

TRUE_N_CSV = os.path.join(BASE_DIR, 'full_grid_1km', '10-3 data (Marten)', 'Constant_detection', 'True_N_per_draw.csv')
PARAMS_CSV = os.path.join(BASE_DIR, 'full_grid_1km', '10-3 data (Marten)', 'Constant_detection', 'param_values_for_each_draw300_marten.csv')
AC_RDS = os.path.join(BASE_DIR, 'full_grid_1km', '10-3 data (Marten)', '500m_mask_marten.RDS')
TRAP_GRID_CSV = os.path.join(BASE_DIR, 'full_grid_1km', '10-3 data (Marten)', '1000m_trap_grid_marten.csv')
DENSITY_BASE = os.path.join(BASE_DIR, 'full_grid_1km', '10-3 data (Marten)', 'Constant_detection', 'D_mod')

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
    "SA20": [34, 68, 80, 93, 105, 155, 197, 219, 282, 298]
}

true_n = pd.read_csv(TRUE_N_CSV)
draw_to_trueN = dict(zip(true_n['Parameter_draw'], true_n['N']))

params = pd.read_csv(PARAMS_CSV)
params = params.rename(columns={'Unnamed: 0': 'index'})

ac_coords = pyreadr.read_r(AC_RDS)
ac_coords = ac_coords[None]
ac_coords_list = ac_coords[['x', 'y']].values.tolist()
ac_coords_list = np.array(ac_coords_list)

trap_coords = pd.read_csv(TRAP_GRID_CSV)
if 'Unnamed: 0' in trap_coords.columns:
    trap_coords = trap_coords.drop(columns=['Unnamed: 0'])
trap_coords_list = []
for i in range(trap_coords.shape[0]):
    trap_coords_list.append((trap_coords['x'].iloc[i], trap_coords['y'].iloc[i]))
trap_coords_list = np.array(trap_coords_list)
print(f"{trap_coords_list.shape} candidate trap locations")

traps = trap_coords_list[:, np.newaxis, :]
centers = ac_coords_list[np.newaxis, :, :]
differences = traps - centers
distances = np.linalg.norm(differences, axis=2)


#################################################################################################################################
############                                        RUN ALL SA GROUPS                                                ############
#################################################################################################################################

os.makedirs(TOPK_BASE_DIR, exist_ok=True)
summary_rows = []

for sa_name, sa_draws in sa_groups_dict.items():
    print('\n' + '=' * 100)
    print(f'Running {sa_name} with draws: {sa_draws}')
    print('=' * 100)

    sa_out_dir = os.path.join(TOPK_BASE_DIR, sa_name)
    os.makedirs(sa_out_dir, exist_ok=True)

    sa_params = params[params['index'].isin(sa_draws)].copy()
    sa_params = sa_params.set_index('index').loc[sa_draws].reset_index()

    D = sa_params['D'].values
    g0 = sa_params['g0'].values
    sigma = sa_params['sigma'].values
    K = 5
    draw = sa_params['index'].values.tolist()

    scenarios = list(zip(*[D, g0, sigma, draw]))

    start_date = datetime.now()
    start_time = time.time()

    topk_traps, RSE_single = topk_single_site_baseline(
        scenarios,
        trap_coords_list,
        ac_coords_list,
        K,
        distances,
        k=70,
        out_dir=sa_out_dir,
        trap_grid_csv=TRAP_GRID_CSV,
        density_base=DENSITY_BASE
    )

    mean_rse, rse_per_scenario, en_per_scenario, ec_per_scenario = evaluate_layout_rse(
        topk_traps,
        scenarios,
        trap_coords_list,
        ac_coords_list,
        K,
        distances,
        out_dir=sa_out_dir,
        prefix=f'{sa_name}_70cam_layout',
        density_base=DENSITY_BASE
    )

    end_date = datetime.now()
    end_time = time.time()

    with open(os.path.join(sa_out_dir, 'topk_runtime.txt'), 'w') as f:
        f.write(f"Start time: {start_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {end_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Start seconds: {start_time}\n")
        f.write(f"End seconds: {end_time}\n")
        f.write(f"Total runtime: {end_time - start_time} seconds\n")

    with open(os.path.join(sa_out_dir, 'sa_draws.txt'), 'w') as f:
        for d in sa_draws:
            f.write(f"{d}\n")

    summary_rows.append({
        'SA': sa_name,
        'n_scenarios': len(scenarios),
        'mean_rse_full_70cam_layout': mean_rse,
        'selected_traps_file': os.path.join(sa_out_dir, 'topk_selected_traps_70.npy')
    })

    print(f'{sa_name}: Mean RSE for full 70-camera top-k layout = {mean_rse}')
    print(f'{sa_name}: Outputs saved to {sa_out_dir}')

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(TOPK_BASE_DIR, 'SA_topk_summary.csv'), index=False)
print('\nFinished all SA groups.')
print(f'Summary saved to {os.path.join(TOPK_BASE_DIR, "SA_topk_summary.csv")}')