import numpy as np
import os
import pandas as pd
import pyreadr
import sys
sys.path.append(os.path.abspath('../src/'))
from functools import partial
import pickle
import multiprocessing as mp
from scipy import stats
from tqdm import tqdm
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")



#################################################################################################################################
############                                        UTILITY FUNCTIONS                                                ############
#################################################################################################################################



def compute_expected_n(ac_locs, trap_locs, g0_vec, sigma_vec, K, density, distances, trap_x):
    """
    Expected number of unique individuals detected using Efford/Boulanger approx
    with g0 and sigma allowed to vary by mask cell/AC location
        ac_locs: x and y coordinates of potential activity center locations based on mask
        trap_locs: x and y coordinates of potential trap locations
        g0_vec: varying g0 values for each potnetial AC location
        sigma_vec: varying sigma values for each potential AC location
        K: number of sampling time periods
        density: varying density at each pixel in the study area
        distances: stored distances between each potential trap location and each potential AC location
        trap_x: binary 0/1 value if trap is placed at that location or not (length = number of potential trap locations)
    """
    # convert per-AC sigma to per-(trap,AC) alpha1 via broadcasting
    sigma_sq = sigma_vec[np.newaxis, :] ** 2              # (1, n_AC)
    alpha1 = 1.0 / (2.0 * sigma_sq)                       # (1, n_AC)
    prob_cap = g0_vec[np.newaxis, :] * np.exp(-alpha1 * (distances ** 2))

    # zero out inactive traps
    inactive = (trap_x == 0)
    prob_cap[inactive, :] = 0.0

    # probability of never being captured across all traps and K occasions
    i_cap_hist = np.squeeze(np.zeros((len(trap_locs), 1)))
    p_empty_cap_hist = compute_cond_lik_ind(
        ac_locs.shape[0], prob_cap, K, len(trap_locs), i_cap_hist
    )
    p_nonempty = 1.0 - p_empty_cap_hist
    expected_n = np.sum(p_nonempty * density)

    return expected_n



def compute_cond_lik_ind(num_activity_centers, est_prob_cap, K, num_traps, ind_cap_hist):
    broadcast_i_cap_hist = np.broadcast_to(ind_cap_hist[:, np.newaxis],
                                           (num_traps, num_activity_centers))
    probs = stats.binom.pmf(broadcast_i_cap_hist, K, est_prob_cap)
    zero_mask = probs == 0.0
    log_probs = np.log(probs, where=~zero_mask)
    log_probs[zero_mask] = -sys.maxsize - 1
    log_cond_lik_sums = np.sum(log_probs, axis=0)
    return np.exp(log_cond_lik_sums)



def compute_expected_n_across_scenarios(ac_locs, trap_locs, g0_list, sigma_list,
                                        K, density_list, distances, trap_x):
    """
    Expected n across scenarios, allowing g0/sigma to vary by AC and scenario.
        g0_list, sigma_list, density_list: lists of length n_scenarios,
        each element is a vector over ACs.
    """
    nscenarios = len(g0_list)
    e_n = np.zeros((nscenarios, 1))
    for s in range(nscenarios):
        e_n[s, 0] = compute_expected_n(ac_locs, trap_locs,
                                       g0_list[s], sigma_list[s],
                                       K, density_list[s], distances, trap_x)
    return e_n



def compute_expected_c(ac_locs, trap_locs, g0_vec, sigma_vec, K, density, distances, trap_x):
    """
    Expected total captures with g0/sigma varying by AC.
    """
    sigma_sq = sigma_vec[np.newaxis, :] ** 2
    alpha1 = 1.0 / (2.0 * sigma_sq)
    prob_cap = g0_vec[np.newaxis, :] * np.exp(-alpha1 * (distances ** 2))

    inactive = (trap_x == 0)
    prob_cap[inactive, :] = 0.0

    density_array = np.array(density).flatten()
    broadcast_density = np.broadcast_to(density_array, (len(trap_locs), len(density_array)))
    expected_c = np.sum(prob_cap * broadcast_density) * K
    return expected_c



def compute_expected_c_across_scenarios(ac_locs, trap_locs, g0_list, sigma_list,
                                        K, density_list, distances, trap_x):
    nscenarios = len(g0_list)
    e_c = np.zeros((nscenarios, 1))
    for s in range(nscenarios):
        e_c[s, 0] = compute_expected_c(ac_locs, trap_locs,
                                       g0_list[s], sigma_list[s],
                                       K, density_list[s], distances, trap_x)
    return e_c



#################################################################################################################################
############                                         GREEDY FUNCTIONS                                                ############
#################################################################################################################################



def forward_greedy(scenarios, trap_loc, centers, K, distances,
                   draw, draw_to_trueN, max_traps,
                   mask_det_dir):

    print("Starting Forward Greedy Algorithm for RSE Minimization")
    trap_x = np.zeros((len(trap_loc),))
    g0_list = []
    sigma_list = []
    density_prior = []   # density per AC per scenario

    prec = 6  # rounding precision for coordinate alignment

    # 1) Build per-scenario density and detection parameters on the mask
    for s, scen in enumerate(scenarios):
        draw_id = scen[3]
        print(f"Building scenario {s}, draw {draw_id}")

        # density on mask
        density_prior_file = (
            f'full_grid_1km/10-3 data (Marten)/Constant_detection/D_mod/'
            f'Dmod_draw_{draw_id}.csv'
        )
        density_df = pd.read_csv(density_prior_file)
        density_df['D_mod'] = density_df['D_mod'] * 25
        density_df['x_round'] = density_df['x'].round(prec)
        density_df['y_round'] = density_df['y'].round(prec)

        # per-mask detection covariates
        det_cov_file = os.path.join(mask_det_dir,
                                    f'Mask_det_covs_draw_{draw_id}.csv')
        det_df = pd.read_csv(det_cov_file)
        det_df['x_round'] = det_df['x'].round(prec)
        det_df['y_round'] = det_df['y'].round(prec)

        # merge to align with mask (centers_df must be global)
        merged = centers_df.merge(density_df[['x_round', 'y_round', 'D_mod']],
                                  on=['x_round', 'y_round'], how='left') \
                           .merge(det_df[['x_round', 'y_round', 'g0', 'sigma']],
                                  on=['x_round', 'y_round'], how='left')

        if merged['D_mod'].isna().any() or merged['g0'].isna().any() or merged['sigma'].isna().any():
            nD = merged['D_mod'].isna().sum()
            ng0 = merged['g0'].isna().sum()
            nsig = merged['sigma'].isna().sum()
            raise ValueError(f"NaNs after rounded merge for draw {draw_id}: "
                             f"D_mod NaNs={nD}, g0 NaNs={ng0}, sigma NaNs={nsig}")

        d_vec = merged['D_mod'].values
        g0_vec = merged['g0'].values
        sigma_vec = merged['sigma'].values

        density_prior.append(d_vec)
        g0_list.append(g0_vec)
        sigma_list.append(sigma_vec)

        print("  sum D:", np.sum(d_vec),
              "g0 range:", g0_vec.min(), g0_vec.max(),
              "sigma range:", sigma_vec.min(), sigma_vec.max())

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

        func1 = partial(compute_expected_n_across_scenarios,
                        centers, trap_loc, g0_list, sigma_list,
                        K, density_prior, distances)
        func2 = partial(compute_expected_c_across_scenarios,
                        centers, trap_loc, g0_list, sigma_list,
                        K, density_prior, distances)

        pool = mp.Pool(min(mp.cpu_count(), 10))
        E_n_per_scenario = np.array(pool.map(func1, trap_x_temp))
        E_c_per_scenario = np.array(pool.map(func2, trap_x_temp))
        pool.close()
        pool.join()

        # shapes: (n_candidates, n_scenarios, 1) -> (n_candidates, n_scenarios)
        E_n_per_scenario = E_n_per_scenario[:, :, 0]
        E_c_per_scenario = E_c_per_scenario[:, :, 0]

        E_r_per_scenario = E_c_per_scenario.copy()
        min_n_r_per_scenario = np.zeros_like(E_r_per_scenario)
        RSE_per_scenario = np.zeros_like(E_r_per_scenario)

        for t in range(len(trap_indices)):
            for s in range(len(scenarios)):
                E_r_per_scenario[t, s] -= E_n_per_scenario[t, s]
                val = min(E_n_per_scenario[t, s], E_r_per_scenario[t, s])
                if val <= 0 or np.isnan(val):
                    min_n_r_per_scenario[t, s] = np.nan
                    RSE_per_scenario[t, s] = np.nan
                else:
                    min_n_r_per_scenario[t, s] = val
                    RSE_per_scenario[t, s] = 1.0 / np.sqrt(val)

        # ignore NaNs when averaging across scenarios
        RSE_temp = np.nanmean(RSE_per_scenario, axis=1).tolist()

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
############                                        READ IN PARAMETERS & SETUP                                      ############
#################################################################################################################################


true_n = pd.read_csv('./full_grid_1km/10-3 data (Marten)/varying_detection_params/2-26 Marten/True_N_per_draw.csv')
draw_to_trueN = dict(zip(true_n['Parameter_draw'], true_n['N']))

params_full = pd.read_csv('./full_grid_1km/10-3 data (Marten)/varying_detection_params/2-26 Marten/param_values_for_each_draw300_marten_g0sigma_covs.csv')
params_full = params_full.rename(columns={'Unnamed: 0': 'index'})

# Store SA groups as a dict so you can choose any order or subset by key
sa_groups_dict = {
    # "SA1":  [64, 74, 76, 78, 109, 125, 164, 204, 238, 250],
    "SA2":  [31, 43, 68, 79, 91, 98, 212, 222, 289, 298],
    "SA3":  [112, 114, 115, 125, 158, 176, 183, 225, 283, 287],
    # "SA4":  [16, 25, 38, 60, 80, 145, 225, 234, 238, 275],
    # "SA5":  [38, 57, 68, 85, 94, 105, 130, 193, 197, 282],
    # "SA6":  [7, 10, 34, 47, 78, 91, 165, 193, 204, 282],
    # "SA7":  [57, 67, 69, 74, 112, 125, 145, 197, 230, 231],
    # "SA8":  [6, 26, 31, 77, 94, 120, 164, 197, 219, 222],
    # "SA9":  [8, 18, 20, 69, 115, 166, 174, 180, 194, 238],
    # "SA10": [11, 83, 158, 197, 212, 216, 222, 239, 257, 287],
    # "SA11": [6, 64, 69, 98, 112, 153, 166, 195, 204, 296],
    # "SA12": [10, 57, 58, 60, 130, 148, 176, 187, 193, 231],
    # "SA13": [18, 57, 61, 78, 105, 130, 183, 227, 233, 287],
    # "SA14": [10, 23, 69, 91, 94, 102, 145, 197, 219, 222],
    # "SA15": [17, 58, 79, 91, 145, 227, 231, 238, 257, 289],
    # "SA16": [18, 34, 69, 114, 174, 176, 181, 195, 197, 282],
    # "SA17": [17, 57, 149, 193, 222, 230, 231, 233, 287, 298],
    # "SA18": [47, 77, 93, 94, 148, 187, 197, 227, 234, 296],
    # "SA19": [7, 77, 80, 98, 109, 204, 225, 238, 257, 287]
    # "SA20": [34, 68, 80, 93, 105, 155, 197, 219, 282, 298]  # already run separately
}

K = 5

# mask (potential AC locations)
ac_coords = pyreadr.read_r('./full_grid_1km/10-3 data (Marten)/500m_mask_marten.RDS')
ac_coords = ac_coords[None]
ac_coords_list = ac_coords[['x', 'y']].values
centers = ac_coords_list
centers_df = pd.DataFrame(centers, columns=['x', 'y'])
prec = 6
centers_df['x_round'] = centers_df['x'].round(prec)
centers_df['y_round'] = centers_df['y'].round(prec)

# traps
trap_coords = pd.read_csv('./full_grid_1km/10-3 data (Marten)/1000m_trap_grid_marten.csv')
trap_coords = trap_coords.drop(columns=['Unnamed: 0'])
trap_coords_list = trap_coords[['x', 'y']].values

# distances trap x mask
traps_3d = trap_coords_list[:, np.newaxis, :]
centers_3d = centers[np.newaxis, :, :]
differences = traps_3d - centers_3d
distances = np.linalg.norm(differences, axis=2)

mask_det_dir = './full_grid_1km/10-3 data (Marten)/varying_detection_params/2-26 Marten/mask_det_covs_mod'



################################################################################################################################
###########                                         RUN GREEDY FOR SELECTED SA GROUPS                              ############
################################################################################################################################

# Choose which SAs to run and in what order:
# e.g., to run all: sa_keys_to_run = list(sa_groups_dict.keys())
# or a subset / custom order: ["SA3", "SA1", "SA9"]
sa_keys_to_run = list(sa_groups_dict.keys())

for sa_key in sa_keys_to_run:
    draw_ids = sa_groups_dict[sa_key]

    print("\n" + "#" * 80)
    print(f"Running {sa_key} with draws {draw_ids}")
    print("#" * 80)

    # filter parameter draws for this SA group
    params = params_full[params_full['index'].isin(draw_ids)].copy()
    params = params.sort_values('index')  # optional, for deterministic order

    D = params['D'].values
    g0_scalar = params['g0'].values   # kept for reference but not used in heuristic now
    sigma_scalar = params['sigma'].values
    draw = params['index'].values.tolist()
    print(f"Parameter draws evaluated over ({sa_key}): {draw}")

    scenarios = list(zip(*[D, g0_scalar, sigma_scalar, draw]))

    start_date = datetime.now()
    start_time = time.time()

    selected_traps, RSE_hist, trap_x = forward_greedy(
        scenarios, trap_coords_list, centers, K, distances,
        draw, draw_to_trueN, 80, mask_det_dir
    )

    end_date = datetime.now()
    end_time = time.time()

    # Output paths for this SA group
    base_dir = f'./secr/Forward Greedy/SA_groups/{sa_key}'
    os.makedirs(base_dir, exist_ok=True)

    # export the total runtime to a text file - Can skip for webapp.
    with open(os.path.join(base_dir, 'runtime.txt'), 'w') as f:
        # write the start time and date
        f.write(f"Start time: {start_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {end_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
        # write the total runtime in seconds
        f.write(f"Start seconds: {start_time} seconds\n")
        f.write(f"End seconds: {end_time} seconds\n")
        f.write(f"Total runtime: {end_time - start_time} seconds\n")
    print(f"[{sa_key}] Total runtime: {end_time - start_time} seconds")

    # Save the results from greedy algorithms
    np.save(os.path.join(base_dir, 'all_selected_traps.npy'), selected_traps)        ### THIS IS THE IMPORTANT FILE OUTPUT TO VISUALIE THE SELECTED TRAPS.

    # All other files are probably not needing to be exported on webapp? Consider commenting out below. This was for my own tracking.
    with open(os.path.join(base_dir, 'rse_hist.txt'), 'w') as f:
        for rse in RSE_hist:
            f.write(f"{rse}\n")
    with open(os.path.join(base_dir, 'considered_trap_locs.pkl'), 'wb') as f:
        pickle.dump(trap_x, f)
    print(f"[{sa_key}] Selected traps:", selected_traps)

    ###############################################################################################################################
    ###########                                      GENERATE FILES FOR SECR (per SA)                                  ############
    ################################################################################################################################
    trap_coords_list_df = pd.DataFrame(trap_coords_list, columns=['x', 'y'])
    trap_coords_list_df.to_csv(os.path.join(base_dir, 'considered_trap_locs.csv'), index=False)

    # All trap coordinates (full grid)
    trap_coords_full = pd.read_csv('./full_grid_1km/10-3 data (Marten)/1000m_trap_grid_marten.csv')
    trap_coords_full = trap_coords_full.rename(columns={'X': 'x', 'Y': 'y'})
    trap_coords_full = trap_coords_full.drop(columns=['Unnamed: 0'])
    all_trap_ids = set(trap_coords_full['Trap_index'])

    for n_cams in [10, 20, 30, 40, 50, 60, 70, 80]:
        traps_subset = selected_traps[:n_cams]

        trap_coords_list_sub_df = trap_coords_full[trap_coords_full['Trap_index'].isin(traps_subset)].copy()
        trap_coords_list_sub_df['Trap_index'] = trap_coords_list_sub_df['Trap_index']  # Keep IDs as-is

        trap_csv_path = os.path.join(base_dir, f'selected_traps_{n_cams}.csv')
        trap_coords_list_sub_df.to_csv(trap_csv_path, index=False)  # Index can be False if IDs are in data

        all_trap_ids = set(trap_coords_full['Trap_index'])
        selected_trap_ids = set(trap_coords_list_sub_df['Trap_index'])
        excluded_trap_ids = sorted(all_trap_ids - selected_trap_ids)

        excluded_txt_path = os.path.join(base_dir, f'{sa_key}-excluded_traps-{n_cams}.txt')
        with open(excluded_txt_path, 'w') as f:
            for item in excluded_trap_ids:
                f.write(f"{item}\n")

        print(f"[{sa_key}] [{n_cams} cams] Selected {len(traps_subset)} traps, "
              f"Excluded {len(excluded_trap_ids)} traps, "
              f"Total traps in full grid: {len(all_trap_ids)}")
        print(f"   Verification: {len(selected_trap_ids) + len(excluded_trap_ids)} = {len(all_trap_ids)}")
