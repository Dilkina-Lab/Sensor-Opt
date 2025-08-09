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

# ----------------------------------------- UTILITY FUNCTIONS --------------------------------------------- 
def compute_expected_n(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    alpha1 = (1 / (2 * sigma * sigma))
    prob_cap = g0 * np.exp(-alpha1 * (distances ** 2))
    for t in range(len(trap_locs)):
        if int(trap_x[t]) == 0:
            prob_cap[t, ...] = 0
    cap_hist = np.zeros(len(trap_locs))
    p_empty = compute_cond_lik_ind(len(ac_locs), prob_cap, K, len(trap_locs), cap_hist)
    return np.sum((1 - p_empty) * density)

def compute_cond_lik_ind(num_centers, cap_probs, K, num_traps, ind_cap_hist):
    broadcasted = np.broadcast_to(ind_cap_hist[:, np.newaxis], (num_traps, num_centers))
    probs = stats.binom.pmf(broadcasted, K, cap_probs)
    with np.errstate(divide='ignore'):
        log_probs = np.log(np.where(probs == 0.0, np.finfo(float).eps, probs))
    return np.exp(np.sum(log_probs, axis=0))

def compute_expected_n_across_scenarios(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    return np.array([[compute_expected_n(ac_locs, trap_locs, g0[s], sigma[s], K,
                                          density[s], distances, trap_x)] for s in range(len(g0))])

# --------------------------------------------- FORWARD GREEDY FUNCTION --------------------------------------------- 
# def forward_greedy_additions(scenarios, trap_locs, ac_locs, K, distances, draw, draw_to_trueN, deployed_indices, budget=10):
#     D, g0, sigma, density_prior = [], [], [], []
#     for s in range(len(scenarios)):
#         D.append(scenarios[s][0])
#         g0.append(scenarios[s][1])
#         sigma.append(scenarios[s][2])
#         path = f'only_trail_1km/D_mod/Dmod_draw_{scenarios[s][3]}.csv'
#         df = pd.read_csv(path)
#         density_prior.append((df['D_mod'] * 25).values)

#     n_traps = len(trap_locs)
#     trap_x = np.zeros(n_traps)
#     trap_x[deployed_indices] = 1

#     selected = list(deployed_indices)
#     en_hist, n_tracker = [], {}

#     # Compute the expected N for the original 68 deployed traps
#     initial_en = compute_expected_n_across_scenarios(ac_locs, trap_locs, g0, sigma, K, density_prior, distances, trap_x)
#     initial_avg_en = np.mean(initial_en)
#     en_hist.append(initial_avg_en)
#     n_tracker[0] = {
#         draw[i]: [draw_to_trueN[draw[i]], initial_en[i][0], initial_en[i][0] - draw_to_trueN[draw[i]]]
#         for i in range(len(draw))
#     }

#     # Begin adding 10 new traps
#     pbar = tqdm(total=budget, desc="Adding new traps")
#     for step in range(budget):

#         curr_en = compute_expected_n_across_scenarios(ac_locs, trap_locs, g0, sigma, K, density_prior, distances, trap_x)
#         current_avg = np.mean(curr_en)

#         candidates = []
#         for i in range(n_traps):
#             if trap_x[i] == 0:
#                 # Check if this candidate is >500m from all currently selected cameras
#                 if np.all(trap_to_trap_dist[i, selected] > 500):
#                     candidates.append(i)
#             # candidates = [i for i in range(n_traps) if trap_x[i] == 0]

#         trap_x_candidates = [trap_x.copy() for _ in candidates]
#         for i, idx in enumerate(candidates):
#             trap_x_candidates[i][idx] = 1

#         func = partial(compute_expected_n_across_scenarios, ac_locs, trap_locs, g0, sigma, K, density_prior, distances)
#         with mp.Pool(min(mp.cpu_count(), 10)) as pool:
#             en_all = np.squeeze(np.array(pool.map(func, trap_x_candidates)))

#         gains = np.mean(en_all, axis=1) - current_avg
#         best = candidates[np.argmax(gains)]
#         trap_x[best] = 1
#         selected.append(best)
#         en_hist.append(np.mean(en_all[np.argmax(gains)]))

#         est_Ns = en_all[np.argmax(gains)]
#         n_tracker[step + 1] = {
#             draw[i]: [draw_to_trueN[draw[i]], est_Ns[i], est_Ns[i] - draw_to_trueN[draw[i]]]
#             for i in range(len(draw))
#         }

#         print(f"Step {step+1}: +Trap {best}, Gain = {gains[np.argmax(gains)]:.2f}")
#         pbar.update(1)
#     pbar.close()

#     with open('./secr/Forward Greedy/FG43/n_tracker.txt', 'w') as f:
#         for step, records in n_tracker.items():
#             f.write(f"Step {step}:\n")
#             for d, values in records.items():
#                 f.write(f"  Draw {d}: TrueN={values[0]}, EstN={values[1]}, Diff={values[2]}\n")

#     return selected, en_hist, trap_x

def forward_greedy_additions(scenarios, trap_locs, ac_locs, K, distances, draw, draw_to_trueN, deployed_indices, budget=10):
    D, g0, sigma, density_prior = [], [], [], []
    for s in range(len(scenarios)):
        D.append(scenarios[s][0])
        g0.append(scenarios[s][1])
        sigma.append(scenarios[s][2])
        path = f'only_trail_1km/D_mod/Dmod_draw_{scenarios[s][3]}.csv'
        df = pd.read_csv(path)
        density_prior.append((df['D_mod'] * 25).values)

    n_traps = len(trap_locs)
    trap_x = np.zeros(n_traps)
    trap_x[deployed_indices] = 1

    selected = list(deployed_indices)

    # Precompute pairwise trap-to-trap distances for enforcing 500m rule
    trap_to_trap_dist = np.linalg.norm(
        trap_locs[:, np.newaxis, :] - trap_locs[np.newaxis, :, :], axis=2
    )

    en_hist = {}
    n_tracker = {}

    # Compute the expected N for originally deployed traps
    initial_en = compute_expected_n_across_scenarios(
        ac_locs, trap_locs, g0, sigma, K, density_prior, distances, trap_x
    )
    initial_avg_en = np.mean(initial_en)
    en_hist = [initial_avg_en]
    n_tracker = {
        0: {
            draw[i]: [draw_to_trueN[draw[i]], initial_en[i][0], initial_en[i][0] - draw_to_trueN[draw[i]]]
            for i in range(len(draw))
        }
    }

    pbar = tqdm(total=budget, desc="Adding new traps")
    for step in range(budget):
        curr_en = compute_expected_n_across_scenarios(
            ac_locs, trap_locs, g0, sigma, K, density_prior, distances, trap_x
        )
        current_avg = np.mean(curr_en)

        # Select candidates not yet deployed and >500m away from all selected traps
        candidates = []
        for i in range(n_traps):
            if trap_x[i] == 0:
                # Check distance to all selected traps
                if np.all(trap_to_trap_dist[i, selected] > 1000):
                    candidates.append(i)
        print("Candidates available for addition:", len(candidates))

        if not candidates:
            print("No further candidates meet the 1km distance constraint. Stopping early.")
            break

        # Create modified trap_x arrays for multiprocessing (for each candidate)
        trap_x_candidates = [trap_x.copy() for _ in candidates]
        for i, idx in enumerate(candidates):
            trap_x_candidates[i][idx] = 1

        func = partial(
            compute_expected_n_across_scenarios, ac_locs, trap_locs, g0, sigma, K, density_prior, distances
        )
        with mp.Pool(min(mp.cpu_count(), 10)) as pool:
            en_all = np.squeeze(np.array(pool.map(func, trap_x_candidates)))

        gains = np.mean(en_all, axis=1) - current_avg
        best = candidates[np.argmax(gains)]

        trap_x[best] = 1
        selected.append(best)
        en_hist.append(np.mean(en_all[np.argmax(gains)]))

        est_Ns = en_all[np.argmax(gains)]
        n_tracker[step + 1] = {
            draw[i]: [draw_to_trueN[draw[i]], est_Ns[i], est_Ns[i] - draw_to_trueN[draw[i]]] for i in range(len(draw))
        }

        print(f"Step {step+1}: +Trap {best}, Gain = {gains[np.argmax(gains)]:.2f}")
        pbar.update(1)

    pbar.close()

    # Save tracking info to a file if needed
    with open('./secr/Forward Greedy/FG43/n_tracker.txt', 'w') as f:
        for step_num, records in n_tracker.items():
            f.write(f"Step {step_num}:\n")
            for d, values in records.items():
                f.write(f"  Draw {d}: TrueN={values[0]}, EstN={values[1]}, Diff={values[2]}\n")

    return selected, en_hist, trap_x


# ---------------------------------------------  LOAD DATA --------------------------------------------- 
true_n = pd.read_csv('./only_trail_1km/True_N_per_draw.csv')
# valid_ids = true_n['Parameter_draw'].tolist()                               # Change this if you want to filter with parameter draws in between some values of N or g0

params = pd.read_csv('./only_trail_1km/param_values_for_each_draw150.csv')
params = params.rename(columns={'Unnamed: 0': 'index'})
# set valid_ids to those with beta1 between -0.8 and 0.99
valid_ids = params[(params['beta1'] >= -0.8) & (params['beta1'] <= 0.99)]['index'].tolist()
params = params[params['index'].isin(valid_ids)].reset_index(drop=True)
params = params.sample(n=25).reset_index(drop=True)                          # Randomly select 50 draws
# params = params[params['index'].isin(valid_ids)].iloc[:5]
print(f"Using {params['index'].tolist()} parameter draws")

D, g0, sigma = params['D'].values, params['g0'].values, params['sigma'].values
draw = params['index'].tolist()
draw_to_trueN = dict(zip(true_n['Parameter_draw'], true_n['N']))

# Gather all potential activity center locations
ac_coords = pyreadr.read_r('./only_trail_1km/mask_7-30-25.RDS')[None]
ac_coords_list = ac_coords[['x', 'y']].values

# Load full possible trap grid
full_trap_grid = pd.read_csv('./only_trail_1km/500m/trail_candidate_traps_spacing500.csv').drop(columns=['Unnamed: 0'])
full_trap_grid = full_trap_grid.rename(columns={'X': 'x', 'Y': 'y'})

# Candidate traps after 50m accessibility filter
# excluded_ids = pd.read_csv('./500m_data/traps_to_remove_50m.csv')['TrapID']
# candidate_grid = full_trap_grid[~full_trap_grid['Trap_index'].isin(excluded_ids)].copy()
candidate_grid = full_trap_grid.copy()

# Load all 68 deployed traps
# deployed_df = pd.read_csv('./secr/Prior Deployment/deployed_cams_2024.csv')
# deployed_df = deployed_df[['x', 'y', 'Trap_index']]  # assuming correct format from file
deployed_df = full_trap_grid[full_trap_grid['Prior_cam']==1].copy()

# Exclude those manual locations by Robin
exclude_indices = set()

# Add ranges
exclude_indices.update(range(973, 996))
exclude_indices.update(range(879, 883))
exclude_indices.update(range(936, 939))
exclude_indices.update(range(1021, 1024))
exclude_indices.update(range(1145, 1152))
exclude_indices.update(range(1189, 1203))
exclude_indices.update(range(722, 737))
exclude_indices.update(range(22, 25))
exclude_indices.update(range(29, 33))
exclude_indices.update(range(712, 716))
exclude_indices.update(range(746,760))
exclude_indices.update(range(1182,1188))

# Add single values
exclude_indices.update([
    1415, 1409, 1396, 1400, 1403,
    124,125,156,157,159,
    348, 349, 393, 394, 934,
    400, 4, 6, 7, 8,
    28, 401,
    743, 771, 781, 782, 803, 18,
    828, 831, 832, 834, 835, 837, 838, 
    839, 842, 844
])


# Filter out deployed traps from candidates
# candidate_exclusive = candidate_grid[~candidate_grid['Trap_index'].isin(deployed_df['Trap_index'])][['x', 'y', 'Trap_index']]
# Filter out deployed traps from candidates
candidate_exclusive = candidate_grid[
    (~candidate_grid['Trap_index'].isin(deployed_df['Trap_index'])) & 
    (~candidate_grid['Trap_index'].isin(exclude_indices))
][['x', 'y', 'Trap_index']]


# Combine deployed and candidates
combined_traps = pd.concat([deployed_df, candidate_exclusive], ignore_index=True)
trap_coords_list = combined_traps[['x', 'y']].values
trap_df = pd.DataFrame(trap_coords_list, columns=['x', 'y'])
trap_df.to_csv('./secr/Forward Greedy/FG43/considered_trap_locs.csv', index=False)
print(f"Total traps including deployed + candidates: {len(combined_traps)}")

# Indices of deployed traps = first 68
deployed_indices = np.arange(len(deployed_df))
print(f"Deployed traps found at rows: {deployed_indices}")

# Compute distances
traps = trap_coords_list[:, np.newaxis, :]
centers = ac_coords_list[np.newaxis, :, :]
distances = np.linalg.norm(traps - centers, axis=2)
scenarios = list(zip(D, g0, sigma, draw))

# Compute pairwise camera-to-camera distances (in meters)
trap_to_trap_dist = np.linalg.norm(
    trap_coords_list[:, np.newaxis, :] - trap_coords_list[np.newaxis, :, :], axis=2
)

# ---------------------------------------------  RUN FORWARD GREEDY ---------------------------------------------
start = time.time()
start_date = datetime.now()

selected_traps, en_hist, trap_x = forward_greedy_additions(
    scenarios, trap_coords_list, ac_coords_list, K=5,
    distances=distances,
    draw=draw,
    draw_to_trueN=draw_to_trueN,
    deployed_indices=deployed_indices,
    budget=10
)

end = time.time()
end_date = datetime.now()
print(f"Runtime: {end - start:.2f} seconds")

with open('./secr/Forward Greedy/FG43/runtime.txt', 'w') as f:
    f.write(f"Start time: {start_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"End time: {end_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Start seconds: {start} seconds\n")
    f.write(f"End seconds: {end} seconds\n")
    f.write(f"Total runtime: {end - start} seconds\n")

print(f"Start time: {start_date}, End time: {end_date}")

# ---------------------------------------------  PREPARE FILES FOR SECR --------------------------------------------- 
# Save results
np.save('./secr/Forward Greedy/FG43/all_selected_traps.npy', selected_traps)
with open('./secr/Forward Greedy/FG43/en_hist.txt', 'w') as f:
    for e in en_hist:
        f.write(f"{e}\n")
with open('./secr/Forward Greedy/FG43/considered_trap_locs.pkl', 'wb') as f:
    pickle.dump(trap_x, f)

# Export selected trap info
selected_arr = np.array(selected_traps)
final_df = trap_df.iloc[selected_arr].copy()
final_df['Trap_index'] = combined_traps.iloc[selected_arr]['Trap_index'].values
final_df.to_csv('./secr/Forward Greedy/FG43/selected_traps.csv', index=False)

# Verification
selected_trap_ids = set(final_df['Trap_index'].astype(int))
deployed_trap_ids = set(deployed_df['Trap_index'].astype(int).tolist())

missing = deployed_trap_ids - selected_trap_ids
extra = selected_trap_ids - deployed_trap_ids
assert len(missing) == 0, f"Missing deployed traps: {sorted(missing)}"
assert len(extra) == 10, f"Expected 10 new traps added but found {len(extra)}: {sorted(extra)}"

print(f"All 70 deployed traps are included.")
print(f"Exactly 10 new traps have been added.")

# Exclude from COMPLETE trap grid (not just candidate + deployed)
universe_ids = set(full_trap_grid['Trap_index'])
excluded_ids = sorted(universe_ids - selected_trap_ids)

with open('./secr/Forward Greedy/FG43/FG43-excluded_traps.txt', 'w') as f:
    for tid in excluded_ids:
        f.write(f"{tid}\n")

print(f"Excluded {len(excluded_ids)} traps out of {len(universe_ids)} total in grid.")
print(f"Selected {len(selected_trap_ids)} total traps.")
