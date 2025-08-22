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
import heapq
from datetime import datetime
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
    # print(f"Expected N: {expected_n}")
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
    broadcast_i_cap_hist = np.broadcast_to(ind_cap_hist[:, np.newaxis], (num_traps, num_activity_centers))      # Reshapes to array of shaep (num_traps, num_activity_centers)
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

def lazy_greedy_max_en(scenarios, trap_locs, ac_locs, K, distances, draw, draw_to_trueN, budget=100):
    """
    Lazy Greedy algorithm for selecting 'budget' trap locations maximizing E(n).
    """
    # Extract scenario parameters
    D, g0, sigma, density_prior = [], [], [], []
    for s in range(len(scenarios)):
        D.append(scenarios[s][0])
        g0.append(scenarios[s][1])
        sigma.append(scenarios[s][2])
        density_prior_file = f'full_grid_500m/8-1 data/D_mod/Dmod_draw_{scenarios[s][3]}.csv'
        density_df = pd.read_csv(density_prior_file)
        density_df['cell_density'] = density_df['D_mod'] * 25
        density_prior.append(density_df['cell_density'].values.tolist())


    n_traps = len(trap_locs)
    trap_x = np.zeros(n_traps)
    selected_traps, en_hist, n_tracker = [], [], {}

    # Precompute trap-to-trap distances for spacing constraint
    trap_to_trap_dist = np.linalg.norm(
        trap_locs[:, np.newaxis, :] - trap_locs[np.newaxis, :, :], axis=2
    )

    # Initial E(n) with no traps
    current_en_per_scenario = compute_expected_n_across_scenarios(
        ac_locs, trap_locs, g0, sigma, K, density_prior, distances, trap_x
    )
    current_avg_en = np.mean(current_en_per_scenario)

    # Build candidate pool (respect spacing constraint against empty set ≡ all traps eligible)
    initial_candidates = list(range(n_traps))
    # Compute initial marginal gains for all candidates
    func = partial(compute_expected_n_across_scenarios,
                   ac_locs, trap_locs, g0, sigma, K, density_prior, distances)

    trap_x_candidates = []
    for i in initial_candidates:
        tx = np.copy(trap_x)
        tx[i] = 1
        trap_x_candidates.append(tx)

    with mp.Pool(min(mp.cpu_count(), 10)) as pool:
        en_per_candidate_init = np.squeeze(np.array(pool.map(func, trap_x_candidates)))

    avg_en_init = np.mean(en_per_candidate_init, axis=1)
    marginal_gains = avg_en_init - current_avg_en

    heap = [(-marginal_gains[i], initial_candidates[i], en_per_candidate_init[i, :]) 
            for i in range(len(initial_candidates))]
    import heapq
    heapq.heapify(heap)

    from tqdm import tqdm
    pbar = tqdm(total=budget, desc="Lazy Greedy Progress")

    for step in range(budget):
        while heap:
            neg_gain, candidate, cached_en_per_scenario = heapq.heappop(heap)
            cached_gain = -neg_gain

            # Enforce 1km spacing constraint
            if selected_traps and not np.all(trap_to_trap_dist[candidate, selected_traps] > 1000):
                continue

            # Recompute marginal gain for this candidate
            trap_x_temp = np.copy(trap_x)
            trap_x_temp[candidate] = 1
            new_en_per_scenario = compute_expected_n_across_scenarios(
                ac_locs, trap_locs, g0, sigma, K, density_prior, distances, trap_x_temp
            )
            new_avg_en = np.mean(new_en_per_scenario)
            new_gain = new_avg_en - current_avg_en

            if not heap or new_gain >= -heap[0]:
                trap_x[candidate] = 1
                selected_traps.append(candidate)
                en_hist.append(new_avg_en)
                current_avg_en = new_avg_en

                n_tracker[step + 1] = {}
                for i, draw_id in enumerate(draw):
                    true_N = draw_to_trueN[draw_id]
                    est_N = new_en_per_scenario[i]
                    diff = est_N - true_N
                    n_tracker[step + 1][draw_id] = [true_N, est_N, diff]

                print(f"Step {step+1}: Added trap {candidate}, E(n)={new_avg_en:.2f}, Gain={new_gain:.2f}")
                pbar.update(1)
                break
            else:
                heapq.heappush(heap, (-new_gain, candidate, new_en_per_scenario))

        if not heap:
            print("No eligible candidates remain.")
            break

    pbar.close()
    return selected_traps, en_hist, trap_x, n_tracker


# Read in True N file
# true_n = pd.read_csv('./500m_data/True_N_per_draw.csv')
true_n = pd.read_csv('./full_grid_500m/8-1 data/True_N_per_draw.csv')
true_n_filtered = true_n[true_n['N'].between(1, 120)]                    # CHANGE DEPENDING ON NEED
true_n_filtered = true_n_filtered['Parameter_draw'].values.tolist()      # Get the parameter draw IDS

# Read in parameter draws
params = pd.read_csv('./full_grid_500m/8-1 data/param_values_for_each_draw150.csv')
params = params.rename(columns={'Unnamed: 0': 'index'})
params = params[params['index'].isin(true_n_filtered)]                   # Filter for parameter draws with True N between 30 and 80 
params = params.sample(n=50)                                                 # Randomly select 5 draws for testing

# Extract parameter values
D, g0, sigma = params['D'].values, params['g0'].values, params['sigma'].values
draw = params['index'].tolist()
draw_to_trueN = dict(zip(true_n['Parameter_draw'], true_n['N']))
K= 5                                        # Number of sampling periods
print(f"Parameter draws evaluated over: {draw}")

# Read in potential activity center locations
ac_coords = pyreadr.read_r('./full_grid_500m/500m_mask.RDS')
ac_coords = ac_coords[None]
ac_coords_list = ac_coords[['x', 'y']].values.tolist()
ac_coords_list = np.array(ac_coords_list)

# Read in potential trap locations
trap_coords = pd.read_csv('./full_grid_500m/500m_trap_grid.csv')
trap_coords = trap_coords.drop(columns = ['Unnamed: 0'])
# exclude_trap_coords = pd.read_csv('./full_grid_500m/traps_to_remove_PLUS_2km_boundary.csv')       # Read in trap locations that are from robin or > 2km away from prior deployment           
exclude_trap_coords = pd.read_csv('./full_grid_500m/traps_to_remove_UTM10N_updated.csv')            # remove traps that robin would never travel to
# exclude_trap_coords = pd.read_csv('./full_grid_500m/traps_to_remove_cost_500m.csv')               # robin removal and greater than 500m to trail
# exclude_trap_coords = pd.read_csv('./full_grid_500m/traps_to_remove_50m.csv')                     # robin removal and greater than 50m to trail
# exclude_trap_coords = pd.read_csv('./full_grid_500m/traps_to_remove_prior.csv')              # Read in trap locations that are from prior deployment
trap_coords = trap_coords[~trap_coords['Trap_index'].isin(exclude_trap_coords['Trap_index'])]
trap_coords_list = []
print(f"Total potential trap locations: {trap_coords.shape[0]}")
for i in range(trap_coords.shape[0]):
    trap_coords_list.append((trap_coords['x'].iloc[i], trap_coords['y'].iloc[i]))
trap_coords_list = np.array(trap_coords_list)

# Randomly select trap locations and activity centers -- Comment out when not testing
# np.random.shuffle(trap_coords_list)              # Randomly Shuffle
# trap_coords_list = (trap_coords_list)[:750]      # select the first 750 locations
# trap_coords_list = np.load('./secr/Forward Greedy/FG38/considered_trap_locs.npy')         # If needed, upload an existing considered traps - keep consistent with backward greedy considered locations
trap_coords_list_df = pd.DataFrame(trap_coords_list, columns=['x', 'y'])
trap_coords_list_df.to_csv('./secr/Lazy Greedy/LG1/considered_trap_locs.csv', index=False)

# Calculate euclidean distances from traps to activity centers
traps = trap_coords_list[:, np.newaxis, :]  # Add a new axis to traps to make it 3D
centers = ac_coords_list[np.newaxis, :, :]  # Add a new axis to centers to make it 3D
differences = traps - centers
distances = np.linalg.norm(differences, axis=2)

scenarios = list(zip(*[D, g0, sigma, draw]))

# log the starting time of backward greedy
start_date = datetime.now()
start_time = time.time()

# Run the forward greedy algorithm
selected_traps, en_hist, trap_x, n_tracker = lazy_greedy_max_en(scenarios, trap_coords_list, ac_coords_list, K, distances, draw, draw_to_trueN, budget=100)

# log the ending time of backward greedy
end_date = datetime.now()
end_time = time.time()

# export the total runtime to a text file
with open('secr/Lazy Greedy/LG1/runtime.txt', 'w') as f:
    # write the start time and date
    f.write(f"Start time: {start_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"End time: {end_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    # write the total runtime in seconds
    f.write(f"Start seconds: {start_time} seconds\n")
    f.write(f"End seconds: {end_time} seconds\n")
    f.write(f"Total runtime: {end_time - start_time} seconds\n")
print(f"Total runtime: {end_time - start_time} seconds")

# Save the results
np.save('./secr/Lazy Greedy/LG1/all_selected_traps.npy', selected_traps)
# Save the expected number of detections history as a txt file
with open('./secr/Lazy Greedy/LG1/en_hist.txt', 'w') as f:
    for en in en_hist:
        f.write(f"{en}\n")
# Save the trap_x configuration
with open('./secr/Lazy Greedy/LG1/considered_trap_locs.pkl', 'wb') as f:
    pickle.dump(trap_x, f)
# Print the selected traps
print("Selected traps:", selected_traps)
# Print the expected number of detections history
print("Expected number of detections history:", en_hist)


# Generate Files for SECR Analysis
trap_coords_list = pd.read_csv('./secr/Lazy Greedy/LG1/considered_trap_locs.csv')
selected_traps = np.load('./secr/Lazy Greedy/LG1/all_selected_traps.npy')
selected_traps = np.array(selected_traps)
selected_traps = selected_traps[:70]  # Limit to first 70 traps for testing
selected_traps = np.sort(selected_traps)

# subset trap_coords_list to include ONLY the selected trap indices
trap_coords_list_sub_df = trap_coords_list.iloc[selected_traps]
trap_coords_list_sub_df['Trap_index'] = trap_coords_list_sub_df.index + 1  # Adjust index to match SECR requirements
trap_coords_list_sub_df.to_csv('secr/Lazy Greedy/LG1/selected_traps.csv', index=True)

# get all the potential trap coordinates
trap_coords = pd.read_csv('full_grid_500m/500m_trap_grid.csv')
trap_coords = trap_coords.rename(columns={'X': 'x', 'Y': 'y'})
trap_coords = trap_coords.drop(columns=['Unnamed: 0'])

# CORRECTED LOGIC: Find excluded trap IDs using set operations on Trap_index values
# Get the Trap_index values for selected traps
selected_trap_ids = set(trap_coords_list_sub_df['Trap_index'])

# Get all Trap_index values from the full grid
all_trap_ids = set(trap_coords['Trap_index'])

# Excluded trap IDs are those in the full grid but NOT in selected_trap_ids
excluded_trap_ids = sorted(all_trap_ids - selected_trap_ids)

# convert to txt file
with open('secr/Lazy Greedy/LG1/LG1-excluded_traps-70.txt', 'w') as f:
    for item in excluded_trap_ids:
        f.write("%s\n" % item)

print(f"Selected {len(selected_traps)} traps")
print(f"Excluded {len(excluded_trap_ids)} traps")
print(f"Total traps in full grid: {len(all_trap_ids)}")
print(f"Verification: {len(selected_trap_ids) + len(excluded_trap_ids)} should equal {len(all_trap_ids)}")
