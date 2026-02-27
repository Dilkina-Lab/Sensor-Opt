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
    g0, sigma, density: 1D arrays length = num_activity_centers (one value per pixel).
    distances: shape (num_traps, num_activity_centers).
    """
    # alpha1_j = 1 / (2 * sigma_j^2)
    alpha1 = 1.0 / (2.0 * sigma * sigma)                     # (n_pixels,)
    alpha1 = np.broadcast_to(alpha1, distances.shape)        # (n_traps, n_pixels)

    g0_b = np.broadcast_to(g0, distances.shape)              # (n_traps, n_pixels)

    prob_cap = g0_b * np.exp(-alpha1 * (distances ** 2))     # (n_traps, n_pixels)

    # zero out inactive traps
    for t in range(len(trap_locs)):
        if int(trap_x[t]) == 0:
            prob_cap[t, :] = 0.0

    i_cap_hist = np.zeros(len(trap_locs))                    # one entry per trap
    p_empty_cap_hist = compute_cond_lik_ind(
        len(ac_locs), prob_cap, K, len(trap_locs), i_cap_hist
    )
    p_nonempty = 1.0 - p_empty_cap_hist

    density_array = np.array(density).flatten()              # (n_pixels,)
    expected_n = np.sum(p_nonempty * density_array)
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


def compute_expected_n_across_scenarios(ac_locs, trap_locs, g0_list, sigma_list, K, density_list, distances, trap_x):
    nscenarios = len(g0_list)
    e_n = np.zeros((nscenarios, 1))
    for s in range(nscenarios):
        e_n[s, 0] = compute_expected_n(
            ac_locs, trap_locs,
            g0_list[s], sigma_list[s], K, density_list[s], distances, trap_x
        )
    return e_n


def compute_expected_c(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    """
    g0, sigma, density: 1D arrays length = num_activity_centers.
    """
    alpha1 = 1.0 / (2.0 * sigma * sigma)
    alpha1 = np.broadcast_to(alpha1, distances.shape)        # (n_traps, n_pixels)
    g0_b = np.broadcast_to(g0, distances.shape)              # (n_traps, n_pixels)

    prob_cap = g0_b * np.exp(-alpha1 * (distances ** 2))

    for t in range(len(trap_locs)):
        if int(trap_x[t]) == 0:
            prob_cap[t, :] = 0.0

    density_array = np.array(density).flatten()
    broadcast_density = np.broadcast_to(density_array, prob_cap.shape)

    expected_c = np.sum(prob_cap * broadcast_density) * K
    return expected_c


def compute_expected_c_across_scenarios(ac_locs, trap_locs, g0_list, sigma_list, K, density_list, distances, trap_x):
    nscenarios = len(g0_list)
    e_c = np.zeros((nscenarios, 1))
    for s in range(nscenarios):
        e_c[s, 0] = compute_expected_c(
            ac_locs, trap_locs,
            g0_list[s], sigma_list[s], K, density_list[s], distances, trap_x
        )
    return e_c

#################################################################################################################################
############                                         GREEDY FUNCTIONS                                                ############
#################################################################################################################################

def forward_greedy(scenarios, trap_loc, centers, K, distances, draw, draw_to_trueN, max_traps):
    print("Starting Forward Greedy Algorithm for RSE Minimization")
    trap_x = np.zeros((len(trap_loc),))
    D = []
    density_prior = []
    g0_prior = []
    sigma_prior = []

    for s in range(len(scenarios)):
        D.append(scenarios[s][0])

        draw_id = scenarios[s][3]

        # Density surface
        density_prior_file = f'full_grid_1km/10-3 data (Marten)/Constant_detection/D_mod/Dmod_draw_{draw_id}.csv'
        density_df = pd.read_csv(density_prior_file)
        density_df["D_mod"] = density_df["D_mod"] * 25  # scaling as before
        density_prior.append(density_df["D_mod"].values)  # shape (n_pixels,)

        # g0 surface
        g0_prior_file = f'full_grid_1km/10-3 data (Marten)/Constant_detection/g0_mod/g0mod_draw_{draw_id}.csv'
        g0_df = pd.read_csv(g0_prior_file)
        g0_prior.append(g0_df["g0_mod"].values)          # shape (n_pixels,)

        # sigma surface
        sigma_prior_file = f'full_grid_1km/10-3 data (Marten)/Constant_detection/sigma_mod/sigmamod_draw_{draw_id}.csv'
        sigma_df = pd.read_csv(sigma_prior_file)
        sigma_prior.append(sigma_df["sigma_mod"].values) # shape (n_pixels,)

    RSE_hist = []       # Track RSE history at each trap addition
    add_hist = []       # Track which traps were added at each step
    activated_trap_hist = []    # Track which traps are active at each step
    rse_tracker = {}    # Track RSE estimates at each step
    n_tracker = {}      # Track N estimates at each step

    counter = 0
    pbar = tqdm(total=max_traps, desc="Forward Greedy Progress")
    while sum(trap_x) < max_traps:      # max_traps is your budget
        pbar.update(1)
        counter += 1

        trap_indices = [i for i, x in enumerate(trap_x) if int(x) == 0]                 # inactive traps
        activated_trap_hist.append([i for i, x in enumerate(trap_x) if int(x) == 1])    # active traps

        trap_x_temp = np.copy(np.broadcast_to(trap_x, (len(trap_indices), trap_x.shape[0])))    # temp array to try adding each inactive trap
        for pos in range(len(trap_indices)):
            trap_idx = trap_indices[pos]
            trap_x_temp[pos, trap_idx] = 1  # try adding this trap

        func1 = partial(compute_expected_n_across_scenarios, centers, trap_loc, g0_prior, sigma_prior, K, density_prior, distances)
        func2 = partial(compute_expected_c_across_scenarios, centers, trap_loc, g0_prior, sigma_prior, K, density_prior, distances)
        pool = mp.Pool(min(mp.cpu_count(), 10))         # I limited to 10 cores since my machine was getting overworked, but this could be adjusted.
        E_n_per_scenario = np.squeeze(np.array(pool.map(func1, trap_x_temp)))   
        E_c_per_scenario = np.squeeze(np.array(pool.map(func2, trap_x_temp)))
        pool.close()
        pool.join()

        E_r_per_scenario = E_c_per_scenario     # Compute E_r based on E_c
        min_n_r_per_scenario = np.zeros(E_r_per_scenario.shape)     # Initialize min(N,R) array
        RSE_per_scenario = np.zeros(E_r_per_scenario.shape)         # Initialize RSE array

        for t in range(len(trap_indices)):  
            for s in range(len(scenarios)):
                E_r_per_scenario[t, s] -= E_n_per_scenario[t, s]    # E_r = E_c - E_n, We set E_r to equal E_c above for ease of computation.
                # if E_r_per_scenario[t, s] < E_n_per_scenario[t, s]:
                #     print(f"Scenario {s} (Draw {scenarios[s][3]}): E_r < E_n ({E_r_per_scenario[t, s]:.2f} < {E_n_per_scenario[t, s]:.2f}) when adding trap {trap_indices[t]}")
                # else:
                #     print(f"Scenario {s} (Draw {scenarios[s][3]}): E_n <= E_r ({E_n_per_scenario[t, s]:.2f} <= {E_r_per_scenario[t, s]:.2f}) when adding trap {trap_indices[t]}")
                min_n_r_per_scenario[t, s] = min(E_n_per_scenario[t, s], E_r_per_scenario[t, s]) 
                RSE_per_scenario[t, s] = 1 / np.sqrt(min_n_r_per_scenario[t, s])

        RSE_temp = np.mean(RSE_per_scenario, axis=1).tolist()   # Mean RSE across scenarios for each candidate trap addition
        variances = np.var(RSE_per_scenario, axis=1)
        average_variance = variances.mean()
        # print(f"Iteration {counter} RSE candidates: {RSE_temp}")
        # print(f"Iteration {counter} average variance: {average_variance}")

        rse_tracker[counter] = {
            cam_id: RSE_temp[i]
            for i, cam_id in enumerate(trap_indices)
        }

        min_change_idx = RSE_temp.index(min(RSE_temp))      # pick the trap to add to the layout that has the lowest RSE
        add = trap_indices[min_change_idx]
        trap_x[add] = 1

        # Track N estimates)
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
true_n = pd.read_csv('./full_grid_1km/10-3 data (Marten)/Constant_detection/True_N_per_draw.csv')
draw_to_trueN = dict(zip(true_n['Parameter_draw'], true_n['N']))

# Read in parameter draws
params = pd.read_csv('./full_grid_1km/10-3 data (Marten)/param_values_for_each_draw300_marten.csv')
params = params.rename(columns={'Unnamed: 0': 'index'})

# Set IDs of parameters to evaluate over
params = params[params['index'].isin([34, 68, 80, 93, 105, 155, 197, 219, 282, 298])]       ### I hardcoded this, not sure how we want to do this for the webapp?

D = params['D'].values
g0 = params['g0'].values
sigma = params['sigma'].values
K = 5
draw = params['index'].values.tolist()
print(f"Parameter draws evaluated over: {draw}")        # sanity check print statement

ac_coords = pyreadr.read_r('./full_grid_1km/10-3 data (Marten)/500m_mask_marten.RDS')       # read in potential AC locations
ac_coords = ac_coords[None]
ac_coords_list = ac_coords[['x', 'y']].values.tolist()
ac_coords_list = np.array(ac_coords_list)

trap_coords = pd.read_csv('./full_grid_1km/10-3 data (Marten)/1000m_trap_grid_marten.csv')
trap_coords = trap_coords.drop(columns = ['Unnamed: 0'])

# Remove trap locations which will never have a deployment -- Not sure if this will be functionality supported in webapp? Right now, not using it so commenting out.
# exclude_trap_coords = pd.read_csv('./full_grid_1km/traps_to_remove_1km.csv')
# trap_coords = trap_coords[~trap_coords['Trap_index'].isin(exclude_trap_coords['Trap_index'])]

trap_coords_list = []       # Create list of candidate trap locations
for i in range(trap_coords.shape[0]):
    trap_coords_list.append((trap_coords['x'].iloc[i], trap_coords['y'].iloc[i]))
trap_coords_list = np.array(trap_coords_list)
print(f"{trap_coords_list.shape} candidate trap locations")     # sanity check on number of candidate trap locations
trap_coords_list_df = pd.DataFrame(trap_coords_list, columns=['x', 'y'])    # convert to dataframe

traps = trap_coords_list[:, np.newaxis, :]  # Add a new axis to traps to make it 3D
centers = ac_coords_list[np.newaxis, :, :]  # Add a new axis to centers to make it 3D
differences = traps - centers
distances = np.linalg.norm(differences, axis=2)  # compute the EUCLIDEAN distances between all traps and all activity centers

scenarios = list(zip(*[D, g0, sigma, draw]))

start_date = datetime.now()     # track runtimes (maybe not needed for the webapp)
start_time = time.time()

################################################################################################################################
###########                                         RUN GREEDY FUNCTIONS                                            ############
################################################################################################################################

# Run the forward greedy algorithm
selected_traps, RSE_hist, trap_x = forward_greedy(scenarios, trap_coords_list, ac_coords_list, K, distances, draw, draw_to_trueN, 80)


################################################################################################################################
###########                                           PROCESS RESULTS                                               ############
################################################################################################################################
# log the ending time of Forward Greedy
end_date = datetime.now()
end_time = time.time()

# export the total runtime to a text file - Can skip for webapp.
with open('secr/Sample Average Approximation/SA2/SA2-20/runtime.txt', 'w') as f:
    # write the start time and date
    f.write(f"Start time: {start_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"End time: {end_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    # write the total runtime in seconds
    f.write(f"Start seconds: {start_time} seconds\n")
    f.write(f"End seconds: {end_time} seconds\n")
    f.write(f"Total runtime: {end_time - start_time} seconds\n")
print(f"Total runtime: {end_time - start_time} seconds")

# Save the results from greedy algorithms
np.save('./secr/Sample Average Approximation/SA2/SA2-20/all_selected_traps.npy', selected_traps)        ### THIS IS THE IMPORTANT FILE OUTPUT TO VISUALIE THE SELECTED TRAPS.
# Save the expected number of detections history as a txt file
# with open('./secr/Greedy Removal/FGR2/en_hist.txt', 'w') as f:
#     for en in en_hist:
#         f.write(f"{en}\n")
# same RSE history as txt file

# All other files are probably not needing to be exported on webapp? Consider commenting out below. This was for my own tracking.
with open('./secr/Sample Average Approximation/SA2/SA2-20/rse_hist.txt', 'w') as f:
    for rse in RSE_hist:
        f.write(f"{rse}\n")
with open('./secr/Sample Average Approximation/SA2/SA2-20/considered_trap_locs.pkl', 'wb') as f:
    pickle.dump(trap_x, f)
print("Selected traps:", selected_traps)


###############################################################################################################################
###########                                      GENERATE FILES FOR SECR                                            ############
################################################################################################################################
"""
This section is probably not needed as we aren't using SECR in the webapp loop anywhere. 
Can probably disregard this as its just setting up file formatting to work with Arielle's SECR code.
"""

trap_coords_list_df = pd.DataFrame(trap_coords_list, columns=['x', 'y'])
trap_coords_list_df.to_csv(f'./secr/Sample Average Approximation/SA2/SA2-20/considered_trap_locs.csv', index=False)

# Generate Files for SECR Analysis
base_dir = './secr/Sample Average Approximation/SA2/SA2-20'
# trap_coords_list = pd.read_csv(f'./secr/Sample Average Approximation/SA2/SA2-20/considered_trap_locs.csv')

# All trap coordinates (full grid)
trap_coords = pd.read_csv('./full_grid_1km/10-3 data (Marten)/1000m_trap_grid_marten.csv')
trap_coords = trap_coords.rename(columns={'X': 'x', 'Y': 'y'})
trap_coords = trap_coords.drop(columns=['Unnamed: 0'])
all_trap_ids = set(trap_coords['Trap_index'])

# Make sure output directory exists
os.makedirs(base_dir, exist_ok=True)

for n_cams in [10, 20, 30, 40, 50, 60, 70, 80]:
    traps_subset = selected_traps[:n_cams]

    trap_coords_list_sub_df = trap_coords[trap_coords['Trap_index'].isin(traps_subset)].copy()
    trap_coords_list_sub_df['Trap_index'] = trap_coords_list_sub_df['Trap_index']  # Keep IDs as-is

    trap_csv_path = f'{base_dir}/selected_traps_{n_cams}.csv'
    trap_coords_list_sub_df.to_csv(trap_csv_path, index=False)  # Index can be False if IDs are in data

    all_trap_ids = set(trap_coords['Trap_index'])
    selected_trap_ids = set(trap_coords_list_sub_df['Trap_index'])
    excluded_trap_ids = sorted(all_trap_ids - selected_trap_ids)

    excluded_txt_path = f'{base_dir}/SA2-20-excluded_traps-{n_cams}.txt'
    with open(excluded_txt_path, 'w') as f:
        for item in excluded_trap_ids:
            f.write(f"{item}\n")

    print(f"[{n_cams} cams] Selected {len(traps_subset)} traps, "
          f"Excluded {len(excluded_trap_ids)} traps, "
          f"Total traps in full grid: {len(all_trap_ids)}")
    print(f"   Verification: {len(selected_trap_ids) + len(excluded_trap_ids)} = {len(all_trap_ids)}")
