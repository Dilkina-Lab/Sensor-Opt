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

def compute_expected_n_indiv(ac_x, ac_y, g0, sigma, K, trap_locs, trap_x):
    """
    Computes expected number of unique individuals detected in a spatial capture-recapture study
    using simulated activity centers with individual-level g0 and sigma.

        ac_x, ac_y (1D arrays): Length (N_indiv) - x and y coordinates of simulated activity centers.
        g0 (1D array): Length (N_indiv) - detection probability at the activity center for each individual.
        sigma (1D array): Length (N_indiv) - scale parameter of the detection function for each individual.
        K (int): Number of sampling periods.
        trap_locs (2D array): Shape (num_traps, 2) - x and y coordinates of all potential trap locations.
        trap_x (1D array): Length (num_traps) - 1 if trap is active, 0 otherwise.
    """
    # Identify active traps
    active_idx = np.where(trap_x.astype(int) == 1)[0]
    if active_idx.size == 0:
        return 0.0

    active_traps = trap_locs[active_idx, :]  # (n_active_traps, 2)

    # Compute squared distances from active traps to each individual AC: shape (n_active_traps, N_indiv)
    diff_x = active_traps[:, 0][:, np.newaxis] - ac_x[np.newaxis, :]
    diff_y = active_traps[:, 1][:, np.newaxis] - ac_y[np.newaxis, :]
    d2 = diff_x**2 + diff_y**2

    # Broadcast per-individual sigma and g0 across traps
    sigma_b = np.broadcast_to(sigma[np.newaxis, :], d2.shape)
    g0_b = np.broadcast_to(g0[np.newaxis, :], d2.shape)

    # Per-trap, per-individual detection probability
    prob_cap = g0_b * np.exp(-d2 / (2.0 * sigma_b * sigma_b))  # (n_active_traps, N_indiv)

    # Probability of no detection over K occasions at each active trap: (1 - p)^K
    p_no_det_per_trap = (1.0 - prob_cap) ** K

    # Probability of no detection at any trap for each individual: product over traps
    p_no_det_i = np.prod(p_no_det_per_trap, axis=0)  # length N_indiv

    # Probability of being detected at least once
    p_det_i = 1.0 - p_no_det_i

    # Expected number of unique individuals detected
    expected_n = np.sum(p_det_i)
    return expected_n


def compute_expected_c_indiv(ac_x, ac_y, g0, sigma, K, trap_locs, trap_x):
    """
    Computes expected number of captures in a spatial capture-recapture study
    using simulated activity centers with individual-level g0 and sigma.

        ac_x, ac_y (1D arrays): Length (N_indiv).
        g0 (1D array): Length (N_indiv).
        sigma (1D array): Length (N_indiv).
        K (int): Number of sampling periods.
        trap_locs (2D array): Shape (num_traps, 2).
        trap_x (1D array): Length (num_traps).
    """
    # Identify active traps
    active_idx = np.where(trap_x.astype(int) == 1)[0]
    if active_idx.size == 0:
        return 0.0

    active_traps = trap_locs[active_idx, :]  # (n_active_traps, 2)

    # Compute squared distances from active traps to each individual AC: shape (n_active_traps, N_indiv)
    diff_x = active_traps[:, 0][:, np.newaxis] - ac_x[np.newaxis, :]
    diff_y = active_traps[:, 1][:, np.newaxis] - ac_y[np.newaxis, :]
    d2 = diff_x**2 + diff_y**2

    # Broadcast per-individual sigma and g0 across traps
    sigma_b = np.broadcast_to(sigma[np.newaxis, :], d2.shape)
    g0_b = np.broadcast_to(g0[np.newaxis, :], d2.shape)

    # Per-trap, per-individual detection probability
    prob_cap = g0_b * np.exp(-d2 / (2.0 * sigma_b * sigma_b))  # (n_active_traps, N_indiv)

    # Expected captures = K * sum_t sum_i p_ti
    expected_c = K * np.sum(prob_cap)
    return expected_c


def compute_expected_n_across_scenarios_indiv(ac_list, trap_locs, g0_list, sigma_list, K, trap_x):
    """
    Computes expected number of unique individuals detected across multiple scenarios
    using simulated activity centers.
    """
    nscenarios = len(g0_list)
    e_n = np.zeros((nscenarios, 1))
    for s in range(nscenarios):
        ac_x, ac_y = ac_list[s]
        e_n[s, 0] = compute_expected_n_indiv(
            ac_x, ac_y, g0_list[s], sigma_list[s], K, trap_locs, trap_x
        )
    return e_n


def compute_expected_c_across_scenarios_indiv(ac_list, trap_locs, g0_list, sigma_list, K, trap_x):
    """
    Computes expected number of captures across multiple scenarios
    using simulated activity centers.
    """
    nscenarios = len(g0_list)
    e_c = np.zeros((nscenarios, 1))
    for s in range(nscenarios):
        ac_x, ac_y = ac_list[s]
        e_c[s, 0] = compute_expected_c_indiv(
            ac_x, ac_y, g0_list[s], sigma_list[s], K, trap_locs, trap_x
        )
    return e_c


#################################################################################################################################
############                                         GREEDY FUNCTIONS                                                ############
#################################################################################################################################

def forward_greedy(scenarios, trap_loc, K, ac_list, g0_indiv_list, sigma_indiv_list, draw, draw_to_trueN, max_traps):
    print("Starting Forward Greedy Algorithm for RSE Minimization")
    trap_x = np.zeros((len(trap_loc),))
    D = []  # still available if you need it later

    # Store D per scenario (not used in computations below but kept for consistency)
    for s in range(len(scenarios)):
        D.append(scenarios[s][0])

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

        # partials for individual-level expectations
        func1 = partial(
            compute_expected_n_across_scenarios_indiv,
            ac_list, trap_loc, g0_indiv_list, sigma_indiv_list, K
        )
        func2 = partial(
            compute_expected_c_across_scenarios_indiv,
            ac_list, trap_loc, g0_indiv_list, sigma_indiv_list, K
        )

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
true_n = pd.read_csv('./full_grid_1km/10-3 data (Marten)/varying_detection_params/2-26 Marten/True_N_per_draw.csv')
draw_to_trueN = dict(zip(true_n['Parameter_draw'], true_n['N']))

# Read in parameter draws
params = pd.read_csv('./full_grid_1km/10-3 data (Marten)/varying_detection_params/2-26 Marten/param_values_for_each_draw300_marten_g0sigma_covs.csv')
params = params.rename(columns={'Unnamed: 0': 'index'})

# Set IDs of parameters to evaluate over
params = params[params['index'].isin([64, 74, 76, 78, 109, 125, 164, 204, 238, 250])]

D = params['D'].values
g0 = params['g0'].values
sigma = params['sigma'].values
K = 5
draw = params['index'].values.tolist()
print(f"Parameter draws evaluated over: {draw}")        # sanity check print statement

# ac_coords = pyreadr.read_r('./full_grid_1km/10-3 data (Marten)/500m_mask_marten.RDS')       # read in potential AC locations
# ac_coords = ac_coords[None]
# ac_coords_list = ac_coords[['x', 'y']].values.tolist()
# ac_coords_list = np.array(ac_coords_list)

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

# traps = trap_coords_list[:, np.newaxis, :]  # Add a new axis to traps to make it 3D
# centers = ac_coords_list[np.newaxis, :, :]  # Add a new axis to centers to make it 3D
# differences = traps - centers
# distances = np.linalg.norm(differences, axis=2)  # compute the EUCLIDEAN distances between all traps and all activity centers

scenarios = list(zip(*[D, g0, sigma, draw]))

# Build per-scenario lists of simulated activity centers and their g0, sigma
ac_list = []          # each element: (ac_x, ac_y)
g0_indiv_list = []    # each: 1D array of g0_i
sigma_indiv_list = [] # each: 1D array of sigma_i

for s in range(len(scenarios)):
    draw_id = scenarios[s][3]   # this is the index for Parameter_draw

    ac_file = f'./full_grid_1km/10-3 data (Marten)/varying_detection_params/2-26 Marten/AC_locations/AC_draw_{draw_id}.csv'
    ac_df = pd.read_csv(ac_file)

    ac_x = ac_df["x"].values
    ac_y = ac_df["y"].values
    g0_i = ac_df["g0"].values
    sigma_i = ac_df["sigma"].values

    ac_list.append((ac_x, ac_y))
    g0_indiv_list.append(g0_i)
    sigma_indiv_list.append(sigma_i)

start_date = datetime.now()     # track runtimes (maybe not needed for the webapp)
start_time = time.time()

################################################################################################################################
###########                                         RUN GREEDY FUNCTIONS                                            ############
################################################################################################################################

# Run the forward greedy algorithm
selected_traps, RSE_hist, trap_x = forward_greedy(
    scenarios, trap_coords_list, K,
    ac_list, g0_indiv_list, sigma_indiv_list,
    draw, draw_to_trueN, 80
)

################################################################################################################################
###########                                           PROCESS RESULTS                                               ############
################################################################################################################################
# log the ending time of Forward Greedy
end_date = datetime.now()
end_time = time.time()

# export the total runtime to a text file - Can skip for webapp.
with open('secr/Sample Average Approximation/SA9/SA9-1/runtime.txt', 'w') as f:
    # write the start time and date
    f.write(f"Start time: {start_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"End time: {end_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    # write the total runtime in seconds
    f.write(f"Start seconds: {start_time} seconds\n")
    f.write(f"End seconds: {end_time} seconds\n")
    f.write(f"Total runtime: {end_time - start_time} seconds\n")
print(f"Total runtime: {end_time - start_time} seconds")

# Save the results from greedy algorithms
np.save('./secr/Sample Average Approximation/SA9/SA9-1/all_selected_traps.npy', selected_traps)        ### THIS IS THE IMPORTANT FILE OUTPUT TO VISUALIE THE SELECTED TRAPS.

# All other files are probably not needing to be exported on webapp? Consider commenting out below. This was for my own tracking.
with open('./secr/Sample Average Approximation/SA9/SA9-1/rse_hist.txt', 'w') as f:
    for rse in RSE_hist:
        f.write(f"{rse}\n")
with open('./secr/Sample Average Approximation/SA9/SA9-1/considered_trap_locs.pkl', 'wb') as f:
    pickle.dump(trap_x, f)
print("Selected traps:", selected_traps)


###############################################################################################################################
###########                                      GENERATE FILES FOR SECR                                            ############
################################################################################################################################
trap_coords_list_df = pd.DataFrame(trap_coords_list, columns=['x', 'y'])
trap_coords_list_df.to_csv(f'./secr/Sample Average Approximation/SA9/SA9-1/considered_trap_locs.csv', index=False)


# Generate Files for SECR Analysis
base_dir = './secr/Sample Average Approximation/SA9/SA9-1'
# trap_coords_list = pd.read_csv(f'./secr/Sample Average Approximation/SA9/SA9-1/considered_trap_locs.csv')


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

    excluded_txt_path = f'{base_dir}/SA9-1-excluded_traps-{n_cams}.txt'
    with open(excluded_txt_path, 'w') as f:
        for item in excluded_trap_ids:
            f.write(f"{item}\n")

    print(f"[{n_cams} cams] Selected {len(traps_subset)} traps, "
          f"Excluded {len(excluded_trap_ids)} traps, "
          f"Total traps in full grid: {len(all_trap_ids)}")
    print(f"   Verification: {len(selected_trap_ids) + len(excluded_trap_ids)} = {len(all_trap_ids)}")
