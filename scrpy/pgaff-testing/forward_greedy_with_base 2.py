import os
import sys
import time
import pickle
import logging
import warnings
import multiprocessing as mp
from datetime import datetime
from functools import partial
import numpy as np
import pandas as pd
import pyreadr
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.getLogger("distributed").setLevel(logging.ERROR)

sys.path.append(os.path.abspath('../src/'))


#################################################################################################################################
############                                         RUN SETTINGS                                                   ############
#################################################################################################################################

"""
Arielle, maybe we can read in and set the paths to the variables and the parameter details at the top of the code?
This isn't an exhaustive list since we have the define the paths to density, true N, param locations, etc. but this is a thought?
"""

# Number of parameter draws to evaluate over
N_PARAM_DRAWS = 10

# Number of NEW traps to add on top of the prior deployment
BUDGET = 10

# Number of sampling occasions
K = 5

#################################################################################################################################
############                                        UTILITY FUNCTIONS                                                ############
#################################################################################################################################


def compute_expected_n(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    """
    Computes expected number of unique individuals detected in a spatial capture-recapture study BASED ON THE FAST HEURISTIC from Efford/Boulanger.

        ac_locs (2D array): Shape (num_activity_centers, 2) - x and y coordinates of activity centers.
        trap_locs (2D array): Shape (num_traps, 2) - x and y coordinates of all potential trap locations.
        g0 (float): Detection probability at the activity center.
        sigma (float): Scale parameter of the detection function.
        K (int): Number of sampling periods.
        density (1D array): Density of animals
        distances (2D array): Shape (num_traps, num_activity_centers) - distance between each trap and activity center.
        trap_x (1D array): 1D array of activated trap locations.
    """
    alpha1 = 1 / (2 * sigma * sigma)
    prob_cap = g0 * np.exp(-alpha1 * (distances ** 2))        # Array of size (# traps, # potential activity centers)

    for t in range(len(trap_locs)):                           # for trap locations that are not selected, set all capture probs to 0
        if int(trap_x[t]) == 0:
            prob_cap[t, ...] = np.zeros(ac_locs.shape[0])

    i_cap_hist = np.squeeze(np.zeros((len(trap_locs), 1)))    # Initialize empty capture history
    p_empty_cap_hist = compute_cond_lik_ind(                  # compute probability of never being captured (call below function)
        len(ac_locs), prob_cap, K, len(trap_locs), i_cap_hist
    )
    p_nonempty = 1 - p_empty_cap_hist                         # probability of being captured at least once
    expected_n = np.sum(p_nonempty * density)                 # En equation based on Efford/Boulanger heuristic
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
    """
    Computes expected number of unique individuals detected across multiple scenarios based on the fast heuristic from Efford/Boulanger.
    """
    nscenarios = len(g0)
    e_n = np.zeros((nscenarios, 1))
    for s in range(nscenarios):
        e_n[s, 0] = compute_expected_n(ac_locs, trap_locs, g0[s], sigma[s], K, density[s], distances, trap_x)
    return e_n


def compute_expected_c(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    """
    Computes expected number of captures in a spatial capture-recapture study BASED ON THE FAST HEURISTIC from Efford/Boulanger.

        ac_locs (2D array): Shape (num_activity_centers, 2)
        trap_locs (2D array): Shape (num_traps, 2)
        g0 (float): Detection probability at the activity center.
        sigma (float): Scale parameter of the detection function.
        K (int): Number of sampling periods.
        density (1D array): Density of animals
        distances (2D array): Shape (num_traps, num_activity_centers)
        trap_x (1D array): 1D array of activated trap locations.
    """
    alpha1 = 1 / (2 * sigma * sigma)                          # convert sigma to alpha1
    prob_cap = g0 * np.exp(-alpha1 * (distances ** 2))       # calculate probability of capture
    density_array = np.array(density).flatten()              # make density a 1D array

    for t in range(len(trap_locs)):                           # for trap locations that are not selected, set all capture probs to 0
        if int(trap_x[t]) == 0:
            prob_cap[t, ...] = np.zeros(ac_locs.shape[0])

    broadcast_density = np.broadcast_to(density_array, (len(trap_locs), len(density_array)))
    expected_c = np.sum(prob_cap * broadcast_density) * K     # expected captures equation based on Efford/Boulanger heuristic
    return expected_c


def compute_expected_c_across_scenarios(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    """
    Computes expected number of captures across multiple scenarios based on the fast heuristic from Efford/Boulanger.
    """
    nscenarios = len(g0)
    e_c = np.zeros((nscenarios, 1))
    for s in range(nscenarios):
        e_c[s, 0] = compute_expected_c(ac_locs, trap_locs, g0[s], sigma[s], K, density[s], distances, trap_x)
    return e_c


#################################################################################################################################
############                                  FORWARD GREEDY WITH BASE TRAPS                                        ############
#################################################################################################################################


def forward_greedy_additions(scenarios, trap_locs, ac_locs, K, distances, draw, draw_to_trueN, deployed_indices, budget):
    """
    Runs the forward greedy algorithm starting from an existing set of deployed traps.
    Uses the SAME RSE objective as the original forward_greedy code, but initializes trap_x with the prior deployed traps already active.

        scenarios (list): List of tuples (D, g0, sigma, draw_id) for each parameter draw.
        trap_locs (2D array): All trap locations considered in the optimization.
        ac_locs (2D array): Potential activity center locations.
        K (int): Number of sampling periods.
        distances (2D array): Trap-to-activity-center distances.
        draw (list): Parameter draw IDs.
        draw_to_trueN (dict): Map from draw ID to true abundance N.
        deployed_indices (1D array): Row indices of previously deployed traps in trap_locs.
        budget (int): Number of NEW traps to add.
    """
    print("Starting Forward Greedy Algorithm from prior deployment")

    trap_x = np.zeros((len(trap_locs),))          # start with zero traps
    trap_x[deployed_indices] = 1                  # activate prior deployed traps

    D = []
    g0 = []
    sigma = []
    density_prior = []

    # Precompute density prior information for all scenarios
    for s in range(len(scenarios)):
        D.append(scenarios[s][0])
        g0.append(scenarios[s][1])
        sigma.append(scenarios[s][2])

        density_prior_file = f'only_trail_1km/D_mod/Dmod_draw_{scenarios[s][3]}.csv'
        density_df = pd.read_csv(density_prior_file)
        density_df['D_mod'] = density_df['D_mod'] * 25      # scale up
        density_prior.append(density_df['D_mod'].values.tolist())

    RSE_hist = []               # Track RSE history at each trap addition
    add_hist = []               # Track which NEW traps were added at each step
    activated_trap_hist = []    # Track which traps are active at each step
    rse_tracker = {}            # Track RSE estimates at each step
    n_tracker = {}              # Track N estimates at each step

    counter = 0
    pbar = tqdm(total=budget, desc="Forward Greedy Progress")

    while len(add_hist) < budget:
        pbar.update(1)
        counter += 1

        trap_indices = [i for i, x in enumerate(trap_x) if int(x) == 0]                 # inactive traps
        activated_trap_hist.append([i for i, x in enumerate(trap_x) if int(x) == 1])    # active traps

        trap_x_temp = np.copy(np.broadcast_to(trap_x, (len(trap_indices), trap_x.shape[0])))    # temp array to try adding each inactive trap
        for pos in range(len(trap_indices)):
            trap_idx = trap_indices[pos]
            trap_x_temp[pos, trap_idx] = 1      # try adding this trap

        func1 = partial(compute_expected_n_across_scenarios, ac_locs, trap_locs, g0, sigma, K, density_prior, distances)
        func2 = partial(compute_expected_c_across_scenarios, ac_locs, trap_locs, g0, sigma, K, density_prior, distances)

        pool = mp.Pool(min(mp.cpu_count(), 10))         # I limited to 10 cores since my machine was getting overworked, but this could be adjusted.
        E_n_per_scenario = np.squeeze(np.array(pool.map(func1, trap_x_temp)))
        E_c_per_scenario = np.squeeze(np.array(pool.map(func2, trap_x_temp)))
        pool.close()
        pool.join()

        # If there is only one candidate trap left, keep arrays 2D for downstream indexing
        if E_n_per_scenario.ndim == 1:
            E_n_per_scenario = E_n_per_scenario[np.newaxis, :]
        if E_c_per_scenario.ndim == 1:
            E_c_per_scenario = E_c_per_scenario[np.newaxis, :]

        E_r_per_scenario = E_c_per_scenario.copy()      # Compute E_r based on E_c
        min_n_r_per_scenario = np.zeros(E_r_per_scenario.shape)
        RSE_per_scenario = np.zeros(E_r_per_scenario.shape)

        for t in range(len(trap_indices)):
            for s in range(len(scenarios)):
                E_r_per_scenario[t, s] -= E_n_per_scenario[t, s]    # E_r = E_c - E_n, We set E_r to equal E_c above for ease of computation.
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
        print(f"Newly selected traps so far: {add_hist}")

    pbar.close()

    # Export n tracker file - probably not needed for webapp, but useful for my own debugging / tracking.
    os.makedirs('./secr/Forward Greedy/FG53', exist_ok=True)
    with open('./secr/Forward Greedy/FG53/n_tracker.txt', 'w') as f:
        for step_num, records in n_tracker.items():
            f.write(f"Step {step_num}:\n")
            for d, values in records.items():
                f.write(f"  Draw {d}: TrueN={values[0]}, EstN={values[1]}, Diff={values[2]}\n")

    selected_traps = [i for i, x in enumerate(trap_x) if int(x) == 1]
    print(f"Final selected traps (including deployed): {selected_traps}")
    return selected_traps, RSE_hist, trap_x


#################################################################################################################################
############                                        READ IN PARAMETERS                                               ############
#################################################################################################################################

# Read in True N file
true_n = pd.read_csv('./only_trail_1km/True_N_per_draw.csv')
draw_to_trueN = dict(zip(true_n['Parameter_draw'], true_n['N']))

# Read in parameter draws
params = pd.read_csv('./only_trail_1km/param_values_for_each_draw150.csv')
params = params.rename(columns={'Unnamed: 0': 'index'})

# Randomly select 10 draws
params = params.sample(n=N_PARAM_DRAWS, random_state=42).reset_index(drop=True)
print(f"Using parameter draws: {params['index'].tolist()}")

D = params['D'].values
g0 = params['g0'].values
sigma = params['sigma'].values
draw = params['index'].tolist()

# Read in potential AC locations
ac_coords = pyreadr.read_r('./only_trail_1km/mask_7-30-25.RDS')
ac_coords = ac_coords[None]
ac_coords_list = ac_coords[['x', 'y']].values
ac_coords_list = np.array(ac_coords_list)

# Load full possible trap grid
full_trap_grid = pd.read_csv('./only_trail_1km/500m/trail_candidate_traps_spacing500.csv')
full_trap_grid = full_trap_grid.drop(columns=['Unnamed: 0'])
full_trap_grid = full_trap_grid.rename(columns={'X': 'x', 'Y': 'y'})

# Remove trap locations which will never have a deployment -- not using here right now, so leaving commented out.
# excluded_ids = pd.read_csv('./500m_data/traps_to_remove_50m.csv')['TrapID']
# candidate_grid = full_trap_grid[~full_trap_grid['Trap_index'].isin(excluded_ids)].copy()

candidate_grid = full_trap_grid.copy()

# Load all previously deployed traps
deployed_df = full_trap_grid[full_trap_grid['Prior_cam'] == 1].copy()
deployed_df = deployed_df[['x', 'y', 'Trap_index', 'Prior_cam']]

# Filter out deployed traps from candidates
candidate_exclusive = candidate_grid[
    ~candidate_grid['Trap_index'].isin(deployed_df['Trap_index'])
][['x', 'y', 'Trap_index']]

# Combine deployed and candidate traps into one set of considered trap locations
combined_traps = pd.concat([deployed_df, candidate_exclusive], ignore_index=True)
trap_coords_list = combined_traps[['x', 'y']].values
print(f"Total traps including deployed + candidates: {len(combined_traps)}")

# Save considered trap locations - probably not needed for webapp, but useful for file tracking.
os.makedirs('./secr/Forward Greedy/FG53', exist_ok=True)
trap_coords_list_df = pd.DataFrame(trap_coords_list, columns=['x', 'y'])
trap_coords_list_df.to_csv('./secr/Forward Greedy/FG53/considered_trap_locs.csv', index=False)

# Indices of deployed traps are first rows of combined_traps
deployed_indices = np.arange(len(deployed_df))
print(f"Deployed traps found at rows: {deployed_indices}")

# Compute distances between all traps and all activity centers
traps = trap_coords_list[:, np.newaxis, :]
centers = ac_coords_list[np.newaxis, :, :]
differences = traps - centers
distances = np.linalg.norm(differences, axis=2)      # compute the EUCLIDEAN distances between all traps and all activity centers

scenarios = list(zip(D, g0, sigma, draw))

start_date = datetime.now()     # track runtimes (maybe not needed for the webapp)
start_time = time.time()


#################################################################################################################################
############                                        RUN FORWARD GREEDY                                               ############
#################################################################################################################################

# Run the forward greedy algorithm from the base deployment
selected_traps, RSE_hist, trap_x = forward_greedy_additions(
    scenarios=scenarios,
    trap_locs=trap_coords_list,
    ac_locs=ac_coords_list,
    K=K,
    distances=distances,
    draw=draw,
    draw_to_trueN=draw_to_trueN,
    deployed_indices=deployed_indices,
    budget=BUDGET
)


#################################################################################################################################
############                                           PROCESS RESULTS                                               ############
#################################################################################################################################

# log the ending time of Forward Greedy
end_date = datetime.now()
end_time = time.time()

# export the total runtime to a text file - Can skip for webapp.
with open('./secr/Forward Greedy/FG53/runtime.txt', 'w') as f:
    f.write(f"Start time: {start_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"End time: {end_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Start seconds: {start_time} seconds\n")
    f.write(f"End seconds: {end_time} seconds\n")
    f.write(f"Total runtime: {end_time - start_time} seconds\n")
print(f"Total runtime: {end_time - start_time} seconds")

# Save the results from greedy algorithm
np.save('./secr/Forward Greedy/FG53/all_selected_traps.npy', selected_traps)      # THIS IS THE IMPORTANT FILE OUTPUT TO VISUALIZE THE SELECTED TRAPS.

with open('./secr/Forward Greedy/FG53/rse_hist.txt', 'w') as f:
    for rse in RSE_hist:
        f.write(f"{rse}\n")

with open('./secr/Forward Greedy/FG53/considered_trap_locs.pkl', 'wb') as f:
    pickle.dump(trap_x, f)

print("Selected traps:", selected_traps)


#################################################################################################################################
############                                      GENERATE FILES FOR SECR                                            ############
#################################################################################################################################
"""
This section is probably not needed as we aren't using SECR in the webapp loop anywhere.
Can probably disregard this as it's just setting up file formatting to work with Arielle's SECR code.
"""

# Export selected trap info
selected_arr = np.array(selected_traps)
final_df = trap_coords_list_df.iloc[selected_arr].copy()
final_df['Trap_index'] = combined_traps.iloc[selected_arr]['Trap_index'].values
final_df.to_csv('./secr/Forward Greedy/FG53/selected_traps.csv', index=False)

# Verification
selected_trap_ids = set(final_df['Trap_index'].astype(int))
deployed_trap_ids = set(deployed_df['Trap_index'].astype(int).tolist())

missing = deployed_trap_ids - selected_trap_ids
extra = selected_trap_ids - deployed_trap_ids

assert len(missing) == 0, f"Missing deployed traps: {sorted(missing)}"
assert len(extra) == BUDGET, f"Expected {BUDGET} new traps added but found {len(extra)}: {sorted(extra)}"

print(f"All deployed traps are included.")
print(f"Exactly {BUDGET} new traps have been added.")

# Exclude from COMPLETE trap grid (not just candidate + deployed)
universe_ids = set(full_trap_grid['Trap_index'])
excluded_ids = sorted(universe_ids - selected_trap_ids)

with open('./secr/Forward Greedy/FG53/FG53-excluded_traps.txt', 'w') as f:
    for tid in excluded_ids:
        f.write(f"{tid}\n")

print(f"Excluded {len(excluded_ids)} traps out of {len(universe_ids)} total in grid.")
print(f"Selected {len(selected_trap_ids)} total traps.")