import numpy as np
import os
import pandas as pd
import pyreadr
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

def compute_expected_c(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    """
    Computes expected number of captures.
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

    density_array = np.array(density).flatten()
    
    # For trap locations that are not selected, set all capture probs to 0
    for t in range(len(trap_locs)):
        if int(trap_x[t]) == 0:
            prob_cap[t,...] = np.zeros(ac_locs.shape[0])
    
    # Compute the expected number of captures
    broadcast_density = np.broadcast_to(density_array, (len(trap_locs), len(density_array)))    # Repeats density to shape of (num_traps, num_activity_centers)
    expected_c = np.sum(prob_cap*broadcast_density)*K
    # print(f"Expected C: {expected_c}")
    return expected_c

def compute_expected_c_across_scenarios(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    nscenarios = len(g0)
    e_c = np.zeros((nscenarios,1))
    for s in range(nscenarios):
        e_c[s,0] = compute_expected_c(ac_locs, trap_locs, g0[s], sigma[s], K, density[s], distances, trap_x)
    return(e_c)

def backward_greedy(scenarios, trap_loc, centers, K, distances):
    # Initailize all potential trap locations to have a camera.
    trap_x = np.ones((len(trap_loc),))

    # Initailize parameter storage across scenarios
    D = []
    g0 = []
    sigma = []
    density_prior = []

    # Intiailize storage of loss functions
    E_n_curr = []
    E_c_curr = []
    E_r_curr = []
    RSE_curr = []
    RSE_hist = []

    # For each scenario, calculate the RSE in the event all potential trap locations are activated.
    for s in range(len(scenarios)):
        D.append(scenarios[s][0])
        g0.append(scenarios[s][1])
        sigma.append(scenarios[s][2])

        # Read in density prior file from /data/density if not using uniform density
        # density_prior_file =  f'500m_data/density/Dmod_draw_{scenarios[s][3]}.csv'
        density_prior_file =  f'500m_data/6-9 data/D_mod/Dmod_draw_{scenarios[s][3]}.csv'
        density_df = pd.read_csv(density_prior_file)
        # density_df['cell_density'] = density_df['cell_density'] * 25            # Multiply cell density value by cell area in hectares (25)
        # density_prior.append(density_df['cell_density'].values.tolist())        # List of Length # of potential activity centers
        density_df['D_mod'] = density_df['D_mod'] * 25            # Multiply cell density value by cell area in hectares (25)
        density_prior.append(density_df['D_mod'].values.tolist())        # List of Length # of potential activity centers

        # Compute RSE for the current scenario when all trap locations are activated
        E_n_curr.append(compute_expected_n(centers, trap_loc, g0[s], sigma[s], K, density_prior[s], distances, trap_x))
        E_c_curr.append(compute_expected_c(centers, trap_loc, g0[s], sigma[s], K, density_prior[s], distances, trap_x))
        E_r_curr.append(E_c_curr[s] - E_n_curr[s])
        RSE_curr.append(1/np.sqrt(min([E_n_curr[s], E_r_curr[s]])))
       
    # Average performance across all scenarios when all trap locations are activated
    RSE_hist.append(np.mean(RSE_curr))    
    remove_hist = []
    activated_trap_hist = []
    rse_tracker = {}
    n_tracker = {}

    # Begin backward greedy approach - removing 1 camera at a time
    counter = 0
    max_iters = int(sum(trap_x) - 1)
    pbar = tqdm(total=max_iters, desc="Backward Greedy Progress")
    while sum(trap_x) > 2:    # Set mininum number of cameras -- dependent on scenario
        pbar.update(1)
        counter += 1           # Tracks number of camera removals
        trap_indices = [i for i, x in enumerate(trap_x) if int(x) == 1]       # Locations where traps which are still activated
        activated_trap_hist.append(trap_indices)
        trap_x_temp = np.copy(np.broadcast_to(trap_x, (len(trap_indices),trap_x.shape[0])))    # 2D array of shape (num_activated_trap_locs, num_possible_trap_locs)

        # Temporarily remove each currently active location to 0 and simulate the performance
        for pos in range(len(trap_indices)):
            trap_idx = trap_indices[pos]
            trap_x_temp[pos, trap_idx] = 0 

        # Multiprocess the E_n and E_c calculations
        func1 = partial(compute_expected_n_across_scenarios, centers, trap_loc, g0, sigma, K, density_prior, distances)   # Setup all parameters but the activated trap locations
        func2 = partial(compute_expected_c_across_scenarios, centers, trap_loc, g0, sigma, K, density_prior, distances)
        pool = mp.Pool(min(mp.cpu_count(), 10))
        E_n_per_scenario = np.squeeze(np.array(pool.map(func1, trap_x_temp)))
        E_c_per_scenario = np.squeeze(np.array(pool.map(func2, trap_x_temp)))
        pool.close()
        pool.join()

        # Calculate E_r
        E_r_per_scenario = E_c_per_scenario                     # E_r = E_c_per_scenario - E_n_per_scenario            
        min_n_r_per_scenario = np.zeros(E_r_per_scenario.shape) 

        # Calculate RSE
        RSE_per_scenario = np.zeros(E_r_per_scenario.shape)
        for t in range(len(trap_indices)):
            for s in range(len(scenarios)):
                E_r_per_scenario[t,s] -= E_n_per_scenario[t,s]
                min_n_r_per_scenario[t,s] = min(E_n_per_scenario[t,s], E_r_per_scenario[t,s])
                RSE_per_scenario[t,s] = 1/np.sqrt(min_n_r_per_scenario[t,s])
        RSE_temp = np.mean(RSE_per_scenario, axis=1).tolist()

        # Calculate per-trap variances across scenarios & store in dictionary
        variances = np.var(RSE_per_scenario, axis=1)
        average_variance = variances.mean()
        print(f"Iteration {counter} average variance: {average_variance}")
        rse_tracker[counter] = {
            cam_id: RSE_temp[i] 
            for i, cam_id in enumerate(trap_indices)
        }

        # add E_n per scenario to the n tracker
        n_tracker[counter] = {
            cam_id: E_n_per_scenario[i, s] 
            for i, cam_id in enumerate(trap_indices)
            for s in range(len(scenarios))
        }

        # Select which trap to remove based on RSE
        min_change_idx = RSE_temp.index(min(RSE_temp)) 
        remove = trap_indices[min_change_idx]
        trap_x[remove] = 0

        # Track removal history
        RSE_hist.append(RSE_temp[min_change_idx])
        remove_hist.append(remove)
        print(remove, RSE_temp[min_change_idx])
    
    # Output final remaining traps
    remaining_traps = [i for i, x in enumerate(trap_x) if int(x) == 1]
    print(f"Final remaining traps: {remaining_traps}")

    # Write results to files
    with open('secr/Backward Greedy/BG10/activated_trap_hist.txt', 'w') as f:
        for item in activated_trap_hist:
            f.write("%s\n" % item)
    with open('secr/Backward Greedy/BG10/remove_hist.txt', 'w') as f:
        for item in remove_hist:
            f.write("%s\n" % item)
    with open('secr/Backward Greedy/BG10/rse_hist.txt', 'w') as f:
        for item in RSE_hist:
            f.write("%s\n" % item)
    with open('secr/Backward Greedy/BG10/rse_tracker.txt', 'w') as f:
        for key, value in rse_tracker.items():
            f.write(f"Iteration {key}: {value}\n")
    with open('secr/Backward Greedy/BG10/n_tracker.txt', 'w') as f:
        for key, value in n_tracker.items():
            f.write(f"Iteration {key}: {value}\n")

    pbar.close()
    
    return(remove_hist, RSE_hist, activated_trap_hist)

# Read in True N file
# true_n = pd.read_csv('./500m_data/True_N_per_draw.csv')
true_n = pd.read_csv('./500m_data/6-9 data/True_N_per_draw.csv')
true_n_filtered = true_n[true_n['N'].between(30, 80)]                    # Only consider scenarios with True N between 30 and 80
true_n_filtered = true_n_filtered['Parameter_draw'].values.tolist()      # Get the parameter draw IDS

# Read in parameter draws
# params = pd.read_csv('./500m_data/param_values_for_each_draw300.csv')
params = pd.read_csv('./500m_data/6-9 data/param_values_for_each_draw.csv')
params = params.rename(columns={'Unnamed: 0': 'index'})
params = params[params['index'].isin(true_n_filtered)]                   # Filter for parameter draws with True N between 30 and 80 
params = params.iloc[:25, :]                                             # Only keep the first ten draws for testing

# Extract parameter values
D = params['D'].values
g0 = params['g0'].values
sigma = params['sigma'].values
K= 5                                        # Number of sampling periods
draw = params['index'].values.tolist()      # Get the IDs of the parameter draws we are evaluating over
print(f"Parameter draws evaluated over: {draw}")

# Read in potential activity center locations
ac_coords = pyreadr.read_r('./500m_data/500m_mask.RDS')
ac_coords = ac_coords[None]
ac_coords_list = ac_coords[['x', 'y']].values.tolist()
ac_coords_list = np.array(ac_coords_list)

# Read in potential trap locations
trap_coords = pd.read_csv('./500m_data/500m_trap_grid.csv')
trap_coords = trap_coords.drop(columns = ['Unnamed: 0'])
# exclude_trap_coords = pd.read_csv('./500m_data/traps_to_remove_PLUS_2km_boundary.csv')            ## excludes those > 2km from a prior deployment            
# exclude_trap_coords = pd.read_csv('./500m_data/traps_to_remove_UTM10N_updated.csv')               ## excludes those that Robin said they would never travel      
# exclude_trap_coords = pd.read_csv('./500m_data/traps_to_remove_cost_500m.csv')                    ## excludes those that Robin said would never travel, and those that are > 500m from a trail
exclude_trap_coords = pd.read_csv('./500m_data/traps_to_remove_50m.csv')                            ## excludes those that Robin said would never travel, and those that are > 50m from a trail/road, and those with a relative cost > 0.3
trap_coords = trap_coords[~trap_coords['Trap_index'].isin(exclude_trap_coords['TrapID'])]        # Remove trap locations that are not accessible or > 2km away from prior deployment
trap_coords_list = []
for i in range(trap_coords.shape[0]):
    trap_coords_list.append((trap_coords['x'].iloc[i], trap_coords['y'].iloc[i]))
trap_coords_list = np.array(trap_coords_list)
print(f"{trap_coords_list.shape} candidate trap locations")

# Randomly select trap locations and activity centers -- Comment out when not testing
np.random.shuffle(trap_coords_list)              # Randomly Shuffle
# trap_coords_list = (trap_coords_list)[:750]      # select the first 750 locations
# trap_coords_list = np.load('./secr/New Density (x25)/200trap-2scenario/trap_coords_list.npy')         # If needed, upload an existing considered traps
np.save('./secr/Backward Greedy/BG10/considered_trap_locs.npy', trap_coords_list)
trap_coords_list_df = pd.DataFrame(trap_coords_list, columns=['x', 'y'])
trap_coords_list_df.to_csv('./secr/Backward Greedy/BG10/considered_trap_locs.csv', index=False)

# Calculate euclidean distances from traps to activity centers
traps = trap_coords_list[:, np.newaxis, :]  # Add a new axis to traps to make it 3D
centers = ac_coords_list[np.newaxis, :, :]  # Add a new axis to centers to make it 3D
differences = traps - centers
distances = np.linalg.norm(differences, axis=2)

scenarios = list(zip(*[D, g0, sigma, draw]))

# log the starting time of backward greedy
start_date = datetime.now()
start_time = time.time()

# Run backward greedy algorithm
backward_greedy(scenarios, trap_coords_list, ac_coords_list, K, distances)

# log the ending time of backward greedy
end_time = time.time()
end_date = datetime.now()

# export the total runtime to a text file
with open('secr/Backward Greedy/BG10/runtime.txt', 'w') as f:
    # write the start time and date
    f.write(f"Start time: {start_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"End time: {end_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    # write the total runtime in seconds
    f.write(f"Start seconds: {start_time} seconds\n")
    f.write(f"End seconds: {end_time} seconds\n")
    f.write(f"Total runtime: {end_time - start_time} seconds\n")
print(f"Total runtime: {end_time - start_time} seconds")

# Generate Files for SECR Analysis
trap_coords_list = np.load('secr/Backward Greedy/BG10/considered_trap_locs.npy')     # read in the traps considered in the algorithm run

with open('secr/Backward Greedy/BG10/activated_trap_hist.txt', 'r') as f:
    lines = f.readlines()

# Find the first line with exactly 60 elements
selected_line = None
for line in lines:
    # Clean and validate the line
    cleaned = line.strip().replace('[', '').replace(']', '').replace(' ', '')
    if cleaned:  # Skip empty lines
        elements = cleaned.split(',')
        if len(elements) == 60:
            selected_line = cleaned
            break  # Remove this line if you want the LAST occurrence instead

if not selected_line:
    raise ValueError("No line with 60 elements found in the file")

# convert string to list of integers
selected_traps = [int(x) for x in selected_line.split(',')]

# # Convert selected_traps to a numpy array and sort it
selected_traps = np.array(selected_traps)
selected_traps = np.sort(selected_traps)

# subset trap_coords_list to include ONLY the selected trap indices
trap_coords_list_sub = trap_coords_list[selected_traps]
trap_coords_list_sub_df = pd.DataFrame(trap_coords_list_sub, columns=['x', 'y'])

# get all the potential trap coordinates
trap_coords = pd.read_csv('500m_data/500m_trap_grid.csv')
trap_coords = trap_coords.rename(columns={'X': 'x', 'Y': 'y'})
trap_coords = trap_coords.drop(columns=['Unnamed: 0'])

# get the x,y coords and index of the traps that were selected
trap_coords_list_sub_df = trap_coords_list_sub_df.merge(trap_coords, on=['x', 'y'], how='left')
trap_coords_list_sub_df.to_csv('secr/Backward Greedy/BG10/selected_traps.csv', index=False)
trap_coords_list_sub_df

# CORRECTED LOGIC: Find excluded trap IDs using set operations on Trap_index values
# Get the Trap_index values for selected traps
selected_trap_ids = set(trap_coords_list_sub_df['Trap_index'])

# Get all Trap_index values from the full grid
all_trap_ids = set(trap_coords['Trap_index'])

# Excluded trap IDs are those in the full grid but NOT in selected_trap_ids
excluded_trap_ids = sorted(all_trap_ids - selected_trap_ids)

# convert to txt file
with open('secr/Backward Greedy/BG10/BG10-excluded_traps.txt', 'w') as f:
    for item in excluded_trap_ids:
        f.write("%s\n" % item)

print(f"Selected {len(selected_traps)} traps")
print(f"Excluded {len(excluded_trap_ids)} traps")
print(f"Total traps in full grid: {len(all_trap_ids)}")
print(f"Verification: {len(selected_trap_ids) + len(excluded_trap_ids)} should equal {len(all_trap_ids)}")
