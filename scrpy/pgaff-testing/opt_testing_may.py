import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy as sp
import pyreadr
import sys
sys.path.append(os.path.abspath('../src/'))
import argparse
from functools import partial
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy import stats
import sys
from tqdm import tqdm
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
        # prob_cap = get_prob_cap_filename(s)
        # print("Got prob_cap")
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
    return expected_c

def compute_expected_c_across_scenarios(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    nscenarios = len(g0)
    e_c = np.zeros((nscenarios,1))
    for s in range(nscenarios):
        # prob_cap = get_prob_cap_filename(s)
        # print("Got prob_cap")
        e_c[s,0] = compute_expected_c(ac_locs, trap_locs, g0[s], sigma[s], K, density[s], distances, trap_x)
    return(e_c)

# def get_prob_cap_filename(s):
#     start = (s // 10) * 10 + 1
#     end = start + 9
#     filename = f'500m_data/prob_cap/prob_cap_{start}_to_{end}.txt.npy'
#     return np.load(filename, mmap_mode='r').copy()     # Read-only memory mapping to avoid loading the entire file into memory

def backward_greedy(scenarios, trap_loc, centers, K, distances):
    # Initailize all potential trap locations to have a camera.
    trap_x = np.ones((len(trap_loc),))

    # Initailize parameter storage across scenarios
    D = []
    g0 = []
    sigma = []
    density_prior = []
    # alpha1 = []

    # Initialize temporary storage of capture probabilities
    # prob_cap_temp = []
    # prior_s = 1
    # batch_counter = 0

    # Intiailize storage of loss functions
    E_n_curr = []
    E_c_curr = []
    E_r_curr = []
    RSE_curr = []
    RSE_hist = []

    # For each scenario, calculate the RSE in the event all potential trap locations are activated.
    for s in range(len(scenarios)):
        # s_counter = s+1
        # print(s)
        D.append(scenarios[s][0])
        g0.append(scenarios[s][1])
        sigma.append(scenarios[s][2])
        # density_prior.append(np.ones((centers.shape[0]))*(47/float(centers.shape[0])))      # List of Length # of potential activity centers
        # Read in density prior file from /data/density if not using uniform density
        density_prior_file =  f'500m_data/density/Dmod_draw_{s+1}.csv'
        density_df = pd.read_csv(density_prior_file)
        density_prior.append(density_df['cell_density'].values.tolist())  # List of Length # of potential activity centers

        # alpha1.append(1/(2*sigma[s]*sigma[s]))
        
        # Write large prob_cap arrays to disk
        # prob_cap_temp.append((g0[s])*np.exp(-alpha1[s]*(distances**2)))      # Array of size (# traps, # poential activity centers)
        
        # Compute RSE for the current scenario when all trap locations are activated
        E_n_curr.append(compute_expected_n(centers, trap_loc, g0[s], sigma[s], K, density_prior[s], distances, trap_x))
        # E_n_curr = compute_expected_n(centers, trap_loc, g0[s], sigma[s], K, density_prior[s], prob_cap_temp[s%10], trap_x)
        # with open('500m_data/e_n.txt', 'a') as f:
        #     f.write(f'{E_n_curr}\n')
        E_c_curr.append(compute_expected_c(centers, trap_loc, g0[s], sigma[s], K, density_prior[s], distances, trap_x))
        # E_c_curr = compute_expected_c(centers, trap_loc, g0[s], sigma[s], K, density_prior[s], prob_cap_temp[s%10], trap_x)
        # with open('500m_data/e_c.txt', 'a') as f:
        #     f.write(f'{E_c_curr}\n')
        E_r_curr.append(E_c_curr[s] - E_n_curr[s])
        # E_r_curr = E_c_curr - E_n_curr
        # with open('500m_data/e_r.txt', 'a') as f:
        #     f.write(f'{E_r_curr}\n')
        RSE_curr.append(1/np.sqrt(min([E_n_curr[s], E_r_curr[s]])))
        # RSE_curr = 1/np.sqrt(min([E_n_curr, E_r_curr]))
        # with open('500m_data/rse.txt', 'a') as f:
        #     f.write(f'{RSE_curr}\n')

        # if s_counter% 10 == 0:
        #     np.save(f'500m_data/prob_cap/prob_cap_{prior_s}_to_{s_counter}.txt', prob_cap_temp)
        #     prior_s = s_counter+1
        #     prob_cap_temp = []
        #     batch_counter = 0

    # Average performance across all scenarios when all trap locations are activated
    # print("Computing Avg RSE Across Scenarios")
    RSE_hist.append(np.mean(RSE_curr))
    # RSE_hist = np.mean(RSE_curr)
    # with open('500m_data/rse_hist.txt', 'a') as f:
    #     f.write(f'{RSE_hist}\n')
    remove_hist = []
    activated_trap_hist = []
    rse_tracker = {}

    # Begin backward greedy approach - removing 1 camera at a time
    counter = 0
    max_iters = int(sum(trap_x) - 60)
    pbar = tqdm(total=max_iters, desc="Backward Greedy Progress")
    while sum(trap_x) > 60:    # Set mininum number of cameras -- dependent on scenario
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
        # print(f"Multiprocessing E_n and E_c for {len(trap_indices)} traps")
        func1 = partial(compute_expected_n_across_scenarios, centers, trap_loc, g0, sigma, K, density_prior, distances)   # Setup all parameters but the activated trap locations
        func2 = partial(compute_expected_c_across_scenarios, centers, trap_loc, g0, sigma, K, density_prior, distances)
        pool = mp.Pool(min(mp.cpu_count(), 6))
        E_n_per_scenario = np.squeeze(np.array(pool.map(func1, trap_x_temp)))
        E_c_per_scenario = np.squeeze(np.array(pool.map(func2, trap_x_temp)))
        pool.close()
        pool.join()

        # Calculate RSE
        # print(f"Calculating RSE for {len(trap_indices)} traps")
        E_r_per_scenario = E_c_per_scenario
        min_n_r_per_scenario = np.zeros(E_r_per_scenario.shape)
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

        # Select which trap to remove based on RSE
        min_change_idx = RSE_temp.index(min(RSE_temp)) 
        remove = trap_indices[min_change_idx]
        trap_x[remove] = 0

        # Track removal history
        RSE_hist.append(RSE_temp[min_change_idx])
        remove_hist.append(remove)
        print(remove, RSE_temp[min_change_idx])

    # Write results to files
    with open('500m_data/activated_trap_hist.txt', 'w') as f:
        for item in activated_trap_hist:
            f.write("%s\n" % item)
    with open('500m_data/remove_hist.txt', 'w') as f:
        for item in remove_hist:
            f.write("%s\n" % item)
    with open('500m_data/rse_hist.txt', 'w') as f:
        for item in RSE_hist:
            f.write("%s\n" % item)
    with open('500m_data/rse_tracker.txt', 'w') as f:
        for key, value in rse_tracker.items():
            f.write(f"Iteration {key}: {value}\n")

    pbar.close()
    
    return(remove_hist, RSE_hist, activated_trap_hist)

# Read in parameter draws
params = pd.read_csv('./500m_data/param_values_for_each_draw300.csv')
params = params.rename(columns={'Unnamed: 0': 'index'})

# FOR TESTING - Only use those with an index between 1 and 100
params = params[params['index'].between(1, 100)]

# Extract parameter values
D = params['D'].values
g0 = params['g0'].values
sigma = params['sigma'].values
K= 5    # Number of sampling periods

# Read in potential activity center locations
ac_coords = pyreadr.read_r('./500m_data/500m_mask.RDS')
ac_coords = ac_coords[None]
ac_coords_list = ac_coords[['x', 'y']].values.tolist()
ac_coords_list = np.array(ac_coords_list)

# Read in potential trap locations
trap_coords = pd.read_csv('./500m_data/500m_trap_grid.csv')
trap_coords = trap_coords.drop(columns = ['Unnamed: 0'])
trap_coords = trap_coords.rename(columns = {'X': 'x', 'Y': 'y'})
trap_coords_list = []
for i in range(trap_coords.shape[0]):
    trap_coords_list.append((trap_coords['x'][i], trap_coords['y'][i]))
trap_coords_list = np.array(trap_coords_list)

# Randomly select 1000 trap locations and activity centers -- Comment out when not testing
# np.random.seed(42)
# np.random.shuffle(ac_coords_list)
# ac_coords_list = (ac_coords_list)[:100]
np.random.shuffle(trap_coords_list)
trap_coords_list = (trap_coords_list)[:200]
np.save('./500m_data/trap_coords_list.npy', trap_coords_list)

# Calculate euclidean distances from traps to activity centers
traps = trap_coords_list[:, np.newaxis, :]  # Add a new axis to traps to make it 3D
centers = ac_coords_list[np.newaxis, :, :]  # Add a new axis to centers to make it 3D
differences = traps - centers
distances = np.linalg.norm(differences, axis=2)

scenarios = list(zip(*[D, g0, sigma]))

backward_greedy(scenarios, trap_coords_list, ac_coords_list, K, distances)