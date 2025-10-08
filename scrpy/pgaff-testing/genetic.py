import numpy as np
import pandas as pd
import pyreadr
import os
import random
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# --- Your Existing Utility Functions (Compute Expected N, C, etc.) ---

def compute_expected_n(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    alpha1 = (1/(2*sigma*sigma))
    prob_cap = ((g0)*np.exp(-alpha1*(distances**2)))  
    for t in range(len(trap_locs)): 
        if int(trap_x[t]) == 0:
            prob_cap[t,...] = np.zeros(ac_locs.shape[0])
    i_cap_hist = np.squeeze(np.zeros((len(trap_locs), 1)))  
    p_empty_cap_hist = compute_cond_lik_ind(len(ac_locs), prob_cap, K, len(trap_locs), i_cap_hist)
    p_nonempty = 1 - p_empty_cap_hist
    expected_n = np.sum(p_nonempty*density)
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
    e_n = np.zeros((nscenarios,1))
    for s in range(nscenarios):
        e_n[s,0] = compute_expected_n(ac_locs, trap_locs, g0[s], sigma[s], K, density[s], distances, trap_x)
    return(e_n)

def compute_expected_c(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    alpha1 = (1/(2*sigma*sigma))
    prob_cap = ((g0)*np.exp(-alpha1*(distances**2)))
    density_array = np.array(density).flatten()
    for t in range(len(trap_locs)):
        if int(trap_x[t]) == 0:
            prob_cap[t,...] = np.zeros(ac_locs.shape[0])
    broadcast_density = np.broadcast_to(density_array, (len(trap_locs), len(density_array)))
    expected_c = np.sum(prob_cap*broadcast_density)*K
    return expected_c

def compute_expected_c_across_scenarios(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    nscenarios = len(g0)
    e_c = np.zeros((nscenarios,1))
    for s in range(nscenarios):
        e_c[s,0] = compute_expected_c(ac_locs, trap_locs, g0[s], sigma[s], K, density[s], distances, trap_x)
    return(e_c)


# --- Genetic Algorithm Functions adapted for RSE optimization ---

def compute_rse_across_scenarios(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    e_n = compute_expected_n_across_scenarios(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x)
    e_c = compute_expected_c_across_scenarios(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x)
    e_r = e_c - e_n
    rse = 1 / np.sqrt(np.minimum(e_n, e_r))
    return np.mean(rse)  # Mean RSE

def fitness_function_rse(trap_x, ac_locs, trap_locs, g0, sigma, K, density, distances):
    # Minimize RSE => maximize negative RSE
    return -compute_rse_across_scenarios(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x)

def initialize_population(pop_size, num_traps, n_active):
    population = []
    for _ in range(pop_size):
        indiv = np.zeros(num_traps, dtype=int)
        active_indices = np.random.choice(num_traps, n_active, replace=False)
        indiv[active_indices] = 1
        population.append(indiv)
    return population

def crossover(parent1, parent2):
    child = np.copy(parent1)
    mask = np.random.rand(len(parent1)) > 0.5
    child[mask] = parent2[mask]
    return child

def mutate(indiv, mutation_rate=0.01):
    for i in range(len(indiv)):
        if random.random() < mutation_rate:
            indiv[i] = 1 - indiv[i]
    return indiv

def evaluate_population(population, ac_locs, trap_locs, g0, sigma, K, density, distances):
    compute_func = partial(fitness_function_rse, ac_locs=ac_locs, trap_locs=trap_locs,
                           g0=g0, sigma=sigma, K=K, density=density, distances=distances)
    with mp.Pool(min(mp.cpu_count(), 10)) as pool:
        fitness_vals = pool.map(compute_func, population)
    return fitness_vals

def genetic_algorithm(ac_locs, trap_locs, g0, sigma, K, density, distances,
                      pop_size=50, n_active=50, generations=100, mutation_rate=0.01):
    num_traps = len(trap_locs)
    population = initialize_population(pop_size, num_traps, n_active)
    best_indiv = None
    best_fitness = float('-inf')
    
    for gen in tqdm(range(generations), desc="Genetic Algorithm Progress"):
        fitnesses = evaluate_population(population, ac_locs, trap_locs, g0, sigma, K, density, distances)
        
        max_idx = np.argmax(fitnesses)
        if fitnesses[max_idx] > best_fitness:
            best_fitness = fitnesses[max_idx]
            best_indiv = population[max_idx]
        print(f"Generation {gen}: Best fitness (negative mean RSE) = {best_fitness}, Mean RSE = {-best_fitness}")
        
        new_population = []
        for _ in range(pop_size):
            contenders = random.sample(list(zip(population, fitnesses)), 3)
            parent1 = max(contenders, key=lambda x: x[1])[0]
            contenders = random.sample(list(zip(population, fitnesses)), 3)
            parent2 = max(contenders, key=lambda x: x[1])[0]
            
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            
            # Enforce fixed number of active traps
            active_count = child.sum()
            if active_count > n_active:
                on_indices = np.where(child == 1)[0]
                to_off = np.random.choice(on_indices, active_count - n_active, replace=False)
                child[to_off] = 0
            elif active_count < n_active:
                off_indices = np.where(child == 0)[0]
                to_on = np.random.choice(off_indices, n_active - active_count, replace=False)
                child[to_on] = 1

            new_population.append(child)
        population = new_population
    return best_indiv, -best_fitness



# --- Parameter loading & data preparation ---

params = pd.read_csv('./full_grid_1km/10-3 data (Marten)/param_values_for_each_draw300_marten.csv')
params = params.rename(columns={'Unnamed: 0': 'index'})
params = params.iloc[:10, :]    # First 10 draws for testing

D = params['D'].values
g0 = params['g0'].values
sigma = params['sigma'].values
K = 5
draw = params['index'].values.tolist()
print(f"Parameter draws evaluated over: {draw}")

ac_coords = pyreadr.read_r('./full_grid_1km/10-3 data (Marten)/500m_mask_marten.RDS')
ac_coords = ac_coords[None]
ac_coords_list = ac_coords[['x', 'y']].values.tolist()
ac_coords_list = np.array(ac_coords_list)

trap_coords = pd.read_csv('./full_grid_1km/10-3 data (Marten)/1000m_trap_grid_marten.csv')
trap_coords = trap_coords.drop(columns = ['Unnamed: 0'])
exclude_trap_coords = pd.read_csv('./full_grid_1km/traps_to_remove_1km.csv')
trap_coords = trap_coords[~trap_coords['Trap_index'].isin(exclude_trap_coords['Trap_index'])]

trap_coords_list = []
for i in range(trap_coords.shape[0]):
    trap_coords_list.append((trap_coords['x'].iloc[i], trap_coords['y'].iloc[i]))
trap_coords_list = np.array(trap_coords_list)

print(f"{trap_coords_list.shape} candidate trap locations")

traps = trap_coords_list[:, np.newaxis, :]
centers = ac_coords_list[np.newaxis, :, :]
differences = traps - centers
distances = np.linalg.norm(differences, axis=2)

scenarios = list(zip(*[D, g0, sigma, draw]))


# --- Run Genetic Algorithm with RSE fitness ---

best_trap_config, best_rse = genetic_algorithm(
    ac_coords_list,
    trap_coords_list,
    g0,
    sigma,
    K,
    D,
    distances,
    pop_size=100,
    n_active=50,     # set desired number of traps to select
    generations=200,
    mutation_rate=0.02
)

print(f"Best mean RSE found: {best_rse}")
selected_traps_final = np.where(best_trap_config == 1)[0].tolist()
print(f"Selected trap indices for deployment: {selected_traps_final}")

