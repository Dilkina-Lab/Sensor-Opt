import argparse
import copy
import csv
from functools import partial
import itertools
import multiprocessing as mp
import numpy as np
import os
import pickle
import rasterio
from scipy.optimize import minimize
import sys
from utils import compute_expected_n, compute_expected_c, compute_scr_neg_log_likelihood, callback
# from utils import find_lcp_to_pts


def compute_expected_n_across_scenarios(raster, raster_cell_size, trap_locs, alpha0, alpha1, alpha2, K, density, trap_x):
    nscenarios = len(alpha0)
    e_n = np.zeros((nscenarios,1))
    for s in range(nscenarios):
        e_n[s,0] = compute_expected_n(raster[s], raster_cell_size[s], trap_locs, alpha0[s], alpha1[s], alpha2[s], K, density[s], trap_x)
    return(e_n)

def compute_expected_c_across_scenarios(raster, raster_cell_size, trap_locs, alpha0, alpha1, alpha2, K, density, trap_x):
    nscenarios = len(alpha0)
    e_c = np.zeros((nscenarios,1))
    for s in range(nscenarios):
        e_c[s,0] = compute_expected_c(raster[s], raster_cell_size[s], trap_locs, alpha0[s], alpha1[s], alpha2[s], K, density[s], trap_x)
    return(e_c)

def compute_rel_err_Nhat_across_scenarios(raster, raster_cell_size, trap_locs, alpha0, alpha1, alpha2, K, N, trap_x):
    print('num trap locs', len(trap_locs))
    nscenarios = len(alpha0)
    rel_err_Nhat = np.zeros((nscenarios, 1))

    # Remove traps that have been excluded
    trap_locs_copy = copy.deepcopy(trap_locs)
    trap_ind_to_remove = [i for i,v in enumerate(trap_x) if trap_x[v] == 0]
    for t_ind in sorted(trap_ind_to_remove, reverse=True):
        del trap_locs_copy[t_ind]

    for s in range(nscenarios):#TODO the scenarios don't match up any more; need to pass scenario number; load a random one for now
        sno = np.random.randint(0,81)

        # Load capture history
        capture_history_path = '../data/simulation/capture_histories/%s_%d_0%s.pkl'%('lowfrag_train', sno, '_grid_14_14')
        capture_histories = pickle.load(open(capture_history_path, 'rb'))
        detections = capture_histories[0] # only use one capture history realization for now
        # Remove detections from excluded traps
        detections = np.delete(detections, trap_ind_to_remove, axis=1) # reuse trap_ind_to_remove from above
        # Remove rows for individuals that are never detected in the included traps
        detections = detections[~np.all(detections==0, axis=1)]

        param_init = [np.log(alpha0[s]), np.log(alpha1[s]), np.log(alpha2[s]), np.log(N[s]-len(detections))]
        # print('param_init')
        # print(param_init)
        scr_args = (raster[s], raster_cell_size[s], trap_locs_copy, detections, K)
        # print('trap_locs')
        # print(trap_locs)
        result = minimize(compute_scr_neg_log_likelihood, x0 = param_init, args = scr_args, method='Nelder-Mead', options={'maxiter': 100, 'disp': True})
        rel_err_Nhat[s,0] = np.abs(N[s] - (len(detections) + np.exp(result.x[3])))/N[s]
        print(rel_err_Nhat[s,0])
    return(rel_err_Nhat)

def backward_greedy(scenarios, trap_loc, K, budget):
    trap_x = np.ones((len(trap_loc),))

    landscape_ndarr = []
    raster_cell_size = []
    ALPHA0 = []
    ALPHA1 = []
    ALPHA2 = []
    density_prior = []
    E_n_curr = []
    E_c_curr = []
    E_r_curr = []
    RSE_curr = []
    RSE_hist = []
    for s in range(len(scenarios)):
        landscape_ndarr.append(scenarios[s][0])
        raster_cell_size.append(scenarios[s][1])
        ALPHA0.append(scenarios[s][2])
        ALPHA1.append(scenarios[s][3])
        ALPHA2.append(scenarios[s][4])
        density_prior.append(scenarios[s][5])
        E_n_curr.append(compute_expected_n(landscape_ndarr[s], raster_cell_size[s], trap_loc, ALPHA0[s], ALPHA1[s], ALPHA2[s], K, density_prior[s], trap_x))
        E_c_curr.append(compute_expected_c(landscape_ndarr[s], raster_cell_size[s], trap_loc, ALPHA0[s], ALPHA1[s], ALPHA2[s], K, density_prior[s], trap_x))
        E_r_curr.append(E_c_curr[s] - E_n_curr[s])
        RSE_curr.append(1/np.sqrt(min([E_n_curr[s], E_r_curr[s]])))
    RSE_hist.append(np.mean(RSE_curr))
    # scenario = scenarios[0]
    # landscape_ndarr = scenario[0]
    # raster_cell_size = scenario[1]
    # ALPHA0 = scenario[2]
    # ALPHA1 = scenario[3]
    # ALPHA2 = scenario[4]
    # density_prior = scenario[5]
    # E_n_curr = compute_expected_n(landscape_ndarr, raster_cell_size, trap_loc, ALPHA0, ALPHA1, ALPHA2, K, density_prior, trap_x)
    # E_c_curr = compute_expected_c(landscape_ndarr, raster_cell_size, trap_loc, ALPHA0, ALPHA1, ALPHA2, K, density_prior, trap_x)
    # E_r_curr = E_c_curr - E_n_curr
    # RSE_curr = 1/np.sqrt(min([E_n_curr, E_r_curr]))
    # RSE_hist = [RSE_curr]
    remove_hist = []
    print('Computed RSE for each scenario with all possible trap locations included.')

    counter = 0
    while sum(trap_x) > budget:
        counter += 1
        print(counter)
        trap_indices = [i for i, x in enumerate(trap_x) if int(x) == 1]
        # trap_indices = trap_indices[0:10]# truncate early while testing
        E = np.zeros((len(trap_indices), 2))
        trap_x_temp = np.copy(np.broadcast_to(trap_x, (len(trap_indices),trap_x.shape[0])))
        for pos in range(len(trap_indices)):
            trap_idx = trap_indices[pos]
            trap_x_temp[pos, trap_idx] = 0
        func1 = partial(compute_expected_n_across_scenarios, landscape_ndarr, raster_cell_size, trap_loc, ALPHA0, ALPHA1, ALPHA2, K, density_prior)
        func2 = partial(compute_expected_c_across_scenarios, landscape_ndarr, raster_cell_size, trap_loc, ALPHA0, ALPHA1, ALPHA2, K, density_prior)
        pool = mp.Pool(min(mp.cpu_count(), 10))
        E_n_per_scenario = np.squeeze(np.array(pool.map(func1, trap_x_temp)))
        # E[...,0] = np.mean(E_n_per_scenario, axis=1)
        E_c_per_scenario = np.squeeze(np.array(pool.map(func2, trap_x_temp)))
        pool.close()
        pool.join()
        E_r_per_scenario = E_c_per_scenario
        min_n_r_per_scenario = np.zeros(E_r_per_scenario.shape)
        RSE_per_scenario = np.zeros(E_r_per_scenario.shape)
        for t in range(len(trap_indices)):
            for s in range(len(scenarios)):
                E_r_per_scenario[t,s] -= E_n_per_scenario[t,s]
                min_n_r_per_scenario[t,s] = min(E_n_per_scenario[t,s], E_r_per_scenario[t,s])
                RSE_per_scenario[t,s] = 1/np.sqrt(min_n_r_per_scenario[t,s])
        # E[...,1] = np.mean(E_r_per_scenario, axis=1)
        # min_n_r = np.min(E, axis=1).tolist()
        RSE_temp = np.mean(RSE_per_scenario, axis=1).tolist()
        # RSE_temp = [1/np.sqrt(i) for i in min_n_r]
        # print(RSE_temp)
        # assert 2==3, 'break now'
        min_change_idx = RSE_temp.index(min(RSE_temp))
        remove = trap_indices[min_change_idx]
        trap_x[remove] = 0
        RSE_hist.append(RSE_temp[min_change_idx])
        remove_hist.append(remove)
        print(remove, RSE_temp[min_change_idx])
    
    return(remove_hist, RSE_hist)

def backward_greedy_Nhat_via_simulation(scenarios, trap_loc, K, budget):
    trap_x = np.ones((len(trap_loc),))
    
    landscape_ndarr = []
    raster_cell_size = []
    ALPHA0 = []
    ALPHA1 = []
    ALPHA2 = []
    N = []
    rel_bias_Nhat_hist = []
    rel_bias_Nhat_curr = []
    for s in range(len(scenarios)):
    #     print('scenario %d'%s)
        landscape_ndarr.append(scenarios[s][0])
        raster_cell_size.append(scenarios[s][1])
        ALPHA0.append(scenarios[s][2])
        ALPHA1.append(scenarios[s][3])
        ALPHA2.append(scenarios[s][4])
        N.append(scenarios[s][5])
        
    #     # Load capture history
    #     capture_history_path = '../data/simulation/capture_histories/%s_%d_0%s.pkl'%('lowfrag_train', s, '_grid_14_14')
    #     capture_histories = pickle.load(open(capture_history_path, 'rb'))
    #     detections = capture_histories[0] # only use one capture history realization for now
    #     # if args.traps_to_remove is not None:
    #     #     detections = np.delete(detections, trap_ind_to_remove, axis=1) # reuse trap_ind_to_remove from above
        
    #     param_init = [np.log(scenarios[s][2]), np.log(scenarios[s][3]), np.log(scenarios[s][4]), np.log(scenarios[s][5]-len(detections))]
    #     scr_args = (scenarios[s][0], scenarios[s][1], trap_loc, detections, K)
    #     result = minimize(compute_scr_neg_log_likelihood, x0 = param_init, args = scr_args, method='Nelder-Mead', options={'maxiter': 100, 'disp': True}, callback=callback)
    #     rel_bias_Nhat_curr.append(np.abs(scenarios[s][5] - (len(detections) + np.exp(result.x[3])))/scenarios[s][5])
    # rel_bias_Nhat_hist.append(np.mean(rel_bias_Nhat_curr))
    # print('Computed relative absolute bias for each scenario with all possible trap locations included.')

    remove_hist = []
    counter = 0
    while sum(trap_x) > budget:
        counter += 1
        print(counter)
        trap_indices = [i for i, x in enumerate(trap_x) if int(x) == 1]
        trap_x_temp = np.copy(np.broadcast_to(trap_x, (len(trap_indices),trap_x.shape[0]))).astype(int)
        for pos in range(len(trap_indices)):
            trap_idx = trap_indices[pos]
            trap_x_temp[pos, trap_idx] = 0

        results_for_all_scenarios = []
        for pos in range(len(trap_indices)):
            results_for_all_scenarios.append(compute_rel_err_Nhat_across_scenarios(landscape_ndarr, raster_cell_size, trap_loc, ALPHA0, ALPHA1, ALPHA2, K, N, trap_x_temp[pos]))
        rel_err_Nhat_per_scenario = np.squeeze(np.array(results_for_all_scenarios))
        # func1 = partial(compute_rel_err_Nhat_across_scenarios, landscape_ndarr, raster_cell_size, trap_loc, ALPHA0, ALPHA1, ALPHA2, K, N)
        # pool = mp.Pool(min(mp.cpu_count(), 10))
        # rel_err_Nhat_per_scenario = np.squeeze(np.array(pool.map(func1, trap_x_temp)))
        # pool.close()
        # pool.join()
        
        rel_err_Nhat_temp = np.mean(rel_err_Nhat_per_scenario, axis=1).tolist()
        min_change_idx = rel_err_Nhat_temp.index(min(rel_err_Nhat_temp))
        remove = trap_indices[min_change_idx]
        trap_x[remove] = 0
        remove_hist.append(remove)
        rel_bias_Nhat_hist.append(rel_err_Nhat_temp[min_change_idx])
        
        print(remove, rel_err_Nhat_temp[min_change_idx])
    
    return(remove_hist, rel_bias_Nhat_hist)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_prefix', type=str, required=True, help='Prefix to configuration files for multiple scenarios of SCR simulation settings.')
    parser.add_argument('--trap_layout_file', required=True, help='Path to trap layout file for candidate trap locations.')
    parser.add_argument('--K', required=True, type=int, help='Number of sampling occasions.')
    # parser.add_argument('--ac_realization_no', type=int, required=True, help='Integer index for activity center realization number.')
    args = parser.parse_args()
    
    budget = 50

    log_file_name = '../logs/%s_%s_%d.log'%(args.config_file_prefix, 'greedy_backward_Nhatsim', budget)
    sys.stdout = open(log_file_name, 'w')


    # Get list of train scenario config files
    scenario_config_files = []
    config_dir_path = '../data/simulation/config/'
    for f in os.listdir(config_dir_path):
        if f.startswith(args.config_file_prefix):
            scenario_config_files.append(config_dir_path+f)
    
    # Get configuration parameters for each scenario
    scenario_config_params_dict = dict()
    for scf in scenario_config_files:
        config_file_name = scf.split('/')[-1].split('.csv')[0]
        scenario_num = int(config_file_name.split('_')[-1])
        config_params = {}
        with open(scf, 'r') as cf:
            paramreader = csv.reader(cf)
            for line in paramreader:
                if line[0] in ['landscape_raster_file']:
                    config_params[line[0]] = line[1]
                elif line[0] in ['alpha0', 'alpha1', 'alpha2', 'raster_cell_size']:
                    config_params[line[0]] = float(line[1])
                else:
                    config_params[line[0]] = int(line[1])
        scenario_config_params_dict[scenario_num] = config_params
    scenario_config_params = [scenario_config_params_dict[k] for k in sorted(scenario_config_params_dict.keys())]
    random_subset = np.random.permutation(range(len(scenario_config_params)))[0:1]
    scenario_config_params = [scenario_config_params[i] for i in random_subset]
    
    # For each variable create list of values in each scenario 
    landscape_raster_file = [scenario_config_params[s]['landscape_raster_file'] for s in range(len(scenario_config_params))]
    landscape_raster_name = [landscape_raster_file[s].split('/')[-1].split('.tif')[0] for s in range(len(scenario_config_params))]
    raster_cell_size = [scenario_config_params[s]['raster_cell_size'] for s in range(len(scenario_config_params))]
    trap_loc_path = [args.trap_layout_file for s in range(len(scenario_config_params))]
    ALPHA0 = [scenario_config_params[s]['alpha0'] for s in range(len(scenario_config_params))]
    ALPHA1 = [scenario_config_params[s]['alpha1'] for s in range(len(scenario_config_params))]
    ALPHA2 = [scenario_config_params[s]['alpha2'] for s in range(len(scenario_config_params))]
    N = [scenario_config_params[s]['N'] for s in range(len(scenario_config_params))]
    K = [args.K for s in range(len(scenario_config_params))]
    

    landscape_ndarr = []
    trap_loc = []
    for s in range(len(scenario_config_params)):
        landscape_raster = rasterio.open(landscape_raster_file[s])
        scenario_landscape_ndarr = np.squeeze(np.array(landscape_raster.read()))
        landscape_ndarr.append(scenario_landscape_ndarr)
        scenario_trap_loc = [(eval(line[0]), eval(line[1])) for line in csv.reader(open(trap_loc_path[s], 'r'))]
        trap_loc.append(scenario_trap_loc)
    
    # density prior
    # uniform:
    density_prior = []
    for s in range(len(scenario_config_params)):
        density_prior.append(np.ones((landscape_ndarr[s].shape[0], landscape_ndarr[s].shape[1]))*(N[s]/float(landscape_ndarr[s].size)))     ### We will not know what N is in practice - 
    # true realized density:
    # ac_loc_path = '../data/simulation/activity_centers/%s_%d.csv'%(config_file_name, args.ac_realization_no)
    # ac = [(eval(p[0]), eval(p[1])) for p in list(csv.reader(open(ac_loc_path, 'r')))]
    # density_prior = np.zeros((landscape_ndarr.shape[0], landscape_ndarr.shape[1]))*(N/float(landscape_ndarr.size))
    # for p in ac:
    #     sx = int(np.floor(p[0]))
    #     sy = int(np.floor(p[1]))
    #     density_prior[sx, sy] += 1
    
    # scenarios = list(zip(*[landscape_ndarr, raster_cell_size, ALPHA0, ALPHA1, ALPHA2, density_prior])) # for RSE(n0)
    scenarios = list(zip(*[landscape_ndarr, raster_cell_size, ALPHA0, ALPHA1, ALPHA2, N])) # for relative bias Nhat
    trap_loc = trap_loc[0] # trap locations are the same for all scenarios
    K = K[0] # K is the same for all scenarios
    # removed_traps, RSE_trace = backward_greedy(scenarios, trap_loc, K, budget) # for RSE(n0)
    removed_traps, RSE_trace = backward_greedy_Nhat_via_simulation(scenarios, trap_loc, K, budget)
    pickle.dump([removed_traps, RSE_trace], open('backward_greedy_Nhatsim_budget_%d_results.pkl'%budget, 'wb'))

if __name__ == '__main__':
    main()
