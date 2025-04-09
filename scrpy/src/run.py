import argparse
import csv
import numpy as np
import pickle
import rasterio
from scipy.optimize import minimize
import sys
import time


from utils import find_lcp_to_pts, compute_scr_neg_log_likelihood, callback, compute_expected_n, compute_expected_c
from visualize import visualize_trap_layout, visualize_activity_centers, visualize_activity_centers_and_trap_layout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='Path to configuration file for SCR simulation settings.')
    parser.add_argument('--ac_realization_no', type=int, required=True, help='Integer index for activity center realization number.')
    parser.add_argument('--caphist_realization_no', type=int, required=True, help='Integer index for capture history realization number.')
    parser.add_argument('--trap_layout_file', required=True, help='Path to trap layout file.')
    parser.add_argument('--K', required=True, type=int, help='Number of sampling occasions.')
    parser.add_argument('--traps_to_remove', default=None, help='Pickled object containing list of trap indices in order of removal from backwards greedy.')
    parser.add_argument('--num_traps_to_remove', default=None, type=int, help='Remove traps_to_remove in order until a specified number have been excluded.')
    args = parser.parse_args()
    config_file_name = args.config_file.split('/')[-1].split('.csv')[0]
    # log_file_name = '../logs/%s_%d_%d.log'%(config_file_name, args.ac_realization_no, args.caphist_realization_no)

    # Read in configuration parameters
    config_params = {}
    with open(args.config_file, 'r') as cf:
        paramreader = csv.reader(cf)
        for line in paramreader:
            if line[0] in ['landscape_raster_file']:
                config_params[line[0]] = line[1]
            elif line[0] in ['alpha0', 'alpha1', 'alpha2', 'raster_cell_size']:
                config_params[line[0]] = float(line[1])
            else:
                config_params[line[0]] = int(line[1])
    
    # Load into variables
    landscape_raster_file = config_params['landscape_raster_file']
    landscape_raster_name = landscape_raster_file.split('/')[-1].split('.tif')[0]
    trap_layout_name = args.trap_layout_file.split('/')[-1].split(landscape_raster_name)[-1].split('.csv')[0]
    raster_cell_size = config_params['raster_cell_size']
    alpha0 = config_params['alpha0']
    alpha1 = config_params['alpha1']
    alpha2 = config_params['alpha2']
    N = config_params['N']

    config_file_name = args.config_file.split('/')[-1].split('.csv')[0]
    
    print('Ground truth parameter values:\nalpha0: %0.4f\nalpha1: %0.4f\nalpha2: %0.4f\nN: %d'%(alpha0, alpha1, alpha2, N))

    # Load landscape and trap locations
    landscape_raster = rasterio.open(landscape_raster_file)
    landscape_ndarr = np.squeeze(np.array(landscape_raster.read()))
    trap_loc = [(eval(line[0]), eval(line[1])) for line in csv.reader(open(args.trap_layout_file, 'r'))]
    if args.traps_to_remove is not None:
        trap_removal_results = pickle.load(open(args.traps_to_remove, 'rb'))
        removed_traps_trace = trap_removal_results
        trap_ind_to_remove = removed_traps_trace[0:args.num_traps_to_remove]
        for t_ind in sorted(trap_ind_to_remove, reverse=True):
            del trap_loc[t_ind]
    visualize_trap_layout(landscape_ndarr, trap_loc, display=False)
    
    # density prior
    # density_prior = np.ones((landscape_ndarr.shape[0], landscape_ndarr.shape[1]))*(N/float(landscape_ndarr.size))
    ac_loc_path = '../data/simulation/activity_centers/%s_%d.csv'%(config_file_name, args.ac_realization_no)
    ac = [(eval(p[0]), eval(p[1])) for p in list(csv.reader(open(ac_loc_path, 'r')))]
    visualize_activity_centers(landscape_ndarr, ac, display=False)
    visualize_activity_centers_and_trap_layout(landscape_ndarr, ac, trap_loc, display=False)
    assert 2==3, 'break'
    
    density_prior = np.zeros((landscape_ndarr.shape[0], landscape_ndarr.shape[1]))*(N/float(landscape_ndarr.size))
    for p in ac:
        sx = int(np.floor(p[0]))
        sy = int(np.floor(p[1]))
        density_prior[sx, sy] += 1
    print('Expected num unique animals detected and total captures:')
    E_n = compute_expected_n(landscape_ndarr, raster_cell_size, trap_loc, ALPHA0, ALPHA1, ALPHA2, K, density_prior, np.ones((len(trap_loc),)))
    print(E_n)
    E_c = compute_expected_c(landscape_ndarr, raster_cell_size, trap_loc, ALPHA0, ALPHA1, ALPHA2, K, density_prior, np.ones((len(trap_loc),)))
    print(E_c)
    E_r = E_c - E_n
    # print(E_r)
    print('---')

    # Load capture histories
    capture_history_path = '../data/simulation/capture_histories/%s_%d.pkl'%(config_file_name, args.ac_realization_no)
    capture_histories = pickle.load(open(capture_history_path, 'rb'))
    n = []
    c = []
    for ch_rno in range(100):
        detections = capture_histories[ch_rno]
        n.append(len(detections))
        c.append(sum(sum(detections)))
    print('Average num unique animals detected and total captures:')
    print(np.mean(n))
    print(np.mean(c))
    print('---')

    RSE_est = 1/np.sqrt(min([E_n, E_r]))
    RSE_est_2 = np.sqrt(1/min([E_n, E_r]) - 1/sum(sum(density_prior)))
    print(RSE_est, RSE_est_2)

    # Load capture history
    # capture_history_path = '../data/simulation/capture_histories/%s_%d.pkl'%(config_file_name, args.ac_realization_no)
    # capture_histories = pickle.load(open(capture_history_path, 'rb'))
    detections = capture_histories[args.caphist_realization_no]
    
    # Minimize
    param_init = [ALPHA0, np.log(ALPHA1), np.log(ALPHA2), np.log(N-len(detections))]
    print('Initial parameter values:\nalpha0: %d\nalpha1: %0.4f\nalpha2: %0.2f\nN: %d'%(ALPHA0, ALPHA1, ALPHA2, N))
    scr_args = (landscape_ndarr, raster_cell_size, trap_loc, detections, 10)

    initial_neg_log_likelihood = compute_scr_neg_log_likelihood(param_init, *scr_args)
    print('Initial negative log likelihood: %0.6f'%initial_neg_log_likelihood)
    
    timer = time.time()
    print('Starting minimization...')
    result = minimize(compute_scr_neg_log_likelihood, x0 = param_init, args = scr_args, method='BFGS', options={'maxiter': 100, 'disp': True})
    print('Stopped minimizing after %0.4f seconds'%(time.time() - timer))
    stderr_log_n0 = np.sqrt(result.hess_inv[3,3])
    std_err_n0 = np.sqrt(np.exp(2*result.x[3])*result.hess_inv[3,3])
    rel_stderr_log_n0 = stderr_log_n0/result.x[3]
    rel_std_err_n0 = std_err_n0/np.exp(result.x[3])
    print(stderr_log_n0, rel_stderr_log_n0, std_err_n0, rel_std_err_n0)
    # print('hess_inv:')
    # print(np.exp(result.x[3]), np.sqrt(result.hess_inv[3,3])/result.x[3])
if __name__ == '__main__':
    main()
    # cProfile.run('main()')