import argparse
import csv
import numpy as np
import pickle
import rasterio
from scipy.optimize import minimize
import sys
import time


from utils import find_lcp_to_pts, compute_scr_neg_log_likelihood, callback
from visualize import visualize_trap_layout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='Path to configuration file for SCR simulation settings.')
    parser.add_argument('--ac_realization_no', type=int, required=True, help='Integer index for activity center realization number.')
    parser.add_argument('--trap_layout_file', required=True, help='Path to trap layout file.')
    parser.add_argument('--K', required=True, type=int, help='Number of sampling occasions.')
    parser.add_argument('--caphist_realization_no', type=int, required=True, help='Integer index for capture history realization number.')
    parser.add_argument('--traps_to_remove', default=None, help='Pickled object containing list of trap indices in order of removal from backwards greedy.')
    parser.add_argument('--num_traps_to_remove', default=None, type=int, help='Remove traps_to_remove in order until a specified number have been excluded.')
    args = parser.parse_args()
    
    # Read in configuration parameters
    config_params = {}
    with open(args.config_file, 'r') as cf:
        paramreader = csv.reader(cf)
        for line in paramreader:
            if line[0] in ['landscape_raster_file', 'trap_config']:
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
    if args.traps_to_remove is None:
        log_file_name = '../logs/%s_%d%s_%d.log'%(config_file_name, args.ac_realization_no, trap_layout_name, args.caphist_realization_no)
    else:
        log_file_name = '../logs/%s_%d%s_%d_remove_%d.log'%(config_file_name, args.ac_realization_no, trap_layout_name, args.caphist_realization_no, args.num_traps_to_remove)
    sys.stdout = open(log_file_name, 'w')
    
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
    
    # Load capture history
    capture_history_path = '../data/simulation/capture_histories/%s_%d%s.pkl'%(config_file_name, args.ac_realization_no, trap_layout_name)
    capture_histories = pickle.load(open(capture_history_path, 'rb'))
    detections = capture_histories[args.caphist_realization_no]
    if args.traps_to_remove is not None:
        detections = np.delete(detections, trap_ind_to_remove, axis=1) # reuse trap_ind_to_remove from above
    
    # Minimize
    param_init = [np.log(alpha0), np.log(alpha1), np.log(alpha2), np.log(N-len(detections))]
    print('Initial parameter values:\nalpha0: %0.4f\nalpha1: %0.4f\nalpha2: %0.4f\nN: %d'%(alpha0, alpha1, alpha2, N))
    scr_args = (landscape_ndarr, raster_cell_size, trap_loc, detections, args.K)

    initial_neg_log_likelihood = compute_scr_neg_log_likelihood(param_init, *scr_args)
    print('Initial negative log likelihood: %0.6f'%initial_neg_log_likelihood)
    
    timer = time.time()
    print('Starting minimization...')
    result = minimize(compute_scr_neg_log_likelihood, x0 = param_init, args = scr_args, method='BFGS', options={'maxiter': 100, 'disp': True}, callback=callback)
    print('Stopped minimizing after %0.4f seconds'%(time.time() - timer))
    print('Minimization status code: %s'%result.status)
    
    alpha0_hat = np.exp(result.x[0])
    alpha1_hat = np.exp(result.x[1])
    alpha2_hat = np.exp(result.x[2])
    N_hat = len(detections) + np.exp(result.x[3])
    neg_log_likelihood = result.fun
    print('Final negative log likelihood: %0.6f'%neg_log_likelihood)
    print('MLE parameter values:\nalpha0_hat: %0.4f\nalpha1_hat: %0.4f\nalpha2_hat: %0.4f\nN_hat: %0.4f'%(alpha0_hat, alpha1_hat, alpha2_hat, N_hat))

    stderr_log_n0 = np.sqrt(result.hess_inv[3,3])
    print('SE(log n0): %0.4f'%stderr_log_n0)
    rel_stderr_log_n0 = stderr_log_n0/result.x[3]
    print('RSE(log n0): %0.4f'%rel_stderr_log_n0)
    
    stderr_n0 = np.sqrt(np.exp(2*result.x[3])*result.hess_inv[3,3])
    print('SE(n0): %0.4f'%stderr_n0)
    rel_stderr_n0 = stderr_n0/np.exp(result.x[3])
    print('RSE(n0): %0.4f'%rel_stderr_n0)

    # # < CONS BIO DATA > #
    # landscape_raster_file = '../data/simulation/covariate_rasters/lowfrag_covariate.tif'
    # landscape_raster_name = landscape_raster_file.split('/')[-1].split('.tif')[0]
    # landscape_raster = rasterio.open(landscape_raster_file)
    # landscape_ndarr = np.squeeze(np.array(landscape_raster.read()))
    
    # ## set ground truth landscape parameters
    # ALPHA0 = 2
    # ALPHA1 = 1/(2*0.3851879*0.3851879)
    # ALPHA2 = 2.25
    # N = 100
    # print('Ground truth:\nalpha0: %d\nalpha1: %0.4f\nalpha2: %0.2f\nN: %d'%(ALPHA0, ALPHA1, ALPHA2, N))
    # print('------------------------------')

    # ## load trap locations
    # trap_loc_path = '../data/simulation/trap_locations/lowfrag_covariate_grid_14_14_consbio.csv'
    # trap_loc = [(10*(eval(line[0])-0.5), 10*(eval(line[1])-0.5)) for line in csv.reader(open(trap_loc_path, 'r'))]
    # # sort the order
    # trap_loc = sorted(trap_loc, key=lambda element:(element[1], element[1]))

    # ## compute least cost paths through this landscape
    # lcp_distances = find_lcp_to_pts(landscape_ndarr, ALPHA2, trap_loc, raster_cell_size=0.1)

    # ## load capture history
    # capture_history_path = '../data/simulation/capture_histories/lowfrag_N100_a2225_consbio.csv'
    # detections = np.array([row for row in csv.reader(open(capture_history_path, 'r'))]).astype(np.float)
    # # < \CONS BIO DATA > #

    # # ------------------------------ #
    # #  < TEST CODE AREA >
    # est_density_uniform = 100/float(landscape_ndarr.size)*np.ones((landscape_ndarr.shape[0], landscape_ndarr.shape[1]))
    # e_n = compute_expected_n(landscape_ndarr, 0.1, trap_loc, ALPHA0, ALPHA1, ALPHA2, 10, est_density_uniform)
    # e_c = compute_expected_c(landscape_ndarr, 0.1, trap_loc, ALPHA0, ALPHA1, ALPHA2, 10, est_density_uniform)
    # e_r = e_c - e_n
    # print(e_c, e_n, e_r)
    # print(sum(sum(detections)), len(detections), sum(sum(detections)) - len(detections))
    # assert 2==3, 'end test code'
    # #  < \TEST CODE AREA >
    # # ------------------------------ #

    

if __name__ == '__main__':
    main()
    # cProfile.run('main()')