# Import Libraries
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import re
import scipy as sp
from functools import partial
import itertools
import multiprocessing as mp
import pyreadr
import sys
sys.path.append(os.path.abspath('../src/'))
from visualize import *
from utils import *
import argparse
import copy


def build_raster():
    """
    Takes x,y coordinates and builds a raster from them.   
    """
    # Load the grid information from data/grid/, reading in the file with 'mask' in the name
    grid_dir = './data/grid/'
    grid_files = os.listdir(grid_dir)
    grid_files = [f for f in grid_files if 'mask' in f]
    grid = pyreadr.read_r(grid_dir + grid_files[0])[None]
    grid

    pixel_size = 100  ## Represents dense grid of 100m
    x_min, y_max = grid['x'].min(), grid['y'].max()
    width = int((grid['x'].max() - x_min) / pixel_size) + 1
    height = int((y_max - grid['y'].min()) / pixel_size) + 1

    # Create an empty raster
    raster = np.zeros((height, width))

    # Create the transform
    transform = from_origin(x_min, y_max, pixel_size, pixel_size)

    # Rasterize the points
    for _, row in grid.iterrows():
        col = int((row['x'] - x_min) / pixel_size)
        row_idx = int((y_max - row['y']) / pixel_size)
        if 0 <= col < width and 0 <= row_idx < height:
            # Assign a constant value (e.g., 1) or increment for counts
            raster[row_idx, col] = 1  # Increment for counts

    output_folder = 'data/raster'
    output_file = f"{grid_files[0].replace('.RDS', '')}.tif"
    os.makedirs(output_folder, exist_ok=True)

    # Write the raster to a file
    with rasterio.open(output_file, 'w', driver='GTiff', height=height, width=width,
                    count=1, dtype=raster.dtype, crs='+proj=utm +zone=33 +datum=WGS84',
                    transform=transform) as dst:
        dst.write(raster, 1)

    return raster



def convert_dense_to_sigma(raster, sigma):
    """
    Takes the 100m x 100m dense grid and converts it into a sigma-spaced grid. 
    """    
    current_spacing = 100
    
    step = int(sigma/current_spacing)

    x_unique = sorted(grid['x'].unique())
    y_unique = sorted(grid['y'].unique())

    x_subsampled = x_unique[::step]
    y_subsampled = y_unique[::step]

    subsampled_grid = pd.DataGrame([(x, y) for x in x_subsampled for y in y_subsampled], columns=['x', 'y'])

    subsampled_grid.to_csv(f'data/grid/subsampled_grid_{sigma}.csv', index=False)



def find_lcp_to_pts(raster, alpha2, ref_pts, raster_cell_size=1):
    """
    Computes least cost path lengths from every cell in a raster to each of a list of reference points.  
    """
    def get_cell_cost(x):
        return np.exp(x*alpha2)
    get_all_cell_costs = np.vectorize(get_cell_cost)
    cost_raster = get_all_cell_costs(raster)
    lcp_graph = graph.MCP_Geometric(cost_raster, fully_connected=True)

    lcp_distances = np.zeros((len(ref_pts), raster.shape[0], raster.shape[1]))
    for rpidx in range(len(ref_pts)):
        rp = ref_pts[rpidx]
        rp_int = (int(np.floor(rp[0])), int(np.floor(rp[1])))
        distances = lcp_graph.find_costs([rp_int])[0]
        lcp_distances[rpidx, ...] = distances*raster_cell_size

    return lcp_distances


def find_euclid_dist(raster, ref_pts, raster_cell_size):
    """
    Computes euclidean distance from every cell in a raster to each of a list of reference points.
    """
    euclid_distances = np.zeros((len(ref_pts), raster.shape[0], raster.shape[1]))
    for rpidx in range(len(ref_pts)):
        rp = ref_pts[rpidx]
        for i in range(raster.shape[0]):
            for j in range(raster.shape[1]):
                euclid_distances[rpidx, i, j] = np.sqrt((i-rp[0])**2 + (j-rp[1])**2)*raster_cell_size
    return euclid_distances


def compute_expected_n(raster, raster_cell_size, dist = ['lcp','euclid'], trap_locs, g0, sigma, K, alpha2 = None, density, trap_x):         # If using lcp distances, pass in an alpha2 value.
    """
    Computes expected number of unique individuals detected in a spatial capture-recapture study.
    """
    alpha1 = 1/(2*sigma**2)

    # Compute Distances
    if dist == 'lcp':
        distances = find_lcp_to_pts(raster, alpha2, trap_locs, raster_cell_size=raster_cell_size)
    elif dist == 'euclid':
        distances = find_euclid_dist(raster, trap_locs, raster_cell_size=raster_cell_size)

    # Probability animal with given activity center is captured at each trap
    prob_cap = (1/(1+np.exp(g0)))*np.exp(-alpha1*(distances**2))

    for t in range(len(trap_locs)): # for trap locations that are not selected, set all capture probs to 0
        if int(trap_x[t]) == 0:
            prob_cap[t,...] = np.zeros((raster.shape[0], raster.shape[1]))
    i_cap_hist = np.squeeze(np.zeros((len(trap_locs), 1))) # empty capture history
    p_empty_cap_hist = compute_cond_lik_ind(prob_cap, K, len(trap_locs), raster, i_cap_hist)
    p_nonempty = 1 - p_empty_cap_hist
    expected_n = sum(sum(p_nonempty*density))
    return expected_n

def compute_expected_n_across_scenarios(raster, raster_cell_size, dist = ['lcp','eucld'], trap_locs, g0, sigma, K, alpha2 = None, density, trap_x):
    nscenarios = len(g0)
    e_n = np.zeros((nscenarios,1))
    for s in range(nscenarios):
        e_n[s,0] = compute_expected_n(raster[s], raster_cell_size[s], trap_locs, g0[s], sigma[s], alpha2 = None, K, density[s], trap_x)
    return(e_n)


def compute_expected_c(raster, raster_cell_size, dist = ['lcp', 'euclid'], trap_locs, g0, sigma, K, alpha2 = None, density, trap_x):
    """
    Computes expected number of captures.
    """
    alpha1 = 1/(2*sigma**2)

    # Compute Distances
    if dist == 'lcp':
        distances = find_lcp_to_pts(raster, alpha2, trap_locs, raster_cell_size=raster_cell_size)
    elif dist == 'euclid':
        distances = find_euclid_dist(raster, trap_locs, raster_cell_size=raster_cell_size)   
    
    prob_cap = (1/(1+np.exp(g0)))*np.exp(-alpha1*(distances**2))

    for t in range(len(trap_locs)): # for trap locations that are not selected, set all capture probs to 0
        if int(trap_x[t]) == 0:
            prob_cap[t,...] = np.zeros((raster.shape[0], raster.shape[1]))
    broadcast_density = np.broadcast_to(density, (len(trap_locs), density.shape[0], density.shape[1]))
    expected_c = sum(sum(sum(prob_cap*broadcast_density)))*K
    return expected_c

def compute_expected_c_across_scenarios(raster, raster_cell_size, trap_locs, alpha0, alpha1, alpha2, K, density, trap_x):
    nscenarios = len(alpha0)
    e_c = np.zeros((nscenarios,1))
    for s in range(nscenarios):
        e_c[s,0] = compute_expected_c(raster[s], raster_cell_size[s], dist = ['lcp','euclid'], trap_locs, g0[s], sigma[s], alpha2[s] = None, K, density[s], trap_x)
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
        capture_histories = 
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
