"""
simulate_scr.py

Simulation code for conducting spatial capture-recapture experiments in silico.

"""
import argparse
import cProfile
import csv
from functools import partial
import itertools
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import rasterio
from scipy import stats
from scipy.optimize import minimize
# from scipy.special import binom
from skimage import graph
import sys
import time
import tqdm
# from rasterio.plot import show
#import rpy2.robjects as robjects
#from rpy2.robjects.packages import importr

class Scenario:
    """
    A Scenario defines a ground-truth set of values.
    """
    def __init__(self, landscape, alpha0, alpha1, alpha2, beta0, beta1):
        self.alpha0 = alpha0                # Probability animal is captured in that pixel
        self.alpha1 = alpha1                # Animal's budget for movement
        self.alpha2 = alpha2                # Scales effects of the movement costs.
        self.beta0 = beta0                  # *** Aren't betas only needed for the simulation? ***
        self.beta1 = beta1
        self.lcp_distances_to_points = None
    


def find_lcp(raster, alpha2, raster_cell_size=1):
    """
    Computes least cost path lengths between every pair of cells in a raster.
    """
    def get_cell_cost(x):
        return np.exp(x*alpha2)             # C(x,y) = exp(alpha2 * z(x,y)) -- z(x,y) is the habitat suitability at cell (x,y). The higher the value of z(x,y), the LOWER the suitability.
    get_all_cell_costs = np.vectorize(get_cell_cost)
    cost_raster = get_all_cell_costs(raster)
    lcp_graph = graph.MCP_Geometric(cost_raster, fully_connected=True)  # *** Why is this step necessary? For simulation? ***
    
    lcp_distances = np.zeros((raster.shape[0], raster.shape[1], raster.shape[0], raster.shape[1]))
    # pool = mp.Pool(mp.cpu_count())
    for x_start in range(raster.shape[0]):
        # results = pool.starmap(lcp_graph.find_costs, [(x_start, y_start) for y_start in range(raster.shape[1])])
        # print(len(results))
        # assert 2==3, 'break break'
        for y_start in range(raster.shape[1]):
            distances = lcp_graph.find_costs([(x_start, y_start)])[0]
            for x_end in range(raster.shape[0]):
                for y_end in range(raster.shape[1]):
                    lcp_distances[x_start, y_start, x_end, y_end] = distances[x_end, y_end]*raster_cell_size
    return lcp_distances

def find_lcp_to_pts(raster, alpha2, ref_pts, raster_cell_size=1):               # *** What are ref_pts? The activity centers? Seems like these are potential trap locations***
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

# def binom_logpmf(N, k, p):
#     prob = binom(int(N), int(k))*math.pow(p,k)*math.pow((1-p),(N-k))
#     # binom_rv = stats.binom(N,p)
#     # assert prob == binom_rv.pmf(k), (prob, binom_rv.pmf(k))
#     if prob == 0: # log(0) = -inf, which will make any sum of logs = -inf
#         return -sys.maxsize - 1
#     return np.log(prob)

def compute_cond_lik_ind(est_prob_cap, K, num_traps, raster, ind_cap_hist):
    """
    Computes the likelihood of an individual's capture history conditional on their activity center location.
    """
    broadcast_i_cap_hist = np.broadcast_to(ind_cap_hist[..., np.newaxis, np.newaxis], (num_traps, raster.shape[0], raster.shape[1]))
    probs = stats.binom.pmf(broadcast_i_cap_hist, K, est_prob_cap)
    zero_mask = probs == 0.0
    log_probs = np.log(probs, where=np.invert(zero_mask))
    log_probs[zero_mask] = -sys.maxsize - 1
    log_cond_lik_sums = np.sum(log_probs, axis=0)
    return np.exp(log_cond_lik_sums)


def compute_scr_neg_log_likelihood(params, *args, parallelize=True):
    """
    Computes the log-likelihood of observed capture histories, given parameter values.                     
    """
    # timer = time.time()                                                       ### *** Are these log-liklihoods something that are provided to us by the simulaiton team? ***

    alpha0 = np.exp(params[0])  # pass log(alpha0) in as parameter
    alpha1 = np.exp(params[1])  # pass log(alpha1) in as parameter
    alpha2 = np.exp(params[2])  # pass log(alpha2) in as parameter
    n0 = np.exp(params[3])      # pass log(n0) in as parameter

    raster = args[0]
    raster_cell_size = args[1]
    tl = args[2]
    cap_hists = args[3]
    K = args[4]

    num_traps = len(tl)
    num_detected_ind = len(cap_hists)
    
    # ------------------------------------------------------------------------------------------#
    # estimate the least cost path distances using current parameter values
    # ------------------------------------------------------------------------------------------#
    est_lcp_distances = find_lcp_to_pts(raster, alpha2, tl, raster_cell_size=raster_cell_size)

    # ------------------------------------------------------------------------------------------#
    # compute the capture probability p(i, x, y) at each trap i of individuals from pixel (x,y)
    # ------------------------------------------------------------------------------------------#
    est_prob_cap = (1/(1+np.exp(alpha0)))*np.exp(-alpha1*(est_lcp_distances**2))

    # ------------------------------------------------------------------------------------------#
    # compute conditional likelihood of each individual's capture history + undetected individuals' capture history
    # ------------------------------------------------------------------------------------------#    
    conditional_likelihoods = np.ndarray((num_detected_ind+1, raster.shape[0], raster.shape[1]))
    if parallelize:
        func = partial(compute_cond_lik_ind, est_prob_cap, K, num_traps, raster)
        pool = mp.Pool(min(mp.cpu_count(), 10))
        conditional_likelihoods[0:num_detected_ind,...] = np.array(pool.map(func, cap_hists))
        pool.close()
        pool.join()
        i_cap_hist = np.squeeze(np.zeros((len(tl), 1)))
        conditional_likelihoods[-1, ...] = compute_cond_lik_ind(est_prob_cap, K, num_traps, raster, i_cap_hist)
    else:
        for ni in range(num_detected_ind):
            i_cap_hist = cap_hists[ni]
            conditional_likelihoods[ni,...] = compute_cond_lik_ind(est_prob_cap, K, num_traps, raster, i_cap_hist)
        i_cap_hist = np.squeeze(np.zeros((len(tl), 1)))
        conditional_likelihoods[-1, ...] = compute_cond_lik_ind(est_prob_cap, K, num_traps, raster, i_cap_hist)        

    # ------------------------------------------------------------------------------------------#
    # compute marginal likelihood of each individual's capture history + undetected individuals' capture history
    # ------------------------------------------------------------------------------------------#
    marginal_likelihoods = np.ndarray((1, num_detected_ind+1))
    for ni in range(num_detected_ind):
        marginal_likelihoods[0, ni] = sum([(1/float(raster.size))*conditional_likelihoods[ni, cx, cy] for cx in range(raster.shape[0]) for cy in range(raster.shape[1])])
    marginal_likelihoods[0, -1] = sum([(1/float(raster.size))*conditional_likelihoods[-1, cx, cy] for cx in range(raster.shape[0]) for cy in range(raster.shape[1])])
    
    
    ## compute log likelihood
    nv = np.ones((1, len(cap_hists)+1))
    nv[0, -1] = n0
    part1 = math.lgamma(len(cap_hists)+n0+1) - math.lgamma(n0+1)
    part2 = sum(np.multiply(nv[0,], np.log(marginal_likelihoods[0,])))

    # print('Computing log likelihood took %0.4f seconds'%(time.time() - timer))
    
    return(-1*(part1 + part2))

def callback(x):
    print(x)

def compute_expected_n(raster, raster_cell_size, trap_locs, alpha0, alpha1, alpha2, K, density, trap_x):
    """
    Computes expected number of unique individuals detected in a spatial capture-recapture study.
    """
    lcp_distances = find_lcp_to_pts(raster, alpha2, trap_locs, raster_cell_size=raster_cell_size)
    prob_cap = (1/(1+np.exp(alpha0)))*np.exp(-alpha1*(lcp_distances**2))
    for t in range(len(trap_locs)): # for trap locations that are not selected, set all capture probs to 0
        if int(trap_x[t]) == 0:
            prob_cap[t,...] = np.zeros((raster.shape[0], raster.shape[1]))
    i_cap_hist = np.squeeze(np.zeros((len(trap_locs), 1))) # empty capture history
    p_empty_cap_hist = compute_cond_lik_ind(prob_cap, K, len(trap_locs), raster, i_cap_hist)
    p_nonempty = 1 - p_empty_cap_hist
    expected_n = sum(sum(p_nonempty*density))
    return expected_n

def compute_expected_c(raster, raster_cell_size, trap_locs, alpha0, alpha1, alpha2, K, density, trap_x):
    """
    Computes expected number of captures.
    """
    lcp_distances = find_lcp_to_pts(raster, alpha2, trap_locs, raster_cell_size=raster_cell_size)
    prob_cap = (1/(1+np.exp(alpha0)))*np.exp(-alpha1*(lcp_distances**2))
    for t in range(len(trap_locs)): # for trap locations that are not selected, set all capture probs to 0
        if int(trap_x[t]) == 0:
            prob_cap[t,...] = np.zeros((raster.shape[0], raster.shape[1]))
    broadcast_density = np.broadcast_to(density, (len(trap_locs), density.shape[0], density.shape[1]))
    expected_c = sum(sum(sum(prob_cap*broadcast_density)))*K
    return expected_c
