"""
This module provides functions for simulating spatial capture-recapture studies.
"""

import numpy as np
from skimage import graph
from typing import List, Tuple

def find_lcp_to_pts(
    raster: np.ndarray,
    alpha2: float,
    ref_pts: List[Tuple[float, float]],
    raster_cell_size: int = 1,
) -> np.ndarray:
    """
    Computes least cost path lengths from every cell in a raster to each of a list of reference points.

    Args:
        raster (np.ndarray): The raster over which to compute the least cost paths.
        alpha2 (float): Parameter for the cost function.
        ref_pts (List[Tuple[float, float]]): List of reference points (x, y) to which the least cost paths are computed.
        raster_cell_size (int, optional): Size of each raster cell. Defaults to 1.

    Returns:
        np.ndarray: A 3D array where each slice along the first dimension represents the least cost path distances
        from every cell in the raster to one of the reference points.
    """
    def get_cell_cost(x: float) -> float:
        return np.exp(x * alpha2)

    get_all_cell_costs = np.vectorize(get_cell_cost)
    cost_raster = get_all_cell_costs(raster)
    lcp_graph = graph.MCP_Geometric(cost_raster, fully_connected=True)

    lcp_distances = np.zeros((len(ref_pts), raster.shape[0], raster.shape[1]))
    for rpidx, rp in enumerate(ref_pts):
        rp_int = (int(np.floor(rp[0])), int(np.floor(rp[1])))
        distances = lcp_graph.find_costs([rp_int])[0]
        lcp_distances[rpidx, ...] = distances * raster_cell_size

    return lcp_distances

def simulate_capture_histories(
    ac: List[Tuple[float, float]],
    tl: List[Tuple[float, float]],
    K: int,
    lcp_dist: np.ndarray,
    alpha0: float,
    alpha1: float,
    n_real: int = 50,
    seed: int = 1111,
) -> List[np.ndarray]:
    """
    Simulates a capture history for a simulated spatial capture-recapture study.

    Assumes multiple individuals can be detected at each detector in a given sampling
    occasion--but multiple detections of the same individual at the same trap in a
    single sampling occasion are indistinguishable.

    Args:
        ac (List[Tuple[float, float]]): List of tuples (x, y) storing activity center locations.
        tl (List[Tuple[float, float]]): List of tuples (x, y) storing trap locations.
        K (int): Number of sampling occasions.
        lcp_dist (np.ndarray): Least cost paths between each trap and every raster pixel.
        alpha0 (float): Capture probability parameter.
        alpha1 (float): Home range parameter.
        n_real (int, optional): Number of realizations of the capture history to simulate. Defaults to 50.
        seed (int, optional): Random seed for the simulation. Defaults to 1111.

    Returns:
        List[np.ndarray]: List of matrices, each representing the number of times each individual was detected at
        each detector for each realization. Each matrix has dimensions (n_individuals, n_traps), where rows represent
        individuals and columns represent traps.
    """
    np.random.seed(seed)
    n_individuals = len(ac)
    n_traps = len(tl)
    detections: List[np.ndarray] = []

    for _ in range(n_real):
        rdetections = np.zeros((n_individuals, n_traps))
        for ni, s in enumerate(ac):
            sr, sc = int(np.floor(s[0])), int(np.floor(s[1]))
            for nt in range(n_traps):
                dist = lcp_dist[nt, sr, sc]
                prob_cap = (1 / (1 + np.exp(alpha0))) * np.exp(-alpha1 * dist * dist)
                n_det = np.random.binomial(K, prob_cap)
                rdetections[ni, nt] = n_det
        rdetections = rdetections[~np.all(rdetections == 0, axis=1)]
        detections.append(rdetections)

    return detections