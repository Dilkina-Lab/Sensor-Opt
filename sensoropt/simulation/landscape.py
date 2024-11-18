"""
This module provides functions for simulating spatial inhomogeneous Poisson processes
and creating grid layouts over a raster representing a landscape.
"""

import itertools
import numpy as np
from typing import List, Tuple

def log_linear_ipp_rate(x: np.ndarray, beta0: float, beta1: float) -> np.ndarray:
    """
    Rate function for an inhomogeneous poisson process, in which
    the intensity is a log-linear function of x.

    Args:
        x (np.ndarray): Input array.
        beta0 (float): Intercept parameter for the log-linear rate function.
        beta1 (float): Slope parameter for the log-linear rate function.

    Returns:
        np.ndarray: The computed rate function.
    """
    return np.exp(beta0 + beta1 * x)

def simulate_log_linear_ipp(
    N: int,
    raster: np.ndarray,
    beta0: float,
    beta1: float,
    n_real: int = 1,
    seed: int = 1111,
) -> List[List[Tuple[float, float]]]:
    """
    Simulates a spatial inhomogeneous Poisson process over 2D raster.
    Simulation is done using the Lewis and Schedler thinning algorithm.

    Args:
        N (int): Number of point events to simulate per realization.
        raster (np.ndarray): Raster over which to simulate the IPP.
        beta0 (float): Intercept parameter for the log-linear rate function.
        beta1 (float): Slope parameter for the log-linear rate function.
        n_real (int, optional): Number of realizations of the IPP to simulate. Defaults to 1.
        seed (int, optional): Random seed for the simulation. Defaults to 1111.

    Returns:
        List[List[Tuple[float, float]]]: List of n_real lists; each sublist contains N tuples (r, c)
        specifying spatial points for one realization of the IPP.
    """
    np.random.seed(seed)

    # Compute the rate function over the entire raster
    lambda_raster = log_linear_ipp_rate(raster, beta0, beta1)
    lambda_max = np.amax(lambda_raster)

    def generate_point() -> Tuple[float, float]:
        row_coord = np.random.uniform(0, raster.shape[0])
        col_coord = np.random.uniform(0, raster.shape[1])
        return row_coord, col_coord

    points: List[List[Tuple[float, float]]] = []
    for _ in range(n_real):
        rpoints: List[Tuple[float, float]] = []
        while len(rpoints) < N:
            row_coord, col_coord = generate_point()
            lambda_rc = lambda_raster[int(np.floor(row_coord)), int(np.floor(col_coord))]
            if np.random.uniform(0, 1) < lambda_rc / lambda_max:
                rpoints.append((row_coord, col_coord))
        points.append(rpoints)
    return points

def create_grid_layout(
    n: int, m: int, raster: np.ndarray, buffer: float = 0.1
) -> List[Tuple[float, float]]:
    """
    Creates a grid layout over a raster, leaving a buffer zone around the edges.

    Args:
        n (int): Number of grid points along the x-axis.
        m (int): Number of grid points along the y-axis.
        raster (np.ndarray): The raster over which to create the grid layout.
        buffer (float, optional): Buffer zone as a fraction of the raster dimensions. Defaults to 0.1.

    Returns:
        List[Tuple[float, float]]: List of tuples representing the grid points (x, y).
    """
    n_pix_h, n_pix_w = raster.shape
    grid_xmin = n_pix_h * buffer
    grid_xmax = n_pix_h * (1 - buffer)
    grid_ymin = n_pix_w * buffer
    grid_ymax = n_pix_w * (1 - buffer)

    grid_x = np.linspace(grid_xmin, grid_xmax, n)
    grid_y = np.linspace(grid_ymin, grid_ymax, m)
    grid_loc = list(itertools.product(grid_x, grid_y))

    return grid_loc