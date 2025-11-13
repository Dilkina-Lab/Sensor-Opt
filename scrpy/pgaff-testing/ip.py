import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import os
import pyreadr
import sys
import argparse
from functools import partial
import pickle
import math
import multiprocessing as mp
from scipy import stats
from tqdm import tqdm
from datetime import datetime
import warnings

# # Load scenario parameters for one draw
# params = pd.read_csv('./full_grid_1km/10-3 data (Marten)/param_values_for_each_draw300_marten.csv')
# param = params.iloc[0]  # just look at first param
# g0 = param['g0']
# sigma = param['sigma']
# K = 5  # fixed constant -- not using in our relaxation

# # Load activity center
# ac_coords = pyreadr.read_r('./full_grid_1km/10-3 data (Marten)/500m_mask_marten.RDS')
# ac_coords = ac_coords[None]
# ac_coords_list = ac_coords[['x', 'y']].values.tolist()
# ac_coords_list = np.array(ac_coords_list)
# # print(f"{ac_coords_list.shape} activity centers")

# # Load candidate trap locations
# trap_coords = pd.read_csv('./full_grid_1km/10-3 data (Marten)/1000m_trap_grid_marten.csv')
# trap_coords = trap_coords.drop(columns = ['Unnamed: 0'])
# trap_coords_list = []
# for i in range(trap_coords.shape[0]):
#     trap_coords_list.append((trap_coords['x'].iloc[i], trap_coords['y'].iloc[i]))
# trap_coords_list = np.array(trap_coords_list)
# # print(f"{trap_coords_list.shape} candidate trap locations")

# # pre-compute distances between candidate trap locations and activity centers
# traps = trap_coords_list[:, np.newaxis, :]  # Add a new axis to traps to make it 3D
# centers = ac_coords_list[np.newaxis, :, :]  # Add a new axis to centers to make it 3D
# differences = traps - centers
# distances = np.linalg.norm(differences, axis=2)

# # Compute prob_cap matrix
# alpha1 = 1/(2 * sigma**2)
# prob_cap = g0 * np.exp(-alpha1 * distances)
# # print(prob_cap.shape)  # should be [n_traps x n_ac]
# p_j = prob_cap.sum(axis=1)  # sum over l (activity centers)
# print(p_j.shape)

# # Compute weighted 'p_j' vector weighted by D(l)
# # Assuming D is a vector across l, matching activity centers
# # D_vector = np.array([D] * prob_cap.shape[1]) # shape aligned to prob_cap if D(l) constant
# # weighted_prob_cap = prob_cap * D_vector  # element-wise multiply [n_traps x n_ac]
# # p_j = weighted_prob_cap.sum(axis=1)  # sum over l (activity centers)

# # Define budget
# budget = 10

# # Setup Model
# m = gp.Model("IP Sensor Placement")

# # Variables: binary trap decision variables
# x = m.addVars(len(traps), vtype=GRB.BINARY, name="x")

# # Objective linearized surrogate: sum over j of -x_j * log(p_j)
# # Since original is K sum_l D(l) sum_j log(1 - x_j Pr[l,j])
# # Approximated linear objective uses precomputed weights: -log(probability weighted by D)
# weights = -np.log(p_j + 1e-10)  # Add small epsilon to avoid log(0)
# obj = gp.quicksum(weights[j] * x[j] for j in range(len(traps)))

# m.setObjective(obj, GRB.MAXIMIZE)

# # Budget constraint
# m.addConstr(x.sum() == budget, "budget_constr")

# print("Starting optimization...")
# start_time = time.time()
# m.optimize()
# end_time = time.time()

# print("\nOptimization results:")
# if m.Status == GRB.INFEASIBLE:
#     print("Model infeasible")
#     m.computeIIS()
#     m.write("model.ilp")
# elif m.Status == GRB.UNBOUNDED:
#     print("Model unbounded")
# elif m.Status == GRB.OPTIMAL:
#     print(f"Optimal objective: {m.ObjVal:.4f}")
#     selected = [j for j in range(len(traps)) if x[j].X > 0.5]
#     print(f"Selected trap indices: {selected}")
# else:
#     print("No optimal solution found")

# print(f"Solver runtime: {m.Runtime:.4f} seconds")
# print(f"Total execution time: {end_time - start_time:.4f} seconds")

# # Get the original Trap_index for each selected index
# selected_trap_ids = trap_coords.iloc[selected]['Trap_index'].tolist()
# print("Original Trap IDs:", selected_trap_ids)


#################################################################################################################################
############                                         READ IN PARAMETERS                                              ############
#################################################################################################################################

param_draws = [6, 7, 8, 10, 11, 16, 17, 18, 20, 23, 25, 26, 31, 34, 38, 43, 46, 47, 57, 58, 60, 61, 64, 67, 68, 69, 73,         # training parameters 
      74, 76, 77, 78, 79, 80, 83, 85, 91, 93, 94, 98, 102, 105, 109, 110, 112, 114, 115, 119, 120, 125, 127, 130, 
      145, 148, 149, 153, 155, 158, 164, 165, 166, 174, 176, 180, 181, 183, 186, 187, 193, 194, 195, 197, 204, 212, 
      216, 219, 222, 225, 227, 230, 231, 232, 233, 234, 235, 238, 239, 240, 250, 251, 257, 267, 275, 279, 282, 283, 
      287, 289, 290, 296, 298]

budgets = [10, 20, 30, 40, 50, 60, 70, 80]

base_dir = "/Users/hannahmurray/Documents/GitHub/Sensor-Opt/scrpy/pgaff-testing/secr/Integer Programming/Marten (A)"            # location where all the selected/excluded IDs should be saved

# Activity centers
ac_coords = pyreadr.read_r('./full_grid_1km/10-3 data (Marten)/500m_mask_marten.RDS')
ac_coords = ac_coords[None]
ac_coords_list = ac_coords[['x', 'y']].values

# Candidate trap locations
trap_coords = pd.read_csv('./full_grid_1km/10-3 data (Marten)/1000m_trap_grid_marten.csv').drop(columns = ['Unnamed: 0'])
trap_coords_list = trap_coords[['x', 'y']].values

# Distance calculation
traps = trap_coords_list[:, np.newaxis, :]
centers = ac_coords_list[np.newaxis, :, :]
differences = traps - centers
distances = np.linalg.norm(differences, axis=2)

# Load all parameters
params = pd.read_csv('./full_grid_1km/10-3 data (Marten)/param_values_for_each_draw300_marten.csv')
for param_id in param_draws:
    # Find densities per candidate trap location
    dmod_path = f'./full_grid_1km/10-3 data (Marten)/Constant_detection/D_mod/Dmod_draw_{param_id}.csv'
    density_df = pd.read_csv(dmod_path)
    D_vec = density_df['D_mod'].values * 25  # scaling densities to keep consistent with greedy approach, not really needed.

    param = params.iloc[param_id-1]
    g0 = param['g0']
    sigma = param['sigma']

    # Distance computation
    traps = trap_coords_list[:, np.newaxis, :]
    centers = ac_coords_list[np.newaxis, :, :]
    distances = np.linalg.norm(traps - centers, axis=2)  # shape (n_traps, n_cells)
    alpha1 = 1/(2 * sigma**2)
    prob_cap = g0 * np.exp(-alpha1 * distances)  # shape (n_traps, n_cells)

    # Compute weighted log-detection for linear surrogate: sum over l [D(l) * -log(prob_cap[j, l])]
    log_prob = np.log(prob_cap + 1e-10)
    weights = np.dot(-log_prob, D_vec)  # shape (n_traps,)

#################################################################################################################################
############                                             RUN OPTIMIZATION                                            ############
#################################################################################################################################
    for budget in budgets:
        m = gp.Model("IP Sensor Placement")
        x = m.addVars(len(traps), vtype=GRB.BINARY, name="x")
        obj = gp.quicksum(weights[j] * x[j] for j in range(len(traps)))
        m.setObjective(obj, GRB.MAXIMIZE)
        m.addConstr(x.sum() == budget, "budget_constr")
        m.Params.OutputFlag = 0

        print(f"Param {param_id}, budget {budget}: Starting optimization...")
        start_time = time.time()
        m.optimize()
        end_time = time.time()

        selected_indices = [j for j in range(len(traps)) if x[j].X > 0.5]
        excluded_indices = list(set(range(len(traps))) - set(selected_indices))

        selected_df = trap_coords.iloc[selected_indices][['Trap_index', 'x', 'y']]
        excluded_ids = trap_coords.iloc[excluded_indices]['Trap_index'].tolist()

        budget_dir = os.path.join(base_dir, f"Budget = {budget}")
        os.makedirs(budget_dir, exist_ok=True)

        selected_csv_path = os.path.join(budget_dir, f"IP-selected_ids-budget{budget}-param{param_id}.csv")
        selected_df.to_csv(selected_csv_path, index=False)

        excluded_txt_path = os.path.join(budget_dir, f"IP-excluded_ids-budget{budget}-param{param_id}.txt")
        with open(excluded_txt_path, "w") as f:
            for trap_id in excluded_ids:
                f.write(f"{trap_id}\n")

        print(f"Param {param_id}, budget {budget}: Done. Selected {len(selected_indices)}, excluded {len(excluded_ids)}. Time: {end_time-start_time:.6f} s")

print("All optimizations complete.")


#################################################################################################################################
############                                            COMPUTE E_N & E_R                                            ############
#################################################################################################################################
