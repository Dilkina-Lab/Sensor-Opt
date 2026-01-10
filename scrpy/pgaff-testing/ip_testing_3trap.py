import time
import os
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from scipy import stats
import pyreadr
import sys


#################################################################################################################################
############                                          UTILITY FUNCTIONS                                              ############
#################################################################################################################################

def compute_expected_n(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x, j1_l_spatial, j2_l_spatial, j3_l_spatial):
    """
    En^s = sum_l ekn_l^s using 3 closest traps

    ekn_l^s = D_l^s ( Q1 x1 + Q2 x2 + Q3 x3
                        - Q1 Q2 z12 - Q1 Q3 z13 - Q2 Q3 z23
                        + Q1 Q2 Q3 y )
    where:
        Qk = 1 - (1 - Pk)^K 
        z12 = x1 AND x2, z13 = x1 AND x3, z23 = x2 AND x3, y = x1 AND x2 AND x3
    """
    alpha1 = 1.0 / (2.0 * sigma * sigma)
    prob_cap = g0 * np.exp(-alpha1 * (distances ** 2))  # (J, L)

    for t in range(len(trap_locs)):
        if int(trap_x[t]) == 0:
            prob_cap[t, ...] = 0.0   # Inactive traps have a probability of capture = 0

    L = prob_cap.shape[1]
    D_vec = np.array(density).flatten()
    ekn_l = np.zeros(L)

    for l in range(L):
        j1 = j1_l_spatial[l]
        j2 = j2_l_spatial[l]
        j3 = j3_l_spatial[l]

        P1 = prob_cap[j1, l]
        P2 = prob_cap[j2, l]
        P3 = prob_cap[j3, l]

        Q1 = 1.0 - (1.0 - P1) ** K
        Q2 = 1.0 - (1.0 - P2) ** K
        Q3 = 1.0 - (1.0 - P3) ** K

        x1 = trap_x[j1]
        x2 = trap_x[j2]
        x3 = trap_x[j3]

        z12 = min(x1, x2)
        z13 = min(x1, x3)
        z23 = min(x2, x3)
        y   = min(x1, x2, x3)

        ekn_l[l] = D_vec[l] * (
            (Q1 * x1) + (Q2 * x2) + (Q3 * x3)
            - (Q1 * Q2 * z12) - (Q1 * Q3 * z13) - (Q2 * Q3 * z23)
            + (Q1 * Q2 * Q3 * y))

    return float(np.sum(ekn_l))

def compute_expected_c(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x, j1_l_spatial, j2_l_spatial, j3_l_spatial):
    """
    Expected captures Ec^s = sum_l ekc_l^s with 3 closest traps:

    ekc_l^s = D_l^s K ( P1 x1 + P2 x2 + P3 x3 )
    """
    alpha1 = 1.0 / (2.0 * sigma * sigma)
    prob_cap = g0 * np.exp(-alpha1 * (distances ** 2))  # (J, L)
    D_vec = np.array(density).flatten()
    L = prob_cap.shape[1]

    expected_c = 0.0
    for l in range(L):
        j1 = j1_l_spatial[l]
        j2 = j2_l_spatial[l]
        j3 = j3_l_spatial[l]

        P1 = prob_cap[j1, l]
        P2 = prob_cap[j2, l]
        P3 = prob_cap[j3, l]

        ekc_l = D_vec[l] * K * (
            (P1 * trap_x[j1]) + (P2 * trap_x[j2]) + (P3 * trap_x[j3]))
        expected_c += ekc_l

    return float(expected_c)

def compute_expected_r(expected_c, expected_n):
    """Expected recaptures Er^s = Ec^s - En^s"""
    return expected_c - expected_n


#################################################################################################################################
############                                         READ IN PARAMETERS                                              ############
#################################################################################################################################

sa_groups = [
    [64, 74, 76, 78, 109, 125, 164, 204, 238, 250],   # SA1
    [31, 43, 68, 79, 91, 98, 212, 222, 289, 298],     # SA2
    [112, 114, 115, 125, 158, 176, 183, 225, 283, 287],
    [16, 25, 38, 60, 80, 145, 225, 234, 238, 275],
    [38, 57, 68, 85, 94, 105, 130, 193, 197, 282],
    [7, 10, 34, 47, 78, 91, 165, 193, 204, 282],
    [57, 67, 69, 74, 112, 125, 145, 197, 230, 231],
    [6, 26, 31, 77, 94, 120, 164, 197, 219, 222],
    [8, 18, 20, 69, 115, 166, 174, 180, 194, 238],
    [11, 83, 158, 197, 212, 216, 222, 239, 257, 287],
    [6, 64, 69, 98, 112, 153, 166, 195, 204, 296],
    [10, 57, 58, 60, 130, 148, 176, 187, 193, 231],
    [18, 57, 61, 78, 105, 130, 183, 227, 233, 287],
    [10, 23, 69, 91, 94, 102, 145, 197, 219, 222],
    [17, 58, 79, 91, 145, 227, 231, 238, 257, 289],
    [18, 34, 69, 114, 174, 176, 181, 195, 197, 282],
    [17, 57, 149, 193, 222, 230, 231, 233, 287, 298],
    [47, 77, 93, 94, 148, 187, 197, 227, 234, 296],
    [7, 77, 80, 98, 109, 204, 225, 238, 257, 287],
    [34, 68, 80, 93, 105, 155, 197, 219, 282, 298]
]

budgets = [10, 20, 30, 40, 50, 60, 70, 80]
K = 5

base_dir = "/Users/hannahmurray/Documents/GitHub/Sensor-Opt/scrpy/pgaff-testing/secr/Integer Programming/Marten/Jan9_3trap_TESTING"

# Read in activity centers
ac_coords = pyreadr.read_r('./full_grid_1km/10-3 data (Marten)/500m_mask_marten.RDS')
ac_coords = ac_coords[None]     # extract the dataframe from the returned dict
ac_coords_list = ac_coords[['x', 'y']].values

# Read in candidate trap locations
trap_coords = pd.read_csv('./full_grid_1km/10-3 data (Marten)/1000m_trap_grid_marten.csv').drop(columns=['Unnamed: 0'])
trap_coords_list = trap_coords[['x', 'y']].values
num_traps = len(trap_coords_list)

# Calculate distances between all candidate traps and potential activity centers
traps = trap_coords_list[:, np.newaxis, :]
centers = ac_coords_list[np.newaxis, :, :]
distances = np.linalg.norm(traps - centers, axis=2)
num_pixels = distances.shape[1]  # L

# Read in parameters per scenario
params = pd.read_csv('./full_grid_1km/10-3 data (Marten)/param_values_for_each_draw300_marten.csv')


#################################################################################################################################
############                                          FIND J1, J2, J3                                               ############
#################################################################################################################################

print("Computing closest traps j1(l), j2(l), j3(l) to each activity center l")

j1_l_spatial = np.argmin(distances, axis=0)  # Closest trap -- row index j with the smallest distance in each column l

distances_masked = distances.copy()
distances_masked[j1_l_spatial, np.arange(num_pixels)] = np.inf
j2_l_spatial = np.argmin(distances_masked, axis=0)  # 2nd closest trap

distances_masked2 = distances_masked.copy()
distances_masked2[j2_l_spatial, np.arange(num_pixels)] = np.inf
j3_l_spatial = np.argmin(distances_masked2, axis=0)  # 3rd closest trap


#################################################################################################################################
############                                        PRE-COMPUTE P AND Q                                              ############
#################################################################################################################################

# Precompute per-scenario P and Q using 3 closest traps
scenario_precomp = {}  # sa_idx -> param_id -> dict with P1,P2,P3,Q1,Q2,Q3
for sa_idx, group in enumerate(sa_groups, start=1):
    print(f"Precomputing P/Q for SA group {sa_idx}...")
    scenario_precomp[sa_idx] = {}

    for param_id in group:
        param_row = params.iloc[param_id - 1]
        g0_val = param_row['g0']
        sigma_val = param_row['sigma']
        alpha1 = 1.0 / (2.0 * sigma_val ** 2)
        prob_cap = g0_val * np.exp(-alpha1 * (distances ** 2))  # shape: (num_traps, num_pixels)

        P1 = prob_cap[j1_l_spatial, np.arange(num_pixels)]
        P2 = prob_cap[j2_l_spatial, np.arange(num_pixels)]
        P3 = prob_cap[j3_l_spatial, np.arange(num_pixels)]

        Q1 = 1.0 - (1.0 - P1) ** K
        Q2 = 1.0 - (1.0 - P2) ** K
        Q3 = 1.0 - (1.0 - P3) ** K

        scenario_precomp[sa_idx][param_id] = {
            "P1": P1,
            "P2": P2,
            "P3": P3,
            "Q1": Q1,
            "Q2": Q2,
            "Q3": Q3}


#################################################################################################################################
############                                                  OPTIMIZE                                               ############
#################################################################################################################################

results = []

for sa_idx, group in enumerate(sa_groups, start=1):
    print(f"Processing SA group {sa_idx} (params: {group})")

    draw_info = {}
    for param_id in group:
        param_row = params.iloc[param_id - 1]
        g0_val = param_row['g0']
        sigma_val = param_row['sigma']
        draw_info[param_id] = (g0_val, sigma_val)

    for budget in budgets:
        m = gp.Model()
        m.ModelName = f"SA{sa_idx}_budget{budget}"
        x = m.addVars(num_traps, vtype=GRB.BINARY)
        z12 = m.addVars(num_pixels, vtype=GRB.BINARY)       # z12 = x1 AND x2
        z13 = m.addVars(num_pixels, vtype=GRB.BINARY)       # z13 = x1 AND x3
        z23 = m.addVars(num_pixels, vtype=GRB.BINARY)       # z23 = x2 AND x3
        y   = m.addVars(num_pixels, vtype=GRB.BINARY)       # y = x1 AND x2 AND x3
        Ekmin = m.addVars(len(group), vtype=GRB.CONTINUOUS, lb=0)

        # Objective: max average Ekmin over scenarios in group
        obj = gp.LinExpr()
        for s_idx in range(len(group)):
            obj += Ekmin[s_idx]
        m.setObjective(obj / len(group), GRB.MAXIMIZE)

        # Budget constraint
        m.addConstr(gp.quicksum(x[j] for j in range(num_traps)) <= budget)

        # Range constraints for z and y variables
        for l in range(num_pixels):
            j1 = j1_l_spatial[l]
            j2 = j2_l_spatial[l]
            j3 = j3_l_spatial[l]

            # z12 = x1 AND x2
            m.addConstr(z12[l] <= x[j1])
            m.addConstr(z12[l] <= x[j2])
            m.addConstr(z12[l] >= x[j1] + x[j2] - 1)

            # z13 = x1 AND x3
            m.addConstr(z13[l] <= x[j1])
            m.addConstr(z13[l] <= x[j3])
            m.addConstr(z13[l] >= x[j1] + x[j3] - 1)

            # z23 = x2 AND x3
            m.addConstr(z23[l] <= x[j2])
            m.addConstr(z23[l] <= x[j3])
            m.addConstr(z23[l] >= x[j2] + x[j3] - 1)

            # y = x1 AND x2 AND x3
            m.addConstr(y[l] <= x[j1])
            m.addConstr(y[l] <= x[j2])
            m.addConstr(y[l] <= x[j3])
            m.addConstr(y[l] >= x[j1] + x[j2] + x[j3] - 2)

        # Constraints per scenario
        for s_idx, param_id in enumerate(group):
            g0_val, sigma_val = draw_info[param_id]
            alpha1 = 1.0 / (2.0 * sigma_val ** 2)
            prob_cap = g0_val * np.exp(-alpha1 * (distances ** 2))

            dmod_path = f'./full_grid_1km/10-3 data (Marten)/Constant_detection/D_mod/Dmod_draw_{param_id}.csv'
            density_df = pd.read_csv(dmod_path)
            D_vec = density_df['D_mod'].values * 25

            # En constraint: Ekmin^s <= sum_l ekn_l^s
            ekn_sum = gp.LinExpr()
            for l in range(num_pixels):
                j1 = j1_l_spatial[l]
                j2 = j2_l_spatial[l]
                j3 = j3_l_spatial[l]

                P1 = prob_cap[j1, l]
                P2 = prob_cap[j2, l]
                P3 = prob_cap[j3, l]

                Q1 = 1.0 - (1.0 - P1) ** K
                Q2 = 1.0 - (1.0 - P2) ** K
                Q3 = 1.0 - (1.0 - P3) ** K

                ekn_l = D_vec[l] * (
                    Q1 * x[j1] + Q2 * x[j2] + Q3 * x[j3]
                    - Q1 * Q2 * z12[l]
                    - Q1 * Q3 * z13[l]
                    - Q2 * Q3 * z23[l]
                    + Q1 * Q2 * Q3 * y[l]
                )
                ekn_sum += ekn_l

            m.addConstr(Ekmin[s_idx] <= ekn_sum, name=f"Ekn_min{s_idx}")

            # Ekr constraint using gamma, lambda, xi from your derivation
            pre = scenario_precomp[sa_idx][param_id]
            P1_all = pre["P1"]
            P2_all = pre["P2"]
            P3_all = pre["P3"]
            Q1_all = pre["Q1"]
            Q2_all = pre["Q2"]
            Q3_all = pre["Q3"]

            ekr_sum = gp.LinExpr()
            for l in range(num_pixels):
                j1 = j1_l_spatial[l]
                j2 = j2_l_spatial[l]
                j3 = j3_l_spatial[l]

                P1_l = P1_all[l]
                P2_l = P2_all[l]
                P3_l = P3_all[l]
                Q1_l = Q1_all[l]
                Q2_l = Q2_all[l]
                Q3_l = Q3_all[l]

                # gamma, lambda, xi as defined in screenshot [file:44]
                gamma1 = K * P1_l - Q1_l
                gamma2 = K * P2_l - Q2_l
                gamma3 = K * P3_l - Q3_l

                lambda12 = Q1_l * Q2_l
                lambda13 = Q1_l * Q3_l
                lambda23 = Q2_l * Q3_l

                xi = Q1_l * Q2_l * Q3_l

                ekr_l = D_vec[l] * (
                    gamma1 * x[j1] + gamma2 * x[j2] + gamma3 * x[j3]
                    + lambda12 * z12[l] + lambda13 * z13[l] + lambda23 * z23[l]
                    - xi * y[l]
                )
                ekr_sum += ekr_l

            m.addConstr(Ekmin[s_idx] <= ekr_sum, name=f"Ekr_min{s_idx}")

        m.Params.OutputFlag = 0
        start_time = time.time()
        m.optimize()
        end_time = time.time()


#################################################################################################################################
############                                              PRINT RESULTS                                              ############
#################################################################################################################################

        if m.status == GRB.OPTIMAL:
            selected_idx = [j for j in range(num_traps) if x[j].X > 0.5]
            trap_selection_array = np.zeros(num_traps)
            trap_selection_array[selected_idx] = 1

            results.append({
                "sa_group": sa_idx,
                "group_ids": str(group),
                "budget": budget,
                "runtime": round(end_time - start_time, 9),
                "obj_val": m.ObjVal,
            })

            budget_folder = os.path.join(base_dir, f"Budget = {budget}")
            os.makedirs(budget_folder, exist_ok=True)

            # Save selected trap IDs to csv file
            selected_csv_path = os.path.join(budget_folder, f"IP-selected_ids-budget{budget}-SA{sa_idx}.csv")
            selected_ids = trap_coords.iloc[selected_idx]['Trap_index'].tolist()
            selected_df = trap_coords.iloc[selected_idx][['Trap_index', 'x', 'y']]
            selected_df.to_csv(selected_csv_path, index=False)

            # Save excluded trap IDs to txt file
            excluded_txt_path = os.path.join(budget_folder, f"IP-excluded_ids-budget{budget}-SA{sa_idx}.txt")
            all_ids = set(trap_coords['Trap_index'])
            excluded_ids = sorted(all_ids - set(selected_ids))
            with open(excluded_txt_path, "w") as f:
                for eid in excluded_ids:
                    f.write(f"{eid}\n")

            print(f"SA{sa_idx} budget{budget} -> obj:{m.ObjVal:.6f}")

results_df = pd.DataFrame(results)
results_csv_path = os.path.join(base_dir, "IP4_Opt_3traps.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"Results saved to {results_csv_path}")
