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

def compute_expected_n(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x, j1_l_spatial, j2_l_spatial):
    """Compute expected unique animals ekn^s = sum_l ekn^s_l using IP formulas"""
    alpha1 = (1/(2*sigma*sigma))
    prob_cap = (g0) * np.exp(-alpha1 * (distances**2))  # (#traps, #ac)

    # Mask inactive traps
    for t in range(len(trap_locs)):
        if int(trap_x[t]) == 0:
            prob_cap[t, ...] = np.zeros(ac_locs.shape[0])

    # Compute Q^s_l,j1 = 1 - (1 - P^s_l,j1)^K for closest traps only
    Q1 = 1 - (1 - prob_cap[j1_l_spatial, np.arange(len(density))])**K
    Q2 = 1 - (1 - prob_cap[j2_l_spatial, np.arange(len(density))])**K

    # ekn^s_l = D^s_l (Q1 x_j1 + Q2 x_j2 - Q1 Q2 z_l) using spatial closest traps
    z_l = np.minimum(trap_x[j1_l_spatial], trap_x[j2_l_spatial])  # both active
    ekn_l = density * (Q1 * trap_x[j1_l_spatial] + Q2 * trap_x[j2_l_spatial] - Q1 * Q2 * z_l)

    return np.sum(ekn_l)


def compute_expected_c(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x,
                       j1_l_spatial, j2_l_spatial):
    """
    Expected total captures ec^s = sum_l ekc_l^s, with

        ekc_l^s = D_l^s K ( P_{l,j1(l)}^s x_{j1(l)} + P_{l,j2(l)}^s x_{j2(l)} )

    using only the two closest traps to each pixel.
    """
    alpha1 = (1/(2*sigma*sigma))
    prob_cap = g0 * np.exp(-alpha1 * (distances**2))  # (#traps, #ac)
    D_vec = np.array(density).flatten()               # length = num_pixels

    num_pixels = prob_cap.shape[1]
    expected_c = 0.0

    for l in range(num_pixels):
        j1 = j1_l_spatial[l]
        j2 = j2_l_spatial[l]

        P1 = prob_cap[j1, l]
        P2 = prob_cap[j2, l]

        ekc_l = D_vec[l] * K * (P1 * trap_x[j1] + P2 * trap_x[j2])
        expected_c += ekc_l

    return expected_c


def compute_expected_r(expected_c, expected_n):
    """Expected recaptures: er^s = ec^s - en^s (from IP)"""
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

base_dir = "/Users/hannahmurray/Documents/GitHub/Sensor-Opt/scrpy/pgaff-testing/secr/Integer Programming/Marten/Jan8_TESTING"

# Read in activity centers
ac_coords = pyreadr.read_r('./full_grid_1km/10-3 data (Marten)/500m_mask_marten.RDS')
ac_coords = ac_coords[None]
ac_coords_list = ac_coords[['x', 'y']].values

# Read in candidate trap locations
trap_coords = pd.read_csv('./full_grid_1km/10-3 data (Marten)/1000m_trap_grid_marten.csv').drop(columns = ['Unnamed: 0'])
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
############                                              FIND J1, J2                                                ############
#################################################################################################################################

# Find the spatially closest traps to each potential activity center (SCENARIO-INDEPENDENT)
print("Computing spatial closest traps j1(l), j2(l)")
j1_l_spatial = np.argmin(distances, axis=0)  # Closest trap by Euclidean distance
distances_masked = distances.copy()
distances_masked[j1_l_spatial, np.arange(num_pixels)] = np.inf
j2_l_spatial = np.argmin(distances_masked, axis=0)  # 2nd closest by distance

# Precompute per-scenario P and Q using SPATIAL closest traps
scenario_precomp = {}  # sa_idx -> param_id -> dict with P1, P2, Q1, Q2
for sa_idx, group in enumerate(sa_groups, start=1):
    print(f"Precomputing P/Q for SA group {sa_idx}...")
    scenario_precomp[sa_idx] = {}

    for param_id in group:
        param_row = params.iloc[param_id - 1]
        g0_val = param_row['g0']
        sigma_val = param_row['sigma']
        alpha1 = 1/(2 * sigma_val**2)
        prob_cap = g0_val * np.exp(-alpha1 * (distances**2))  # shape: (num_traps, num_pixels)

        # P^s_l,j1, P^s_l,j2 using SPATIAL closest traps
        P1 = prob_cap[j1_l_spatial, np.arange(num_pixels)]
        P2 = prob_cap[j2_l_spatial, np.arange(num_pixels)]

        Q1 = 1 - (1 - P1)**K
        Q2 = 1 - (1 - P2)**K

        scenario_precomp[sa_idx][param_id] = {
            "P1": P1,
            "P2": P2,
            "Q1": Q1,
            "Q2": Q2
        }

results = []


#################################################################################################################################
############                                                  OPTIMIZE                                               ############
#################################################################################################################################

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
        z = m.addVars(num_pixels, vtype=GRB.BINARY)
        Ekmin = m.addVars(len(group), vtype=GRB.CONTINUOUS, lb=0)  # Ekmin^s constraints

        # Objective: max (1/|g|)sum{scenarios} Ekmin^s
        obj = gp.LinExpr()
        for s_idx, param_id in enumerate(group):
            obj += Ekmin[s_idx]
        m.setObjective(obj / len(group), GRB.MAXIMIZE)

        # Budget constraint
        m.addConstr(gp.quicksum(x[j] for j in range(num_traps)) <= budget)

        # z_l linearization constraints
        for l in range(num_pixels):
            j1 = j1_l_spatial[l]
            j2 = j2_l_spatial[l]
            m.addConstr(z[l] <= x[j1], name=f"z_l{l}_upper1")
            m.addConstr(z[l] <= x[j2], name=f"z_l{l}_upper2")
            m.addConstr(z[l] >= x[j1] + x[j2] - 1, name=f"z_l{l}_lower")

        # ekn^s_l and ekr^s_l constraints for each scenario s in group
        for s_idx, param_id in enumerate(group):
            g0_val, sigma_val = draw_info[param_id]
            alpha1 = 1/(2 * sigma_val**2)
            prob_cap = g0_val * np.exp(-alpha1 * (distances**2))

            # Load density D^s_l for this scenario
            dmod_path = f'./full_grid_1km/10-3 data (Marten)/Constant_detection/D_mod/Dmod_draw_{param_id}.csv'
            density_df = pd.read_csv(dmod_path)
            D_vec = density_df['D_mod'].values * 25  # Scenario-specific densities

            # ekn^s_l constraint: Ekmin^s <= sum_l ekn^s_l
            ekn_sum = gp.LinExpr()
            for l in range(num_pixels):
                j1 = j1_l_spatial[l]
                j2 = j2_l_spatial[l]
                Q1 = 1 - (1 - prob_cap[j1, l])**K
                Q2 = 1 - (1 - prob_cap[j2, l])**K
                ekn_l = D_vec[l] * (Q1 * x[j1] + Q2 * x[j2] - Q1 * Q2 * z[l])
                ekn_sum += ekn_l

            m.addConstr(Ekmin[s_idx] <= ekn_sum, name=f"Ekn_min{s_idx}")

            # ekr^s_l constraint: Ekmin^s <= sum_l ekr^s_l with gamma1,gamma2,gamma3
            pre = scenario_precomp[sa_idx][param_id]
            P1 = pre["P1"]
            P2 = pre["P2"]
            Q1_all = pre["Q1"]
            Q2_all = pre["Q2"]

            ekr_sum = gp.LinExpr()
            for l in range(num_pixels):
                j1 = j1_l_spatial[l]
                j2 = j2_l_spatial[l]

                P1_l = P1[l]
                P2_l = P2[l]
                Q1_l = Q1_all[l]
                Q2_l = Q2_all[l]

                gamma1 = K * P1_l - Q1_l
                gamma2 = K * P2_l - Q2_l
                gamma3 = Q1_l * Q2_l

                ekr_l = D_vec[l] * (gamma1 * x[j1] + gamma2 * x[j2] + gamma3 * z[l])
                ekr_sum += ekr_l

            m.addConstr(Ekmin[s_idx] <= ekr_sum, name=f"Ekr_min{s_idx}")

        m.Params.OutputFlag = 0
        start_time = time.time()
        m.optimize()
        end_time = time.time()

        if m.status == GRB.OPTIMAL and m.SolCount > 0:
            selected_idx = [j for j in range(num_traps) if x[j].X > 0.5]
            trap_selection_array = np.zeros(num_traps)
            trap_selection_array[selected_idx] = 1

            # Post-optimization evaluation using updated utility functions
            En_list, Er_list = [], []
            for param_id in group:
                g0_val, sigma_val = draw_info[param_id]
                dmod_path = f'./full_grid_1km/10-3 data (Marten)/Constant_detection/D_mod/Dmod_draw_{param_id}.csv'
                density_df = pd.read_csv(dmod_path)
                D_vec = density_df['D_mod'].values * 25
                En = compute_expected_n(ac_coords_list, trap_coords_list, g0_val, sigma_val, K, D_vec,
                                        distances, trap_selection_array, j1_l_spatial, j2_l_spatial)
                Ec = compute_expected_c(ac_coords_list, trap_coords_list, g0_val, sigma_val, K, D_vec,
                                        distances, trap_selection_array, j1_l_spatial, j2_l_spatial)
                Er = compute_expected_r(Ec, En)
                En_list.append(En)
                Er_list.append(Er)

            En_avg = float(np.mean(En_list))
            Er_avg = float(np.mean(Er_list))
            min_metric = "En" if En_avg < Er_avg else "Er"

            results.append({
                "sa_group": sa_idx,
                "group_ids": str(group),
                "budget": budget,
                "runtime": round(end_time - start_time, 6),
                "E_n_avg": En_avg,
                "E_r_avg": Er_avg,
                "obj_val": m.ObjVal,
                "min_metric": min_metric
            })

            # Save results
            budget_folder = os.path.join(base_dir, f"Budget = {budget}")
            os.makedirs(budget_folder, exist_ok=True)

            selected_csv_path = os.path.join(budget_folder, f"IP-selected_ids-budget{budget}-SA{sa_idx}.csv")
            excluded_txt_path = os.path.join(budget_folder, f"IP-excluded_ids-budget{budget}-SA{sa_idx}.txt")

            selected_ids = trap_coords.iloc[selected_idx]['Trap_index'].tolist()
            selected_df = trap_coords.iloc[selected_idx][['Trap_index', 'x', 'y']]
            all_ids = set(trap_coords['Trap_index'])
            excluded_ids = sorted(all_ids - set(selected_ids))

            selected_df.to_csv(selected_csv_path, index=False)
            with open(excluded_txt_path, "w") as f:
                for eid in excluded_ids:
                    f.write(f"{eid}\n")

            print(f"SA{sa_idx} budget{budget} -> En:{En_avg:.6f} Er:{Er_avg:.6f} obj:{m.ObjVal:.6f}")
        else:
            print(f"SA{sa_idx} budget{budget} - NO SOLUTION")

results_df = pd.DataFrame(results)
results_csv_path = os.path.join(base_dir, "IP3_Opt_Results.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"Results saved to {results_csv_path}")