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
    prob_cap = (g0)*np.exp(-alpha1*(distances**2))  # (#traps, #ac)
    
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


def compute_expected_c(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    """Expected total captures ec^s_l (unchanged from original)"""
    alpha1 = (1/(2*sigma*sigma))
    prob_cap = g0*np.exp(-alpha1*(distances**2))
    density_array = np.array(density).flatten()
    for t in range(len(trap_locs)):
        if int(trap_x[t]) == 0:
            prob_cap[t, ...] = np.zeros(ac_locs.shape[0])
    broadcast_density = np.broadcast_to(density_array, (len(trap_locs), len(density_array)))
    expected_c = np.sum(prob_cap*broadcast_density)*K
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

base_dir = "/Users/hannahmurray/Documents/GitHub/Sensor-Opt/scrpy/pgaff-testing/secr/Integer Programming/Marten/Jan7_TESTING"

# Read in activity centers
ac_coords = pyreadr.read_r('./full_grid_1km/10-3 data (Marten)/500m_mask_marten.RDS')
ac_coords = ac_coords[None]
ac_coords_list = ac_coords[['x', 'y']].values

# Read in candidate trap locations
trap_coords = pd.read_csv('./full_grid_1km/10-3 data (Marten)/1000m_trap_grid_marten.csv').drop(columns = ['Unnamed: 0'])
trap_coords_list = trap_coords[['x', 'y']].values
num_traps = len(trap_coords_list)

# Calculate distances between all candidate traps and activity centers
traps = trap_coords_list[:, np.newaxis, :]
centers = ac_coords_list[np.newaxis, :, :]
distances = np.linalg.norm(traps - centers, axis=2)
num_pixels = distances.shape[1]  # L


#################################################################################################################################
############                                          FIND J1, J2, Gamma                                             ############
#################################################################################################################################

# Find the spatially closest traps to each potential activity center(SCENARIO-INDEPENDENT!)
print("Computing spatial closest traps j1(l), j2(l)")
j1_l_spatial = np.argmin(distances, axis=0)  # Closest trap by Euclidean distance
distances_masked = distances.copy()
distances_masked[j1_l_spatial, np.arange(num_pixels)] = np.inf
j2_l_spatial = np.argmin(distances_masked, axis=0)  # 2nd closest by distance

params = pd.read_csv('./full_grid_1km/10-3 data (Marten)/param_values_for_each_draw300_marten.csv')

# Precompute per-scenario gamma coefficients using SPATIAL closest traps
scenario_gamma = {}  # sa_idx -> param_id -> gamma_l array (per pixel!)
for sa_idx, group in enumerate(sa_groups, start=1):
    print(f"Precomputing per-scenario gamma for SA group {sa_idx}...")
    scenario_gamma[sa_idx] = {}
    
    for param_id in group:
        param_row = params.iloc[param_id - 1]
        g0_val = param_row['g0']
        sigma_val = param_row['sigma']
        alpha1 = 1/(2 * sigma_val**2)
        prob_cap = g0_val * np.exp(-alpha1 * (distances**2))
        
        # P^s_l,j1, P^s_l,j2 using SPATIAL closest traps
        p1 = prob_cap[j1_l_spatial, np.arange(num_pixels)]
        p2 = prob_cap[j2_l_spatial, np.arange(num_pixels)]
        
        # gamma^s_l,3 = marginal contribution when BOTH traps active
        # = P(both) - max(P1, P2) where P(both) = 1 - (1-P1)(1-P2)
        p_both = 1 - (1 - p1) * (1 - p2)
        gamma_l = p_both - np.maximum(p1, p2)
        
        scenario_gamma[sa_idx][param_id] = gamma_l

results = []

# Main optimization loop
for sa_idx, group in enumerate(sa_groups, start=1):
    print(f"Processing SA group {sa_idx} (params: {group})")
    
    draw_info = {}
    for param_id in group:
        param_row = params.iloc[param_id - 1]
        g0_val = param_row['g0']
        sigma_val = param_row['sigma']
        draw_info[param_id] = (g0_val, sigma_val)
    
    for budget in budgets:
        rerun_count = 0

        m = gp.Model()
        m.ModelName = f"SA{sa_idx}_budget{budget}"
        x = m.addVars(num_traps, vtype=GRB.BINARY)
        z = m.addVars(num_pixels, vtype=GRB.BINARY)
        Ekmin = m.addVars(len(group), vtype=GRB.CONTINUOUS, lb=0)  # Emin^s constraints
        
        # Objective: max (1/|g|)∑_{s∈g} Emin^s  [IP objective]
        obj = gp.LinExpr()
        for s_idx, param_id in enumerate(group):
            obj += Ekmin[s_idx]
        m.setObjective(obj / len(group), GRB.MAXIMIZE)          # len of group should be 10 in our case
        
        # Budget constraint
        m.addConstr(gp.quicksum(x[j] for j in range(num_traps)) <= budget)
        
        # z_l linearization constraints using SPATIAL closest traps (IP constraints)
        for l in range(num_pixels):
            j1 = j1_l_spatial[l]
            j2 = j2_l_spatial[l]
            m.addConstr(z[l] <= x[j1], name=f"z_l{l}_upper1")
            m.addConstr(z[l] <= x[j2], name=f"z_l{l}_upper2")
            m.addConstr(z[l] >= x[j1] + x[j2] - 1, name=f"z_l{l}_lower")
        
        # ekn^s_l and ekr^s_l constraints for each scenario s in group (IP definitions)
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
                Q1 = 1 - (1 - prob_cap[j1, l])**K  # Q^s_l,j1
                Q2 = 1 - (1 - prob_cap[j2, l])**K  # Q^s_l,j2
                ekn_l = D_vec[l] * (Q1 * x[j1] + Q2 * x[j2] - Q1 * Q2 * z[l])
                ekn_sum += ekn_l
            
            m.addConstr(Ekmin[s_idx] <= ekn_sum, name=f"Ekn_min{s_idx}")
            
            # ekr^s_l constraint: Ekmin^s <= sum_l ekr^s_l using precomputed gamma
            gamma_l = scenario_gamma[sa_idx][param_id]  # γ^s_l,3 from precomputation
            ekr_sum = gp.LinExpr()
            for l in range(num_pixels):
                j1 = j1_l_spatial[l]
                j2 = j2_l_spatial[l]
                p1 = prob_cap[j1, l]  # γ^s_l,1 = P^s_l,j1
                p2 = prob_cap[j2, l]  # γ^s_l,2 = P^s_l,j2
                ekr_l = D_vec[l] * (p1 * x[j1] + p2 * x[j2] + gamma_l[l] * z[l]) * K
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

            # Post-optimization evaluation using UPDATED utility functions (IP formulas)
            En_list, Er_list = [], []
            for param_id in group:
                g0_val, sigma_val = draw_info[param_id]
                dmod_path = f'./full_grid_1km/10-3 data (Marten)/Constant_detection/D_mod/Dmod_draw_{param_id}.csv'
                density_df = pd.read_csv(dmod_path)
                D_vec = density_df['D_mod'].values * 25

                # Use NEW IP-consistent compute_expected_n
                En = compute_expected_n(ac_coords_list, trap_coords_list, g0_val, sigma_val, K, D_vec, 
                                      distances, trap_selection_array, j1_l_spatial, j2_l_spatial)
                Ec = compute_expected_c(ac_coords_list, trap_coords_list, g0_val, sigma_val, K, D_vec, 
                                      distances, trap_selection_array)
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
                "run": rerun_count,
                "runtime": round(end_time - start_time, 6),
                "E_n_avg": En_avg,
                "E_r_avg": Er_avg,
                "obj_val": m.ObjVal,
                "min_metric": min_metric
            })

            # Save results (UNCHANGED)
            budget_folder = os.path.join(base_dir, f"Budget = {budget}")
            group_run_folder = os.path.join(budget_folder, f"SA{sa_idx}_Run{rerun_count}")
            os.makedirs(group_run_folder, exist_ok=True)
            
            selected_csv_path = os.path.join(group_run_folder, f"IP-selected_ids-run{rerun_count}-budget{budget}-SA{sa_idx}.csv")
            excluded_txt_path = os.path.join(group_run_folder, f"IP-excluded_ids-run{rerun_count}-budget{budget}-SA{sa_idx}.txt")

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
results_csv_path = os.path.join(base_dir, "IP_DualCoverage_Results_SA_Groups.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"Results saved to {results_csv_path}")
