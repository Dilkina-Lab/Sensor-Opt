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

def compute_expected_n(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
    alpha1 = (1/(2*sigma*sigma))
    prob_cap = (g0)*np.exp(-alpha1*(distances**2))  # (#traps, #ac)
    for t in range(len(trap_locs)):
        if int(trap_x[t]) == 0:
            prob_cap[t, ...] = np.zeros(ac_locs.shape[0])
    i_cap_hist = np.zeros(len(trap_locs))
    p_empty_cap_hist = compute_cond_lik_ind(len(ac_locs), prob_cap, K, len(trap_locs), i_cap_hist)
    p_nonempty = 1 - p_empty_cap_hist
    expected_n = np.sum(p_nonempty*density)
    return expected_n

def compute_cond_lik_ind(num_activity_centers, est_prob_cap, K, num_traps, ind_cap_hist):
    broadcast_i_cap_hist = np.broadcast_to(ind_cap_hist[:, np.newaxis], (num_traps, num_activity_centers))
    probs = stats.binom.pmf(broadcast_i_cap_hist, K, est_prob_cap)
    zero_mask = probs == 0.0
    log_probs = np.log(probs, where=np.invert(zero_mask))
    log_probs[zero_mask] = -sys.maxsize - 1
    log_cond_lik_sums = np.sum(log_probs, axis=0)
    return np.exp(log_cond_lik_sums)

def compute_expected_c(ac_locs, trap_locs, g0, sigma, K, density, distances, trap_x):
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
    return expected_c - expected_n


#################################################################################################################################
############                                         READ IN PARAMETERS                                              ############
#################################################################################################################################
delta = .001            # small constant for E_r constraint, adjust as needed

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
K = 5    # number of sampling periods

base_dir = "/Users/hannahmurray/Documents/GitHub/Sensor-Opt/scrpy/pgaff-testing/secr/Integer Programming/Marten (A)/TESTING"

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

# Extract information for each parameter draws
params = pd.read_csv('./full_grid_1km/10-3 data (Marten)/param_values_for_each_draw300_marten.csv')
results = []

for sa_idx, group in enumerate(sa_groups, start=1):
    print(f"Processing SA group {sa_idx} (params: {group})...")
    
    # Precompute 10 scenarios per group
    draw_info = {}
    for param_id in group:
        param_row = params.iloc[param_id - 1]
        g0_val = param_row['g0']
        sigma_val = param_row['sigma']
        dmod_path = f'./full_grid_1km/10-3 data (Marten)/Constant_detection/D_mod/Dmod_draw_{param_id}.csv'
        density_df = pd.read_csv(dmod_path)
        D_vec = density_df['D_mod'].values * 25
        draw_info[param_id] = (g0_val, sigma_val, D_vec)
    
    for budget in budgets:
        F = -np.inf
        rerun_count = 0
        exclude_sets = []

        while True:
            m = gp.Model()
            x = m.addVars(num_traps, vtype=GRB.BINARY)
            
            # === SCALARIZED OBJECTIVE: F_λ = (1-λ) E(C) + (2λ-1) E(n) ===
            lambda_param = 0.6  # must be greater than 0.5 for submodularity
            
            # Average across 10 scenarios in group
            E_c_linear = np.zeros(num_traps)
            E_n_submod = np.zeros(num_traps)
            
            for param_id in group:
                g0_val, sigma_val, D_vec = draw_info[param_id]
                alpha1 = 1/(2 * sigma_val**2)
                prob_cap = g0_val * np.exp(-alpha1 * (distances**2))  # J x L
                
                # E(C) = sum_l D_l sum_j x_j p_lj 
                E_c_linear += np.dot(prob_cap, D_vec)
                
                # E(n) marginal approximation
                for j in range(num_traps):
                    E_n_submod[j] += np.sum(D_vec * prob_cap[j, :])
            
            E_c_linear /= len(group)
            E_n_submod /= len(group)
            
            # Objective 
            obj = ((1-lambda_param) * gp.quicksum(E_c_linear[j] * x[j] for j in range(num_traps)) + 
                   max(0, 2*lambda_param-1) * gp.quicksum(E_n_submod[j] * x[j] for j in range(num_traps)))
            m.setObjective(obj, GRB.MAXIMIZE)
            
            # Budget constraint
            m.addConstr(gp.quicksum(x[j] for j in range(num_traps)) == budget)
            
            # # Rerun constraint
            # if rerun_count > 0:
            #     m.addConstr(E_r <= F - delta, name="F_constraint")
            
            # # Exclusion sets (diversity)
            # for prev_sel in exclude_sets:
            #     m.addConstr(gp.quicksum((1 - x[j]) if prev_sel[j] else x[j]
            #                             for j in range(num_traps)) >= 1)
            
            m.Params.OutputFlag = 0
            # m.Params.MIPGap = 0.01
            # m.Params.TimeLimit = 60
            
            start_time = time.time()
            m.optimize()
            end_time = time.time()
            
            if m.status == GRB.OPTIMAL and m.SolCount > 0:
                selected_idx = [j for j in range(num_traps) if x[j].X > 0.5]
                trap_selection_array = np.zeros(num_traps)
                trap_selection_array[selected_idx] = 1
                # exclude_sets.append(trap_selection_array.copy())

                # Exact 10-scenario evaluation
                En_list, Er_list = [], []
                for param_id in group:
                    g0_val, sigma_val, D_vec = draw_info[param_id]
                    En = compute_expected_n(ac_coords_list, trap_coords_list, g0_val, sigma_val, K, D_vec, distances, trap_selection_array)
                    Ec = compute_expected_c(ac_coords_list, trap_coords_list, g0_val, sigma_val, K, D_vec, distances, trap_selection_array)
                    Er = compute_expected_r(Ec, En)
                    En_list.append(En)
                    Er_list.append(Er)

                En_avg = float(np.mean(En_list))
                Er_avg = float(np.mean(Er_list))
                max_metric = "En" if En_avg > Er_avg else "Er"

                results.append({
                    "sa_group": sa_idx,
                    "group_ids": str(group),
                    "budget": budget,
                    "run": rerun_count,
                    "runtime": round(end_time - start_time, 6),
                    "E_n_avg": En_avg,
                    "E_r_avg": Er_avg,
                    "obj_val": m.ObjVal,
                    "lambda": lambda_param,
                    "max_metric": max_metric
                })

                # SAVE SELECTED AND UNSELECTED TRAP FILES
                budget_folder = os.path.join(base_dir, f"Budget = {budget}")
                group_run_folder = os.path.join(budget_folder, f"SA{sa_idx}_Run{rerun_count}")
                os.makedirs(group_run_folder, exist_ok=True)
                
                selected_csv_path = os.path.join(
                    group_run_folder,
                    f"IP-selected_ids-run{rerun_count}-budget{budget}-SA{sa_idx}.csv"
                )
                excluded_txt_path = os.path.join(
                    group_run_folder,
                    f"IP-excluded_ids-run{rerun_count}-budget{budget}-SA{sa_idx}.txt"
                )

                selected_ids = trap_coords.iloc[selected_idx]['Trap_index'].tolist()
                selected_df = trap_coords.iloc[selected_idx][['Trap_index', 'x', 'y']]
                all_ids = set(trap_coords['Trap_index'])
                excluded_ids = sorted(all_ids - set(selected_ids))

                selected_df.to_csv(selected_csv_path, index=False)
                with open(excluded_txt_path, "w") as f:
                    for eid in excluded_ids:
                        f.write(f"{eid}\n")
                
                print(f"SA{sa_idx} budget{budget} run{rerun_count} -> En:{En_avg:.6f} Er:{Er_avg:.6f} obj:{m.ObjVal:.6f} max:{max_metric}")
                
                break
                # # Rerun logic
                # if rerun_count == 0 and Er_avg > En_avg:
                #     # F = Er_avg + delta
                #     # rerun_count += 1
                #     continue
                # else:
                #     continue
                #     # break
            else:
                print(f"SA{sa_idx} budget{budget} run{rerun_count} - NO SOLUTION")
                break


results_df = pd.DataFrame(results)
results_csv_path = os.path.join(base_dir, "IP_Scalarized_Results_SA_Groups.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"Results summary saved to {results_csv_path}")
