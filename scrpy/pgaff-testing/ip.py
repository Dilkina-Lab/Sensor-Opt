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
param_draws = [6, 7, 8, 10, 11, 16, 17, 18, 20, 23, 25, 26, 31, 34, 38, 43, 46, 47, 57, 58, 60, 
              61, 64, 67, 68, 69, 73, 74, 76, 77, 78, 79, 80, 83, 85, 91, 93, 94, 98, 102, 105, 109, 110, 112, 
              114, 115, 119, 120, 125, 127, 130, 145, 148, 149, 153, 155, 158, 164, 165, 166, 174, 176, 180, 181, 
              183, 186, 187, 193, 194, 195, 197, 204, 212, 216, 219, 222, 225, 227, 230, 231, 232, 233, 234, 235, 
              238, 239, 240, 250, 251, 257, 267, 275, 279, 282, 283, 287, 289, 290, 296, 298]
budgets = [10, 20, 30, 40, 50, 60, 70, 80]           # just to test functionality, will extend to all 8 budgets later
K = 5    # number of sampling periods

base_dir = "/Users/hannahmurray/Documents/GitHub/Sensor-Opt/scrpy/pgaff-testing/secr/Integer Programming/Marten (A)"

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
results = []        # store runtimes, E_n, E_r for each parameter draw

for param_id in param_draws:
    param_row = params.iloc[param_id - 1]
    g0_val = param_row['g0']
    sigma_val = param_row['sigma']

    # Read in per-pixel densities
    dmod_path = f'./full_grid_1km/10-3 data (Marten)/Constant_detection/D_mod/Dmod_draw_{param_id}.csv'
    density_df = pd.read_csv(dmod_path)
    D_vec = density_df['D_mod'].values * 25     # scaling * 25 to keep consistent with the greedy approach, not really needed.

    for budget in budgets:
        F = -np.inf
        rerun_count = 0
        exclude_sets = []
        extra_run_flag = False  # For tracking if a second run is needed when Er < En after first run

        while True:
            m = gp.Model()
            x = m.addVars(num_traps, vtype=GRB.BINARY)          # add decision variables, x[j] = 1 if trap j is selected, 0 otherwise

            # Compute weights for the linearized surrogate objective function E_n
            alpha1 = 1/(2 * sigma_val**2)
            prob_cap = g0_val * np.exp(-alpha1 * (distances**2))  # shape (num_traps, num_cells)
            log_prob = np.log(prob_cap+ 1e-50)
            weights = np.dot(-log_prob, D_vec)  # linearized surrogate weights for E_n objective

            # Define objective (linearized E_n)
            obj = gp.quicksum(weights[j] * x[j] for j in range(num_traps))
            m.setObjective(obj, GRB.MINIMIZE)

            # Budget constraint
            m.addConstr(gp.quicksum(x[j] for j in range(num_traps)) == budget, name="budget_constraint")

            # Coefficients for linear E_c expression (already linear)
            coeffs = K * np.dot(prob_cap, D_vec)  # shapes: (num_traps, [l]) x ([l],) -> (num_traps,)
            E_c_expr = gp.quicksum(coeffs[j] * x[j] for j in range(num_traps))

            # Linear constraint for minimum recaptures (E_r = E_c - E_n surrogate)
            if rerun_count > 0:
                # Use the linearized surrogate for E_n in the constraint (consistent with objective)
                E_n_obj_expr = gp.quicksum(weights[j] * x[j] for j in range(num_traps))
                m.addConstr(E_c_expr - E_n_obj_expr <= F - delta, name = "E_r_constraint")


            # Exclude previous solutions to avoid repeats
            for prev_sel in exclude_sets:
                m.addConstr(gp.quicksum((1 - x[j]) if prev_sel[j] else x[j] for j in range(num_traps)) >= 1)

            m.Params.OutputFlag = 0
            start_time = time.time()
            m.optimize()
            end_time = time.time()
            
            if m.status == GRB.OPTIMAL and m.SolCount > 0:
                selected_idx = [j for j in range(num_traps) if x[j].X > 0.5]
                trap_selection_array = np.zeros(num_traps)
                trap_selection_array[selected_idx] = 1
                exclude_sets.append(trap_selection_array.copy())

                # Compute true expected detections and recaptures for reporting only
                En = compute_expected_n(ac_coords_list, trap_coords_list, g0_val, sigma_val, K, D_vec, distances, trap_selection_array)
                Ec = compute_expected_c(ac_coords_list, trap_coords_list, g0_val, sigma_val, K, D_vec, distances, trap_selection_array)
                Er = compute_expected_r(Ec, En)
                min_metric = "En" if En < Er else "Er"
                obj_val = m.ObjVal  # Surrogate objective value (the minimized quantity)

                results.append({
                    "param_id": param_id,
                    "budget": budget,
                    "run": rerun_count,
                    "runtime": round(end_time - start_time, 6),
                    "E_n": En,
                    "E_r": Er,
                    "obj_val": obj_val,   # Store the surrogate objective for each run
                    "min_metric": min_metric
                })
                print(f"Param {param_id}, budget {budget}, run {rerun_count} -> E_n: {En:.6f}, E_r: {Er:.6f}, obj_val: {obj_val:.6f}, min: {min_metric}")

                # Save selected/excluded trap IDs per run, for both rerun_count == 0 (when E_n < E_r) and rerun_count == 1
                if (rerun_count == 0 and En < Er) or rerun_count == 1:
                    budget_folder = os.path.join(base_dir, f"Budget = {budget}")
                    param_run_folder = os.path.join(budget_folder, f"Param{param_id}_Run{rerun_count}")
                    os.makedirs(param_run_folder, exist_ok=True)
                    selected_csv_path = os.path.join(param_run_folder, f"IP-selected_ids-run{rerun_count}-budget{budget}-param{param_id}.csv")
                    excluded_txt_path = os.path.join(param_run_folder, f"IP-excluded_ids-run{rerun_count}-budget{budget}-param{param_id}.txt")

                    selected_ids = trap_coords.iloc[selected_idx]['Trap_index'].tolist()
                    selected_df = trap_coords.iloc[selected_idx][['Trap_index', 'x', 'y']]
                    all_ids = set(trap_coords['Trap_index'])
                    excluded_ids = sorted(all_ids - set(selected_ids))

                    selected_df.to_csv(selected_csv_path, index=False)
                    with open(excluded_txt_path, "w") as f:
                        for eid in excluded_ids:
                            f.write(f"{eid}\n")

                # If after the first run, Er < En, do one more run; otherwise break loop after first run
                if rerun_count == 0 and Er < En:
                    F = Er + delta
                    rerun_count += 1
                    extra_run_flag = True
                    continue
                else:
                    break
            elif m.status == GRB.INFEASIBLE:
                m.computeIIS()
                m.write("model.ilp")
                print("Infeasible model: IIS written to model.ilp")
                print("Constraints in IIS:")
                for c in m.getConstrs():
                    if c.IISConstr > 0:
                        print(f"{c.ConstrName}")

                print("Variables with bounds in IIS:")
                for v in m.getVars():
                    if v.IISLB > 0:
                        print(f"{v.VarName} has infeasible lower bound")
                    if v.IISUB > 0:
                        print(f"{v.VarName} has infeasible upper bound")

                break
            else:
                print(f"Optimization ended with status {m.status}, no solution available.")
                break

results_df = pd.DataFrame(results)
results_csv_path = os.path.join(base_dir, "IP_En_Er_Iterations_Summary.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"Results summary saved to {results_csv_path}")
