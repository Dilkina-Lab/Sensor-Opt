import time
import os
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from scipy import stats
import pyreadr
import sys

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

delta = 1e-3

param_draws = [6, 7, 8, 10]  # Extend your draw list as needed
budgets = [10, 20]
base_dir = "/Users/hannahmurray/Documents/GitHub/Sensor-Opt/scrpy/pgaff-testing/secr/Integer Programming/Marten (A)/TESTING"

ac_coords = pyreadr.read_r('./full_grid_1km/10-3 data (Marten)/500m_mask_marten.RDS')
ac_coords = ac_coords[None]
ac_coords_list = ac_coords[['x', 'y']].values

trap_coords = pd.read_csv('./full_grid_1km/10-3 data (Marten)/1000m_trap_grid_marten.csv').drop(columns = ['Unnamed: 0'])
trap_coords_list = trap_coords[['x', 'y']].values
num_traps = len(trap_coords_list)

traps = trap_coords_list[:, np.newaxis, :]
centers = ac_coords_list[np.newaxis, :, :]
distances = np.linalg.norm(traps - centers, axis=2)

params = pd.read_csv('./full_grid_1km/10-3 data (Marten)/param_values_for_each_draw300_marten.csv')
results = []

for param_id in param_draws:
    param_row = params.iloc[param_id - 1]
    g0_val = param_row['g0']
    sigma_val = param_row['sigma']
    K = 5
    dmod_path = f'./full_grid_1km/10-3 data (Marten)/Constant_detection/D_mod/Dmod_draw_{param_id}.csv'
    if not os.path.exists(dmod_path):
        print(f"Density file missing for param {param_id}. Skipping.")
        continue
    density_df = pd.read_csv(dmod_path)
    D_vec = density_df['D_mod'].values * 25

    rerun_count_limit = 25
    for budget in budgets:
        F = -np.inf
        rerun_count = 0
        exclude_sets = []

        while True:
            m = gp.Model()
            x = m.addVars(num_traps, vtype=GRB.BINARY)

            alpha1 = 1/(2 * sigma_val**2)
            prob_cap = g0_val * np.exp(-alpha1 * (distances**2))  # shape (num_traps, num_cells)
            log_prob = np.log(prob_cap + 1e-10)
            weights = np.dot(-log_prob, D_vec)  # surrogate for objective

            obj = gp.quicksum(weights[j] * x[j] for j in range(num_traps))
            m.setObjective(obj, GRB.MAXIMIZE)
            m.addConstr(gp.quicksum(x[j] for j in range(num_traps)) == budget)

            # Coefficients for linear E_r expression
            Pr = prob_cap  # shape [num_traps, num_cells]
            coeffs = K * np.dot(Pr, D_vec)  # shapes: (num_traps, [l]) x ([l],) -> (num_traps,)
            er_expr = gp.quicksum(coeffs[j] * x[j] for j in range(num_traps))
            # If rerunning, add E_r constraint using previous En as constant. NOTE: This is conservative (uses best current solution's En)
            if rerun_count > 0:
                m.addConstr(er_expr - En_prev >= F + delta)

            # Exclude previous solutions to avoid repeats (if needed)
            for prev_sel in exclude_sets:
                m.addConstr(gp.quicksum((1 - x[j]) if prev_sel[j] else x[j] for j in range(num_traps)) >= 1)

            m.Params.OutputFlag = 0
            start_time = time.time()
            m.optimize()
            end_time = time.time()

            selected_idx = [j for j in range(num_traps) if x[j].X > 0.5]
            trap_selection_array = np.zeros(num_traps)
            trap_selection_array[selected_idx] = 1
            exclude_sets.append(trap_selection_array.copy())

            selected_idx = [j for j in range(num_traps) if x[j].X > 0.5]
            trap_selection_array = np.zeros(num_traps)
            trap_selection_array[selected_idx] = 1

            # Now save selected/excluded trap IDs here:

            selected_ids = trap_coords.iloc[selected_idx]['Trap_index'].tolist()
            selected_df = trap_coords.iloc[selected_idx][['Trap_index', 'x', 'y']]

            all_ids = set(trap_coords['Trap_index'])
            excluded_ids = sorted(all_ids - set(selected_ids))

            output_dir = os.path.join(base_dir, f"Param{param_id}_Budget{budget}_Run{rerun_count}")
            os.makedirs(output_dir, exist_ok=True)

            selected_csv_path = os.path.join(output_dir, "selected_ids.csv")
            excluded_txt_path = os.path.join(output_dir, "excluded_ids.txt")

            # Save files
            selected_df.to_csv(selected_csv_path, index=False)

            with open(excluded_txt_path, "w") as f:
                for eid in excluded_ids:
                    f.write(f"{eid}\n")




            # Compute En and Er using the selected configuration
            En = compute_expected_n(ac_coords_list, trap_coords_list, g0_val, sigma_val, K, D_vec, distances, trap_selection_array)
            Ec = compute_expected_c(ac_coords_list, trap_coords_list, g0_val, sigma_val, K, D_vec, distances, trap_selection_array)
            Er = compute_expected_r(Ec, En)
            min_metric = "En" if En < Er else "Er"
            results.append({
                "param_id": param_id,
                "budget": budget,
                "run": rerun_count,
                "runtime": round(end_time - start_time, 6),
                "E_n": En,
                "E_r": Er,
                "min_metric": min_metric
            })
            print(f"Param {param_id}, budget {budget}, run {rerun_count} -> E_n: {En:.6f}, E_r: {Er:.6f}, min: {min_metric}")

            if En < Er or rerun_count >= rerun_count_limit:
                break
            else:
                F = Er + delta
                En_prev = En  # Store value for constraint in rerun
                rerun_count += 1

results_df = pd.DataFrame(results)
results_csv_path = os.path.join(base_dir, "IP_En_Er_Iterations_Summary.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"Results summary saved to {results_csv_path}")