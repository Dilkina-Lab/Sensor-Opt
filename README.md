# Optimizing Sensor Placement with Greedy Algorithms: A Case Study in Wildlife Camera Trapping for Spatial Capture-Recapture Population Estimation

This repository contains the data and code for the American pine marten case study presented in our IJCAI paper. The goal of this project is to optimize camera trap placement for Spatially Explicit Capture-Recapture (SCR) population estimation using a Sample Average Approximation (SAA) framework with greedy search. The study area is the South Chilcotin Mountains, British Columbia, Canada. While the code and data below are specific to this species and this study area, it can be adopted for various species in different locations.

---

## Repository Structure

### `Marten Data/`

All data used in the marten case study, including simulation outputs, candidate trap locations, and optimization outputs.

#### Folders

- **`AC_locations/`** — Simulated activity center (AC) locations for the marten population across the study area. Each animal is assigned a unique activity center representing the center of its home range, which governs its detection probability relative to each sensor location.
- **`ch/`** — Capture history files generated from simulation. Each capture history records which individual animals were detected at which camera traps and during which of the K=5 sampling occasions, serving as input to the SCR model.
- **`D_mod/`** — Density model outputs containing estimated spatial surfaces of marten density as a function of habitat covariates (percent tree cover β₁ and human footprint index β₂), used to simulate ecologically realistic activity center distributions.
- **`Optimized Layouts/`** — Final optimized camera trap configurations produced by the SAA-Greedy method across all eight camera budget levels (10, 20, ..., 80 cameras) for all 20 SAA groupings.

#### Files

- **`500m_mask_marten.RDS`** — A habitat mask at 500m resolution defining the state space (valid area for activity center distribution and sensor placement) within the South Chilcotin study area, with a buffer of 2σ applied around the detector array.
- **`1000m_trap_grid_marten.csv`** — The 1,023 candidate camera trap locations on a 1km spacing grid across the study area, in CSV format.
- **`1000m_trap_grid_marten.RDS`** — The same 1km candidate trap grid in RDS format for use in R.
- **`param_values_for_each_draw300_marten.csv`** — Parameter values for each of the 300 Latin Hypercube Sampling (LHS) draws from the joint distribution of the five ecological parameters: baseline detection probability (g₀ ∈ [0.01, 0.5]), movement scale (σ ∈ [200, 1000] m), tree cover effect (β₁ ∈ [0.1, 2]), human footprint effect (β₂ ∈ [−1, 0.5]), and average marten density (D ∈ [0.0015, 0.004] animals/ha). These draws form the scenario pool G used in the SAA framework.
- **`True_N_per_draw.csv`** — The true simulated population size N for each of the 300 parameter draws (N ∈ [178, 561]), used to compute mean absolute bias |N̂ − N| when evaluating optimized layouts against the test parameter set.

---

## Pipeline Steps

### `Step1_simulate_AC_and_CH.R`

Simulates American pine marten activity centers across the South Chilcotin study area and generates capture histories for a given sensor layout and parameter draw. Activity center locations are distributed according to the density surface D(l), which depends on habitat covariates. Detection probability at each trap decays with distance from an animal's activity center according to a half-normal function parameterized by g₀ and σ. This step produces the 300 sets of capture histories spanning the full LHS parameter space, which serve as input to the SAA optimization framework.

### `Step2_forward_greedy.py`

Implements the forward greedy algorithm (SAA-Greedy) for camera trap placement. Starting from the empty set, the algorithm iteratively selects the candidate trap location that produces the greatest expected reduction in RSE(N̂) across a set of T=10 training parameter scenarios drawn from the LHS pool. Performance is evaluated using the closed-form approximation of RSE(N̂) from Efford & Boulanger (2019), which avoids the computational cost of full SCR maximum likelihood estimation at each step. A key advantage of this approach is that a single run produces optimized layouts for all budget levels from 1 to B cameras simultaneously. The SAA procedure is repeated M=20 times with independent training draws; the best layout per budget is selected on a validation set of 50 draws and reported on an independent test set of 150 draws.

### `Step3_run_secr_model_SAA.R`

Fits the Spatially Explicit Capture-Recapture (SCR) model to evaluate the performance of candidate camera trap configurations on held-out parameter scenarios. Uses the `secr` R package to estimate population size N̂ and detection parameters via maximum likelihood on the simulated capture histories. Computes evaluation metrics — relative standard error RSE(N̂) and mean absolute bias |N̂ − N| — across the validation and test parameter sets for the SAA-Greedy, SAA-Genetic, and Uniform baseline configurations across all eight camera budget levels.

---

## Citation

If you use this code or data, please cite our IJCAI paper (citation to be added).
