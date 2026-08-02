setwd("/project2/dilkina_438/hannahmu")
######################## Validation secr runs using GA-optimized trap sets #########################
# Load packages & files
list.of.packages <- c("tidyverse", "lubridate", "dplyr",
                      "secrdesign", "sf", "terra", "ggplot2", "lhs")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = TRUE)
rm(list.of.packages)
rm(new.packages)
# # Authenticate GCS
# gcs_auth(json_file = "SensorOpt/secr/pgaff-camera-optim.json", token = NULL, email = NULL)

######################### Load in spatial layers ###############################
TC <- rast("SensorOpt/secr/spatial_data/So_Chilcotin_TC_proj.tif")
plot(TC)
# CHANGE THIS TO HF_log
HF <- rast("SensorOpt/secr/spatial_data/So_Chilcotin_HF_proj.tif")
plot(HF)
SA <- st_read("SensorOpt/secr/spatial_data/study_area_2021_GCS.shp")
SA <- st_set_crs(SA, 4326)
SA_proj <- st_transform(SA, crs = st_crs(TC))
plot(TC)
lines(SA_proj, col = "red")

################## Load base traps ############################
traps_500m <- read.csv("SensorOpt/Bears_ConstantDetection_1km_2018_2/Traps_2018_SCM.csv")
traps_500m <- traps_500m %>%
  mutate(Trap_index = row_number())

########################## Load param vals and true N ##########################
# NOTE: this points at the 2018 data folder while your GA run used 2018 data for
# param_vals/traps_500m - double check this is intentional before running.
param_vals <- read.csv("SensorOpt/Bears_ConstantDetection_1km_2018_2/param_values_for_each_draw300_2018.csv")
param_vals <- param_vals[-1]

true_N <- read.csv("SensorOpt/Bears_ConstantDetection_1km_2018_2/True_N_per_draw.csv")

########################## Sigma for mask ######################################
# max_sigma <- signif(max(param_vals$sigma), 1).  
# max_sigma <- 5000             ### USE FOR 2019 RUNS
max_sigma <- 7500               ### USE FOR 2018 RUNS

#### VALIDATION PARAMETERS
draw_ids <- c(2, 3, 12, 19, 24, 49, 56, 63, 70, 71, 92, 121, 122, 123, 124, 128, 131, 135, 138, 141,
              156, 159, 163, 172, 173, 182, 184, 185, 188, 189, 200, 205, 206, 208, 211, 213, 221, 228, 236, 244,
              255, 256, 260, 261, 265, 270, 281, 284, 293, 299)

########################## SET SA GROUPS TO RUN HERE ##########################
o_fun       <- 4          # matches the objective function used in your GA run
sa_groups   <- c(16:20)      
trap_counts <- c(30)  # the completed budgets
###############################################################################

for (sa in sa_groups) {
  for (n_traps in trap_counts) {

    message("\n========== SA", sa, " | ", n_traps, " traps ==========")

    # Point at the unselected-traps output from the GA run
    # exclude_path <- paste0("Output/Obj_fun_", o_fun, "/Budget_", n_traps,
    #                       "/Unselected_traps/SA", sa, "-unselected_traps_", n_traps, "traps.csv")

    # Unselected traps from the greedy runs
    exclude_path <- paste0("SensorOpt/secr/Sample Average Approximation/Bears2018_1km_SubsetGrid_2/SA", sa,
                      "/SA", sa, "-excluded_traps-", n_traps, ".txt")


    if (!file.exists(exclude_path)) {
      message("Skipping SA", sa, " / ", n_traps, " traps — file not found: ", exclude_path)
      next
    }

    #genetic
    # exclude_traps <- read.csv(exclude_path)

    #greedy
    exclude_traps <- read.table(exclude_path, col.names = "Trap_index")

    # Filter out excluded (unselected) traps
    optim_cams <- traps_500m %>% filter(!Trap_index %in% exclude_traps$Trap_index)

    # Select only optim_cams traps
    traps12 <- traps_500m %>% filter(Trap_index %in% optim_cams$Trap_index)

    # Prepare traps data frame for secr input (no 'usage' column)
    traps_df <- traps12 %>%
      mutate(trapID = Trap_index) %>%
      select(trapID, x, y)

    # Ensure numeric coordinates and correct column names
    traps_df$x <- as.numeric(traps_df$x)
    traps_df$y <- as.numeric(traps_df$y)
    names(traps_df) <- c("trapID", "x", "y")

    # Build mask for this trap set
    SA_buffered <- st_buffer(SA_proj, max_sigma * 2)
    mask1 <- make.mask(traps = traps12, type = "polybuffer",
                       poly = SA_proj, buffer = max_sigma * 2, spacing = 500)       ## make sure this is consistent with all the files
    plot(mask1)
    ### population between 2018/19 isn't changing very much, so its really just going to rexplore the full parameter set.
    ### do full **opt/val/test 2018** (test 2019 sims) and (test 2019 empirical)
            ### 1. for 2018 Validation: use the 2018 mask, 2018 max_sigma (RERUN WITH SA_proj IN MASK)
            ### 4 .for 2018 Test: use the 2018 mask, 2018 max_sigma (NOT CURRENTLY BEEN RUN, USE SA_PROJ)
                    ### to see if the chronic underestimation is also happening here. If it is, then theres some other issue?
                    ### if there isn't an issue, then the conclusion is that optimization on 2018 does not translate to 2019, although we don't expect much change from yr to yr.
            ### 2. for 2019 Test: use the 2019 mask, 2019 max_sigma (RERUN WITH THE SA_PROJ file)
                    ### chronic underestimation. make sure the mask is correct. If it still underestimates, 
            ### 3. for 2019 Empirical: 2019 mask, 2019 max_sigma
            ### 5. for 2019 Empirical: 10000 max_sigma


    coords_mask <- vect(data.frame(x = mask1$x, y = mask1$y),
                        geom = c("x", "y"), crs = crs(SA_proj))

    TC_extract <- extract(TC, coords_mask)
    HF_extract <- extract(HF, coords_mask)

    TC_extract$p047r024_TC_2015[is.na(TC_extract$p047r024_TC_2015)] <-
      mean(TC_extract$p047r024_TC_2015, na.rm = TRUE)
    HF_extract$`hii_2020-01-01`[is.na(HF_extract$`hii_2020-01-01`)] <-
      mean(HF_extract$`hii_2020-01-01`, na.rm = TRUE)

    covariates(mask1) <- data.frame(
      TC = scale(TC_extract[, 2])[, 1],
      HF = scale(HF_extract[, 2])[, 1]
    )
    summary(covariates(mask1))

    results_list <- list()

    # file_name <- paste0("Output/Obj_fun_", o_fun, "/Budget_", n_traps,
    #                 "/Validation/SA", sa, "-", n_traps, "traps-validation.csv")


    file_name <- paste0("Output/Greedy/Budget_", n_traps,
                    "/Validation/SA", sa, "-", n_traps, "traps-validation.csv")

    dir.create(dirname(file_name), recursive = TRUE, showWarnings = FALSE)

    for (draw in draw_ids) {
      gc()
      message("Draw ", draw)
 
      ch <- read.csv(paste0("SensorOpt/Bears_ConstantDetection_1km_2018_2/ch/ch_draw_", draw, ".csv"))
 
      ch2 <- ch %>% filter(trap_id %in% traps12$Trap_index) %>%
        mutate(animal = individual, trap = trap_id, session = 1) %>%
        select(session, animal, occasion, trap)
 
      dets2 <- ch2 %>%
        mutate(ID = animal, Detector = trap, Occasion = occasion,
               Session = paste("Draw", draw, sep = "")) %>%
        select(Session, ID, Occasion, Detector)
 
      write.table(dets2, file = paste0("dets2_SA", sa, "_budget", n_traps, ".txt"), sep = "\t", row.names = FALSE, quote = FALSE)
 
      traps_df <- traps12 %>% mutate(trapID = Trap_index) %>% select(trapID, x, y)
      write.table(traps_df, file = paste0("traps_SA", sa, "_budget", n_traps, ".txt"), sep = "\t", row.names = FALSE, quote = FALSE)
 
      out <- tryCatch({
 
        if (nrow(dets2) == 0) {
          stop("No detections at any selected trap for this draw")
        }
 
        ch <- read.capthist(captfile = paste0("dets2_SA", sa, "_budget", n_traps, ".txt"),
                    trapfile = paste0("traps_SA", sa, "_budget", n_traps, ".txt"),
                    detector = "proximity", skip = 1)
 
        system.time(fit_model <- secr.fit(capthist = ch, mask = mask1,
                                          model = list(D ~ TC + HF, g0 ~ 1, sigma ~ 1),
                                          detectfn = 'HN', method = "Nelder-Mead",
                                          start = list(D = 0.0001, g0 = 0.5, sigma = 3000)))
 
        N_mod_mat <- region.N(fit_model)
        pred      <- summary(fit_model)$predicted
        coefs     <- summary(fit_model)$coef
 
        data.frame(
          Draw              = draw,
          N_mod             = N_mod_mat[2, 1],
          N_lower_95CI      = N_mod_mat[2, 3],
          N_upper_95CI      = N_mod_mat[2, 4],
          N_true            = true_N$N[draw],
          N_abs_error       = abs(N_mod_mat[2, 1] - true_N$N[draw]),
          beta0             = coefs[1, "beta"],
          beta1             = coefs[2, "beta"],
          beta1_lower_95CI  = coefs[2, "lcl"],
          beta1_upper_95CI  = coefs[2, "ucl"],
          beta1_true        = param_vals[draw, 'beta1'],
          beta1_abs_error   = abs(coefs[2, "beta"] - param_vals[draw, 'beta1']),
          sigma_mod         = pred["sigma", "estimate"],
          sigma_lower_95CI  = pred["sigma", "lcl"],
          sigma_upper_95CI  = pred["sigma", "ucl"],
          sigma_true        = param_vals[draw, 'sigma'],
          sigma_abs_error   = abs(pred["sigma", "estimate"] - param_vals[draw, 'sigma']),
          g0_mod            = pred["g0", "estimate"],
          g0_lower_95CI     = pred["g0", "lcl"],
          g0_upper_95CI     = pred["g0", "ucl"],
          g0_true           = param_vals[draw, 'g0'],
          g0_abs_error      = abs(pred["g0", "estimate"] - param_vals[draw, 'g0'])
        )
 
      }, error = function(e) {
        message(":x: Error on draw ", draw, ": ", e$message)
 
        data.frame(
          Draw              = draw,
          N_mod             = NA,
          N_lower_95CI      = NA,
          N_upper_95CI      = NA,
          N_true            = true_N$N[draw],
          N_abs_error       = NA,
          beta0             = NA,
          beta1             = NA,
          beta1_lower_95CI  = NA,
          beta1_upper_95CI  = NA,
          beta1_true        = param_vals[draw, 'beta1'],
          beta1_abs_error   = NA,
          sigma_mod         = NA,
          sigma_lower_95CI  = NA,
          sigma_upper_95CI  = NA,
          sigma_true        = param_vals[draw, 'sigma'],
          sigma_abs_error   = NA,
          g0_mod            = NA,
          g0_lower_95CI     = NA,
          g0_upper_95CI     = NA,
          g0_true           = param_vals[draw, 'g0'],
          g0_abs_error      = NA
        )
      })
 
      results_list[[as.character(draw)]] <- out
 
      # Save incrementally after every draw so a crash on any later draw
      # doesn't lose progress already made
      results <- bind_rows(results_list)
      write.csv(results, file = file_name, row.names = FALSE)
    }
 
    message("Saved: ", file_name)
  }
}