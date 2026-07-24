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
SA_rect <- st_as_sf(st_as_sfc(st_bbox(SA_proj))) #This gets the rectangular bounding box of the study area
plot(TC)
lines(SA_rect, col = "white")
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
max_sigma <- signif(max(param_vals$sigma), 1)

#### VALIDATION PARAMETERS
draw_ids <- c(2, 3, 12, 19, 24, 49, 56, 63, 70, 71, 92, 121, 122, 123, 124, 128, 131, 135, 138, 141,
              156, 159, 163, 172, 173, 182, 184, 185, 188, 189, 200, 205, 206, 208, 211, 213, 221, 228, 236, 244,
              255, 256, 260, 261, 265, 270, 281, 284, 293, 299)

########################## SET SA GROUPS TO RUN HERE ##########################
o_fun       <- 4          # matches the objective function used in your GA run
sa_groups   <- 19:20      # all 20 SA draws
trap_counts <- c(30)  # the completed budgets
###############################################################################

for (sa in sa_groups) {
  for (n_traps in trap_counts) {

    message("\n========== SA", sa, " | ", n_traps, " traps ==========")

    # Point at the unselected-traps output from the GA run
    exclude_path <- paste0("Output/Obj_fun_", o_fun, "/Budget_", n_traps,
                          "/Unselected_traps/SA", sa, "-unselected_traps_", n_traps, "traps.csv")

    if (!file.exists(exclude_path)) {
      message("Skipping SA", sa, " / ", n_traps, " traps — file not found: ", exclude_path)
      next
    }

    exclude_traps <- read.csv(exclude_path)

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
    SA_buffered <- st_buffer(SA_rect, max_sigma * 2)
    mask1 <- make.mask(traps = traps12, type = "polybuffer",
                       poly = SA_rect, buffer = max_sigma * 2, spacing = 500)
    plot(mask1)

    coords_mask <- vect(data.frame(x = mask1$x, y = mask1$y),
                        geom = c("x", "y"), crs = crs(SA_rect))

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

    results <- matrix(nrow = 0, ncol = 14)

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

      write.table(dets2, file = paste0("dets2_SA", sa, ".txt"), sep = "\t", row.names = FALSE, quote = FALSE)

      traps_df <- traps12 %>% mutate(trapID = Trap_index) %>% select(trapID, x, y)
      write.table(traps_df, file = paste0("traps_SA", sa, ".txt"), sep = "\t", row.names = FALSE, quote = FALSE)

      tryCatch({
        ch <- read.capthist(captfile = paste0("dets2_SA", sa, ".txt"),
                            trapfile = paste0("traps_SA", sa, ".txt"),
                            detector = "proximity", skip = 1)

        system.time(fit_model <- secr.fit(capthist = ch, mask = mask1,
                                          model = list(D ~ TC + HF, g0 ~ 1, sigma ~ 1),
                                          detectfn = 'HN', method = "Nelder-Mead",
                                          start = list(D = 0.0001, g0 = 0.5, sigma = 3000)))

        out <- data.frame(
          Draw            = draw,
          N_mod           = region.N(fit_model)[2, 1],
          N_true          = true_N$N[draw],
          N_abs_error     = abs(region.N(fit_model)[2, 1] - true_N$N[draw]),
          beta0           = summary(fit_model)$coef[1, 1],
          beta1           = summary(fit_model)$coef[2, 1],
          beta1_true      = param_vals[draw, 'beta1'],
          beta1_abs_error = abs(summary(fit_model)$coef[2, 1] - param_vals[draw, 'beta1']),
          sigma_mod       = summary(fit_model)$predicted[3, 2],
          sigma_true      = param_vals[draw, 'sigma'],
          sigma_abs_error = abs(summary(fit_model)$predicted[3, 2] - param_vals[draw, 'sigma']),
          g0_mod          = summary(fit_model)$predicted[2, 2],
          g0_true         = param_vals[draw, 'g0'],
          g0_abs_error    = abs(summary(fit_model)$predicted[2, 2] - param_vals[draw, 'g0'])
        )
        results <<- rbind(results, out)

      }, error = function(e) {
        message(":x: Error on draw ", draw, ": ", e$message)

        out <- data.frame(
          Draw            = draw,
          N_mod           = NA,
          N_true          = true_N$N[draw],
          N_abs_error     = NA,
          beta0           = NA,
          beta1           = NA,
          beta1_true      = param_vals[draw, 'beta1'],
          beta1_abs_error = NA,
          sigma_mod       = NA,
          sigma_true      = param_vals[draw, 'sigma'],
          sigma_abs_error = NA,
          g0_mod          = NA,
          g0_true         = param_vals[draw, 'g0'],
          g0_abs_error    = NA
        )
        results <<- rbind(results, out)
      })
    }

    file_name <- paste0("Output/Obj_fun_", o_fun, "/Budget_", n_traps,
                        "/Validation/SA", sa, "-", n_traps, "traps-validation.csv")
    dir.create(dirname(file_name), recursive = TRUE, showWarnings = FALSE)
    write.csv(results, file = file_name, row.names = FALSE)
    message("Saved: ", file_name)
  }
}