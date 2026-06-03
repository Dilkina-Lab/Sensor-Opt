######################## Uniform Deployment Baseline with tryCatch #########################
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
# traps_500m <- read.csv("SensorOpt/Bears_ConstantDetection_1km_2018/1000m_trapping_grid.csv")
# traps_500m <- traps_500m[-c(1)]

traps_500m <- read.csv("SensorOpt/Bears_ConstantDetection_1km_2018/Traps_2018_SCM.csv")
traps_500m <- traps_500m %>%
  rename(x = X, y = Y) %>%
  mutate(Trap_index = row_number())

# Remove the first unnamed index column if present (based on your earlier code)
# if ("X" %in% colnames(traps_500m)) traps_500m <- traps_500m[-1]

########################## Load param vals and true N ##########################
# gcs_get_object(
#     "MS1_5/grizz_2018/param_values_for_each_draw300_2018.csv",
#     bucket = "pgaff_simulations",
#     saveToDisk = "param_values_for_each_draw300_2018.csv",
#     overwrite = TRUE
# )
# param_vals <- read.csv("param_values_for_each_draw300_2018.csv")

param_vals <- read.csv("SensorOpt/Bears_ConstantDetection_1km_2018/param_values_for_each_draw300_2018.csv")
param_vals <- param_vals[-1]

# gcs_get_object(
#     "MS1_5/grizz_2018/True_N_per_draw.csv",
#     bucket = "pgaff_simulations",
#     saveToDisk = "True_N_per_draw.csv",
#     overwrite = TRUE
# )
# true_N <- read.csv("True_N_per_draw.csv")

true_N <- read.csv("SensorOpt/Bears_ConstantDetection_1km_2018/True_N_per_draw.csv")

########################## Sigma for mask ######################################
max_sigma <- signif(max(param_vals$sigma), 1)

#### VALIDATION PARAMETERS
# draw_ids <- c(2, 3, 12, 19, 24, 49, 56, 63, 70, 71, 92, 121, 122, 123, 124, 128, 131, 135, 138, 141,
#               156, 159, 163, 172, 173, 182, 184, 185, 188, 189, 200, 205, 206, 208, 211, 213, 221, 228, 236, 244,
#               255, 256, 260, 261, 265, 270, 281, 284, 293, 299)

# ### TEST PARAMETERS
draw_ids <- c(1, 4, 5, 9, 13, 14, 15, 21, 22, 27, 28, 29, 30, 32, 33, 35, 36, 37, 39, 40, 41,
 42, 44, 45, 48, 50, 51, 52, 53, 54, 55, 59, 62, 65, 66, 72, 75, 81, 82, 84, 86, 87, 88, 89, 90, 95,
 96, 97, 99, 100, 101, 103, 104, 106, 107, 108, 111, 113, 116, 117, 118, 126, 129, 132, 133, 134, 136, 137,
  139, 140, 142, 143, 144, 146, 147, 150, 151, 152, 154, 157, 160, 161, 162, 167, 168, 169, 170, 171, 175, 177,
   178, 179, 190, 191, 192, 196, 198, 199, 201, 202, 203, 207, 209, 210, 214, 215, 217, 218, 220, 223, 224, 226, 229,
    237, 241, 242, 243, 245, 246, 247, 248, 249, 252, 253, 254, 258, 259, 262, 263, 264, 266, 268, 269, 271, 272, 273,
     274, 276, 277, 278, 280, 285, 286, 288, 291, 292, 294, 295, 297, 300)

########################## SET SA GROUPS TO RUN HERE ##########################
sa_groups   <- c(3)      # e.g. 1, 2, 10:11, c(10, 15)
trap_counts <- c(60)
###############################################################################

for (sa in sa_groups) {
  for (n_traps in trap_counts) {

    message("\n========== SA", sa, " | ", n_traps, " traps ==========")

    # Load excluded traps for this SA group and trap count
    # exclude_path <- paste0("SensorOpt/secr/Sample Average Approximation/Bears2018_1km_FullGrid/SA", sa, "/SA", sa, "-excluded_traps-", n_traps, ".txt")
    exclude_path <- paste0("SensorOpt/secr/Sample Average Approximation/Bears2018_1km_SubsetGrid/SA", sa, "/SA", sa, "-excluded_traps-", n_traps, ".txt")


    if (!file.exists(exclude_path)) {
      message("Skipping SA", sa, " / ", n_traps, " traps — file not found: ", exclude_path)
      next
    }

    exclude_traps <- read.table(exclude_path, col.names = "Trap_index")

    # Filter out excluded traps
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
    # mask1 <- make.mask(traps = traps1, types = "polybuffer",
    #                    poly = SA_proj, buffer = max_sigma * 2, spacing = 1000)
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

    # results <- matrix(nrow = 0, ncol = 17)
    results <- matrix(nrow = 0, ncol = 14)

    for (draw in draw_ids) {
      gc()
      message("Draw ", draw)

      # ch <- read.csv(paste0("SensorOpt/full_grid_1km/ch/ch_draw_", draw, ".csv"))
      ch <- read.csv(paste0("SensorOpt/Bears_ConstantDetection_1km_2018/ch/ch_draw_", draw, ".csv"))
      # ch <- read.csv(paste0("SensorOpt/full_grid_1km/10-3 data (Marten)/Constant_detection/ch/ch_draw_", draw, ".csv"))

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

        # saveRDS(fit_model, file = paste("model_U_80A_1km_SAA2.RDS"))

        # out <- data.frame(
        #     Draw = draw,
        #     N_mod = region.N(fit_model)[2, 1],
        #     N_true = true_N$N[draw],
        #     N_abs_error = abs(region.N(fit_model)[2, 1] - true_N$N[draw]),
        #     beta0 = summary(fit_model)$coef[1, 1],
        #     beta1 = summary(fit_model)$coef[2, 1],
        #     beta2 = summary(fit_model)$coef[3, 1],
        #     beta1_true = param_vals[draw, 'beta1'],
        #     beta2_true = param_vals[draw, 'beta2'],
        #     beta1_abs_error = abs(summary(fit_model)$coef[2, 1] - param_vals[draw, 'beta1']),
        #     beta2_abs_error = abs(summary(fit_model)$coef[3, 1] - param_vals[draw, 'beta2']),
        #     sigma_mod = summary(fit_model)$predicted[3, 2],
        #     sigma_true = param_vals[draw, 'sigma'],
        #     sigma_abs_error = abs(summary(fit_model)$predicted[3, 2] - param_vals[draw, 'sigma']),
        #     g0_mod = summary(fit_model)$predicted[2, 2],
        #     g0_true = param_vals[draw, 'g0'],
        #     g0_abs_error = abs(summary(fit_model)$predicted[2, 2] - param_vals[draw, 'g0'])
        # )

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

        # out <- data.frame(
        #     Draw = draw,
        #     N_mod = NA,
        #     N_true = true_N$N[draw],
        #     N_abs_error = NA,
        #     beta0 = NA,
        #     beta1 = NA,
        #     beta2 = NA,
        #     beta1_true = param_vals[draw, 'beta1'],
        #     beta2_true = param_vals[draw, 'beta2'],
        #     beta1_abs_error = NA,
        #     beta2_abs_error = NA,
        #     sigma_mod = NA,
        #     sigma_true = param_vals[draw, 'sigma'],
        #     sigma_abs_error = NA,
        #     g0_mod = NA,
        #     g0_true = param_vals[draw, 'g0'],
        #     g0_abs_error = NA
        # )

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

    file_name <- paste0("Bear2018-subsetgrid-SA", sa, "-", n_traps, "traps-test.csv")
    write.csv(results, file = file_name, row.names = FALSE)
    message("Saved: ", file_name)
  }
}