# Load required packages
list.of.packages <- c(
  "tidyverse", "lubridate", "googleCloudStorageR",
  "dplyr", "secrdesign", "sf", "terra", "ggplot2", "lhs"
)
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = TRUE)
rm(list.of.packages); rm(new.packages)

# GCS authentication
gcs_auth(
  json_file = "SensorOpt/secr/pgaff-camera-optim.json",
  token = NULL,
  email = NULL
)


######################### Load spatial layers ###############################
TC <- rast("SensorOpt/secr/spatial_data/So_Chilcotin_TC_proj.tif")
plot(TC)

HF <- rast("SensorOpt/secr/spatial_data/So_Chilcotin_HF_proj.tif")
plot(HF)

HF_log<-log(HF+1)

SA <- st_read("SensorOpt/secr/spatial_data/study_area_2021_GCS.shp")
SA <- st_set_crs(SA, 4326)
study_area <- st_transform(SA, crs = st_crs(TC))
plot(study_area, add = TRUE, col = "red")


################ Make a traps object by subsetting the 500m grid ################
# traps_500m <- read.csv("SensorOpt/only_trail_1km/500m/trail_candidate_traps_spacing500.csv")
# traps_500m <- read.csv("SensorOpt/full_grid_500m/500m_trap_grid.csv")
traps_500m <- read.csv("SensorOpt/full_grid_1km/1000m_trap_grid.csv")


# Remove the first unnamed index column if present (based on your earlier code)
if ("X" %in% colnames(traps_500m)) traps_500m <- traps_500m[-1]

# exclude_traps <- read.table("SensorOpt/secr/Forward Greedy/FG47/FG47-excluded_traps.txt", col.names = "Trap_index")
exclude_traps <- read.table("SensorOpt/secr/Random/full_grid_1km/10traps_1_excluded.txt", col.names = "Trap_index")



# Filter out excluded traps
optim_cams <- traps_500m %>% 
  filter(!Trap_index %in% exclude_traps$Trap_index)

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


########################## Make a mask #########################################
gcs_get_object(
  "sim_8-14-2025_1kmgrid/param_values_for_each_draw150_7-24-25.csv",
  bucket = "pgaff_simulations",
  saveToDisk = "param_values_for_each_draw150_7-24-25.csv",
  overwrite = TRUE
)

# gcs_get_object(
#   "sim_update_8-1-2025_500mgrid/param_values_for_each_draw150_7-24-25.csv",
#   bucket = "pgaff_simulations",
#   saveToDisk = "param_values_for_each_draw150_7-24-25.csv",
#   overwrite = TRUE
# )

# gcs_get_object(
#   "sim_update_7-30-2025/param_values_for_each_draw150_7-24-25.csv",
#   bucket = "pgaff_simulations",
#   saveToDisk = "param_values_for_each_draw150_7-24-25.csv",
#   overwrite = TRUE
# )

param_vals <- read.csv("param_values_for_each_draw150_7-24-25.csv")
param_vals <- param_vals[-1]

max_sigma <- signif(max(param_vals$sigma), 1) # meters
SA_buffered <- st_buffer(study_area, max_sigma * 2)
plot(SA_buffered$geometry)
plot(study_area$geometry, col = "red", add = TRUE)

# **Important:** Since traps are not passed as an object now, 
# we need to create a temporary traps object for mask and plotting only:
# Use in-memory traps object here with traps_df:
traps_temp <- read.traps(data = traps_df, detector = "proximity")

mask1 <- make.mask(
  traps = traps_temp,
  type = "polybuffer",
  poly = study_area,
  buffer = max_sigma * 2,
  spacing = 500
)
plot(mask1)

coords_mask <- vect(data.frame(x = mask1$x, y = mask1$y), geom = c("x", "y"), crs = crs(study_area))

TC_extract <- extract(TC, coords_mask)
HF_extract <- extract(HF_log, coords_mask)

# Fill missing values with means
TC_extract$p047r024_TC_2015[is.na(TC_extract$p047r024_TC_2015)] <- mean(TC_extract$p047r024_TC_2015, na.rm = TRUE)
HF_extract$`hii_2020-01-01`[is.na(HF_extract$`hii_2020-01-01`)] <- mean(HF_extract$`hii_2020-01-01`, na.rm = TRUE)

covariates(mask1) <- data.frame(
  TC = scale(TC_extract[, 2])[, 1],
  HF = scale(HF_extract[, 2])[, 1]
)
summary(covariates(mask1))


############################# load true Ns for comparison #####################
gcs_get_object(
  "sim_8-14-2025_1kmgrid/True_N_per_draw.csv",
  bucket = "pgaff_simulations",
  saveToDisk = "True_N_per_draw.csv",
  overwrite = TRUE
)

# gcs_get_object(
#   "sim_update_8-1-2025_500mgrid/True_N_per_draw.csv",
#   bucket = "pgaff_simulations",
#   saveToDisk = "True_N_per_draw.csv",
#   overwrite = TRUE
# )

# gcs_get_object(
#   "sim_update_7-30-2025/True_N_per_draw.csv",
#   bucket = "pgaff_simulations",
#   saveToDisk = "True_N_per_draw.csv",
#   overwrite = TRUE
# )
true_N <- read.csv("True_N_per_draw.csv")

start_draw <- 1
end_draw <- 150

results <- matrix(nrow = 0, ncol = 16)


# ##################### Main loop ###############################################

# for (draw in start_draw:end_draw) {
  
#   gc()
#   print(draw)
  
#   ch <- read.csv(paste0("SensorOpt/full_grid_500m/8-1 data/ch/ch_draw_", draw, ".csv"))
#     # ch <- read.csv(paste0("SensorOpt/only_trail_1km/500m/ch/ch_draw_", draw, ".csv"))

  
#   # Filter detections for only traps in traps12 subset
#   ch2 <- ch %>% 
#     filter(trap_id %in% traps12$Trap_index) %>% 
#     mutate(animal = individual,
#            trap = trap_id,
#            session = 1) %>%
#     select(session, animal, occasion, trap)
  
  
#   # Prepare captures file for secr input
#   dets2 <- ch2 %>% 
#     mutate(ID = animal,
#            Detector = trap,
#            Occasion = occasion,
#            Session = paste0("Draw", draw)) %>% 
#     select(Session, ID, Occasion, Detector)
  
#   # Write detections & traps files to disk (required for file-based reading)
#   write.table(dets2, file = "dets2.txt", sep = "\t", row.names = FALSE, quote = FALSE)
#   write.table(traps_df, file = "traps.txt", sep = "\t", row.names = FALSE, quote = FALSE)
  
  
#   # Read capthist using file-based trap input (trapfile argument) to avoid errors
#   ch_secr <- read.capthist(
#     captfile = "dets2.txt",
#     trapfile = "traps.txt",
#     skip = 1,
#     detector = "proximity"
#   )
  
  
#   ################# Fit an inhomogeneous SECR model ##########################
#   system.time(fit_model <- secr.fit(
#     capthist = ch_secr,
#     mask = mask1,
#     model = list(
#       D ~ TC + HF,    # Density as function of covariates
#       g0 ~ 1,         # Constant detection probability
#       sigma ~ 1       # Constant spatial scale
#     ),
#     detectfn = "HN",
#     method = "Nelder-Mead",
#     start = list(D = 0.0001, g0 = 0.5, sigma = 3000)
#   ))
  
  
#   # Extract results for comparison
#   out <- data.frame(
#     Draw = draw,
#     N_mod = region.N(fit_model)[2, 1],
#     N_true = true_N$N[draw],
#     N_abs_error = abs(region.N(fit_model)[2, 1] - true_N$N[draw]),
#     beta1 = summary(fit_model)$coef[2, 1],
#     beta2 = summary(fit_model)$coef[3, 1],
#     beta1_true = param_vals[draw, "beta1"],
#     beta2_true = param_vals[draw, "beta2"],
#     beta1_abs_error = abs(summary(fit_model)$coef[2, 1] - param_vals[draw, "beta1"]),
#     beta2_abs_error = abs(summary(fit_model)$coef[3, 1] - param_vals[draw, "beta2"]),
#     sigma_mod = summary(fit_model)$predicted[3, 2],
#     sigma_true = param_vals[draw, "sigma"],
#     sigma_abs_error = abs(summary(fit_model)$predicted[3, 2] - param_vals[draw, "sigma"]),
#     g0_mod = summary(fit_model)$predicted[2, 2],
#     g0_true = param_vals[draw, "g0"],
#     g0_abs_error = abs(summary(fit_model)$predicted[2, 2] - param_vals[draw, "g0"])
#   )
  
#   results <- rbind(results, out)
  
# }

# file_name <- paste0("Random_80traps_10_", start_draw, "-", end_draw, ".csv")

# write.csv(results, file = file_name, row.names = FALSE)


##################### Main loop with error handling ###########################

for (draw in start_draw:end_draw) {
  
  cat("\n---- Processing draw", draw, "----\n")
  
  gc()
  
  # Load detection history
  # ch <- read.csv(paste0("SensorOpt/full_grid_500m/8-1 data/ch/ch_draw_", draw, ".csv"))
  ch <- read.csv(paste0("SensorOpt/full_grid_1km/ch/ch_draw_", draw, ".csv"))

  
  # Filter to traps of interest
  ch2 <- ch %>% 
    filter(trap_id %in% traps12$Trap_index) %>% 
    mutate(animal = individual,
           trap = trap_id,
           session = 1) %>%
    select(session, animal, occasion, trap)
  
  # Build detections table
  dets2 <- ch2 %>% 
    mutate(ID = animal,
           Detector = trap,
           Occasion = occasion,
           Session = paste0("Draw", draw)) %>% 
    select(Session, ID, Occasion, Detector)
  
  # Write files for secr input
  write.table(dets2, file = "dets2.txt", sep = "\t", row.names = FALSE, quote = FALSE)
  write.table(traps_df, file = "traps.txt", sep = "\t", row.names = FALSE, quote = FALSE)
  
  # Wrap capture history creation + model fitting in tryCatch
  tryCatch({
    
    # Read capture history
    ch_secr <- read.capthist(
      captfile = "dets2.txt",
      trapfile = "traps.txt",
      skip = 1,
      detector = "proximity"
    )
    
    # Fit SECR model
    fit_model <- secr.fit(
      capthist = ch_secr,
      mask = mask1,
      model = list(
        D ~ TC + HF,
        g0 ~ 1,
        sigma ~ 1
      ),
      detectfn = "HN",
      method = "Nelder-Mead",
      start = list(D = 0.0001, g0 = 0.5, sigma = 3000)
    )
    
    # Extract results normally
    out <- data.frame(
      Draw = draw,
      N_mod = region.N(fit_model)[2, 1],
      N_true = true_N$N[draw],
      N_abs_error = abs(region.N(fit_model)[2, 1] - true_N$N[draw]),
      beta1 = summary(fit_model)$coef[2, 1],
      beta2 = summary(fit_model)$coef[3, 1],
      beta1_true = param_vals[draw, "beta1"],
      beta2_true = param_vals[draw, "beta2"],
      beta1_abs_error = abs(summary(fit_model)$coef[2, 1] - param_vals[draw, "beta1"]),
      beta2_abs_error = abs(summary(fit_model)$coef[3, 1] - param_vals[draw, "beta2"]),
      sigma_mod = summary(fit_model)$predicted[3, 2],
      sigma_true = param_vals[draw, "sigma"],
      sigma_abs_error = abs(summary(fit_model)$predicted[3, 2] - param_vals[draw, "sigma"]),
      g0_mod = summary(fit_model)$predicted[2, 2],
      g0_true = param_vals[draw, "g0"],
      g0_abs_error = abs(summary(fit_model)$predicted[2, 2] - param_vals[draw, "g0"])
    )
    
    results <- rbind(results, out)
    
  }, error = function(e) {
    
    message("❌ Error on draw ", draw, ": ", e$message)
    
    # Fill NA values for this draw
    out <- data.frame(
      Draw = draw,
      N_mod = NA,
      N_true = true_N$N[draw],
      N_abs_error = NA,
      beta1 = NA,
      beta2 = NA,
      beta1_true = param_vals[draw, "beta1"],
      beta2_true = param_vals[draw, "beta2"],
      beta1_abs_error = NA,
      beta2_abs_error = NA,
      sigma_mod = NA,
      sigma_true = param_vals[draw, "sigma"],
      sigma_abs_error = NA,
      g0_mod = NA,
      g0_true = param_vals[draw, "g0"],
      g0_abs_error = NA
    )
    
    results <- rbind(results, out)
    
  })
}

# Save results
file_name <- paste0("Random_1km_10traps_1_", start_draw, "-", end_draw, ".csv")
write.csv(results, file = file_name, row.names = FALSE)
