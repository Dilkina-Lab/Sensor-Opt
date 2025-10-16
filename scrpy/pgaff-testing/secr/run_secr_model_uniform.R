######################### Uniform Deployment Baseline with tryCatch #########################
# Load packages & files
list.of.packages <- c("tidyverse", "lubridate", "googleCloudStorageR", "dplyr",
                      "secrdesign", "sf", "terra", "ggplot2", "lhs")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = TRUE)
rm(list.of.packages)
rm(new.packages)
# Authenticate GCS
gcs_auth(json_file = "SensorOpt/secr/pgaff-camera-optim.json", token = NULL, email = NULL)
######################### Load in spatial layers ###############################
TC <- rast("SensorOpt/secr/spatial_data/So_Chilcotin_TC_proj.tif")
plot(TC)
#CHANGE THIS TO HF_log
HF <- rast("SensorOpt/secr/spatial_data/So_Chilcotin_HF_proj.tif")
plot(HF)
SA <- st_read("SensorOpt/secr/spatial_data/study_area_2021_GCS.shp")
SA <- st_set_crs(SA, 4326)
SA_proj <- st_transform(SA, crs = st_crs(TC))
SA_rect<-st_as_sf(st_as_sfc(st_bbox(SA_proj))) #This gets the rectangular bounding box of the study area
plot(TC)
lines(SA_rect, col = "white")
lines(SA_proj, col = "red")
################## Load & filter eligible traps ############################
# traps_500m <- read.csv("SensorOpt/full_grid_500m/500m_trap_grid.csv")
# traps_500m <- traps_500m[-c(1)]
# traps_500m <- read.csv("SensorOpt/full_grid_1km/1000m_trap_grid.csv")
# traps_500m <- read.csv("SensorOpt/full_grid_1km/9-5 data/1000m_trap_grid_8-14-25.csv")
traps_500m <- read.csv("SensorOpt/full_grid_1km/10-3 data (Marten)/1000m_trap_grid_marten.csv")
traps_500m <- traps_500m[-c(1)]
# # traps_to_remove <- read.csv("SensorOpt/secr/traps_to_remove_500m.csv")
# traps_to_remove <- read.csv("SensorOpt/full_grid_1km/traps_to_remove_1km.csv")
# remove_indices <- traps_to_remove$Trap_index
# traps_eligible <- traps_500m %>%
#     filter(!(Trap_index %in% remove_indices))
traps_eligible <- traps_500m
##################### Alternate trap generation ################################
n_cams <- 80
side <- ceiling(sqrt(st_area(SA_rect)))
grid_cols <- ceiling(sqrt(n_cams))
new_spacing <- as.numeric(floor(side / grid_cols))
trap_grid_unclip <- st_make_grid(
    SA_rect,
    cellsize = new_spacing,
    square = TRUE,
    what = "centers"
)
traps_sf <- st_intersection(st_sf(geometry = trap_grid_unclip), SA_rect)
orig_grid_sf <- st_as_sf(traps_eligible, coords = c("x", "y"), crs = st_crs(TC))
dist_matrix <- st_distance(traps_sf, orig_grid_sf)
nearest_idx <- apply(dist_matrix, 1, which.min)
coarse_grid_sf <- orig_grid_sf[nearest_idx, ]
# Print the initial number of candidate cameras
message("Initial number of candidate cameras: ", nrow(coarse_grid_sf))
if(nrow(coarse_grid_sf) > n_cams){
    buffer_dist <- 10
    inner_SA <- st_buffer(SA_rect, dist = -buffer_dist)
    buffer_traps <- which(!st_within(coarse_grid_sf, inner_SA, sparse = FALSE))
    n_trim <- nrow(coarse_grid_sf) - n_cams
    # If not enough buffer traps, sample from non-buffer traps
    if(length(buffer_traps) < n_trim){
        # Sample whatever exists, then sample remaining from rest
        remove_traps <- buffer_traps
        extra_needed <- n_trim - length(buffer_traps)
        remaining_indices <- setdiff(seq_len(nrow(coarse_grid_sf)), buffer_traps)
        if(length(remaining_indices) > 0 && extra_needed > 0){
            remove_traps <- c(remove_traps, sample(remaining_indices, extra_needed))
        }
    } else if(n_trim > 0) {
        remove_traps <- sample(buffer_traps, n_trim)
    } else {
        remove_traps <- integer(0) # nothing to remove
    }
    coarse_grid_sf <- coarse_grid_sf[-remove_traps, ]
    message("After buffer-based trimming, working with ", nrow(coarse_grid_sf), " cameras.")
}
# Defensive check: if still too few, sample from what remains
if(nrow(coarse_grid_sf) < n_cams){
    warning("Fewer traps than requested—sampling as many as possible")
    message("Final number of cameras used: ", nrow(coarse_grid_sf))
} else {
    message("Final number of cameras used: ", nrow(coarse_grid_sf))
}
if (!"Trap_index" %in% names(coarse_grid_sf)) {
    coords_match <- st_coordinates(coarse_grid_sf)
    coords_orig <- st_coordinates(orig_grid_sf)
    idx <- match(
        paste(coords_match[,1], coords_match[,2]),
        paste(coords_orig[,1], coords_orig[,2])
    )
    coarse_grid_sf$Trap_index <- traps_eligible$Trap_index[idx]
}
traps12 <- traps_eligible %>% filter(Trap_index %in% coarse_grid_sf$Trap_index)
if (nrow(traps12) == 0) stop("No matching traps found in traps12 after alternative trap selection.")
traps1 <- read.traps(data = traps12, detector = "proximity")
# traps12 holds your selected trap points as a plain data frame with columns x and y
# Just write it out as CSV to save the trap positions with their coordinates
write.csv(traps12 %>% dplyr::select(Trap_index, x, y),
          "U_80A_1km_SAA2_SelectedTrapCoordinates.csv",
          row.names = FALSE)
########################## Make a mask #########################################
# gcs_get_object(
#   "sim_8-14-2025_1kmgrid/param_values_for_each_draw150_7-24-25.csv",
#   bucket = "pgaff_simulations",
#   saveToDisk = "param_values_for_each_draw150_7-24-25.csv",
#   overwrite = TRUE
# )
# param_vals <- read.csv("param_values_for_each_draw150_7-24-25.csv")
# param_vals <- param_vals[-c(1)]
gcs_get_object(
    "Synthetic_sim/param_values_for_each_draw300_marten.csv",
    #   "sim_9-5-25_1kmgrid_300samps/param_values_for_each_draw300_9-5-25.csv",
    bucket = "pgaff_simulations",
    saveToDisk = "param_values_for_each_draw300_marten.csv",
    overwrite = TRUE
)
param_vals <- read.csv("param_values_for_each_draw300_marten.csv")
param_vals <- param_vals[-1]
max_sigma <- signif(max(param_vals$sigma), 1)
SA_buffered <- st_buffer(SA_rect, max_sigma * 2)
plot(TC)
lines(SA_buffered, col="white")
lines(SA_rect, col = "red")
mask1<-make.mask(traps=traps1, type="polybuffer",
                 poly=SA_rect, buffer=max_sigma*2, spacing = 500)
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
############################# load up a ch #####################################
# gcs_get_object(
#   "sim_8-14-2025_1kmgrid/True_N_per_draw.csv",
#   bucket = "pgaff_simulations",
#   saveToDisk = "True_N_per_draw.csv",
#   overwrite = TRUE
# )
# true_N <- read.csv("True_N_per_draw.csv")
gcs_get_object(
    "Synthetic_sim/Constant_detection/True_N_per_draw.csv",
    #   "sim_9-5-25_1kmgrid_300samps/True_N_per_draw.csv",
    bucket = "pgaff_simulations",
    saveToDisk = "True_N_per_draw.csv",
    overwrite = TRUE
)
true_N <- read.csv("True_N_per_draw.csv")
# start_draw <- 151
# end_draw <- 300
# results <- matrix(nrow = 0, ncol = 16)
# for (i in start_draw:end_draw) {
#     gc()
#     draw <- i
#     print(draw)
draw_ids <- c(1, 4, 5, 9, 13, 14, 15, 21, 22, 27, 28, 29, 30, 32, 33, 35, 36, 37, 39, 40, 41, 42, 44, 45, 48, 50, 51, 52, 53, 54, 55, 59, 62, 65, 66, 72, 75, 81,
              82, 84, 86, 87, 88, 89, 90, 95, 96, 97, 99, 100, 101, 103, 104, 106, 107, 108, 111, 113, 116, 117, 118, 126, 129, 132, 133, 134, 136, 137, 139, 140, 142, 143, 144,
              146, 147, 150, 151, 152, 154, 157, 160, 161, 162, 167, 168, 169, 170, 171, 175, 177, 178, 179, 190, 191, 192, 196, 198, 199, 201, 202, 203, 207, 209, 210, 214, 215, 217,
              218, 220, 223, 224, 226, 229, 237, 241, 242, 243, 245, 246, 247, 248, 249, 252, 253, 254, 258, 259, 262, 263, 264, 266, 268, 269, 271, 272, 273, 274, 276, 277, 278, 280, 285,
              286, 288, 291, 292, 294, 295, 297, 300)
results <- matrix(nrow = 0, ncol = 16)
for (draw in draw_ids) {
    gc()
    print(draw)

    # ch <- read.csv(paste0("SensorOpt/full_grid_1km/ch/ch_draw_", draw, ".csv"))
    ch <- read.csv(paste0("SensorOpt/full_grid_1km/10-3 data (Marten)/Constant_detection/ch/ch_draw_", draw, ".csv"))
    ch2 <- ch %>% filter(trap_id %in% traps12$Trap_index) %>%
        mutate(animal = individual,
            trap = trap_id,
            session = 1) %>%
        select(session, animal, occasion, trap)
    dets2 <- ch2 %>%
        mutate(ID = animal,
            Detector = trap,
            Occasion = occasion,
            Session = paste("Draw", draw, sep = "")) %>%
        select(Session, ID, Occasion, Detector)
    write.table(dets2, file = "dets2.txt", sep = "\t", row.names = FALSE, quote = FALSE)
    traps_df <- traps12 %>% mutate(trapID = Trap_index) %>% select(trapID, x, y)
    write.table(traps_df, file = "traps.txt", sep = "\t", row.names = FALSE, quote = FALSE)
    tryCatch({
        ch <- read.capthist(captfile = "dets2.txt",
                            trapfile = "traps.txt",
                            detector = "proximity",
                            skip = 1)
        system.time(fit_model <- secr.fit(capthist = ch,
                                        mask = mask1,
                                        model = list(D ~ TC + HF, g0 ~ 1, sigma ~ 1),
                                        detectfn = 'HN',
                                        method = "Nelder-Mead",
                                        start = list(D = 0.0001, g0 = 0.5, sigma = 3000)))
        saveRDS(fit_model, file = paste("model_U_80A_1km_SAA2.RDS"))
        out <- data.frame(
            Draw = draw,
            N_mod = region.N(fit_model)[2, 1],
            N_true = true_N$N[draw],
            N_abs_error = abs(region.N(fit_model)[2, 1] - true_N$N[draw]),
            beta1 = summary(fit_model)$coef[2, 1],
            beta2 = summary(fit_model)$coef[3, 1],
            beta1_true = param_vals[draw, 'beta1'],
            beta2_true = param_vals[draw, 'beta2'],
            beta1_abs_error = abs(summary(fit_model)$coef[2, 1] - param_vals[draw, 'beta1']),
            beta2_abs_error = abs(summary(fit_model)$coef[3, 1] - param_vals[draw, 'beta2']),
            sigma_mod = summary(fit_model)$predicted[3, 2],
            sigma_true = param_vals[draw, 'sigma'],
            sigma_abs_error = abs(summary(fit_model)$predicted[3, 2] - param_vals[draw, 'sigma']),
            g0_mod = summary(fit_model)$predicted[2, 2],
            g0_true = param_vals[draw, 'g0'],
            g0_abs_error = abs(summary(fit_model)$predicted[2, 2] - param_vals[draw, 'g0'])
        )
        results <- rbind(results, out)
    }, error = function(e) {
        message(":x: Error on draw ", draw, ": ", e$message)
        out <- data.frame(
            Draw = draw,
            N_mod = NA,
            N_true = true_N$N[draw],
            N_abs_error = NA,
            beta1 = NA,
            beta2 = NA,
            beta1_true = param_vals[draw, 'beta1'],
            beta2_true = param_vals[draw, 'beta2'],
            beta1_abs_error = NA,
            beta2_abs_error = NA,
            sigma_mod = NA,
            sigma_true = param_vals[draw, 'sigma'],
            sigma_abs_error = NA,
            g0_mod = NA,
            g0_true = param_vals[draw, 'g0'],
            g0_abs_error = NA
        )
        results <- rbind(results, out)
})
}

# file_name <- paste0("U_80A_1km_SAA2_", start_draw, "-", end_draw, ".csv")
file_name <- paste0("U_80A_1km_SAA2_A.csv")
write.csv(results, file = file_name, row.names = FALSE)