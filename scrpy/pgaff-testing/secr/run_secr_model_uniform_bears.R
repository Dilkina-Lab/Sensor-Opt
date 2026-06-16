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
HF <- rast("SensorOpt/secr/spatial_data/So_Chilcotin_HF_proj.tif")
plot(HF)
SA <- st_read("SensorOpt/secr/spatial_data/study_area_2021_GCS.shp")
SA <- st_set_crs(SA, 4326)
SA_proj <- st_transform(SA, crs = st_crs(TC))
SA_rect <- st_as_sf(st_as_sfc(st_bbox(SA_proj)))
plot(TC)
lines(SA_rect, col = "white")
lines(SA_proj, col = "red")

################## Load & filter eligible traps ############################
traps_500m <- read.csv("SensorOpt/Bears_ConstantDetection_1km_2018/1000m_trapping_grid.csv")
traps_500m <- traps_500m[-c(1)]
traps_eligible <- traps_500m

########################## Load param vals and true N ##########################
param_vals <- read.csv("SensorOpt/Bears_ConstantDetection_1km_2018/param_values_for_each_draw300_2018.csv")
param_vals <- param_vals[-1]

true_N <- read.csv("SensorOpt/Bears_ConstantDetection_1km_2018/True_N_per_draw.csv")

########################## Sigma for mask ######################################
max_sigma <- signif(max(param_vals$sigma), 1)

########################## Draw IDs ############################################
draw_ids <- c(1, 4, 5, 9, 13, 14, 15, 21, 22, 27, 28, 29, 30, 32, 33, 35, 36, 37, 39, 40, 41, 42, 44, 45, 48, 50, 51, 52, 53, 54, 55, 59, 62, 65, 66, 72, 75, 81,
              82, 84, 86, 87, 88, 89, 90, 95, 96, 97, 99, 100, 101, 103, 104, 106, 107, 108, 111, 113, 116, 117, 118, 126, 129, 132, 133, 134, 136, 137, 139, 140, 142, 143, 144,
              146, 147, 150, 151, 152, 154, 157, 160, 161, 162, 167, 168, 169, 170, 171, 175, 177, 178, 179, 190, 191, 192, 196, 198, 199, 201, 202, 203, 207, 209, 210, 214, 215, 217,
              218, 220, 223, 224, 226, 229, 237, 241, 242, 243, 245, 246, 247, 248, 249, 252, 253, 254, 258, 259, 262, 263, 264, 266, 268, 269, 271, 272, 273, 274, 276, 277, 278, 280, 285,
              286, 288, 291, 292, 294, 295, 297, 300)

########################## SET TRAP COUNTS TO RUN HERE ########################
trap_counts <- c(20)
###############################################################################

for (n_cams in trap_counts) {

  message("\n========== Uniform deployment | ", n_cams, " traps ==========")

  # ── Build uniform grid for this trap count ──────────────────────────────
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

  message("Initial number of candidate cameras: ", nrow(coarse_grid_sf))

  if (nrow(coarse_grid_sf) > n_cams) {
      buffer_dist <- 10
      inner_SA <- st_buffer(SA_rect, dist = -buffer_dist)
      buffer_traps <- which(!st_within(coarse_grid_sf, inner_SA, sparse = FALSE))
      n_trim <- nrow(coarse_grid_sf) - n_cams

      if (length(buffer_traps) < n_trim) {
          remove_traps <- buffer_traps
          extra_needed <- n_trim - length(buffer_traps)
          remaining_indices <- setdiff(seq_len(nrow(coarse_grid_sf)), buffer_traps)
          if (length(remaining_indices) > 0 && extra_needed > 0) {
              remove_traps <- c(remove_traps, sample(remaining_indices, extra_needed))
          }
      } else if (n_trim > 0) {
          remove_traps <- sample(buffer_traps, n_trim)
      } else {
          remove_traps <- integer(0)
      }
      coarse_grid_sf <- coarse_grid_sf[-remove_traps, ]
      message("After buffer-based trimming, working with ", nrow(coarse_grid_sf), " cameras.")
  }

  if (nrow(coarse_grid_sf) < n_cams) {
      warning("Fewer traps than requested — sampling as many as possible")
  }
  message("Final number of cameras used: ", nrow(coarse_grid_sf))

  if (!"Trap_index" %in% names(coarse_grid_sf)) {
      coords_match <- st_coordinates(coarse_grid_sf)
      coords_orig  <- st_coordinates(orig_grid_sf)
      idx <- match(
          paste(coords_match[, 1], coords_match[, 2]),
          paste(coords_orig[, 1],  coords_orig[, 2])
      )
      coarse_grid_sf$Trap_index <- traps_eligible$Trap_index[idx]
  }

  traps12 <- traps_eligible %>% filter(Trap_index %in% coarse_grid_sf$Trap_index)
  if (nrow(traps12) == 0) {
      message("No matching traps found for ", n_cams, " traps — skipping.")
      next
  }

  traps1 <- read.traps(data = traps12, detector = "proximity")

  write.csv(traps12 %>% dplyr::select(Trap_index, x, y),
            paste0("Bear2018-uniform-", n_cams, "traps-SelectedTrapCoordinates.csv"),
            row.names = FALSE)

  # ── Build mask ──────────────────────────────────────────────────────────
  mask1 <- make.mask(traps = traps1, type = "polybuffer",
                     poly = SA_rect, buffer = max_sigma * 2, spacing = 500)
  plot(mask1)

  coords_mask <- vect(data.frame(x = mask1$x, y = mask1$y),
                      geom = c("x", "y"), crs = crs(SA_rect))

  TC_extract <- extract(TC, coords_mask)

  TC_extract$p047r024_TC_2015[is.na(TC_extract$p047r024_TC_2015)] <-
      mean(TC_extract$p047r024_TC_2015, na.rm = TRUE)

  covariates(mask1) <- data.frame(
      TC = scale(TC_extract[, 2])[, 1]
  )
  summary(covariates(mask1))

  # ── Loop over draws ─────────────────────────────────────────────────────
  results <- matrix(nrow = 0, ncol = 14)

  for (draw in draw_ids) {
      gc()
      message("Draw ", draw)

      ch <- read.csv(paste0("SensorOpt/Bears_ConstantDetection_1km_2018/ch/ch_draw_", draw, ".csv"))

      ch2 <- ch %>% filter(trap_id %in% traps12$Trap_index) %>%
          mutate(animal = individual, trap = trap_id, session = 1) %>%
          select(session, animal, occasion, trap)

      dets2 <- ch2 %>%
          mutate(ID = animal, Detector = trap, Occasion = occasion,
                 Session = paste("Draw", draw, sep = "")) %>%
          select(Session, ID, Occasion, Detector)

      write.table(dets2,   file = paste0("dets2_uniform_", n_cams, ".txt"),  sep = "\t", row.names = FALSE, quote = FALSE)
      write.table(traps12 %>% mutate(trapID = Trap_index) %>% select(trapID, x, y),
                  file = paste0("traps_uniform_", n_cams, ".txt"), sep = "\t", row.names = FALSE, quote = FALSE)

      tryCatch({
          ch_fit <- read.capthist(captfile = paste0("dets2_uniform_", n_cams, ".txt"),
                                  trapfile  = paste0("traps_uniform_", n_cams, ".txt"),
                                  detector  = "proximity", skip = 1)

          system.time(fit_model <- secr.fit(capthist = ch_fit, mask = mask1,
                                            model    = list(D ~ TC, g0 ~ 1, sigma ~ 1),
                                            detectfn = 'HN', method = "Nelder-Mead",
                                            start    = list(D = 0.0001, g0 = 0.5, sigma = 3000)))

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

  file_name <- paste0("Bear2018-uniform-", n_cams, "traps-test.csv")
  write.csv(results, file = file_name, row.names = FALSE)
  message("Saved: ", file_name)
}