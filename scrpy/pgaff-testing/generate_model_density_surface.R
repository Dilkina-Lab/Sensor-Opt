############################# FULL SCRIPT (CLIPPED COLOURS, GLOBAL MIN/MAX LEGEND) ##############################
# Clear memory
rm(list = ls())

# Load required packages
list.of.packages <- c(
  "tidyverse", "lubridate", "googleCloudStorageR",
  "dplyr", "sf", "secr", "terra", "ggplot2", "cowplot", "scales"
)
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[, "Package"])]
if (length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = TRUE)

# Authenticate GCS
gcs_auth(json_file = "SensorOpt/secr/pgaff-camera-optim.json")

# Download required files
gcs_get_object(
  "Synthetic_sim/param_values_for_each_draw300_marten.csv",
  bucket = "pgaff_simulations",
  saveToDisk = "param_values_for_each_draw300_marten.csv",
  overwrite = TRUE
)
gcs_get_object(
  "Synthetic_sim/500m_mask_marten.RDS",
  bucket = "pgaff_simulations",
  saveToDisk = "500m_mask_marten.RDS",
  overwrite = TRUE
)

# Load mask and covariates
mask <- readRDS("500m_mask_marten.RDS")
mask_covs <- covariates(mask)

# === BETA FILES ===
beta_files <- list(
  "Genetic" = "SAA3-SA19-70traps_test_RSE.csv",
  "Greedy"  = "SA2-20-70traps_test_RSE_CELLDENSITIES.csv",
  "Uniform" = "U_70traps_test.csv"
)

# Process each method
results_list  <- list()
summary_stats <- data.frame()

for (method_name in names(beta_files)) {
  cat("Processing", method_name, "...\n")
  
  # Load betas and CHECK COLUMN NAMES
  betas <- read.csv(beta_files[[method_name]])
  cat("Column names in", method_name, ":\n")
  print(colnames(betas)[1:10])
  cat("\n")
  
  betas <- betas %>% filter(N_abs_error <= 1000)
  cat("Loaded", nrow(betas), "draws for", method_name, "\n")
  
  # Storage matrices
  all_Dhat  <- matrix(NA, nrow = nrow(mask), ncol = nrow(betas))
  all_error <- matrix(NA, nrow = nrow(mask), ncol = nrow(betas))
  
  # Loop through draws - ROBUST COEFS EXTRACTION
  for (i in 1:nrow(betas)) {
    # Find beta columns dynamically
    beta_cols <- intersect(
      colnames(betas),
      c("beta0", "beta1", "beta2",
        "Beta0", "Beta1", "Beta2",
        "b0", "b1", "b2")
    )
    
    if (length(beta_cols) != 3) {
      cat("ERROR: Could not find 3 beta columns in", method_name, "\n")
      cat("Available columns:", paste(colnames(betas), collapse = ", "), "\n")
      next
    }
    
    coefs <- as.numeric(betas[i, beta_cols[1:3]])
    D_tmp <- exp(coefs[1] + coefs[2] * mask_covs$TC + coefs[3] * mask_covs$HF)
    
    pixel_area_ha <- 0.5 * 0.5 * 100
    EN_pred <- sum(D_tmp) * pixel_area_ha
    scale_factor <- EN_pred / betas[i, "N_mod"]
    Dhat <- D_tmp / scale_factor
    
    draw <- betas[i, "Draw"]
    gcs_get_object(
      paste0("Synthetic_sim/Constant_detection/D_mod/Dmod_draw_", draw, ".csv"),
      bucket = "pgaff_simulations",
      saveToDisk = "density_temp.csv",
      overwrite = TRUE
    )
    Dtrue <- read.csv("density_temp.csv")
    
    Dout <- Dtrue %>%
      mutate(xy = paste(x, y, sep = "_")) %>%
      left_join(
        data.frame(
          xy   = paste(mask$x, mask$y, sep = "_"),
          Dhat = Dhat
        ),
        by = "xy"
      ) %>%
      select(-xy) %>%
      mutate(signed_error = Dhat - D_mod)
    
    all_Dhat[, i]  <- Dhat
    all_error[, i] <- Dout$signed_error
  }
  
  # Compute means
  mean_Dhat  <- rowMeans(all_Dhat,  na.rm = TRUE)
  mean_error <- rowMeans(all_error, na.rm = TRUE)
  
  # Store results
  results_list[[method_name]] <- data.frame(
    x          = mask$x,
    y          = mask$y,
    mean_Dhat  = mean_Dhat,
    mean_error = mean_error,
    method     = method_name
  )
  
  # Summary stats (per method)
  summary_stats <- rbind(
    summary_stats,
    data.frame(
      method    = method_name,
      Dhat_min  = min(mean_Dhat,  na.rm = TRUE),
      Dhat_max  = max(mean_Dhat,  na.rm = TRUE),
      error_min = min(mean_error, na.rm = TRUE),
      error_max = max(mean_error, na.rm = TRUE)
    )
  )
  
  cat(method_name, "complete.\n\n")
}

# Combine all results
all_results <- bind_rows(results_list)

# Optional: CSV outputs
write.csv(all_results, "all_methods_density_error_summary.csv", row.names = FALSE)
write.csv(summary_stats, "summary_statistics.csv", row.names = FALSE)

# ================== FACETTED MEAN-BIAS MAP (CLIPPED COLOURS, GLOBAL MIN/MAX LEGEND) ==================

err_vec <- all_results$mean_error

# 1) Colour clipping: focus colours on inner quantiles of all mean_error
q <- quantile(err_vec, probs = c(0.01, 0.99), na.rm = TRUE)
clip_min <- q[1]
clip_max <- q[2]

# 2) Global limits for legend end labels (from summary_stats)
err_min <- min(summary_stats$error_min, na.rm = TRUE)
err_max <- max(summary_stats$error_max, na.rm = TRUE)
err_mid <- 0  # white at zero

# Legend labels use global min/max
legend_breaks <- c(err_min, err_max)
legend_labels <- sprintf("%.4f", legend_breaks)

p_error <- ggplot(all_results, aes(x, y, fill = mean_error)) +
  geom_tile() +
  scale_fill_gradient2(
    low      = "blue4",
    mid      = "white",
    high     = "red3",
    midpoint = err_mid,
    limits   = c(clip_min, clip_max),
    breaks   = legend_breaks,
    labels   = legend_labels,
    oob      = squish,
    na.value = "white",
    name     = "Mean Bias",
    guide    = guide_colourbar(direction = "horizontal")
  ) +
  coord_equal() +
  theme_void() +
  facet_wrap(~method, ncol = 2) +
  theme(
    panel.spacing   = grid::unit(0, "lines"),
    legend.position = "bottom"   # back to below the panels
  )


print(p_error)

ggsave(
  "error_only_facetted_clipped.tiff",
  p_error,
  width  = 10,
  height = 5,
  dpi    = 600,
  compression = "lzw",
  bg = "white"
)

print(summary_stats)
