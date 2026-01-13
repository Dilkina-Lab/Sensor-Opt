############################# HANNAH'S NEW VERSIONS ##############################
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

# === ONLY USE FIRST TWO BETA FILES (NO IP METHODS) ===
beta_files <- list(
    "Genetic" = "SAA3-SA19-70traps_test_RSE.csv",
    "Greedy"  = "SA2-20-70traps_test_RSE_CELLDENSITIES.csv"
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
    
    # Summary stats (optional, but kept)
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

# === DATA-DRIVEN SCALES (MIN/MAX ACROSS BOTH METHODS) ===
global_dhat_min  <- min(all_results$mean_Dhat,  na.rm = TRUE)
global_dhat_max  <- max(all_results$mean_Dhat,  na.rm = TRUE)
global_error_min <- min(all_results$mean_error, na.rm = TRUE)
global_error_max <- max(all_results$mean_error, na.rm = TRUE)

# Keep these if you like having min/mid/max in the legend
dhat_min    <- global_dhat_min
dhat_max    <- global_dhat_max
dhat_mid    <- (dhat_min + dhat_max) / 2
dhat_breaks <- c(dhat_min, dhat_mid, dhat_max)
dhat_labels <- sprintf("%.2f", dhat_breaks)

err_min      <- global_error_min
err_max      <- global_error_max
err_mid      <- (err_min + err_max) / 2
error_breaks <- c(err_min, err_mid, err_max)
error_labels <- sprintf("%.2f", error_breaks)

# === ERROR PLOTS ONLY (FACETTED BY METHOD) ===
p_error <- ggplot(all_results, aes(x, y, fill = mean_error)) +
    geom_tile() +
    scale_fill_gradientn(
        colors = c("blue4", "blue", "lightblue", "white", "yellow", "orange", "red3"),
        limits = c(global_error_min, global_error_max),
        breaks = error_breaks,
        labels = error_labels,
        values = scales::rescale(c(
            global_error_min,
            (global_error_min * 0.8 + 0 * 0.2),
            0,
            (0 * 0.5 + global_error_max * 0.5),
            global_error_max
        )),
        oob    = scales::squish,
        name   = "Mean Bias",
        guide  = guide_colourbar(direction = "horizontal")  # horizontal legend
    ) +
    coord_equal() +
    theme_void() +
    facet_wrap(~method, ncol = 2) +
    theme(
        panel.spacing   = grid::unit(0, "lines"),  # tightly packed facets
        legend.position = "bottom"                 # legend underneath
    )

# Show in console / RStudio Plots pane
print(p_error)

# Save ONLY one TIFF containing the facetted error plots
ggsave(
    "error_only_facetted.tiff",
    p_error,
    width  = 10,
    height = 5,
    dpi    = 600,
    compression = "lzw",
    bg = "white"
)

# Inspect summary stats
print(summary_stats)
