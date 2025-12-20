# Clear memory (if necessary)
rm(list=ls())

# Load packages & files
list.of.packages <- c("tidyverse", "lubridate", "googleCloudStorageR","dplyr",
                      "secrdesign", "sf", "terra", "ggplot2", "lhs")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = TRUE)
rm(list.of.packages)
rm(new.packages)

gcs_auth(json_file = "SensorOpt/secr/pgaff-camera-optim.json", token = NULL, email = NULL)

################ Get the relevant Marten sim data inputs #######################
# gcs_get_object("Synthetic_sim/1000m_trap_grid_marten.csv", 
#                bucket = "pgaff_simulations",
#                saveToDisk = "1000m_trap_grid_marten.csv",
#                overwrite=TRUE)
# traps_500m <- read.csv("SensorOpt/full_grid_1km/10-3 data (Marten)/1000m_trap_grid_marten.csv")


gcs_get_object("Synthetic_sim/param_values_for_each_draw300_marten.csv", 
               bucket =  "pgaff_simulations", 
               saveToDisk = "param_values_for_each_draw300_marten.csv",
               overwrite=TRUE)

#Get the study area shapefile - FIXED
files <- gcs_list_objects(bucket = "gs://pgaff-camera-optim_storage", 
                          prefix = "Partner_data/Spatial_data/So_Chilcotin_Mtns/Boundary/Study_area_boundary")

destination_folder <- "./spatial_data/"
dir.create(destination_folder, showWarnings = FALSE)

# Filter for actual files only (skip directories ending in /)
file_list <- files$name[!grepl("/$", files$name)]

for (file in file_list) {
    destination_path <- file.path(destination_folder, basename(file))
    gcs_get_object(file, bucket = "gs://pgaff-camera-optim_storage", 
                   saveToDisk = destination_path,
                   overwrite = TRUE)
    cat("Downloaded:", file, "to", destination_path, "\n")
}


SA<-st_read("./spatial_data/study_area_2021_GCS.shp")
study_area <- st_transform(SA, crs = 32610)
bbox<-st_bbox(study_area)
plot(study_area$geometry)

scaled_samples<-read.csv("param_values_for_each_draw300_marten.csv")

alltrapsx<-read.csv("SensorOpt/full_grid_1km/10-3 data (Marten)/1000m_trap_grid_marten.csv")
trap_coords1<-alltrapsx%>%select(x,y)
alltraps<-read.traps(data = trap_coords1,
                     detector="proximity")

max_sigma<-ceiling(max(scaled_samples$sigma)/1000)*1000
mask1<-make.mask(traps=alltraps, type="polybuffer",
                 poly=st_as_sfc(bbox), buffer=max_sigma*2, spacing = 500)

plot(mask1)
plot(alltraps, add=TRUE, col="red")

########################### GAoptim settings ###################################
# Budgets to evaluate (number of detectors)
budgets <- seq(10, 80, by = 10)

# The 10 row indices for the draws (your SA1 group)
param_draws<-c(64, 74, 76, 78, 109, 125, 164, 204, 238, 250)

# Create local output directories
selected_dir <- "./selected_traps/"
unselected_dir <- "./unselected_traps/"
dir.create(selected_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(unselected_dir, showWarnings = FALSE, recursive = TRUE)

# Runtime tracking
runtime <- data.frame()

# Outer loop: budgets; inner loop: parameter draws
for (ntraps in budgets) {
    cat("\n============================\n",
        "Running GAoptim for budget (ntraps) =", ntraps, "\n",
        "============================\n")
    
    # Create budget subdirectories
    budget_selected_dir <- file.path(selected_dir, paste0("Budget_", ntraps))
    budget_unselected_dir <- file.path(unselected_dir, paste0("Budget_", ntraps))
    dir.create(budget_selected_dir, showWarnings = FALSE, recursive = TRUE)
    dir.create(budget_unselected_dir, showWarnings = FALSE, recursive = TRUE)
    
    for (i in seq_along(param_draws)) {
        draw_idx <- param_draws[i]
        start_time <- Sys.time()
        
        g0    <- scaled_samples$g0[draw_idx]
        sigma <- scaled_samples$sigma[draw_idx]
        
        # Cell density for this draw
        gcs_get_object(
            paste0("Synthetic_sim/Constant_detection/D_mod/Dmod_draw_", i, ".csv"),
            bucket     = "pgaff_simulations",
            saveToDisk = "density.csv",
            overwrite  = TRUE
        )
        
        D <- read.csv("density.csv")
        D <- as.numeric(D[,1])
        
        cat("  Draw", i, "(row", draw_idx, "): g0 =", round(g0, 4), 
            ", sigma =", round(sigma, 0), "\n")
        
        # Run GAoptim with timing
        timing <- system.time({
            ga_result <- GAoptim(mask      = mask1,
                                alltraps  = alltraps,
                                ntraps    = ntraps,
                                detectpar = list(lambda0 = g0, sigma = sigma),
                                noccasions = 5,
                                detectfn   = "HHN",
                                D          = D,
                                criterion  = 4)
        })
        
        runtime <- rbind(runtime, data.frame(
            Budget = ntraps,
            Draw = i,
            Row = draw_idx,
            runtime_sec = as.numeric(timing["elapsed"])
        ))
        
        # EXTRACT SELECTED AND UNSELECTED TRAPS
        selected_traps <- as.data.frame(ga_result$optimaltraps)
        selected_traps$Trap_index <- row.names(ga_result$optimaltraps)
        
        unselected_traps <- alltrapsx %>%
            filter(!Trap_index %in% selected_traps$Trap_index) %>%
            select(x, y, Trap_index)
        
        # Save local CSV files only
        selected_csv <- file.path(budget_selected_dir, 
                                paste0("selected_traps_B", ntraps, "_draw", i, "_row", draw_idx, ".csv"))
        unselected_csv <- file.path(budget_unselected_dir, 
                                  paste0("unselected_traps_B", ntraps, "_draw", i, "_row", draw_idx, ".csv"))
        
        write.csv(selected_traps, file = selected_csv, row.names = FALSE)
        write.csv(unselected_traps, file = unselected_csv, row.names = FALSE)
        
        cat("    Saved: B", ntraps, "draw", i, "row", draw_idx, "\n")
        cat("    Runtime:", round(timing["elapsed"], 2), "sec\n")
        cat("    Criterion value:", round(ga_result$optimalenrm, 4), "\n\n")

    }
    
    # Save budget runtime summary locally
    budget_runtime_csv <- file.path(budget_selected_dir, "runtime.csv")
    write.csv(runtime[runtime$Budget == ntraps, ], file = budget_runtime_csv, row.names = FALSE)
}

# Final overall runtime summary
write.csv(runtime, file = "overall_runtime.csv", row.names = FALSE)

cat("All results saved locally!\n")
cat("Structure:\n")
cat("  selected_traps/Budget_10/selected_traps_B10_draw1_row64.csv\n")
cat("  unselected_traps/Budget_10/unselected_traps_B10_draw1_row64.csv\n")
cat("  selected_traps/Budget_10/runtime.csv\n")
cat("  overall_runtime.csv\n")
