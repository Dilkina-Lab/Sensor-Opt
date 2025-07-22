####### Uniform Deployment Baseline
############### IMPORTANT:
# Make sure you are using the most recent version of the secr package (at least v 5.2.1)
###############

# Load packages & files
list.of.packages <- c("tidyverse", "lubridate", "googleCloudStorageR","dplyr",
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
SA_proj <- st_transform(SA, crs = st_crs(TC))  # Project to TC's CRS
plot(SA_proj, add = TRUE, col = "red")

################## Load & filter eligible traps ############################

# Load 500m trap grid
traps_500m <- read.csv("SensorOpt/secr/500m_trap_grid_4-24-25.csv")
traps_500m <- traps_500m[-c(1)]  # Drop first column if it's row numbers

# Load list of traps to remove
traps_to_remove <- read.csv("SensorOpt/secr/traps_to_remove_50m.csv")  # UPDATED -- Comment out if not needed.
remove_indices <- traps_to_remove$Trap_index

# Filter traps_500m to remove excluded trap indices  # UPDATED
traps_eligible <- traps_500m %>% 
    filter(!(Trap_index %in% remove_indices))

##################### Alternate trap generation ################################
# Set camera budget and grid spacing
n_cams <- 68
side <- ceiling(sqrt(st_area(SA_proj)))
grid_cols <- ceiling(sqrt(n_cams))
new_spacing <- as.numeric(floor(side / grid_cols))

# Make a new grid in projected CRS
trap_grid_unclip <- st_make_grid(
    SA_proj,
    cellsize = new_spacing,
    square = TRUE,
    what = "centers"
)

# Clip to the study area (all in projected CRS)
traps_sf <- st_intersection(st_sf(geometry = trap_grid_unclip), SA_proj)

# Now find the closest 500m grid point among eligible traps only  # UPDATED
orig_grid_sf <- st_as_sf(traps_eligible, coords = c("x", "y"), crs = st_crs(TC))

# Compute distance matrix (all in projected CRS)
dist_matrix <- st_distance(traps_sf, orig_grid_sf)

# Get the index of the minimum distance (nearest point) for each row
nearest_idx <- apply(dist_matrix, 1, which.min)

# Extract nearest points
coarse_grid_sf <- orig_grid_sf[nearest_idx, ]

# Thin to the trap budget by randomly removing n cameras on the grid edges
if(nrow(coarse_grid_sf) > n_cams){
    buffer_dist <- 1000  # Within 1000m of the SA edge
    inner_SA <- st_buffer(SA_proj, dist = -buffer_dist)
    buffer_traps <- which(!st_within(coarse_grid_sf, inner_SA, sparse = FALSE))
    n_trim <- nrow(coarse_grid_sf) - n_cams
    remove_traps <- sample(buffer_traps, n_trim, replace = FALSE)
    coarse_grid_sf <- coarse_grid_sf[-remove_traps, ]
}

# Ensure Trap_index is present for merging
if (!"Trap_index" %in% names(coarse_grid_sf)) {
    coords_match <- st_coordinates(coarse_grid_sf)
    coords_orig <- st_coordinates(orig_grid_sf)
    idx <- match(
        paste(coords_match[,1], coords_match[,2]),
        paste(coords_orig[,1], coords_orig[,2])
    )
    coarse_grid_sf$Trap_index <- traps_eligible$Trap_index[idx]  # UPDATED
}

# Filter traps_eligible to only those in the selected grid
traps12 <- traps_eligible %>% filter(Trap_index %in% coarse_grid_sf$Trap_index)  # UPDATED
if (nrow(traps12) == 0) stop("No matching traps found in traps12 after alternative trap selection.")
traps1 <- read.traps(data = traps12, detector = "proximity")

########################## Make a mask #########################################
gcs_get_object("sim_update_7-21-2025/param_values_for_each_draw150_7-21-25.csv", 
                bucket = "pgaff_simulations", 
                saveToDisk = "param_values_for_each_draw150_7-21-25.csv", 
                overwrite=TRUE)

param_vals <- read.csv("param_values_for_each_draw150_7-21-25.csv")
param_vals <- param_vals[-c(1)]

# Buffer the SA by 2*sigma
max_sigma <- signif(max(param_vals$sigma),1) # meters
SA_buffered <- st_buffer(SA_proj, max_sigma*2)
plot(SA_buffered$geometry)
plot(SA_proj$geometry, col="red", add=TRUE)

# Generate a mask around the traps
mask1 <- make.mask(traps = traps1, type="polybuffer",
                   poly=SA_proj, buffer=max_sigma*2, spacing = 500)
plot(mask1)

# Extract the values for the mask from covariate layers
coords_mask <- vect(data.frame(x=mask1$x, y=mask1$y),
                    geom = c("x", "y"), crs = crs(SA_proj))

TC_extract <- extract(TC, coords_mask)
HF_extract <- extract(HF, coords_mask)

# Some missing values, fill with the mean
TC_extract$p047r024_TC_2015[is.na(TC_extract$p047r024_TC_2015)] <-
    mean(TC_extract$p047r024_TC_2015, na.rm=TRUE)
HF_extract$`hii_2020-01-01`[is.na(HF_extract$`hii_2020-01-01`)] <-
    mean(HF_extract$`hii_2020-01-01`, na.rm=TRUE)

# Add covariates to mask
covariates(mask1) <- data.frame(TC = scale(TC_extract[,2])[,1],
                                HF = scale(HF_extract[,2])[,1])
summary(covariates(mask1))

############################# load up a ch #####################################
gcs_get_object("sim_update_7-21-2025/True_N_per_draw.csv", 
               bucket = "pgaff_simulations", 
               saveToDisk = "True_N_per_draw.csv", 
               overwrite=TRUE)
true_N <- read.csv("True_N_per_draw.csv")

# Define the range of draws you want to loop over
start_draw <- 26
end_draw <- 50

results <- matrix(nrow=0, ncol=16)
for (i in start_draw:end_draw) {
    gc()
    draw <- i
    print(draw)
    ch <- read.csv(paste("SensorOpt/500m_data/7-21 data/ch/ch_draw_",draw,".csv", sep=""))
    
    # Constrain to only traps in traps1
    ch2 <- ch %>% filter(trap_id %in% traps12$Trap_index) %>%
        mutate(animal=individual,
               trap=trap_id,
               session=1) %>%
        select(session,animal,occasion,trap)
    
    # Make the secr input files
    dets2 <- ch2 %>%
        mutate(ID=animal,
               Detector=trap, 
               Occasion=occasion,
               Session=paste("Draw",draw,sep="")) %>%
        select(Session, ID, Occasion, Detector)
    write.table(dets2, file="dets2.txt", sep = "\t", row.names=FALSE)
    
    traps_df <- traps12 %>% mutate(trapID=Trap_index) %>% select(trapID,x,y)
    write.table(traps_df, file="traps.txt", sep = "\t", row.names=FALSE)
    
    ch <- read.capthist(captfile = "dets2.txt", 
                        trapfile = "traps.txt",
                        detector = "proximity",
                        skip=1)
    
    # Fit an inhomogenous secr model
    system.time(fit_model <- secr.fit(capthist = ch, 
                                      mask = mask1,
                                      model = list(D~TC+HF, g0~1, sigma~1),
                                      detectfn = 'HN',
                                      method="Nelder-Mead", 
                                      start=list(D=0.0001, g0=0.5, sigma=3000)))
    
    saveRDS(fit_model, file=paste("model_U4.RDS"))
    
    out <- data.frame(
        Draw=draw,
        N_mod=region.N(fit_model)[2,1],
        N_true=true_N$N[draw],
        N_abs_error=abs(region.N(fit_model)[2,1] - true_N$N[draw]),
        beta1=summary(fit_model)$coef[2,1],
        beta2=summary(fit_model)$coef[3,1],
        beta1_true=param_vals[draw,'beta1'],
        beta2_true=param_vals[draw,'beta2'],
        beta1_abs_error=abs(summary(fit_model)$coef[2,1] - param_vals[draw,'beta1']),
        beta2_abs_error=abs(summary(fit_model)$coef[3,1] - param_vals[draw,'beta2']),
        sigma_mod=summary(fit_model)$predicted[3,2],
        sigma_true=param_vals[draw,'sigma'],
        sigma_abs_error=abs(summary(fit_model)$predicted[3,2] - param_vals[draw,'sigma']),
        g0_mod=summary(fit_model)$predicted[2,2],
        g0_true=param_vals[draw,'g0'],
        g0_abs_error=abs(summary(fit_model)$predicted[2,2] - param_vals[draw,'g0'])
    )
    
    results <- rbind(results, cbind(out))
}

file_name <- paste0("U4_", start_draw, "-", end_draw, ".csv")
write.csv(results, file = file_name, row.names = FALSE)
