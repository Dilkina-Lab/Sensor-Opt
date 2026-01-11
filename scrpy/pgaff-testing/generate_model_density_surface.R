# Clear memory (if necessary)
rm(list=ls())

# Load packages & files
# List of packages needed
list.of.packages <- c("tidyverse", "lubridate", "googleCloudStorageR","dplyr",
                      "sf", "secr", "ggplot2", "cowplot")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = TRUE)
rm(list.of.packages)
rm(new.packages)

gcs_auth(json_file = "SensorOpt/secr/pgaff-camera-optim.json", token = NULL, email = NULL)

################ Get the relevant Marten sim data inputs #######################
gcs_get_object("Synthetic_sim/param_values_for_each_draw300_marten.csv", 
               bucket =  "pgaff_simulations", 
               saveToDisk = "param_values_for_each_draw300_marten.csv",
               overwrite=TRUE)

#Get the study area shapefile
files <- gcs_list_objects(bucket = "gs://pgaff-camera-optim_storage", 
                          prefix = "Partner_data/Spatial_data/So_Chilcotin_Mtns/Boundary/Study_area_boundary")

# Define the local folder where you want to download the files
# Make this folder in working directory
destination_folder <- "./spatial_data/"

# Loop through each file and download it
for (file in files$name) {
  # Define the destination file path
  destination_path <- file.path(destination_folder, basename(file))
  
  # Download the file from GCS
  gcs_get_object(file, bucket = "gs://pgaff-camera-optim_storage", 
                 saveToDisk = destination_path,
                 overwrite=TRUE)
  
  # Print progress
  cat("Downloaded:", file, "to", destination_path, "\n")
}

gcs_get_object("Synthetic_sim/500m_mask_marten.RDS", 
               bucket =  "pgaff_simulations", 
               saveToDisk = "500m_mask_marten.RDS",
               overwrite=TRUE)

############################# Load data ########################################
scaled_samples<-read.csv("param_values_for_each_draw300_marten.csv")

mask<-readRDS("500m_mask_marten.RDS")

#Extract the mask covariates used to fit the model
mask_covs<-covariates(mask)

################## Load in the beta outputs from the SECR fit ##################
# betas<-read.csv("SAA3-SA19-70traps_test_RSE.csv")
# betas<-read.csv("SA2-20-70traps_test_RSE_CELLDENSITIES.csv")
betas<-read.csv("IP2-70traps_SA18_test_RSE.csv")
head(betas)

#Filter out any N_abs_error>1000
betas<-betas%>%filter(N_abs_error<=1000)
hist(betas$N_abs_error)

################## Use the betas to get cell-specific densities ################

#Storage for mean Dhat from all draws
all_Dhat<-all_error<-matrix(NA, nrow=nrow(mask), ncol=nrow(betas))

#Looping through the betas and using them to generate cell-specific density estimates
for(i in 1:nrow(betas)){
  
  coefs<-betas[i,c("beta0","beta1","beta2")]
  
  #Manually computing the linear predictor for each cell
  D_tmp<-exp(coefs[,1]+coefs[,2]*mask_covs$TC+coefs[,3]*mask_covs$HF)
  
  #Now these need to be scaled to the N_mod
  pixel_area_ha<-0.5*0.5*100  #500m x 500m = 25 ha
  EN_pred<-sum(D_tmp)*pixel_area_ha
  
  #Since the EN_pred is very often larger than N_mod, we scale all cell-densities
  #back to N_mod. This preserves the spatial pattern and maintains the N level
  #indicated by the model.
  scale_factor<-EN_pred/betas[i,"N_mod"]
  
  Dhat<-D_tmp/scale_factor
  
  #Join to the mask coordinates
  mask2<-mask
  mask2$Dhat<-Dhat
  
  #Get the true cell-specific density
  draw<-betas[i,"Draw"]
  
  gcs_get_object(paste("Synthetic_sim/Constant_detection/D_mod/Dmod_draw_",draw,".csv",sep=""), 
                 bucket =  "pgaff_simulations", 
                 saveToDisk = "density_temp.csv",
                 overwrite=TRUE)
  
  Dtrue<-read.csv("density_temp.csv")
  
  #Join Dhat and Dtrue
  Dout<-Dtrue%>%mutate(xy=paste(x,y,sep="_"))%>%
    left_join(mask2%>%mutate(xy=paste(x,y,sep="_"))%>%
                select(xy,Dhat), by="xy")%>%select(!xy)%>%
    mutate(Draw=draw,
           signed_error=Dhat-D_mod)%>%
    select(Draw,D_mod,EN,Dhat,signed_error,x,y)
  
  #Save locally
  write.csv(Dout,file=paste("Draw_",draw,"_predicted_density.csv",sep=""))
  
  #But also store in the matrix for averaging
  all_Dhat[,i]<-Dhat
  all_error[,i]<-Dout$signed_error
  
  #Quick maps
  p1<-ggplot(Dout,aes(x=x,y=y,fill=D_mod))+
    geom_tile()+
    scale_fill_viridis_c()+
    coord_equal()+
    theme_void()
  
  p2<-ggplot(Dout,aes(x=x,y=y,fill=Dhat))+
    geom_tile()+
    scale_fill_viridis_c()+
    coord_equal()+
    theme_void()
  
  p3<-ggplot(Dout,aes(x=x,y=y,fill=signed_error))+
    geom_tile()+
    scale_fill_gradient2(low = "blue",mid = "white",high = "red",
                         midpoint = 0,
                         name = "Dhat-Dmod")+
    coord_equal()+
    theme_void()
  
  plot_grid(p1,p2,p3,ncol=3)
  
  #Save plot
  ggsave(paste("Draw_",draw,"_predicted_density_map.tiff",sep=""), bg="white", 
         height=7, width=10, dpi=600, compression="lzw")
}

#Get the average predicted cell-density across draws
mean_Dhat<-rowMeans(all_Dhat, na.rm=TRUE)

#Add to mask for plotting
mask3<-mask
mask3$mean_Dhat<-mean_Dhat

#Save locally
write.csv(mask3, file="mean_predicted_cell_densities.csv")

#Make a plot and save
ggplot(mask3,aes(x=x,y=y,fill=mean_Dhat))+
  geom_tile()+
  scale_fill_viridis_c()+
  coord_equal()+
  theme_void()

ggsave(file="mean_predicted_cell_densities.tiff", bg="white", 
       height=7, width=10, dpi=600, compression="lzw")

#Get the average error across draws
mean_error<-rowMeans(all_error, na.rm=TRUE)

#Add to mask for plotting
mask4<-Dout
mask4$mean_error<-mean_error

#Save locally
write.csv(mask4, file="mean_error.csv")

#Make a plot and save
ggplot(mask4,aes(x=x,y=y,fill=mean_error))+
  geom_tile()+
  scale_fill_gradient2(low = "blue",mid = "white",high = "red",
                       midpoint = 0,
                       name = "Mean Dhat-Dmod")+
  coord_equal()+
  theme_void()

ggsave(file="mean_error.tiff", bg="white", 
       height=7, width=10, dpi=600, compression="lzw")



############################# HANNAH'S NEW VERSIONS ##############################
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


# Define the four beta files
beta_files <- list(
  "Genetic"     = "SAA3-SA19-70traps_test_RSE.csv",
  "Greedy"      = "SA2-20-70traps_test_RSE_CELLDENSITIES.csv",
  "IP (3 trap)" = "IP4-70traps_SA3_test_RSE.csv",
  "IP (4 trap)" = "IP5-70traps-SA3-test-RSE.csv"
)


# Process each method
results_list <- list()
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
  
  # Save GeoTIFFs
  r_template <- rast(mask)
  r_template$mean_Dhat  <- mean_Dhat
  r_template$mean_error <- mean_error
  
  writeRaster(
    r_template[["mean_Dhat"]],
    paste0(method_name, "_density_geotiff.tif"),
    overwrite = TRUE,
    gdal = c("COMPRESS=LZW", "TILED=YES")
  )
  writeRaster(
    r_template[["mean_error"]],
    paste0(method_name, "_error_geotiff.tif"),
    overwrite = TRUE,
    gdal = c("COMPRESS=LZW", "TILED=YES")
  )
  
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
write.csv(all_results, "all_methods_density_error_summary.csv", row.names = FALSE)
write.csv(summary_stats, "summary_statistics.csv", row.names = FALSE)


# === CLEAN LEGENDS: MIN/MID/MAX ONLY ===
global_dhat_range  <- c(0, 0.30)
global_error_range <- c(-0.10, 0.30)

dhat_min    <- global_dhat_range[1]
dhat_mid    <- mean(global_dhat_range)
dhat_max    <- global_dhat_range[2]
dhat_breaks <- c(dhat_min, dhat_mid, dhat_max)
dhat_labels <- sprintf("%.2f", dhat_breaks)

err_min       <- global_error_range[1]
err_mid       <- mean(global_error_range)
err_max       <- global_error_range[2]
error_breaks  <- c(err_min, err_mid, err_max)
error_labels  <- sprintf("%.2f", error_breaks)


# === DENSITY PLOTS ===
p_density <- ggplot(all_results, aes(x, y, fill = mean_Dhat)) +
  geom_tile() +
  scale_fill_viridis_c(
    limits = global_dhat_range,
    breaks = dhat_breaks,
    labels = dhat_labels,
    oob    = scales::squish,
    name   = "Mean Dhat"
  ) +
  coord_equal() +
  theme_void() +
  facet_wrap(~method, ncol = 2)

print(p_density)
ggsave(
  "all_methods_density.tiff",
  p_density,
  width = 15,
  height = 5,
  dpi = 600,
  compression = "lzw",
  bg = "white"
)


# Individual density plots
for (method_name in names(beta_files)) {
  p <- ggplot(results_list[[method_name]], aes(x, y, fill = mean_Dhat)) +
    geom_tile() +
    scale_fill_viridis_c(
      limits = global_dhat_range,
      breaks = dhat_breaks,
      labels = dhat_labels,
      oob    = scales::squish,
      name   = "Mean Dhat"
    ) +
    coord_equal() +
    theme_void() +
    ggtitle(paste(method_name, "Density"))
  
  print(p)
  ggsave(
    paste0(method_name, "_density.tiff"),
    p,
    width = 10,
    height = 7,
    dpi = 600,
    compression = "lzw",
    bg = "white"
  )
}


# === ERROR PLOTS ===
p_error <- ggplot(all_results, aes(x, y, fill = mean_error)) +
  geom_tile() +
  scale_fill_gradientn(
    colors = c("blue4", "blue", "lightblue", "white", "yellow", "orange", "red3"),
    limits = global_error_range,
    breaks = error_breaks,
    labels = error_labels,
    values = scales::rescale(c(-0.10, -0.02, 0, 0.02, 0.05, 0.10, 0.30)),
    oob    = scales::squish,
    name   = "Mean Error"
  ) +
  coord_equal() +
  theme_void() +
  facet_wrap(~method, ncol = 2)

print(p_error)
ggsave(
  "all_methods_error.tiff",
  p_error,
  width = 15,
  height = 5,
  dpi = 600,
  compression = "lzw",
  bg = "white"
)


# Individual error plots
for (method_name in names(beta_files)) {
  p <- ggplot(results_list[[method_name]], aes(x, y, fill = mean_error)) +
    geom_tile() +
    scale_fill_gradientn(
      colors = c("blue4", "blue", "lightblue", "white", "yellow", "orange", "red3"),
      limits = global_error_range,
      breaks = error_breaks,
      labels = error_labels,
      values = scales::rescale(c(-0.10, -0.02, 0, 0.02, 0.05, 0.10, 0.30)),
      oob    = scales::squish,
      name   = "Mean Error"
    ) +
    coord_equal() +
    theme_void() +
    ggtitle(paste(method_name, "Error"))
  
  print(p)
  ggsave(
    paste0(method_name, "_error.tiff"),
    p,
    width = 10,
    height = 7,
    dpi = 600,
    compression = "lzw",
    bg = "white"
  )
}

print(summary_stats)
