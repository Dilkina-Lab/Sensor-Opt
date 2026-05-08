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

gcs_auth(json_file = "pgaff-camera-optim.json", token = NULL, email = NULL)

########################### Functions needed ###################################
IHP_D <- function(betas, mask, D_mean, covs, poly) {
  
  #Predict density for every cell of the mask based on covariates and simulated betas
  cov_matrix <- as.matrix(covariates(mask))
  colnames(cov_matrix)<-names(covariates(mask))
  cov_matrix2<-as.matrix(cov_matrix[,which(colnames(cov_matrix)%in%covs)])
  
  #Ensure betas and covariates align
  beta_values <- unlist(betas)  # Convert betas list to vector
  if((length(beta_values)-1) != ncol(cov_matrix2)) {
    stop("Number of betas must match the number of covariates in the mask.")
  }
  
  #Compute the linear predictor
  cov_pred<-cov_matrix2%*%beta_values[-1]
  
  D_pred_tmp <- exp(betas$beta0+cov_pred)
  
  ############ Scale the intensity to match the simulated density ##############
  #Area of one pixel in hectares (500m x 500m = 25 ha)
  pixel_area_ha<-0.5*0.5*100
  
  #Total expected number of animals implied by D_pred_tmp
  EN_pred<-sum(D_pred_tmp)*pixel_area_ha
  
  #Get the expected number of animals based on the D_mean sim input
  #Total area in hectares
  A_ha<-dim(mask)[1]*pixel_area_ha
  
  #Expected number of animals
  EN<-D_mean*A_ha#D_mean is in animals/ha
  
  #Calculate the scaling needed to downgrade intensity
  scale_factor<-EN_pred/EN
  
  #Scale the intensity based on that factor
  D_pred<-D_pred_tmp/scale_factor
  
  return(list(D_pred,EN))
  
  rm(AC_init)
  gc()
}

simplify_capture_history <- function(sim_ch) {
  # Get the dimensions of the 3D array: individuals, occasions, and traps
  dims <- dim(sim_ch)
  
  # Find indices where the value is 1 (captures)
  capture_indices <- which(sim_ch == 1, arr.ind = TRUE)
  
  # The capture indices give us the row, column, and slice, which correspond to:
  # - row (individual)
  # - column (occasion)
  # - slice (trap)
  individual <- capture_indices[, 1]  # Individuals
  occasion <- capture_indices[, 2]    # Occasions
  trap_id <- capture_indices[, 3]     # Trap IDs
  
  # Combine the individual, occasion, and trap_id into a matrix
  capture_matrix <- cbind(individual, occasion, trap_id)
  
  return(capture_matrix)
}

####################### Load in the spatial layers #############################
# Define the bucket and folder
bucket_name <- "gs://pgaff-camera-optim_storage"
folder_name <- "Partner_data/Spatial_data/So_Chilcotin_Mtns/"

# List all files in the folder
files <- gcs_list_objects(bucket = bucket_name, 
                          prefix = folder_name)

# Define the local folder where you want to download the files
# Make this folder in working directory
destination_folder <- "./spatial_data/"

# Loop through each file and download it
for (file in files$name) {
  # Define the destination file path
  destination_path <- file.path(destination_folder, basename(file))
  
  # Download the file from GCS
  gcs_get_object(file, bucket = bucket_name, saveToDisk = destination_path)
  
  # Print progress
  cat("Downloaded:", file, "to", destination_path, "\n")
}

# Define the bucket and folder
folder_name <- "Partner_data/Spatial_data/So_Chilcotin_Mtns/Bad_grid_cells"

# List all files in the folder
files <- gcs_list_objects(bucket = bucket_name, 
                          prefix = folder_name)

# Define the local folder where you want to download the files
# Make this folder in working directory
destination_folder <- "./spatial_data/"

# Loop through each file and download it
for (file in files$name) {
  # Define the destination file path
  destination_path <- file.path(destination_folder, basename(file))
  
  # Download the file from GCS
  gcs_get_object(file, bucket = bucket_name, saveToDisk = destination_path)
  
  # Print progress
  cat("Downloaded:", file, "to", destination_path, "\n")
}

######################### Load in spatial layers ###############################
#Made in "prep_secr_covs.R" script
TC<-rast("./spatial_data/So_Chilcotin_TC_proj.tif")
plot(TC)

HF_log<-rast("./spatial_data/HF_log.tif")
plot(HF_log)

SA<-st_read("./spatial_data/study_area_2021_GCS.shp")
study_area <- st_transform(SA, crs = crs(TC))
plot(study_area, add=TRUE, col="red")

##################### Create a super dense trapping grid #######################

#Get the study area bounding box
bbox<-st_bbox(study_area)

#Create a dense grid of traps (500m spacing) within the study area
trap_spacing <- 1000

#Generate a regular trapping grid
trap_grid_unclip<-st_make_grid(bbox,
                               cellsize = trap_spacing, # 500 meters
                               square = TRUE,
                               what = "centers")

points(trap_grid_unclip, col="red")

traps_sf<-st_as_sf(trap_grid_unclip)
traps_sf$Trap_index<-seq(1,nrow(traps_sf), by=1)
traps_sf

trap_coords1 <- as.data.frame(st_coordinates(trap_grid_unclip)) # Extract coordinates
colnames(trap_coords1) <- c("x", "y")

############################## Save traps ######################################
trap_csv<-data.frame(Trap_index=seq(1,nrow(trap_coords1),by=1),
                     x=trap_coords1$x,
                     y=trap_coords1$y)

write.csv(trap_csv, file="1000m_trap_grid_marten.csv")
saveRDS(traps_sf, file="1000m_trap_grid_marten.RDS")

traps<-read.csv("1000m_trap_grid_marten.csv")
trap_coords1<-traps%>%select(x,y) # Extract coordinates

traps_sf<-st_as_sf(as.data.frame(traps), coords=c("x", "y"), crs=crs(study_area))

traps1 <- read.traps(data = trap_coords1, 
                     detector="proximity")

############################# Make mask ########################################
#Buffer the SA by 2*sigma
max_sigma<-1000 #meters
SA_buffered<-st_buffer(st_as_sfc(bbox), max_sigma*2)
plot(SA_buffered)
points(traps_sf, col="red")

#Generate a mask around the traps
mask1<-make.mask(traps=traps1, type="polybuffer",
                 poly=st_as_sfc(bbox), buffer=max_sigma*2, spacing = 500)

plot(mask1)
points(traps_sf, col="red")

#Extract the values for the mask from the projected LC_reclass raster
coords_mask<-vect(data.frame(x=mask1$x,y=mask1$y),
                  geom = c("x", "y"), crs = crs(study_area))

TC_extract<-extract(TC, coords_mask)
HF_extract<-extract(HF_log, coords_mask)

summary(TC_extract)
summary(HF_extract)

#Some missing values, fill with the mean
TC_extract$p047r024_TC_2015[is.na(TC_extract$p047r024_TC_2015)]<-
  mean(TC_extract$p047r024_TC_2015, na.rm=TRUE)
HF_extract$`hii_2020-01-01`[is.na(HF_extract$`hii_2020-01-01`)]<-
  mean(HF_extract$`hii_2020-01-01`, na.rm=TRUE)

#Add covariates to mask
covariates(mask1) <- data.frame(TC = scale(TC_extract[,2])[,1],
                                HF = scale(HF_extract[,2])[,1])
summary(covariates(mask1))

#Check correlation
cor(covariates(mask1)) #36% correlated - OK

#Save the mask and upload to GCS
saveRDS(mask1,file="500m_mask_marten.RDS")

gcs_upload(file="500m_mask_marten.RDS",
           name="Synthetic_sim/500m_mask_marten.RDS", 
           bucket = "pgaff_simulations",
           predefinedAcl = "bucketLevel")

gcs_get_object("Synthetic_sim/500m_mask_marten.RDS", 
               bucket =  "pgaff_simulations", 
               saveToDisk = "500m_mask_marten.RDS",
               overwrite=TRUE)

#for later plotting
hab<-data.frame(x=mask1$x,y=mask1$y,
                TC=covariates(mask1)[,1])

########################## Define parameters ###################################
#Clear out some memory first
ls_list<-ls()
ls_list2<-ls_list[which(!ls_list%in%c("mask1","SA_buffered",
                                      "traps1", "study_area", "bbox",
                                      "simplify_capture_history","IHP_D","hab",
                                      "spacing"))]
rm(list=ls_list2)

#Based on American Marten
D_range<-c(0.0015, 0.004) #animals/ha

#Set ranges for detection parameters
g0_range<-c(0.01,0.5)

#Sigma
sigma_range<-c(200,1000)#in meters

#Set the beta range for the density parameters
beta1_range <- c(0.1, 2) #Tree cover - obligate positive
beta2_range <- c(-1, 0.5) #Human footprint - more negative than grizzlies but still goes +

############################# Set up LHS #######################################
# Define the number of samples
n_samples <- 300

#Generate an LHS design with 5 variables
lhs_sample <- randomLHS(n_samples, 5)

#Define variable ranges (min and max for each variable)
var_ranges <- data.frame(
  min = c(D_range[1],g0_range[1],sigma_range[1],beta1_range[1],
          beta2_range[1]),
  max = c(D_range[2],g0_range[2],sigma_range[2],beta1_range[2],
          beta2_range[2]))

var_ranges

#Transform the LHS values to match variable ranges
scaled_samples <- lhs_sample
for (i in 1:ncol(lhs_sample)) {
  scaled_samples[, i] <- lhs_sample[, i] * (var_ranges$max[i] - var_ranges$min[i]) + var_ranges$min[i]
}

#Convert to a dataframe and name columns
scaled_samples<-as.data.frame(scaled_samples)
colnames(scaled_samples)<-c("D","g0","sigma","beta1","beta2")

dim(scaled_samples)
head(scaled_samples)
summary(scaled_samples)

#Save param table as a .csv
write.csv(scaled_samples, file="./Output/param_values_for_each_draw300_marten.csv")

gcs_upload(file="./Output/param_values_for_each_draw300_marten.csv",
           name="Synthetic_sim/param_values_for_each_draw300_marten.csv", 
           bucket = "pgaff_simulations",
           predefinedAcl = "bucketLevel")

gcs_get_object("Synthetic_sim/param_values_for_each_draw300_marten.csv", 
               bucket =  "pgaff_simulations", 
               saveToDisk = "./Output/param_values_for_each_draw300_marten.csv",
               overwrite=TRUE)

scaled_samples<-read.csv("./Output/param_values_for_each_draw300_marten.csv")

######################### Run the simulations ##################################
gc()

True_N<-matrix(nrow=0, ncol=2)
for(i in 1:n_samples){
  
  gc()
  
  print(i)
  
  params<-scaled_samples[i,]
  
  print(params)
  
  IHP_betas<-data.frame(beta0=log(params$D),
                        beta1=params$beta1,
                        beta2=params$beta2)
  
  D_mod<-IHP_D(betas=IHP_betas,mask=mask1,D_mean=params$D,
               covs=colnames(covariates(mask1))[1:2],
               poly=SA_buffered)
  
  ################################ Simulate ACs ##################################
  #For the drawn combination of parameter values, simulate activity centers 
  #with density varying by habitat suitability
  AC<-sim.popn(
    D=D_mod[[1]][,1],
    core=mask1,
    Ndist = "poisson",
    poly=SA_buffered,
    model2D="IHP")
  
  #Plot the habitat and activity centers
  b1<-round(params$beta1,2)
  b2<-round(params$beta2,2)
  
  plt<-ggplot() +
    geom_tile(data=hab, aes(x=x, y=y, fill=TC))+
    #geom_tile(data=hab, aes(x=x, y=y, fill=HF))+
    geom_sf(data=study_area, fill=NA, color="lightgray", linewidth=0.5)+
    geom_point(data=data.frame(x = AC[, 1], y = AC[, 2]),
               aes(x=x, y=y), fill="red",
               color="black", shape=21, size=1)+
    scale_fill_viridis_c(name="Centered/Scaled TC")+
    theme_void()+
    theme(legend.position="right") +
    labs(title=paste("Draw ",i, 
                     " TC beta=",b1,
                     " HF beta=",b2,
                     sep=""))+
    coord_sf()
  plt
  
  ggsave(plt, file=paste("./Output/AC_plots/AC_plot_",i,".tiff", sep=""), bg="white")
  
  gcs_upload(file=paste("./Output/AC_plots/AC_plot_",i,".tiff", sep=""),
             name=paste("Synthetic_sim/Constant_detection/AC_plots/AC_plot_",i,".tiff", sep=""),
             bucket = "pgaff_simulations",
             predefinedAcl = "bucketLevel")
  
  ################## Simulate the capture histories ##########################
  #right now assuming fully marked
  #Partially marked uses sim.resight()
  simulated_ch<-sim.capthist(
    popn = AC,                              # ACs (inhomogeneous density)
    traps = traps1,                              # Trap layout
    detectfn = "HN",                            # Half-normal detection function
    detectpar = list(g0=params$g0, 
                     sigma=params$sigma),     # Detection parameters
    nsessions = 1,
    noccasions = 5,                              # Number of sampling occasions
    p.available = 1,                             # Availability for capture
    renumber = FALSE)
  
  #Simplify the output and save
  simplified_ch<- simplify_capture_history(simulated_ch)
  
  write.csv(simplified_ch,
            file=paste("./Output/ch/ch_draw_",i,".csv", sep=""),
            row.names = FALSE)
  
  gcs_upload(file=paste("./Output/ch/ch_draw_",i,".csv", sep=""),
             name=paste("Synthetic_sim/Constant_detection/ch/ch_draw_",i,".csv", sep=""),
             bucket = "pgaff_simulations",
             predefinedAcl = "bucketLevel")
  
  write.csv(AC,
            file=paste("./Output/AC_locations/AC_draw_",i,".csv", sep=""),
            row.names = FALSE)
  
  gcs_upload(file=paste("./Output/AC_locations/AC_draw_",i,".csv", sep=""),
             name=paste("Synthetic_sim/Constant_detection/AC_locations/AC_draw_",i,".csv", sep=""),
             bucket = "pgaff_simulations",
             predefinedAcl = "bucketLevel")
  
  D_surface<-data.frame(D_mod=D_mod[[1]],
                        EN=D_mod[[2]],
                        x=mask1$x,
                        y=mask1$y)
  
  write.csv(D_surface,
            file=paste("./Output/D_mod/Dmod_draw_",i,".csv", sep=""),
            row.names = FALSE)
  
  gcs_upload(file=paste("./Output/D_mod/Dmod_draw_",i,".csv", sep=""),
             name=paste("Synthetic_sim/Constant_detection/D_mod/Dmod_draw_",i,".csv", sep=""),
             bucket = "pgaff_simulations",
             predefinedAcl = "bucketLevel")
  
  #Add a .csv with the true N of each realization (the number of ACs in the mask)
  n_ACs<-data.frame(N=nrow(AC),
                    Parameter_draw=i)
  True_N<-rbind(True_N, cbind(n_ACs))
  
}

write.csv(True_N, file="./Output/True_N_per_draw.csv")

gcs_upload(file="./Output/True_N_per_draw.csv",
           name="Synthetic_sim/Constant_detection/True_N_per_draw.csv",
           bucket = "pgaff_simulations",
           predefinedAcl = "bucketLevel")