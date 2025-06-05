############### IMPORTANT:
# Make sure you are using the most recent version of the secr package (at least v 5.2.1)
###############

# Load packages & files
# List of packages needed
list.of.packages <- c("tidyverse", "lubridate", "googleCloudStorageR","dplyr",
                      "secrdesign", "sf", "terra", "ggplot2", "lhs")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = TRUE)
rm(list.of.packages)
rm(new.packages)

#This file gives access to pgaff GCS bucket, makes sure it is in your working
#directory and please do not publish it anywhere
gcs_auth(json_file = "SensorOpt/secr/pgaff-camera-optim.json", token = NULL, email = NULL)

######################### Load in spatial layers ###############################
TC<-rast("SensorOpt/secr/spatial_data/So_Chilcotin_TC_proj.tif")
plot(TC)

HF<-rast("SensorOpt/secr/spatial_data/So_Chilcotin_HF_proj.tif")
plot(HF)
 
# Run the below in command lind
# ogrinfo --config SHAPE_RESTORE_SHX YES /home1/hannahmu/SensorOpt/secr/spatial_data/study_area_2021_GCS.shp
SA<-st_read("SensorOpt/secr/spatial_data/study_area_2021_GCS.shp")
SA <- st_set_crs(SA, 4326)
study_area <- st_transform(SA, crs = st_crs(TC))
plot(study_area, add = TRUE, col = "red")

################ Make a traps object by subsetting the 500m grid ################
traps_500m<-read.csv("SensorOpt/secr/500m_trap_grid_4-24-25.csv")
traps_500m<-traps_500m[-c(1)]

exclude_traps <- read.table("SensorOpt/secr/BG2/excluded_traps.txt")  # Assumes one ID per line

optim_cams <- traps_500m %>% 
  filter(!Trap_index %in% exclude_traps$V1)  # V1 is default column name from read.table

# Then continue with existing code:
traps12 <- traps_500m %>% filter(Trap_index %in% optim_cams$Trap_index)

#Make into a traps object for SECR
traps1<-read.traps(data=traps12, detector="proximity")

points(traps1, pch=16, col="blue")

##################### Alternate trap generation ################################
#Here, we are generating a regular grid of traps, feeding in a trap budget
#This is limited to spacing increments of 500, to subset the existing grid

#This is an example (commented out) where we set the spacing, not the number of cameras
#round sigma to the nearest 500
#new_spacing<-round(param_vals[draw,'sigma']/500)*500

#Here, we set the camera budget (N cameras) instead of the spacing - do not use if
#using the above strategy instead
# bbox<-st_bbox(SA)
# side<-ceiling(sqrt(st_area(SA)))

# n_cams<-60
# grid_cols<-ceiling(sqrt(n_cams))
# new_spacing<-as.numeric(floor(side/grid_cols))
# new_spacing

# #Make a new grid
# trap_grid_unclip<-st_make_grid(SA,
#                                cellsize = new_spacing,
#                                square = TRUE,
#                                what = "centers")

# #Clip to the study area
# traps_sf<-st_intersection(st_sf(geometry = trap_grid_unclip), SA)

# #Now find the closest 500m grid point
# orig_grid_sf<-st_as_sf(as.data.frame(traps_500m), coords = c("x", "y"), crs = crs(TC))
# orig_grid_sf <- st_set_crs(orig_grid_sf, 4326)  # Overwriting to WGS84


# #Compute distance matrix
# dist_matrix <- st_distance(traps_sf, orig_grid_sf)

# #Get the index of the minimum distance (nearest point) for each row
# nearest_idx <- apply(dist_matrix, 1, which.min)

# #Extract nearest points
# coarse_grid_sf <- orig_grid_sf[nearest_idx, ]
# dim(coarse_grid_sf)

# #Thin to the trap budget by randomly removing n cameras on the grid edges
# if(dim(coarse_grid_sf)[1]>n_cams){
#   buffer_dist <- 1000  #Within 1000m of the SA edge
#   inner_SA<-st_buffer(SA, dist = -buffer_dist)

#   #Find the traps within that 1km of the edge
#   buffer_traps<-which(!st_within(coarse_grid_sf, inner_SA, sparse = FALSE))

#   #Randomly select some of those traps based on how many need trimming
#   n_trim<-dim(coarse_grid_sf)[1]-n_cams
#   remove_traps<-sample(buffer_traps, n_trim, replace = FALSE)
#   coarse_grid_sf<-coarse_grid_sf[-remove_traps,]
# }

# dim(coarse_grid_sf)

# plot(orig_grid_sf$geometry)
# points(coarse_grid_sf$geometry, pch=20, col="red")

# traps12<-traps_500m%>%filter(Trap_index%in%coarse_grid_sf$Trap_index)
# traps1<-read.traps(data=traps12, detector="proximity")

########################## Make a mask #########################################
gcs_get_object("sim_update_4-25-2025/param_values_for_each_draw300_4-28-25.csv", 
               bucket = "pgaff_simulations", 
               saveToDisk = "param_values_for_each_draw300_4-28-25.csv", 
               overwrite=TRUE)

param_vals<-read.csv("param_values_for_each_draw300_4-28-25.csv")
param_vals<-param_vals[-c(1)]

#Buffer the SA by 2*sigma
max_sigma<-signif(max(param_vals$sigma),1) #meters
SA_buffered<-st_buffer(study_area, max_sigma*2)
plot(SA_buffered$geometry)
plot(study_area$geometry, col="red", add=TRUE)

#Generate a mask around the traps
mask1 <- make.mask(traps = traps1, type="polybuffer",
                   poly=study_area, buffer=max_sigma*2, spacing = 500)
plot(mask1)

#Extract the values for the mask from covariate layers
coords_mask<-vect(data.frame(x=mask1$x,y=mask1$y),
                  geom = c("x", "y"), crs = crs(study_area))

TC_extract<-extract(TC, coords_mask)
HF_extract<-extract(HF, coords_mask)

#Some missing values, fill with the mean
TC_extract$p047r024_TC_2015[is.na(TC_extract$p047r024_TC_2015)]<-
  mean(TC_extract$p047r024_TC_2015, na.rm=TRUE)
HF_extract$`hii_2020-01-01`[is.na(HF_extract$`hii_2020-01-01`)]<-
  mean(HF_extract$`hii_2020-01-01`, na.rm=TRUE)

#Add covariates to mask
covariates(mask1) <- data.frame(TC = scale(TC_extract[,2])[,1],
                                HF = scale(HF_extract[,2])[,1])
summary(covariates(mask1))

############################# load up a ch #####################################
#Arielle has added a loop here to go through all the parameter draws

#Get true Ns for comparison later
gcs_get_object("sim_update_4-25-2025/True_N_per_draw.csv", 
               bucket = "pgaff_simulations", 
               saveToDisk = "True_N_per_draw.csv", 
               overwrite=TRUE)
true_N<-read.csv("True_N_per_draw.csv")

# Define the range of draws you want to loop over
start_draw <- 21
end_draw <- 30

# Your loop will look like this:
results <- matrix(nrow=0, ncol=16)
for (i in start_draw:end_draw) {

  
  gc()
  
  draw<-i #Choose which param draw to use
  
  print(draw)
  
  ch<-read.csv(paste("ch/ch_draw_",draw,".csv", sep=""))
  
  #Constrain to only traps in traps1. Not that not all traps will appear if some 
  #did not detect any animals
  ch2<-ch%>%filter(trap_id%in%traps12$Trap_index)%>%
    mutate(animal=individual,
           trap=trap_id,
           session=1)%>%
    select(session,animal,occasion,trap)
  
  #################### Make the secr input files #################################
  #############
  # Captures
  #############
  
  #Must convert to session, id, trap, occasion
  dets2<-ch2%>%
    mutate(ID=animal,
           Detector=trap, 
           Occasion=occasion,
           Session=paste("Draw",draw,sep=""))%>%
    select(Session, ID, Occasion, Detector)
  #head(dets2)
  #summary(dets2)
  
  #This writes a temporary file which gets overwritten each time we produe a new capture file
  write.table(dets2, file="dets2.txt", sep = "\t", row.names=FALSE)
  
  #############
  # Traps
  #############
  
  traps_df<-traps12%>%mutate(trapID=Trap_index)%>%select(trapID,x,y)
  
  #This writes a temporary file which gets overwritten each time we produe a new trap input file
  write.table(traps_df, file="traps.txt", sep = "\t", row.names=FALSE)
  
  #Make ch input for the secr package
  ch<-read.capthist(captfile = "dets2.txt", 
                    trapfile = "traps.txt",
                    detector = "proximity",
                    skip=1)
  
  ################# Fit an inhomogenous secr model ###############################
  system.time(fit_model <- secr.fit(capthist = ch, #The input above in secr package format 
                                    mask = mask1, #The new mask specific to selected trap array
                                    model = list(D~TC+HF, #Density, IPPP as function of tree cover and human footprint
                                                 g0~1, sigma~1), #Constant detection parameters (for now)
                                    detectfn = 'HN', #Half normal detection function
                                    #This method improves stability for the betas, but does not return SE
                                    #We can get 95%CIs (below) but it will take a very long time to run
                                    #We can comment out this line and run a less stable method that does return SEs
                                    method="Nelder-Mead", 
                                    start=list(D=0.0001,  #Reasonable initial values for speed
                                               g0=0.5, 
                                               sigma=3000)))
  
  saveRDS(fit_model, file=paste("model_U1.RDS"))
  
  #If we need CIs for Nelder-Mead method - take a while though
  #We only get CIs for the beta parameters, so to get the real parameters we need to 
  #Back transform
  #CIs<-confint(fit_mod, parm=c("D","D.TC","D.HF","g0","sigma"))
  #CI_real<-exp(CIs[c(1,4,5)])
  
  #Get output
  out<-data.frame(Draw=draw,
                  N_mod=region.N(fit_model)[2,1],
                  N_true=true_N$N[draw],
                  N_abs_error=abs(region.N(fit_model)[2,1] - true_N$N[draw]),
                  beta1=summary(fit_model)$coef[2,1],
                  beta2=summary(fit_model)$coef[3,1],
                  beta1_true=param_vals[draw,'beta1'],
                  beta2_true=param_vals[draw,'beta2'],
                  beta1_abs_error=abs(summary(fit_model)$coef[2,1] - param_vals[draw,'beta1']),
                  beta2_abs_errcdor=abs(summary(fit_model)$coef[3,1] - param_vals[draw,'beta2']),
                  sigma_mod=summary(fit_model)$predicted[3,2],
                  sigma_true=param_vals[draw,'sigma'],
                  sigma_abs_error=abs(summary(fit_model)$predicted[3,2] - param_vals[draw,'sigma']),
                  g0_mod=summary(fit_model)$predicted[2,2],
                  g0_true=param_vals[draw,'g0'],
                  g0_abs_error=abs(summary(fit_model)$predicted[2,2] - param_vals[draw,'g0']))
  
  results<-rbind(results, cbind(out))

}

file_name <- paste0("U1_", start_draw, "-", end_draw, ".csv")
write.csv(results, file = file_name, row.names = FALSE)









######## Uniform Deployment Baseline
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

################ Make a traps object by subsetting the 500m grid ################
traps_500m <- read.csv("SensorOpt/secr/500m_trap_grid_4-24-25.csv")
traps_500m <- traps_500m[-c(1)]

##################### Alternate trap generation ################################
# Set camera budget and grid spacing
n_cams <- 60
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

# Now find the closest 500m grid point
orig_grid_sf <- st_as_sf(traps_500m, coords = c("x", "y"), crs = st_crs(TC))

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
  coarse_grid_sf$Trap_index <- traps_500m$Trap_index[idx]
}

# Filter traps_500m to only those in the selected grid
traps12 <- traps_500m %>% filter(Trap_index %in% coarse_grid_sf$Trap_index)
if (nrow(traps12) == 0) stop("No matching traps found in traps12 after alternative trap selection.")
traps1 <- read.traps(data = traps12, detector = "proximity")

########################## Make a mask #########################################
gcs_get_object("sim_update_4-25-2025/param_values_for_each_draw300_4-28-25.csv", 
               bucket = "pgaff_simulations", 
               saveToDisk = "param_values_for_each_draw300_4-28-25.csv", 
               overwrite=TRUE)

param_vals <- read.csv("param_values_for_each_draw300_4-28-25.csv")
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
gcs_get_object("sim_update_4-25-2025/True_N_per_draw.csv", 
               bucket = "pgaff_simulations", 
               saveToDisk = "True_N_per_draw.csv", 
               overwrite=TRUE)
true_N <- read.csv("True_N_per_draw.csv")

# Define the range of draws you want to loop over
start_draw <- 51
end_draw <- 100

results <- matrix(nrow=0, ncol=16)
for (i in start_draw:end_draw) {
  gc()
  draw <- i
  print(draw)
  ch <- read.csv(paste("ch/ch_draw_",draw,".csv", sep=""))
  
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
  
  saveRDS(fit_model, file=paste("model_U1.RDS"))
  
  # Get output (fix typo: beta2_abs_errcdor -> beta2_abs_error)
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

file_name <- paste0("U1_", start_draw, "-", end_draw, ".csv")
write.csv(results, file = file_name, row.names = FALSE)
