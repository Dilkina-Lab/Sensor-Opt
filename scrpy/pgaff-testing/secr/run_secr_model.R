# Sys.setenv(
#     GDAL_DATA = paste0(Sys.getenv("CONDA_PREFIX"), "/share/gdal"),
#     PROJ_LIB = paste0(Sys.getenv("CONDA_PREFIX"), "/share/proj")
# )
# install.packages(
#   "sf",
#   type = "source",
#   configure.args = "--with-proj-lib=$(brew --/usr/local)/lib/"
#   repos = 'https://cloud.r-project.org'
# )

# # install.packages("rgeos", repos="http://R-Forge.R-project.org", type="source")
# # install.packages("rgdal", repos="http://R-Forge.R-project.org", type="source")
# # library(devtools)
# # install_github("r-spatial/sf", configure.args = "--with-proj-lib=/usr/local/lib/")

# # #Run a secr model using a simulated capture history, subsetting for some number of traps
# options(repos = c(CRAN = "https://cloud.r-project.org"))
# install.packages(c("secr", "secrdesign"))

# library(sf)          # For spatial operations (st_read)
# library(secr)        # For SECR analysis
# library(secrdesign)  # For SECR study design

# Clear memory (if necessary)
rm(list=ls())

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
#Made in "prep_secr_covs.R" script
#Make a new folder locally called "spatial_data" to hold these
gcs_get_object("Partner_data/Spatial_data/So_Chilcotin_Mtns/TreeCover/So_Chilcotin_TC_proj.tif", 
               bucket = "pgaff-camera-optim_storage", 
               saveToDisk = "./spatial_data/So_Chilcotin_TC_proj.tif",
               overwrite=TRUE)
gcs_get_object("Partner_data/Spatial_data/So_Chilcotin_Mtns/Human_footprint/So_Chilcotin_HF_proj.tif", 
               bucket = "pgaff-camera-optim_storage", 
               saveToDisk = "./spatial_data/So_Chilcotin_HF_proj.tif",
               overwrite=TRUE)
gcs_get_object("Partner_data/Spatial_data/So_Chilcotin_Mtns/Boundary/Study_area_boundary/study_area_2021_GCS.shp", 
               bucket = "pgaff-camera-optim_storage", 
               saveToDisk = "./spatial_data/study_area_2021_GCS.shp",
               overwrite=TRUE)

#Tree cover
TC<-rast("./spatial_data/So_Chilcotin_TC_proj.tif")
plot(TC)

#Human footprint
HF<-rast("./spatial_data/So_Chilcotin_HF_proj.tif")
plot(HF)

#Study area boundary
SA<-st_read("./spatial_data/study_area_2021_GCS.shp")
study_area <- st_transform(SA, crs = crs(TC))
plot(study_area, add=TRUE, col="red")

############### Make a traps object by subsetting the 500m grid ################
gcs_get_object("sim_update_4-25-2025/500m_trap_grid_4-24-25.csv", 
               bucket = "pgaff_simulations", 
               saveToDisk = "500m_trap_grid_4-24-25.csv")
traps_500m<- read.csv("500m_trap_grid_4-24-25.csv")
traps_500m<-traps_500m[-c(1)]

############
#Select only the cameras needed - based on the optimization results
###########
#This is an example of using randomly selected 100 traps
# optim_cams<-traps_500m[sample(nrow(traps_500m), 100, replace=FALSE),]

# New code for exclusion-based selection:
# Load trap IDs to exclude
exclude_traps <- read.table("non_matching_ids.txt")  # Assumes one ID per line

# Filter traps excluding these IDs
optim_cams <- traps_500m %>% 
  filter(!Trap_index %in% exclude_traps$V1)  # V1 is default column name from read.table

# Then continue with existing code:
traps12 <- traps_500m %>% filter(Trap_index %in% optim_cams$Trap_index)

#Make into a traps object for SECR
traps1<-read.traps(data=traps12, detector="proximity")

points(traps1, pch=16, col="blue")

########################## Make a mask #########################################
#Now we need to make a new mask for the new camera grid
gcs_get_object("sim_update_4-25-2025/param_values_for_each_draw300_4-28-25.csv", 
               bucket = "pgaff_simulations", 
               saveToDisk = "param_values_for_each_draw300_4-28-25.csv")
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
#You will need to define the ch for the parameter draw here
#Or we can make a loop to go through all draws
draw<-194

gcs_get_object(paste("sim_update_4-25-2025/ch/ch_draw_",draw,".csv", sep=""), 
               bucket = "pgaff_simulations", 
               saveToDisk = paste("ch_draw_",draw,".csv", sep=""))

ch<-read.csv(paste("ch_draw_",draw,".csv", sep=""))

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
head(dets2)
summary(dets2)

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
                                  start=list(D=0.0001,  #Reasonable initial values for speed
                                             g0=0.5, 
                                             sigma=3000)))

saveRDS(fit_model, file="modelxxx.RDS")

#Get true Ns for comparison
gcs_get_object("sim_update_4-25-2025/True_N_per_draw.csv", 
               bucket = "pgaff_simulations", 
               saveToDisk = "True_N_per_draw.csv", 
               overwrite=TRUE)
true_N<-read.csv("True_N_per_draw.csv")

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

out

write.csv(out, file = paste0("500m_test_", draw, ".csv"), row.names = FALSE)
