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

gcs_auth(json_file = "pgaff-camera-optim.json", token = NULL, email = NULL)

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
betas<-read.csv("SAA3-SA19-70traps_test_RSE_FILTERED.csv")
head(betas)

################## Use the betas to get cell-specific densities ################

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
    mutate(Draw=draw)%>%
    select(Draw,D_mod,EN,Dhat,x,y)
  
  #Save locally
  write.csv(Dout,file=paste("Draw_",draw,"_predicted_density.csv",sep=""))
  
  #Quick side-by-side map
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
  
  plot_grid(p1,p2,ncol=2)
  
  #Save plot
  ggsave(paste("Draw_",draw,"_predicted_density_map.tiff",sep=""), bg="white", 
         height=7, width=10, dpi=600, compression="lzw")
}