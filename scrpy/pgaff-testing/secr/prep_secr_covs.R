# Clear memory (if necessary)
rm(list=ls())

# Load packages & files
# List of packages needed
list.of.packages <- c("tidyverse", "lubridate", "googleCloudStorageR","dplyr",
                      "sf", "terra", "ggplot2")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
lapply(list.of.packages, require, character.only = TRUE)
rm(list.of.packages)
rm(new.packages)

#Download from GCS if not already local
gcs_auth(json_file = "pgaff-camera-optim.json", token = NULL, email = NULL)

#################### Study area and buffer for mask ############################
SA<-st_read("./spatial_data/study_area_2021_GCS.shp")
SA <- st_transform(SA, crs = 32610)
plot(SA$geometry, col="red")

#Buffer the SA by 2*sigma
max_sigma<- 23000 #meters - extra wide for any calcs needed
SA_buffered<-st_buffer(SA, max_sigma*2)
plot(SA_buffered$geometry)
plot(SA$geometry, col="red", add=TRUE)

#################### Prepping two new layers for PPP ###########################
################
#Human footprint
################
gcs_get_object("Partner_data/Spatial_data/So_Chilcotin_Mtns/TreeCover/hii_2020-01-01.tif", 
               bucket = "gs://pgaff-camera-optim_storage", 
               saveToDisk = "./spatial_data/hii_2020-01-01.tif")

HF<-rast("./spatial_data/hii_2020-01-01.tif")
SA_buffered_proj<-st_transform(SA_buffered, crs = crs(HF))
HF_crop<-crop(HF,SA_buffered_proj)
HF_crop_proj<-project(HF_crop, crs(SA_buffered), method="near")
plot(HF_crop_proj)
plot(SA_buffered$geometry, col="gray", add=TRUE)
plot(SA$geometry, col="red", add=TRUE)

#Save
writeRaster(HF_crop_proj, "So_Chilcotin_HF_proj.tif", overwrite=TRUE)

gcs_upload(file="So_Chilcotin_HF_proj.tif",
           name="Partner_data/Spatial_data/So_Chilcotin_Mtns/Human_footprint/So_Chilcotin_HF_proj.tif", 
           bucket = "pgaff-camera-optim_storage",
           predefinedAcl = "bucketLevel")

################
#Tree Cover
################
#Downloaded from GEE using this script
#// Define your region of interest (ROI) using lat/lon coordinates
#var roi = ee.Geometry.Point([-122.93820966231463, 50.98346092032365]);

#// Load the GFCC30TC dataset (modify the dataset path if needed)
#var dataset = ee.ImageCollection("NASA/MEASURES/GFCC/TC/v3") // Modify this if necessary to your dataset
#.filterBounds(roi)  // Filter by your region of interest
#.first();  // Get the first image (or use other filtering options)

#// Print the information about the region, path, and row
#print(dataset.getInfo());  // This will provide information about the MODIS image including the

#### Then used the above p and r code to download tiles from here:
#https://e4ftl01.cr.usgs.gov/MEASURES/GFCC30TC.003/2015.01.01/index.html

#// Ensure all bands are of the same data type (convert to Float32)
#dataset = dataset.toFloat();  // Convert the entire dataset to Float32 (or use .toByte() if needed for integer values)

#// Display the data
#Map.centerObject(roi, 10); // Center the map on your ROI
#Map.addLayer(dataset, {bands: ['tree_canopy_cover'], min: 0, max: 100}, 'Tree Cover');

#// Export the data to Google Drive (or Google Cloud Storage if needed)
#Export.image.toDrive({
#  image: dataset,
#  description: 'TreeCoverData',
#  scale: 30,  // Adjust depending on the dataset's resolution
#  region: roi
#});

gcs_get_object("Partner_data/Spatial_data/So_Chilcotin_Mtns/TreeCover/p047r024_TC_2015.tif", 
               bucket = "gs://pgaff-camera-optim_storage", 
               saveToDisk = "./spatial_data/p047r024_TC_2015.tif", overwrite=TRUE)

gcs_get_object("Partner_data/Spatial_data/So_Chilcotin_Mtns/TreeCover/p047r025_TC_2015.tif", 
               bucket = "gs://pgaff-camera-optim_storage", 
               saveToDisk = "./spatial_data/p047r025_TC_2015.tif", overwrite=TRUE)

gcs_get_object("Partner_data/Spatial_data/So_Chilcotin_Mtns/TreeCover/p048r024_TC_2015.tif", 
               bucket = "gs://pgaff-camera-optim_storage", 
               saveToDisk = "./spatial_data/p048r024_TC_2015.tif", overwrite=TRUE)

gcs_get_object("Partner_data/Spatial_data/So_Chilcotin_Mtns/TreeCover/p048r025_TC_2015.tif", 
               bucket = "gs://pgaff-camera-optim_storage", 
               saveToDisk = "./spatial_data/p048r025_TC_2015.tif", overwrite=TRUE)

TC1<-rast("./spatial_data/p047r024_TC_2015.tif")
#Eliminate and cloud cover values
reclass_matrix <- cbind(101, Inf, NA)  # Values greater than 100 are replaced with NA

TC1<-classify(TC1, reclass_matrix)
plot(TC1)

TC2<-rast("./spatial_data/p047r025_TC_2015.tif")
TC2<-classify(TC2, reclass_matrix)
TC3<-rast("./spatial_data/p048r024_TC_2015.tif")
TC3<-classify(TC3, reclass_matrix)
TC4<-rast("./spatial_data/p048r025_TC_2015.tif")
TC4<-classify(TC4, reclass_matrix)
TC<-mosaic(TC1, TC2, TC3, TC4, fun = max)
plot(TC)

SA_buffered_proj<-st_transform(SA_buffered, crs = crs(TC1))
lines(SA_buffered_proj)

TC_crop<-crop(TC,SA_buffered_proj)
TC_crop_proj<-project(TC_crop, crs(SA_buffered), method="near")
plot(TC_crop_proj)
plot(SA_buffered$geometry, col="gray", add=TRUE)
plot(SA$geometry, col="red", add=TRUE)

#Save
writeRaster(TC_crop_proj, "So_Chilcotin_TC_proj.tif", overwrite=TRUE)

gcs_upload(file="So_Chilcotin_TC_proj.tif",
           name="Partner_data/Spatial_data/So_Chilcotin_Mtns/TreeCover/So_Chilcotin_TC_proj.tif", 
           bucket = "pgaff-camera-optim_storage",
           predefinedAcl = "bucketLevel")

############### This is old stuff, but still useful for costing ################
########################### Land Cover #########################################
LC<-rast("./spatial_data/landcover-2020-classification.tif")
SA_buffered_proj<-st_transform(SA_buffered, crs = crs(LC))
LC_crop<-crop(LC,SA_buffered_proj)
LC_crop_proj<-project(LC_crop, crs(SA_buffered), method="near")
plot(LC_crop_proj)
plot(SA_buffered$geometry, col="gray", add=TRUE)
plot(SA$geometry, col="red", add=TRUE)

#Save
writeRaster(LC_crop_proj, "So_Chilcotin_LC_2xsig_proj.tif", overwrite=TRUE)

gcs_upload(file="So_Chilcotin_LC_2xsig_proj.tif",
           name="Partner_data/Spatial_data/So_Chilcotin_Mtns/Land_cover/So_Chilcotin_LC_2xsig_proj.tif", 
           bucket = "pgaff-camera-optim_storage",
           predefinedAcl = "bucketLevel")

#Preferred habitat
#Using "2020Canada_LC_ClassIndex
val_table<-data.frame(value=unique(values(LC_crop_proj))[order(unique(values(LC_crop_proj)))],
                      habitat=c("Coniferous forest",
                                "Deciduous Forest",
                                "Deciduous Forest",
                                "Shrubland",
                                "Grassland",
                                "Grassland",
                                "Wetland",
                                "Cropland",
                                "Barren",
                                "Urban",
                                "Water",
                                "Snow_ice",
                                "NoData"))
val_table

# Create a reclassification matrix
# Each row in the matrix represents: [from_value, to_value, new_value]
#Habitat preferences (0-1 with one being more preferred and 0 for avoided habitats)
m_pref <- c(0.9, 6.9, 1,
            7.1, 8.9, 0.5,
            9.1, 12.9, 0.3,
            13.1, 14.9, 0.8,
            14.1, 15.9, 0.3,
            15.91, 19.9, 0)

pref_reclass_matrix<-matrix(m_pref, ncol=3, byrow=TRUE)
pref_reclass_matrix

#Habitat avoidance (0-1 with one being more avoided and 0 for preferred habitats)
m_avoid <- c(0.9,15.1, 0,
             15.2, 16.9, 0.7,
             16.91, 18.9, 1,
             18.91, 19.9, 0.9)

avoid_reclass_matrix<-matrix(m_avoid, ncol=3, byrow=TRUE)
avoid_reclass_matrix

# Reclassify the raster using the matrix
LC_pref <- classify(LC_crop_proj, pref_reclass_matrix)
unique(values(LC_pref))
plot(LC_pref)

LC_avoid <- classify(LC_crop_proj, avoid_reclass_matrix)
unique(values(LC_avoid))
plot(LC_avoid)

#Save
writeRaster(LC_pref, "So_Chilcotin_LC_2xsig_proj_PREF.tif", overwrite=TRUE)
writeRaster(LC_avoid, "So_Chilcotin_LC_2xsig_proj_AVOID.tif", overwrite=TRUE)

gcs_upload(file="So_Chilcotin_LC_2xsig_proj_PREF.tif",
           name="Partner_data/Spatial_data/So_Chilcotin_Mtns/Land_cover/So_Chilcotin_LC_2xsig_proj_PREF.tif", 
           bucket = "pgaff-camera-optim_storage",
           predefinedAcl = "bucketLevel")

gcs_upload(file="So_Chilcotin_LC_2xsig_proj_AVOID.tif",
           name="Partner_data/Spatial_data/So_Chilcotin_Mtns/Land_cover/So_Chilcotin_LC_2xsig_proj_AVOID.tif", 
           bucket = "pgaff-camera-optim_storage",
           predefinedAcl = "bucketLevel")

############################ Elevation #########################################
DEM1<-rast("./spatial_data/DEM/n50_w122_1arc_v3.tif")
DEM2<-rast("./spatial_data/DEM/n50_w123_1arc_v3.tif")
DEM3<-rast("./spatial_data/DEM/n50_w124_1arc_v3.tif")
DEM4<-rast("./spatial_data/DEM/n50_w125_1arc_v3.tif")
DEM5<-rast("./spatial_data/DEM/n51_w122_1arc_v3.tif")
DEM6<-rast("./spatial_data/DEM/n51_w123_1arc_v3.tif")
DEM7<-rast("./spatial_data/DEM/n51_w124_1arc_v3.tif")
DEM8<-rast("./spatial_data/DEM/n51_w125_1arc_v3.tif")

DEM<-terra::merge(DEM1,DEM2,DEM3,DEM4,DEM5,DEM6,DEM7,DEM8)
rm(DEM1,DEM2,DEM3,DEM4,DEM5,DEM6,DEM7,DEM8)
DEM<-project(DEM, crs(SA), method = "near")

DEM_crop <- crop(DEM, SA_buffered)
plot(DEM_crop)
plot(SA_buffered$geometry, col="gray", add=TRUE)

writeRaster(DEM_crop, "So_Chilcotin_DEM_2xsig_proj.tif", overwrite=TRUE)
gcs_upload(file="So_Chilcotin_DEM_2xsig_proj.tif",
           name="Partner_data/Spatial_data/So_Chilcotin_Mtns/DEM/So_Chilcotin_DEM_2xsig_proj.tif", 
           bucket = "pgaff-camera-optim_storage",
           predefinedAcl = "bucketLevel")

############################ Distance to roads ##################################
#Prepping here, but calc will be done in QGIS
roads<-st_read("~/Desktop/PGAFF/spatial_data/Rds_wider/DRA_MPAR_line.shp")
SA_buffered_proj2<-st_transform(SA_buffered, crs=crs(roads))
roads_crop<-st_intersection(roads, SA_buffered_proj2)
plot(roads_crop$geometry)
roads_crop$OBJECTID<-as.character(roads_crop$OBJECTID)

st_write(roads_crop, 
         dsn="~/Desktop/PGAFF/spatial_data/So_Chilcotin_roads_2xsig_proj.shp", 
         driver="ESRI Shapefile",
         append=FALSE)

#In QGIS, converted to raster, then used raster-proximity to get distance
#Need to project the results
dist_roads<-rast("./spatial_data/dist_roads.tif")
dist_roads_proj<-project(dist_roads, crs(SA_buffered), method="near")
plot(dist_roads_proj)
plot(SA, add=TRUE, col="red")

writeRaster(dist_roads_proj, "So_Chilcotin_dist_roads_2xsig_proj.tif", overwrite=TRUE)
gcs_upload(file="So_Chilcotin_dist_roads_2xsig_proj.tif",
           name="Partner_data/Spatial_data/So_Chilcotin_Mtns/Roads/So_Chilcotin_dist_roads_2xsig_proj.tif", 
           bucket = "pgaff-camera-optim_storage",
           predefinedAcl = "bucketLevel")

