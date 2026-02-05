import os
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from pixc2profile import download_pixc_data
from pixc2profile.river import River
from pixc2profile.pixc import PIXC
from pixc2profile.profile import Profile


skip_steps = [
            # "step_1", 
            #  "step_2", 
            #  "step_3", 
              # "step_4"
              ]  # specify which steps to skip for testing


# Take Selingue as an example
# File path settings
home_dir = "/mnt/d/pixc2profile/examples/data/shps/test_river.shp"
river_name = "test_river"
river_shp_path = os.path.join(home_dir, river_name, "reach", "river.shp")
# Output paths for profiles: wse_water (only perform water pixel filtering) and wse_water_qc (perform both water pixel filtering and quality flag filtering)
wse_water_path=os.path.join(home_dir, river_name, "wse_profile_water.csv")
wse_water_qc_path=os.path.join(home_dir, river_name, "wse_profile_water_qc.csv")

# Data parameters
start_date = "2023-01-01"
end_date = "2026-01-31"
pass_tile_list = ["454_082L", "454_083L", "191_227L", "191_227R", "191_226R"]
pixc_version = "SWOT_L2_HR_PIXC_D"
login_strategy = 'netrc'


# Node and buffer parameters
node_spacing = 50 # m
# approximate river channel width for buffer generation
channel_width = 260 # m
pixc_water_dir_name="pixc_water"
pixc_qc_dirname="pixc_water_qc_filtered"
n_partitions = 10


# Water filtering and Quality flag filtering settings
quality_flag_dict = {
    "geolocation_qual": [1, 4, 524356, 68, 524352, 8388612, 524292]
}
classification_categories = [3, 4] # "water near land" and "open water"
prior_water_prob_threshold = 0.5 # Pekel et al. (2016)
water_frac_threshold = 0.2 # fraction of water
wse_upper_limit = 500 # m, upper limit of WSE to exclude outliers; Usually the elevation of river bank can be used here

# Profile building parameters
agg_func = ["median", "std", "mean", "count"] # aggregation function to calculate WSE for each node; currently only support "median"
# keep quality flag values for the median calculation
keep_qual_groups=["interferogram_qual", "classification_qual", "geolocation_qual", "sig0_qual"]
# smooth parameters
frac_list = np.arange(1, 16, 1)/100 # test different fraction parameters from 0.01 to 0.2
it=3 # number of iterations for LOWESS
seg_location = [47] # list of segment locations to apply different smoothing, e.g., [20, 30] to have three segments: 0-20km, 20-30km, 30-end



# ================================== Step 1: Download PIXC data ======================================================================

if "step_1" not in skip_steps:
  pixc_file_paths = download_pixc_data(home_dir=home_dir,
                     pixc_version=pixc_version,
                   riv_name=river_name,
                   start_date=start_date,
                   end_date=end_date,
                   pass_tile_list=pass_tile_list,
                   login_strategy=login_strategy)
else:
  print("Skipping step 1: Download PIXC data")
  pixc_file_paths = sorted(glob.glob(os.path.join(home_dir, river_name, pixc_version, "*.nc")))
  print(f"Found {len(pixc_file_paths)} PIXC files in {os.path.join(home_dir, river_name, pixc_version)}")

# =================================== Step 2: Process river shapefile to generate nodes and buffers ===================================

river = River(home_dir=home_dir,
              riv_name=river_name,
              riv_shp_path = river_shp_path,
              node_spacing = node_spacing,
              riv_width = channel_width,
              )

if "step_2" not in skip_steps:
  # Initiate a river object
  # generate node, create buffer, then export both shapefiles to "home_dir/river_name/nodes/"
  # river.generate_nodes()
  # river.generate_buffers()
  # river.export_shapefiles()
  # or just call process_river to do all above steps
  river.process_river()
else:
  print("Skipping step 2: Process river shapefile to generate nodes and buffers")
  river.buffer_export_path = os.path.join(home_dir, river_name, "nodes", f"buffer_{node_spacing}m.shp")
  river.node_export_path = os.path.join(home_dir, river_name, "nodes", f"nodes_{node_spacing}m.shp")
  print(f"Using existing buffer path: {river.buffer_export_path}")
  print(f"Using existing node path: {river.node_export_path}")


# =================================== Step 3: Convert PIXC netcdf to csv after filtering non-water pixels and qc flags ===================================
# initiate PIXC object

if "step_3" not in skip_steps:
  pixc = PIXC(home_dir=home_dir,
              riv_name=river_name,
              pixc_file_paths=pixc_file_paths,
              var_list=None,
              create_ref_table_on_init=True)

  # extract water pixels within buffers
  pixc_water_paths = pixc.process_water_pixels(node_buffer_path=river.buffer_export_path,
                            pixc_water_dir_name=pixc_water_dir_name,
                            n_parts=n_partitions,
                            classification_categories=classification_categories,
                              prior_water_prob_threshold=prior_water_prob_threshold,
                              water_frac_threshold=water_frac_threshold,
                              dask_strategy = "dask-delay"
                            )
  pixc_water_qc_filtered_paths = pixc.filter_with_quality_flags(
      pixc_water_paths=pixc.pixc_water_paths,
      quality_flag_dict=quality_flag_dict,
      pixc_qc_dirname=pixc_qc_dirname
      )
else:
  print("Skipping step 3: Convert PIXC netcdf to csv after filtering non-water pixels and qc flags")
  pixc_water_paths = sorted(glob.glob(os.path.join(home_dir, river_name, pixc_water_dir_name, "*.csv")))
  pixc_water_qc_filtered_paths = sorted(glob.glob(os.path.join(home_dir, river_name, pixc_qc_dirname, "*.csv")))
  print(f"Found {len(pixc_water_paths)} PIXC water csv files in {os.path.join(home_dir, river_name, pixc_water_dir_name)}")
  print(f"Found {len(pixc_water_qc_filtered_paths)} PIXC water qc filtered csv files in {os.path.join(home_dir, river_name, pixc_qc_dirname)}")

# =================================== Step 4: Build WSE profiles over time ===================================

if "step_4" not in skip_steps:
  # Initiate a Profile object for pixc water data
  profile = Profile(home_dir=home_dir,
                    riv_name=river_name,
                    node_path=river.node_export_path,
                    buffer_path=river.buffer_export_path,
                    pixc_csv_paths=pixc_water_paths,
                    output_path=None
                    )
  # build WSE profiles for all dates
  wse_water = profile.build_wse_profiles_over_time(
                                      agg_func = agg_func,
                                      smooth_target = "wse_median",
                                      keep_qual_groups=keep_qual_groups,
                                      interpolation_method="lowess",
                                      seg_locations=seg_location,
                                      # interpolation params
                                      frac_list=frac_list,
                                      it=it,
                                      save_output=None,
                                      dask_strategy = "sequential"
                                      )
  # Initiate another Profile object for pixc water qc filtered data
  profile_qc = Profile(home_dir=home_dir,
                    riv_name=river_name,
                    node_path=river.node_export_path,
                    buffer_path=river.buffer_export_path,
                    pixc_csv_paths=pixc_water_qc_filtered_paths,
                    output_path=None
                    )
  # build WSE profiles for all dates based on QC filtered data
  wse_water_qc = profile_qc.build_wse_profiles_over_time(
                                      agg_func = agg_func,
                                      smooth_target = "wse_median",
                                      keep_qual_groups=keep_qual_groups,
                                      interpolation_method="lowess",
                                      seg_locations=seg_location,
                                      # interpolation params
                                      frac_list=frac_list,
                                      it=it,
                                      save_output=None,
                                      dask_strategy = "sequential"
                                      )
else:
  print("Skipping step 4: Build WSE profiles over time")
  # read existing csv files
  wse_water = pd.read_csv(os.path.join(home_dir, river_name, "pixc_water_profiles.csv"))
  wse_water_qc = pd.read_csv(os.path.join(home_dir, river_name, "pixc_water_qc_filtered_profiles.csv"))
  print(f"Loaded existing wse_water profile with {len(wse_water)} rows from {os.path.join(home_dir, river_name, 'pixc_water_profiles.csv')}")
  print(f"Loaded existing wse_water_qc profile with {len(wse_water_qc)} rows from {os.path.join(home_dir, river_name, 'pixc_water_qc_filtered_profiles.csv')}")


# The exported profiles dropped the nodes that do not have any WSE values, so we need to re-attach the dist_km from the node shapefile
node_gdf = gpd.read_file(river.node_export_path)
wse_water = wse_water.set_index("date").groupby(level = "date").apply(
    lambda x: node_gdf[["node_id", "dist_km"]].merge(
        x.drop(columns=["dist_km"]), on="node_id", how="left"
    )
).reset_index(drop=False)
wse_water_qc = wse_water_qc.set_index("date").groupby(level = "date").apply(
    lambda x: node_gdf[["node_id", "dist_km"]].merge(
        x.drop(columns=["dist_km"]), on="node_id", how="left"
    )
).reset_index(drop=False)


# export final profiles
wse_water.to_csv(wse_water_path, index=False)
wse_water_qc.to_csv(wse_water_qc_path, index=False)
print(f"Exported wse_water profile to {wse_water_path}")
print(f"Exported wse_water_qc profile to {wse_water_qc_path}")
