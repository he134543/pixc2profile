import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pixc2profile import download_pixc_data
from pixc2profile.river import River
from pixc2profile.pixc import PIXC
from pixc2profile.profile import Profile

# ==================================== Configuration Parameters ====================================
home_dir = "/mnt/d/pixc2profile/examples/data"
river_name = "test_river"
start_date = "2025-08-01"
end_date = "2025-08-30"
pass_tile_list = ["454_082L", "454_083L", "191_227L", "191_227R", "191_226R"]
pixc_version = "SWOT_L2_HR_PIXC_D"
login_strategy = 'netrc'
river_shp_path = os.path.join(home_dir, "shps", "test_river.shp")
# create a node every 50 meters
node_spacing = 50 # m
# approximate river channel width for buffer generation
channel_width = 260 # m
pixc_water_dir_name="pixc_water"
pixc_qc_dirname="pixc_water_qc_filtered"
n_partitions = 10
classification_categories = [3, 4] # "water near land" and "open water"
prior_water_prob_threshold = 0.5 # Pekel et al. (2016)
water_frac_threshold = 0.2 # fraction of water
wse_upper_limit = 400 # m, upper limit of WSE to exclude outliers; Usually the elevation of river bank can be used here
quality_flag_dict = {
    "geolocation_qual": [4, 64, 68]
}
output_path=os.path.join(home_dir, river_name, "profiles.csv")
profile_range=(0, np.inf) # if the profile needs to be cutted, change this parameter
agg_func = ["median", "std", "mean", "count"] # aggregation function to calculate WSE for each node; currently only support "median"
# keep quality flag values for the median calculation
keep_qual_groups=["interferogram_qual", "classification_qual", "geolocation_qual", "sig0_qual"]
# smooth parameters
frac_list=[0.01, 0.05, 0.1] # multiple LOWESS fractions to try
it=3 # number of iterations for LOWESS
seg_location = [47] # list of segment locations to apply different smoothing, e.g., [20, 30] to have three segments: 0-20km, 20-30km, 30-end



# ================================== Step 1: Download PIXC data ======================================================================
pixc_file_paths = download_pixc_data(home_dir=home_dir,
                     pixc_version=pixc_version,
                   riv_name=river_name,
                   start_date=start_date,
                   end_date=end_date,
                   pass_tile_list=pass_tile_list,
                   login_strategy=login_strategy)

# =================================== Step 2: Process river shapefile to generate nodes and buffers ===================================
# Initiate a river object
river = River(home_dir=home_dir,
              riv_name=river_name,
              riv_shp_path = river_shp_path,
              node_spacing = node_spacing,
              riv_width = channel_width,
              )
# generate node, create buffer, then export both shapefiles to "home_dir/river_name/nodes/"
# river.generate_nodes()
# river.generate_buffers()
# river.export_shapefiles()
# or just call process_river to do all above steps
river.process_river()

# =================================== Step 3: Build WSE profiles over time ===================================
# initiate PIXC object
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
                            water_frac_threshold=water_frac_threshold
                          )
pixc_water_qc_filtered_paths = pixc.filter_with_quality_flags(
    pixc_water_paths=pixc.pixc_water_paths,
    quality_flag_dict=quality_flag_dict,
    pixc_qc_dirname=pixc_qc_dirname
    )

# =================================== Step 4: Build WSE profiles over time ===================================
# Initiate a Profile object for pixc water data
profile = Profile(home_dir=home_dir,
                  riv_name=river_name,
                  node_path=river.node_export_path,
                  buffer_path=river.buffer_export_path,
                  pixc_csv_paths=pixc_water_paths,
                  output_path=None
                  )
# build WSE profiles for all dates
wse_water = profile.build_wse_profiles_over_time(profile_range=profile_range, 
                                    agg_func = agg_func,
                                    smooth_target = "wse_median",
                                    keep_qual_groups=keep_qual_groups,
                                    interpolation_method="lowess",
                                    seg_locations=seg_location,
                                    # interpolation params
                                    frac_list=frac_list,
                                    it=it,
                                    save_output=None
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
wse_water_qc = profile_qc.build_wse_profiles_over_time(profile_range=profile_range, 
                                    agg_func = agg_func,
                                    smooth_target = "wse_median",
                                    keep_qual_groups=keep_qual_groups,
                                    interpolation_method="lowess",
                                    seg_locations=seg_location,
                                    # interpolation params
                                    frac_list=frac_list,
                                    it=it,
                                    save_output=None,
                                    )
# CONCAT wse_water and wse_water_qc for comparison plotting
wse_water["data_source"] = "pixc_water"
wse_water_qc["data_source"] = "pixc_water_qc_filtered"
wse_combined = pd.concat([wse_water, wse_water_qc], ignore_index=True)

# Export final profiles
wse_combined.to_csv(output_path, index=False)