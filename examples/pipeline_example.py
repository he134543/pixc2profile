# This is an example to run the all steps in 1 pipeline
import pixc2profile.pipeline as pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Deine configuration dictionary
config = {
  # General settings
  "home_dir":"/mnt/d/pixc2profile/examples/data", # Root directory for read/save data
  "river_name":"test_river", # Name of the river
  # PIXC data aquisition settings for Step 1
  "login_strategy":"netrc", # earthaccess login strategy; recommended to use 'netrc', or 'interactive' for interactive login
  "start_date":"2024-01-01",
  "end_date":"2024-03-01",
  "pass_tile_list":["454_082L","454_083L","191_227L","191_227R","191_226R"], # Pass_Tile
  "pixc_version":"SWOT_L2_HR_PIXC_2.0", # PIXC version

  # Node and channel settings for Step 2
  "node_spacing":50,
  "channel_width":260,
  
  # PIXC filtering settings for Step 3
  "n_partitions":10, # Number of partitions to process PIXC data in parallel.
  "classification_categories":[3,4], # Check PIXC documentation for classification categories.
  "prior_water_prob_threshold":0.5, # Threshold for prior water probability (Pekel et al. 2016).
  "water_frac_threshold":0.2, # Fraction of water pixels threshold.
  "wse_upper_limit":400, # Upper limit for WSE to filter out extreme values.
  "quality_flag_dict":{"geolocation_qual":[4,64,68]}, # Example quality flags to filter out
  "pixc_water_dir_name":"pixc_water", # Directory name to save PIXC data after filtering water pixels
  "pixc_qc_dirname":"pixc_water_qc_filtered", # Directory name to save PIXC data after applying quality flags
  
  # Profile reconstruction settings for Step 4
  "profile_range":[0,"inf"], # Range of profiles to reconstruct; use "inf" or "-inf" for infinity; Distance in km from the first node
  "agg_func":"median", # Aggregate WSE on each node
  "keep_qual_groups":["interferogram_qual","classification_qual","geolocation_qual","sig0_qual"],
  "frac_list":[0.01,0.05,0.1,0.2], # A list of fractions for LOWESS
  "it":3, # LOWESS iterations
  "seg_location":[], # e.g., segment LOWESS smoothing at specific locations. Empty list means no segmentation.
  "log_level":"INFO"
}

# Run the full pipeline
pipeline(config)



# Final profiles are saved in the directory /mnt/d/pixc2profile/examples/data/test_river/profiles.csv
wse_profiles = pd.read_csv("/mnt/d/pixc2profile/examples/data/test_river/profiles.csv")
sns.scatterplot(data=wse_profiles, x='dist_km', y='wse_raw', hue='date')
sns.lineplot(data=wse_profiles, x='dist_km', y='wse_smooth_lowess_0.1', hue='date', legend=False) # fraction number = 0.1 for lowess smoothing
plt.show()