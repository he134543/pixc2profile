# pixc2profile
A Python toolkit for extracting along-river water surface elevation (WSE) profiles directly from SWOT Pixel Cloud (PIXC) data.

## Steps:

- Step 1: Download PIXC data. 
- Step 2: Generate nodes and create buffers along the user-defined river centerline.
- Step 3: Filter PIXC observations: non-water, poor-quality, observations outside the buffers.
- Step 4: Aggregate WSEs on each node to reconstruct WSE profiles, with optional LOWESS smoothing.

![Framework](docs/framework.png)

## Why pixc2profile?

This toolkip is a more detailed and customizable alternative to the official SWOT River SP product, enabling users to extract WSE profiles that better suit their specific research needs: user-defined river reach, node spacing, filtering criteria and smoothing parameters. 

## Quick Start

1. Clone the repository and install the package from source:

   ```bash
   git clone https://github.com/he134543/pixc2profile.git
   cd pixc2profile
   pip install .
   ```

2. Run the full pipeline with your configuration:

    Profiles are saved in the specified `{home_dir}/{river_name}/profiles.csv`.

    ```python
    import pixc2profile.pipeline as piepline
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
    ```

    The final output look like this: 
    | node_id | dist_km |   date  |     wse_raw     | wse_lowess_0.01 | wse_lowess_0.05 | wse_lowess_0.1 | wse_lowess_0.2 | ...qual_groups... |
    |---------|---------|---------|-----------------|-----------------|-----------------|----------------|----------------|-------------------|
    |    0    |   0.0   |20240110 |      120.5      |      120.3      |      120.4      |     120.45     |     120.5      |        ...        |
    |    1    |   0.05  |20240110 |      121.0      |      120.8      |      120.9      |     120.95     |     121.0      |        ...        |
    

3. (Optional) If you want to know each step is implemented, see "examples/step_by_step_Guide.ipynb" for more details.
