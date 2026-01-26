import numpy as np
import pandas as pd
import earthaccess
import os
import datetime

def download_pixc_data(home_dir: str, 
                       riv_name: str,
                       start_date: str,
                       end_date: str,
                       pass_tile_list: list,
                       pixc_version: str = "SWOT_L2_HR_PIXC_2.0",
                       login_strategy: str = 'netrc',
                       ):
    """
    Downloads PIXC2.0 data from Earthdata for a specified river and date range.
    Arguments:
        home_dir (str): Base directory to save the downloaded data.
        riv_name (str): Name of the river to filter data.
        start_date (str): Start date for data retrieval in 'YYYY-MM-DD' format.
        end_date (str): End date for data retrieval in 'YYYY-MM-DD' format.
        pass_tile_list (list): List of pass tile identifiers to filter data. For example, ["300_030R", "301_031R"] means pass 300, tile 030R and pass 301, tile 031R.
        pixc_version (str, optional): Directory name for PIXC data. Defaults to "SWOT_L2_HR_PIXC_2.0".
        login_strategy (str, optional): Login strategy for Earthdata authentication. Defaults to 'netrc'. Use 'interactive' for manual login.
    Returns:
        list: List of downloaded PIXC file paths.
    """

    # Initialize EarthAccess client
    auth = earthaccess.login(strategy=login_strategy)

    # Create a directory for this river
    riv_dir = os.path.join(home_dir, riv_name)
    pixc_dir = os.path.join(riv_dir, pixc_version)

    # Search and Download PIXC data
    # turn start and end date into datetime objects
    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    temporal_range = (start_dt, end_dt)
    for swot_pass_tile in pass_tile_list:
        pixc_results = earthaccess.search_data(short_name = pixc_version, # short name of the product
                                            temporal = temporal_range, # can also specify by time
                                            # bounding_box = tuple(bbox), 
                                            granule_name = f"*_{swot_pass_tile}_*") # Lake
        # download
        download_files = earthaccess.download(pixc_results, pixc_dir)

    print(f"Downloaded {len(download_files)} PIXC files to {pixc_dir}")

    return download_files