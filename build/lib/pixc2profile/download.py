import numpy as np
import pandas as pd
import earthaccess
import os
import datetime
import xarray as xr

def download_pixc_data(home_dir: str, 
                       riv_name: str,
                       start_date: str,
                       end_date: str,
                       pass_tile_list: list,
                       pixc_version: str = "SWOT_L2_HR_PIXC_2.0",
                       login_strategy: str = 'netrc',
                       check_file_integrity: bool = True
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
    download_files = []
    download_links = []
    for swot_pass_tile in pass_tile_list:
        pixc_results = earthaccess.search_data(short_name = pixc_version, # short name of the product
                                            temporal = temporal_range, # can also specify by time
                                            # bounding_box = tuple(bbox), 
                                            granule_name = f"*_{swot_pass_tile}_*") # Lake
        print(f"Found {len(pixc_results)} PIXC files for {swot_pass_tile} between {start_date} and {end_date}")
        print(f"Downloading PIXC files for {swot_pass_tile}...")
        download_links.extend(pixc_results)
        try:
            # download
            downloaded = earthaccess.download(pixc_results, pixc_dir)
            download_files.extend(downloaded)
        except Exception as e:
            print(f"No available PIXC data for {swot_pass_tile}: {e}")
            continue
    # create a dict to store the mapping of files and their corresponding results
    file_result_mapping = {file: result for file, result in zip(download_files, download_links)}
    # check file integrity
    if check_file_integrity:
        print("Checking file integrity...")
        for file in download_files:
            try:
                # open the file with xarray to check if it's valid
                ds = xr.open_dataset(file)
                ds.close()
            except Exception as e:
                print(f"File {file} is corrupted or incomplete")
                # remove the corrupted file
                os.remove(file)
                download_files.remove(file)
                print(f"Redownloading {file}...")
                # redownload the file
                try:
                    redownloaded = earthaccess.download([file_result_mapping[file]], pixc_dir)
                    download_files.extend(redownloaded)
                except Exception as e:
                    print(f"Failed to redownload {file}: {e}")
                    continue

    return download_files