import glob as glob
import xarray as xr 
import pandas as pd 
import numpy as np
import geopandas as gpd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import dask_geopandas as dgpd
import json
from dask.diagnostics import ProgressBar
from typing import Optional, List

_pixc_var_list = [
        "latitude", 
        "longitude", 
        "height",
        "geoid",
        "solid_earth_tide",
        "load_tide_fes",
        "pole_tide",
        "classification",
        "water_frac",
        "prior_water_prob",
        "bright_land_flag",
        "interferogram_qual",
        "classification_qual",
        "geolocation_qual",
        "sig0_qual",
]


class PIXC:
    """
    A class for processing SWOT PIXC (Pixel Cloud) data.
    
    This class provides methods to load, process, and filter PIXC data files,
    creating reference tables and extracting water pixel information.
    """
    
    def __init__(self, 
                 home_dir: str,
                 riv_name: str,
                 pixc_dirname: str = "SWOT_L2_HR_PIXC_2.0",
                 var_list: List[str] = None,
                 create_ref_table_on_init: bool = True):
        """
        Initialize PIXC processor.
        
        Arguments:
            home_dir (str): Base directory where PIXC data is stored.
            riv_name (str): Name of the river.
            pixc_dirname (str, optional): Directory name for PIXC data. Defaults to "SWOT_L2_HR_PIXC_2.0".
            var_list (List[str], optional): List of variables to extract from PIXC files. Defaults to _pixc_var_list.
            create_ref_table_on_init (bool, optional): Whether to create reference table on initialization. Defaults to True.
        """
        self.home_dir = home_dir
        self.riv_name = riv_name
        self.pixc_dirname = pixc_dirname
        self.var_list = var_list if var_list is not None else _pixc_var_list.copy()
        
        # Initialize data storage
        self.ref_table = None
        self.pixc_water_paths = []
        self.pixc_water_qc_filtered_paths = []
        self.version = None
        
        # Create reference table if requested
        if create_ref_table_on_init:
            self.create_ref_table()
    
    @property
    def pixc_dir(self) -> str:
        """Get the full path to the PIXC directory."""
        return os.path.join(self.home_dir, self.riv_name, self.pixc_dirname)
    
    @property
    def pixc_water_dir(self) -> str:
        """Get the full path to the PIXC water directory."""
        return os.path.join(self.home_dir, self.riv_name, "pixc_water")
    
    @property
    def available_dates(self) -> List[str]:
        """Get list of available dates from the reference table."""
        if self.ref_table is not None:
            return self.ref_table.date.unique().tolist()
        return []
    
    def create_ref_table(self) -> pd.DataFrame:
        """
        Load PIXC2.0 file information and create a reference table.
        
        Returns:
            pd.DataFrame: Reference table containing PIXC file paths and associated metadata.
        """
        # read how many pixc files are downloaded
        pixc_files = glob.glob(os.path.join(self.pixc_dir, "*.nc"))
        print(f"Found {len(pixc_files)} PIXC files in {self.pixc_dir}")

        # extract date, tile and pass from pixc filenames
        if self.pixc_dirname == "SWOT_L2_HR_PIXC_2.0":
            self.version = "C"    
        elif self.pixc_dirname == "SWOT_L2_HR_PIXC_D":
            self.version = "D"
        else:
            raise ValueError("Unsupported PIXC version. Supported versions are 'SWOT_L2_HR_PIXC_2.0' and 'SWOT_L2_HR_PIXC_D'.")

        # ATTENTION: this assumes the filename format is consistent with version standard
        self.ref_table = pd.DataFrame()
        self.ref_table["PIXC_paths"] = pixc_files
        self.ref_table["date"] = self.ref_table.PIXC_paths.apply(lambda fp: os.path.basename(fp).split("_")[-3][:8])
        self.ref_table["cycle"] = self.ref_table.PIXC_paths.apply(lambda fp: os.path.basename(fp).split("_")[4])
        self.ref_table["pass_id"] = self.ref_table.PIXC_paths.apply(lambda fp: os.path.basename(fp).split("_")[5])
        self.ref_table["tile"] = self.ref_table.PIXC_paths.apply(lambda fp: os.path.basename(fp).split("_")[6])

        return self.ref_table

    def process_water_pixels(self,
                           node_buffer_path: str,
                           pixc_water_dir_name: str = "pixc_water",
                           n_parts: int = 10,
                           subset_dates: List[str] = None,
                           classification_categories: List[int] = [3, 4],
                           prior_water_prob_threshold: float = 0.5,
                           water_frac_threshold: float = 0.2,
                           wse_upper_limit: float = None,
                           channel_mask_path: Optional[str] = None,
                           ) -> List[str]:
        """
        Convert PIXC netCDF files to CSV files containing water points within the buffer area.
        
        Arguments:
            node_buffer_path (str): Path to the buffer shapefile for spatial filtering.
            pixc_water_dir_name (str): Directory name for water output files.
            n_parts (int, optional): Number of partitions for Dask GeoDataFrame. Defaults to 10.
            subset_dates (List[str], optional): List of dates to process. If None, processes all available dates.
            classification_categories (List[int], optional): List of classification categories to include. Defaults to [3,4].
            prior_water_prob_threshold (float, optional): Minimum prior water probability to include a point. Defaults to 0.5.
            water_frac_threshold (float, optional): Minimum water fraction to include a point. Defaults to 0.2.
            wse_upper_limit (float, optional): Upper limit for water surface elevation (WSE) to filter out erroneous data. Defaults to None.
            channel_mask_path (Optional[str], optional): Path to channel mask file for additional filtering. Defaults to None.
        
        Returns:
            List[str]: List of paths to the created water CSV files.
        """
        if self.ref_table is None:
            raise ValueError("Reference table not created. Call create_ref_table() first.")
        
        # read buffer shapefile; convert utm crs to epsg:4326 for intersection usage below
        nodes_buffer_gdf = gpd.read_file(node_buffer_path).to_crs("EPSG:4326")
        nodes_buffer_gdf = gpd.GeoDataFrame(geometry=[nodes_buffer_gdf.union_all()]).set_crs(epsg=4326)

        # read available dates
        available_dates = self.available_dates
        # if subset_dates is provided, filter available_dates
        if subset_dates is not None:
            available_dates = [dt for dt in available_dates if dt in subset_dates]

        # create output directory if it doesn't exist
        water_dir = os.path.join(self.home_dir, self.riv_name, pixc_water_dir_name)
        os.makedirs(water_dir, exist_ok=True)

        # reset the water paths list
        self.pixc_water_paths = []

        # process each date separately
        for aval_date in tqdm(available_dates, desc="Filtering water pixels within node buffers"):
            # initialize an empty dataframe to save output
            df = pd.DataFrame([])
            # potentially multiple pixc files for this date
            pixc_paths_this_date = self.ref_table.loc[self.ref_table.date == aval_date].PIXC_paths.values

            for i in range(len(pixc_paths_this_date)):
                # load pixc file
                ds_pixc = xr.open_dataset(pixc_paths_this_date[i], 
                                      group = "pixel_cloud", 
                                      chunks = "auto")[self.var_list]
                # filter water pixels
                ds_pixc = ds_pixc.where(ds_pixc.classification.isin(classification_categories) & 
                                      (ds_pixc.prior_water_prob >= prior_water_prob_threshold) & 
                                      (ds_pixc.water_frac >= water_frac_threshold))

                # calculate water surface elevation (WSE)
                ds_pixc["wse"] = ds_pixc.height \
                                - ds_pixc.geoid \
                                - ds_pixc.solid_earth_tide \
                                - ds_pixc.load_tide_fes \
                                - ds_pixc.pole_tide
                # filter out unrealistic WSE values
                if wse_upper_limit is not None:
                    ds_pixc = ds_pixc.where(ds_pixc.wse <= wse_upper_limit)
                # convert to dataframe
                df_pixc = ds_pixc.to_dataframe().dropna()
                # concat to the total df
                df = pd.concat([df, df_pixc])
                # release cache
                del ds_pixc
                del df_pixc

            # drop duplicates
            df = df.drop_duplicates()
            df = gpd.GeoDataFrame(df, 
                                geometry = gpd.points_from_xy(df.longitude,
                                                                df.latitude),
                                crs = "EPSG:4326")
            df = dgpd.from_geopandas(df, npartitions=n_parts)  # Convert to Dask GeoDataFrame for parallel processing

            # only select points within the node buffer
            df = df.loc[df.geometry.intersects(nodes_buffer_gdf.union_all())]
            # only select points within the channel mask if provided
            if channel_mask_path is not None:
                channel_mask_gdf = gpd.read_file(channel_mask_path).to_crs(df.crs)
                df = df.loc[df.geometry.intersects(channel_mask_gdf.union_all())]

            # compute and export to csv
            pixc_water_path = os.path.join(water_dir, f"{aval_date}_water.csv")
            df.compute().to_csv(pixc_water_path, index = None)
            self.pixc_water_paths.append(pixc_water_path)

        return self.pixc_water_paths

    def filter_with_quality_flags(self,
                                pixc_water_paths: List[str] = None,
                                quality_flag_dict: dict = None,
                                pixc_qc_dirname: str = "pixc_water_qc_filtered",
                                ) -> None:
        """
        Filter PIXC water points based on quality flags.
        
        Arguments:
            pixc_water_paths (List[str], optional): List of file paths for the PIXC water CSV files. 
                                                   If None, uses self.pixc_water_paths.
            quality_flag_dict (dict): Dictionary specifying quality flags and their removal values. 
                                    For example, {"geolocation_qual":[4, 64]}
            export_dir (str, optional): Directory to save filtered files. 
        """
        if pixc_water_paths is None:
            if not self.pixc_water_paths:
                raise ValueError("No water paths available. Run process_water_pixels() first or provide pixc_water_paths.")
            pixc_water_paths = self.pixc_water_paths
            
        if quality_flag_dict is None:
            raise ValueError("quality_flag_dict must be provided")
        
        # initialize filtered paths list again
        self.pixc_water_qc_filtered_paths = []

        for pixc_water_path in tqdm(pixc_water_paths, desc="Filtering PIXC water points with quality flags"):
            # load pixc water csv, make sure wse and classification are float64, quality flags are int
            dtypes = {"wse": np.float64, 
                      "classification": np.float64}
            for flag in quality_flag_dict.keys():
                dtypes[flag] = np.int64
            df = pd.read_csv(pixc_water_path, dtype=dtypes).dropna(subset=["wse"])
            # apply quality flag filters, remove values that are within provided value list
            for flag, values in quality_flag_dict.items():
                df = df[~df[flag].isin(values)]
            # export back to csv
            export_dir = os.path.join(self.home_dir, self.riv_name, pixc_qc_dirname)
            export_path = os.path.join(export_dir, os.path.basename(pixc_water_path).replace("_water.csv", "_water_qc_filtered.csv"))
            self.pixc_water_qc_filtered_paths.append(export_path)
            os.makedirs(export_dir, exist_ok=True)
            df.reset_index(drop=True).to_csv(export_path, index=None)

        return self.pixc_water_qc_filtered_paths

    def get_summary_stats(self) -> dict:
        """
        Get summary statistics about the PIXC data.
        
        Returns:
            dict: Dictionary containing summary statistics.
        """
        stats = {
            'home_dir': self.home_dir,
            'riv_name': self.riv_name,
            'pixc_dirname': self.pixc_dirname,
            'version': self.version,
            'pixc_dir': self.pixc_dir,
            'n_available_dates': len(self.available_dates),
            'available_dates': self.available_dates,
            'n_processed_water_files': len(self.pixc_water_paths),
            'var_list': self.var_list
        }
        
        if self.ref_table is not None:
            stats['n_pixc_files'] = len(self.ref_table)
            stats['cycles'] = sorted(self.ref_table.cycle.unique().tolist())
            stats['passes'] = sorted(self.ref_table.pass_id.unique().tolist())
            stats['tiles'] = sorted(self.ref_table.tile.unique().tolist())
        
        return stats

    def __repr__(self) -> str:
        """String representation of the PIXC object."""
        return (f"PIXC(home_dir='{self.home_dir}', "
                f"riv_name='{self.riv_name}', "
                f"pixc_dirname='{self.pixc_dirname}', "
                f"version='{self.version}', "
                f"n_files={len(self.ref_table) if self.ref_table is not None else 0})")