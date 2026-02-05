import glob
import os
from typing import List, Tuple, Optional, Union

import pandas as pd 
import numpy as np
import geopandas as gpd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from dask import delayed
from dask.diagnostics import ProgressBar
from tqdm import tqdm
from pixc2profile.helper import aggregate_wse_with_other_vars
from pixc2profile.interpolator import Interpolator

class Profile:
    # Constants
    DEFAULT_CRS = "EPSG:4326"
    DATE_FORMAT_PATTERN = r"\d{8}"
    
    # Default column names
    DEFAULT_WSE_COL = "wse"
    DEFAULT_DIST_COL = "dist_km"
    
    # Default parameters
    DEFAULT_AGG_FUNCS = ['median', 'mean', 'std', 'count', "q25", "q75"]
    DEFAULT_QUAL_GROUPS = ["interferogram_qual", "classification_qual", "geolocation_qual", "sig0_qual"]
    DEFAULT_FRAC_LIST = [0.01, 0.05, 0.1, 0.2]
    DEFAULT_LOWESS_ITERATIONS = 3


    def __init__(self,
                 riv_name: str,
                 home_dir: str,
                 node_path: str,
                 buffer_path: str,
                 pixc_csv_paths: List[str],
                 output_path: str,
                 wse_col: str = None,
                 dist_col: str = None,
                 ) -> None:
        """
        Initialize the profile class for a specific river.
        
        Args:
            riv_name: Name of the river.
            home_dir: Base directory for the river data.
            node_path: Path to the shapefile containing river nodes.
            buffer_path: Path to the shapefile containing river buffers.
            pixc_csv_paths: List of paths to pixc observations CSV files.
            output_path: File path to save the final output (csv).
            wse_col: Name of the water surface elevation column in PIXC data. Defaults to 'wse'.
            dist_col: Name of the distance column in buffer data. Defaults to 'dist_km'.
            
        Raises:
            FileNotFoundError: If required files don't exist.
        """
        # Validate inputs
        self._validate_initialization_inputs(
            node_path, buffer_path, pixc_csv_paths
        )
        
        self.riv_name = riv_name
        self.home_dir = home_dir
        self.node_path = node_path
        self.buffer_path = buffer_path

        # File paths for PIXC data
        self.pixc_csv_paths = pixc_csv_paths

        # Column names
        self.wse_col = wse_col if wse_col is not None else self.DEFAULT_WSE_COL
        self.dist_col = dist_col if dist_col is not None else self.DEFAULT_DIST_COL

        # Initialize storage
        self.aval_dates: Optional[List[str]] = None
        self.node_gdf: Optional[gpd.GeoDataFrame] = None
        self.node_buffer_gdf: Optional[gpd.GeoDataFrame] = None

        # Output profiles
        self.output_path = output_path
        
        # Initialize interpolator
        self.interpolator = Interpolator()

    def _validate_initialization_inputs(self, node_path: str, buffer_path: str, 
                                       pixc_csv_paths: List[str]) -> None:
        """Validate that required paths exist."""
        # Check shapefiles exist
        shapefiles_to_check = {
            "Node shapefile": node_path,
            "Buffer shapefile": buffer_path,
        }
        
        for name, path in shapefiles_to_check.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found: {path}")
        
        # Validate file path lists
        if not pixc_csv_paths:
            raise ValueError("pixc_csv_paths cannot be empty")
        
        # Check that all files in the lists exist
        for i, path in enumerate(pixc_csv_paths):
            if not os.path.exists(path):
                raise FileNotFoundError(f"PIXC CSV file not found: {path}")
    
    def _extract_dates_from_files(self, file_paths: List[str]) -> List[str]:
        """Extract dates from PIXC file names."""
        return [os.path.basename(path).split("_")[0] for path in file_paths]
    
    def load_available_dates(self) -> None:
        """
        Load available dates from the PIXC CSV file paths.
        """
        # Extract dates from file names
        aval_dates = self._extract_dates_from_files(self.pixc_csv_paths)
        self.aval_dates = sorted(list(set(aval_dates)))

    def load_node_buffer_gdf(self) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Load the river nodes and buffers as GeoDataFrames.
        
        Returns:
            Tuple of (nodes_gdf, buffers_gdf): GeoDataFrames of river nodes and buffers.
            
        Raises:
            FileNotFoundError: If shapefiles cannot be read.
            ValueError: If required columns are missing.
        """
        try:
            nodes_gdf = gpd.read_file(self.node_path)
            nodes_gdf = nodes_gdf.to_crs(crs=self.DEFAULT_CRS)
            
            node_buffer_gdf = gpd.read_file(self.buffer_path)
            node_buffer_gdf = node_buffer_gdf.to_crs(crs=self.DEFAULT_CRS)
            
            # Validate required columns
            required_cols = ['node_id', self.dist_col]
            for col in required_cols:
                if col not in node_buffer_gdf.columns:
                    raise ValueError(f"Required column '{col}' not found in buffer shapefile")
            
            self.node_gdf = nodes_gdf
            self.node_buffer_gdf = node_buffer_gdf
            return nodes_gdf, node_buffer_gdf
            
        except Exception as e:
            raise FileNotFoundError(f"Error loading shapefiles: {str(e)}")
    
    def _create_date_to_file_mapping(self) -> dict:
        """
        Create mapping from dates to file paths.
        
        Returns:
            Dictionary mapping dates to file paths.
        """
        date_map = {}
        for path in self.pixc_csv_paths:
            date = os.path.basename(path).split("_")[0]
            date_map[date] = path
        
        return date_map
    
    def _load_pixc_data_for_date(self, date: str) -> gpd.GeoDataFrame:
        """
        Load PIXC data for a specific date.
        
        Args:
            date: Date string in format YYYYMMDD.
            
        Returns:
            GeoDataFrame with PIXC data.
        """
        # Create date-to-file mapping
        date_map = self._create_date_to_file_mapping()
        
        if date not in date_map:
            raise FileNotFoundError(f"No PIXC file found for date: {date}")
        
        pixc_path = date_map[date]
        
        # Load and convert to GeoDataFrame
        df = pd.read_csv(pixc_path)
        gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df.longitude, df.latitude), 
            crs=self.DEFAULT_CRS
        )
        
        return gdf
    
    def _ensure_dependencies_loaded(self) -> None:
        """Ensure required data (dates and geometries) are loaded."""
        if self.aval_dates is None:
            self.load_available_dates()
        if self.node_gdf is None or self.node_buffer_gdf is None:
            self.load_node_buffer_gdf()
    
    def agg_wse_per_node(self,
                          date: str,
                          agg_func: List[str] = None,
                          keep_qual_groups: List[str] = None,
                          ) -> pd.DataFrame:
        """
        Build aggregated WSE profiles for a specific date from raw and QC PIXC water points.
        
        Args:
            date: Date string in format YYYYMMDD.
            agg_func: List of aggregation functions to apply.
            keep_qual_groups: List of quality groups to preserve.
            
        Returns:
            DataFrame with aggregated WSE data per node.
            
        Raises:
            ValueError: If date is not available.
        """
        # Set defaults
        if agg_func is None:
            agg_func = self.DEFAULT_AGG_FUNCS
        if keep_qual_groups is None:
            keep_qual_groups = self.DEFAULT_QUAL_GROUPS
        
        # Ensure dependencies are loaded
        self._ensure_dependencies_loaded()
        
        # Validate date
        if date not in self.aval_dates:
            raise ValueError(f"Date {date} not available. Available dates: {self.aval_dates}")
        
        # Load PIXC data for the date
        gdf = self._load_pixc_data_for_date(date)
        
        # Find intersected PIXC points with reach node buffer
        pixc_points_in_buffer = gpd.sjoin(
            self.node_buffer_gdf, gdf, predicate='intersects', how="inner"
        )
        
        # Aggregate WSE by node; keep quality flags associated with the reference stat
        node_wse = pixc_points_in_buffer.set_index("node_id").groupby(level = "node_id").apply(
            lambda x: aggregate_wse_with_other_vars(
                x, keep_qual_groups, include_stats=agg_func, reference_stat='median', wse_col=self.wse_col, dist_col=self.dist_col
            ), 
        ).reset_index()
    
        
        # Re-attach node_id and dist_km to ensure all nodes are present
        node_wse = self.node_buffer_gdf.loc[:, ["node_id", self.dist_col]].merge(
            node_wse, on=["node_id", self.dist_col], how='left'
        ).reset_index(drop=True).sort_values(by=self.dist_col)
        
        # Add date column
        node_wse["date"] = date
        
        return node_wse
    
    def smooth_wse_profile(
        self,
        node_wse: pd.DataFrame,
        var_name: str, 
        method: str = 'lowess',
        seg_locations: List[float] = None,
        **method_params
    ) -> pd.DataFrame:
        """
        Smooth the WSE profile using various interpolation methods.
        
        Args:
            node_wse: DataFrame containing the WSE profile with columns 'dist_km' and WSE values.
            var_name: Name of the variable to smooth.
            method: Interpolation method ('lowess', 'gaussian_process').
            seg_locations: List of segment locations (in km) to apply piecewise smoothing.
            **method_params: Method-specific parameters (e.g., frac_list for LOWESS, length_scale for GP).
            
        Returns:
            DataFrame with additional smoothed columns.
            
        Raises:
            ValueError: If required columns are missing or method is not supported.
        """
        # Set defaults for seg_locations
        if seg_locations is None:
            seg_locations = []
        
        # Set default parameters based on method
        if method == 'lowess' and 'frac_list' not in method_params:
            method_params['frac_list'] = self.DEFAULT_FRAC_LIST
        if method == 'lowess' and 'it' not in method_params:
            method_params['it'] = self.DEFAULT_LOWESS_ITERATIONS
        
        # Use the interpolator to smooth the profile
        try:
            result = self.interpolator.interpolate(
                data=node_wse,
                x_col=self.dist_col,
                y_col=var_name,
                method=method,
                seg_locations=seg_locations,
                **method_params
            )
            return result
        except Exception as e:
            print(f"Warning: {method} smoothing failed for {var_name}: {e}")
            return node_wse
    
    def build_wse_profiles_over_time(self,
                                    agg_func: List[str] = None,
                                    smooth_target: str = "wse_median",
                                    keep_qual_groups: List[str] = None,
                                    interpolation_method: str = 'lowess',
                                    seg_locations: List[float] = None,
                                    save_output: bool = True,
                                    dask_strategy: str = "dask-delay",
                                    **interpolation_params
                                    ) -> pd.DataFrame:
        """
        Build WSE profiles over all available dates.
        
        Args:
            agg_func: List of aggregation functions to apply.
            smooth_target: Target variable to smooth (e.g., 'wse_median').
            keep_qual_groups: List of quality groups to preserve.
            interpolation_method: Interpolation method ('lowess', 'gaussian_process', 'moving_average', 'polynomial').
            seg_locations: List of segment locations (in km) for piecewise smoothing.
            save_output: Whether to save the output to CSV file.
            dask_strategy: Dask strategy for parallel processing ('dask-delay' or 'geopandas').
            **interpolation_params: Method-specific parameters for interpolation.
            
        Returns:
            DataFrame containing WSE profiles over time.
            
        Raises:
            ValueError: If no dates are available.
        """
        # Check if available dates are loaded
        if self.aval_dates is None:
            self.load_available_dates()
        
        if not self.aval_dates:
            raise ValueError("No available dates found. Check PIXC data directories.")
        
        print(f"Building WSE profiles for {len(self.aval_dates)} dates...")
        
        # Define delayed function for processing a single date
        def process_single_date(date):
            # Calculate the WSE profile for this date
            node_wse = self.agg_wse_per_node(
                date=date,
                agg_func=agg_func,
                keep_qual_groups=keep_qual_groups,
            )
            
            # Smooth the WSE profile
            node_wse = self.smooth_wse_profile(
                node_wse=node_wse,
                var_name=smooth_target,
                method=interpolation_method,
                seg_locations=seg_locations,
                **interpolation_params
            )
            
            return node_wse
        
        # Create delayed tasks for all dates
        if dask_strategy == "dask-delay":
            delayed_tasks = [delayed(process_single_date)(date) for date in self.aval_dates]
            # Execute all tasks in parallel with progress bar
            print("Processing dates in parallel using Dask delayed...")
            with ProgressBar():
                profile_list = delayed(list)(delayed_tasks).compute()

        else:
            profile_list = []
            for date in tqdm(self.aval_dates, desc="Processing dates"):
                # print(f"Processing date: {date}")
                profile_df = process_single_date(date)
                profile_list.append(profile_df)
        
        print(f"Successfully processed all {len(profile_list)} dates")
        
        # Concatenate all profiles into one dataframe
        all_profiles = pd.concat(profile_list, ignore_index=True).sort_values(by=["date", self.dist_col])
        
        print(f"Successfully built profiles with {len(all_profiles)} records")
        
        # Save to CSV if requested
        if save_output:
            try:
                # Create output directory if it doesn't exist
                output_dir = os.path.dirname(self.output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                all_profiles.to_csv(self.output_path, index=False)
                print(f"Output saved to: {self.output_path}")
                
            except Exception as e:
                print(f"Warning: Failed to save output to {self.output_path}: {e}")
        
        return all_profiles

        

