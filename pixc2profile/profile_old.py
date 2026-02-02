import glob
import os
from typing import List, Tuple, Optional, Union

import pandas as pd 
import numpy as np
import geopandas as gpd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from pixc2profile.helper import aggregate_wse_with_other_vars

class Profile:
    # Constants
    DEFAULT_CRS = "EPSG:4326"
    DATE_FORMAT_PATTERN = r"\d{8}"
    
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
                 pixc_water_paths: List[str],
                 pixc_water_qc_filtered_paths: List[str],
                 output_path: str,
                 ) -> None:
        """
        Initialize the profile class for a specific river.
        
        Args:
            riv_name: Name of the river.
            home_dir: Base directory for the river data.
            node_path: Path to the shapefile containing river nodes.
            buffer_path: Path to the shapefile containing river buffers.
            pixc_water_paths: List of paths to pixc observations of water pixels within river node buffers; CSV files.
            pixc_water_qc_filtered_paths: List of paths to pixc observations after removing poor-quality observations; CSV files.
            output_path: File path to save the final output (csv).
            
        Raises:
            FileNotFoundError: If required files don't exist.
            ValueError: If file path lists don't match.
        """
        # Validate inputs
        self._validate_initialization_inputs(
            node_path, buffer_path, pixc_water_paths, pixc_water_qc_filtered_paths
        )
        
        self.riv_name = riv_name
        self.home_dir = home_dir
        self.node_path = node_path
        self.buffer_path = buffer_path

        # File paths for PIXC data
        self.pixc_water_paths = pixc_water_paths
        self.pixc_water_qc_filtered_paths = pixc_water_qc_filtered_paths

        # Initialize storage
        self.aval_dates: Optional[List[str]] = None
        self.node_gdf: Optional[gpd.GeoDataFrame] = None
        self.node_buffer_gdf: Optional[gpd.GeoDataFrame] = None

        # Output profiles
        self.output_path = output_path

    def _validate_initialization_inputs(self, node_path: str, buffer_path: str, 
                                       pixc_water_paths: List[str], pixc_water_qc_filtered_paths: List[str]) -> None:
        """Validate that required paths exist and file lists match."""
        # Check shapefiles exist
        shapefiles_to_check = {
            "Node shapefile": node_path,
            "Buffer shapefile": buffer_path,
        }
        
        for name, path in shapefiles_to_check.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} not found: {path}")
        
        # Validate file path lists
        if not pixc_water_paths:
            raise ValueError("pixc_water_paths cannot be empty")
        if not pixc_water_qc_filtered_paths:
            raise ValueError("pixc_water_qc_filtered_paths cannot be empty")
        
        # Check that all files in the lists exist
        for i, path in enumerate(pixc_water_paths):
            if not os.path.exists(path):
                raise FileNotFoundError(f"PIXC water file not found: {path}")
        
        for i, path in enumerate(pixc_water_qc_filtered_paths):
            if not os.path.exists(path):
                raise FileNotFoundError(f"PIXC QC filtered file not found: {path}")
        
        # Extract dates and validate they match
        water_dates = self._extract_dates_from_files(pixc_water_paths)
        qc_dates = self._extract_dates_from_files(pixc_water_qc_filtered_paths)
        
        if set(water_dates) != set(qc_dates):
            missing_in_water = set(qc_dates) - set(water_dates)
            missing_in_qc = set(water_dates) - set(qc_dates)
            error_msg = "Dates in PIXC water and QC filtered file lists do not match."
            if missing_in_water:
                error_msg += f" Missing in water files: {missing_in_water}"
            if missing_in_qc:
                error_msg += f" Missing in QC files: {missing_in_qc}"
            raise ValueError(error_msg)
    
    def _extract_dates_from_files(self, file_paths: List[str]) -> List[str]:
        """Extract dates from PIXC file names."""
        return [os.path.basename(path).split("_")[0] for path in file_paths]
    
    def load_available_dates(self) -> None:
        """
        Load available dates from the PIXC water and quality-controlled file paths.
        
        Raises:
            ValueError: If dates don't match between file lists.
        """
        # Extract dates from file names
        aval_dates_water = self._extract_dates_from_files(self.pixc_water_paths)
        aval_dates_qc_filtered = self._extract_dates_from_files(self.pixc_water_qc_filtered_paths)

        # Check if the two lists are the same (this should already be validated in __init__)
        if set(aval_dates_water) != set(aval_dates_qc_filtered):
            missing_in_water = set(aval_dates_qc_filtered) - set(aval_dates_water)
            missing_in_qc = set(aval_dates_water) - set(aval_dates_qc_filtered)
            error_msg = "Available dates in PIXC water and quality-controlled file lists do not match."
            if missing_in_water:
                error_msg += f" Missing in water: {missing_in_water}"
            if missing_in_qc:
                error_msg += f" Missing in QC: {missing_in_qc}"
            raise ValueError(error_msg)
            
        self.aval_dates = sorted(list(set(aval_dates_water)))

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
            required_cols = ['node_id', 'dist_km']
            for col in required_cols:
                if col not in node_buffer_gdf.columns:
                    raise ValueError(f"Required column '{col}' not found in buffer shapefile")
            
            self.node_gdf = nodes_gdf
            self.node_buffer_gdf = node_buffer_gdf
            return nodes_gdf, node_buffer_gdf
            
        except Exception as e:
            raise FileNotFoundError(f"Error loading shapefiles: {str(e)}")
    
    def _create_date_to_file_mapping(self) -> Tuple[dict, dict]:
        """
        Create mappings from dates to file paths.
        
        Returns:
            Tuple of (water_date_map, qc_date_map): Dictionaries mapping dates to file paths.
        """
        water_date_map = {}
        for path in self.pixc_water_paths:
            date = os.path.basename(path).split("_")[0]
            water_date_map[date] = path
        
        qc_date_map = {}
        for path in self.pixc_water_qc_filtered_paths:
            date = os.path.basename(path).split("_")[0]
            qc_date_map[date] = path
        
        return water_date_map, qc_date_map
    
    def _load_pixc_data_for_date(self, date: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Load PIXC water data for a specific date.
        
        Args:
            date: Date string in format YYYYMMDD.
            
        Returns:
            Tuple of (water_gdf, water_qc_gdf): GeoDataFrames for raw and QC water data.
        """
        # Create date-to-file mappings
        water_date_map, qc_date_map = self._create_date_to_file_mapping()
        
        if date not in water_date_map:
            raise FileNotFoundError(f"No PIXC water file found for date: {date}")
        if date not in qc_date_map:
            raise FileNotFoundError(f"No PIXC QC water file found for date: {date}")
        
        pixc_water_path = water_date_map[date]
        pixc_water_qc_path = qc_date_map[date]
        
        # Load and convert to GeoDataFrames
        df_water = pd.read_csv(pixc_water_path)
        gdf_water = gpd.GeoDataFrame(
            df_water, 
            geometry=gpd.points_from_xy(df_water.longitude, df_water.latitude), 
            crs=self.DEFAULT_CRS
        )
        
        df_water_qc = pd.read_csv(pixc_water_qc_path)
        gdf_water_qc = gpd.GeoDataFrame(
            df_water_qc, 
            geometry=gpd.points_from_xy(df_water_qc.longitude, df_water_qc.latitude), 
            crs=self.DEFAULT_CRS
        )
        
        return gdf_water, gdf_water_qc
    
    def _ensure_dependencies_loaded(self) -> None:
        """Ensure required data (dates and geometries) are loaded."""
        if self.aval_dates is None:
            self.load_available_dates()
        if self.node_gdf is None or self.node_buffer_gdf is None:
            self.load_node_buffer_gdf()
    
    def cal_agg_wse_per_node(self,
                          date: str,
                          profile_range: Tuple[float, float] = (0, np.inf),
                          agg_func: List[str] = None,
                          keep_qual_groups: List[str] = None,
                          ) -> pd.DataFrame:
        """
        Build aggregated WSE profiles for a specific date from raw and QC PIXC water points.
        
        Args:
            date: Date string in format YYYYMMDD.
            profile_range: Tuple of (min_km, max_km) to filter the profile.
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
        gdf_water, gdf_water_qc = self._load_pixc_data_for_date(date)
        
        # Find intersected PIXC points with reach node buffer
        pixc_points_in_buffer_water = gpd.sjoin(
            self.node_buffer_gdf, gdf_water, predicate='intersects', how="inner"
        )
        pixc_points_in_buffer_water_qc = gpd.sjoin(
            self.node_buffer_gdf, gdf_water_qc, predicate='intersects', how="inner"
        )
        
        # Aggregate WSE by node
        agg_water = pixc_points_in_buffer_water.groupby("node_id").apply(
            lambda x: aggregate_wse_with_other_vars(
                x, keep_qual_groups, include_stats=agg_func, reference_stat='median'
            ), 
            include_groups=False
        ).reset_index()
        
        agg_water_qc = pixc_points_in_buffer_water_qc.groupby("node_id").apply(
            lambda x: aggregate_wse_with_other_vars(x, keep_qual_groups), 
            include_groups=False
        ).reset_index()
        
        # Merge the two profiles
        node_wse = pd.merge(agg_water, agg_water_qc, on=["node_id", "dist_km"], suffixes=('_raw', '_qc'))
        
        # Fill empty nodes with NaN
        node_wse = self.node_buffer_gdf.loc[:, ["node_id", "dist_km"]].merge(
            node_wse, on=["node_id", "dist_km"], how='left'
        ).reset_index(drop=True).sort_values(by="dist_km")
        
        # Filter the profile based on the profile_range
        node_wse = node_wse[
            (node_wse.dist_km >= profile_range[0]) & (node_wse.dist_km <= profile_range[1])
        ]
        
        # Add date column
        node_wse["date"] = date
        
        return node_wse
    
    def _segment_profile_by_location(self, node_wse: pd.DataFrame, seg_location: List[float]) -> List[pd.DataFrame]:
        """
        Segment the WSE profile based on location breakpoints.
        
        Args:
            node_wse: DataFrame containing the WSE profile.
            seg_location: List of segment locations (in km) to apply piecewise smoothing.
            
        Returns:
            List of segmented DataFrames.
        """
        if not seg_location:
            return [node_wse]
        
        node_wse_list = []
        seg_location_sorted = sorted(seg_location)
        
        for i in range(len(seg_location_sorted) + 1):
            if i == 0:
                # First segment from the start to the first seg_location
                segment = node_wse[node_wse.dist_km <= seg_location_sorted[i]]
            elif i == len(seg_location_sorted):
                # Last segment from the last seg_location to the end
                segment = node_wse[node_wse.dist_km > seg_location_sorted[i-1]]
            else:
                # Middle segments between two seg_locations
                segment = node_wse[
                    (node_wse.dist_km > seg_location_sorted[i-1]) & 
                    (node_wse.dist_km <= seg_location_sorted[i])
                ]
            
            if not segment.empty:
                node_wse_list.append(segment)
        
        return node_wse_list
    
    def smooth_wse_profile(
        self,
        node_wse: pd.DataFrame, 
        var_name: str = "wse_median_raw",
        frac_list: List[float] = None,
        it: int = None, 
        seg_location: List[float] = None,
    ) -> pd.DataFrame:
        """
        Smooth the WSE profile using LOWESS smoothing.
        
        Args:
            node_wse: DataFrame containing the WSE profile with columns 'dist_km' and WSE values.
            var_name: Name of the variable to smooth.
            frac_list: List of fractions for LOWESS smoothing.
            it: Number of robustifying iterations for LOWESS.
            seg_location: List of segment locations (in km) to apply piecewise smoothing.
            
        Returns:
            DataFrame with additional smoothed columns.
            
        Raises:
            ValueError: If required columns are missing or parameters are invalid.
        """
        # Set defaults
        if frac_list is None:
            frac_list = self.DEFAULT_FRAC_LIST
        if it is None:
            it = self.DEFAULT_LOWESS_ITERATIONS
        if seg_location is None:
            seg_location = []
        
        # Validate inputs
        required_cols = ['dist_km', var_name]
        missing_cols = [col for col in required_cols if col not in node_wse.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if not all(isinstance(f, (int, float)) and 0 < f <= 1 for f in frac_list):
            raise ValueError("All fractions must be numeric values between 0 and 1")
        
        # Segment the WSE profile
        node_wse_list = self._segment_profile_by_location(node_wse.copy(), seg_location)
        
        # Apply LOWESS smoothing to each segment
        lowess = sm.nonparametric.lowess
        
        for i, df in enumerate(node_wse_list):
            # Skip if segment has insufficient data
            df_clean = df.dropna(subset=[var_name])
            if len(df_clean) < 3:  # Need at least 3 points for LOWESS
                # Fill with NaN for segments with insufficient data
                for frac in frac_list:
                    df[f"wse_smooth_lowess_{frac}"] = np.nan
                node_wse_list[i] = df
                continue
            
            endog = df_clean[var_name].values
            exog = df_clean.dist_km.values
            xval = df.dist_km.values
            
            for frac in frac_list:
                try:
                    smoothed = lowess(
                        endog=endog, 
                        exog=exog, 
                        frac=frac, 
                        it=it,
                        xvals=xval,
                        return_sorted=False
                    )
                    df[f"wse_smooth_lowess_{frac}"] = smoothed
                except Exception as e:
                    # If LOWESS fails, fill with NaN
                    df[f"wse_smooth_lowess_{frac}"] = np.nan
                    print(f"Warning: LOWESS smoothing failed for segment {i}, frac={frac}: {e}")
            
            node_wse_list[i] = df
        
        # Concatenate all smoothed segments
        result = pd.concat(node_wse_list, ignore_index=True).sort_values(by="dist_km")
        
        return result
    
    def build_wse_profile(self,
                          date: str,
                          profile_range: Tuple[float, float] = (0, np.inf),
                          agg_func: List[str] = None,
                          keep_qual_groups: List[str] = None,
                          frac_list: List[float] = None,
                          it: int = None,
                          seg_location: List[float] = None,
                          ) -> pd.DataFrame:
        """
        Build the WSE profile for a given date with smoothing.
        
        Args:
            date: Date string in format YYYYMMDD.
            profile_range: Tuple of (min_km, max_km) to filter the profile.
            agg_func: List of aggregation functions to apply.
            keep_qual_groups: List of quality groups to preserve.
            frac_list: List of fractions for LOWESS smoothing.
            it: Number of robustifying iterations for LOWESS.
            seg_location: List of segment locations (in km) for piecewise smoothing.
            
        Returns:
            DataFrame with WSE profile including smoothed values.
        """
        # Set defaults
        if frac_list is None:
            frac_list = self.DEFAULT_FRAC_LIST
        if it is None:
            it = self.DEFAULT_LOWESS_ITERATIONS
        if seg_location is None:
            seg_location = []
        
        # Calculate the WSE profile (fix the method name bug)
        node_wse = self.cal_agg_wse_per_node(
            date=date,
            profile_range=profile_range,
            agg_func=agg_func,
            keep_qual_groups=keep_qual_groups,
        )
        
        # Smooth the WSE profile by both wse_median_raw and wse_median_qc
        node_wse = self.smooth_wse_profile(
            node_wse=node_wse,
            var_name="wse_median_raw",
            frac_list=frac_list,
            it=it,
            seg_location=seg_location,
        )
        node_wse = self.smooth_wse_profile(
            node_wse=node_wse,
            var_name="wse_median_qc",
            frac_list=frac_list,
            it=it,
            seg_location=seg_location,
        )


        return node_wse
    
    def build_wse_profiles_over_time(self,
                                    profile_range: Tuple[float, float] = (0, np.inf),
                                    agg_func: List[str] = None,
                                    keep_qual_groups: List[str] = None,
                                    frac_list: List[float] = None,
                                    it: int = None,
                                    seg_location: List[float] = None,
                                    save_output: bool = True,
                                    ) -> pd.DataFrame:
        """
        Build WSE profiles over all available dates.
        
        Args:
            profile_range: Tuple of (min_km, max_km) to filter the profile.
            agg_func: List of aggregation functions to apply.
            keep_qual_groups: List of quality groups to preserve.
            frac_list: List of fractions for LOWESS smoothing.
            it: Number of robustifying iterations for LOWESS.
            seg_location: List of segment locations (in km) for piecewise smoothing.
            save_output: Whether to save the output to CSV file.
            
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
        
        print(f"Building WSE profiles for {len(self.aval_dates)} dates: {self.aval_dates}")
        
        # Loop over all available dates to build WSE profiles
        profile_list = []
        failed_dates = []
        
        for i, date in enumerate(self.aval_dates):
            try:
                print(f"Processing date {i+1}/{len(self.aval_dates)}: {date}")
                node_wse = self.build_wse_profile(
                    date=date,
                    profile_range=profile_range,
                    agg_func=agg_func,
                    keep_qual_groups=keep_qual_groups,
                    frac_list=frac_list,
                    it=it,
                    seg_location=seg_location,
                )
                profile_list.append(node_wse)
                
            except Exception as e:
                print(f"Warning: Failed to process date {date}: {e}")
                failed_dates.append(date)
                continue
        
        if not profile_list:
            raise ValueError("Failed to process any dates successfully")
        
        if failed_dates:
            print(f"Warning: Failed to process {len(failed_dates)} dates: {failed_dates}")
        
        # Concatenate all profiles into one dataframe
        all_profiles = pd.concat(profile_list, ignore_index=True).sort_values(by=["date", "dist_km"])
        
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
    
    def plot_wse_profile(self, 
                         frac: float = 0.1,
                         figsize: Tuple[int, int] = (12, 6),
                         save_path: Optional[str] = None
                         ) -> None:
        """
        Plot the WSE profiles using matplotlib.
        
        Args:
            frac: Fraction value for LOWESS smoothing to plot.
            figsize: Figure size as (width, height).
            save_path: Optional path to save the figure.
            
        Raises:
            FileNotFoundError: If output file doesn't exist.
            KeyError: If expected columns are missing.
        """
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(f"Output file not found: {self.output_path}. Run build_wse_profiles_over_time first.")
        
        wse_profiles = pd.read_csv(self.output_path)
        
        # Validate required columns
        required_cols = ['date', 'dist_km', 'wse_median_raw', 'wse_qc']
        missing_cols = [col for col in required_cols if col not in wse_profiles.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")
        
        # Check if smoothed columns exist
        smooth_col_raw = f'wse_smooth_lowess_{frac}_raw'
        smooth_col_qc = f'wse_smooth_lowess_{frac}_qc'
        if smooth_col_raw not in wse_profiles.columns:
            raise KeyError(f"Smoothed column not found: {smooth_col_raw}. Available fractions might be different.")
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
        
        # Color list with discrete colormap
        unique_dates = wse_profiles['date'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_dates)))
        
        # Left: raw WSE
        for i, date in enumerate(unique_dates):
            df_date = wse_profiles[wse_profiles['date'] == date]
            ax1.scatter(df_date['dist_km'], df_date['wse_median_raw'], 
                       label=date, s=10, alpha=0.2, color=colors[i])
            if smooth_col_raw in df_date.columns:
                ax1.plot(df_date['dist_km'], df_date[smooth_col_raw], 
                        linewidth=2, color=colors[i])
        
        ax1.set_xlabel('Distance along river (km)')
        ax1.set_ylabel('Water Surface Elevation (m)')
        ax1.set_title(f'Raw WSE Profiles for {self.riv_name}')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Right: QC WSE
        for i, date in enumerate(unique_dates):
            df_date = wse_profiles[wse_profiles['date'] == date]
            ax2.scatter(df_date['dist_km'], df_date['wse_median_qc'], 
                       label=date, s=10, alpha=0.2, color=colors[i])
            if smooth_col_qc in df_date.columns:
                ax2.plot(df_date['dist_km'], df_date[smooth_col_qc], 
                        linewidth=2, color=colors[i])
        
        ax2.set_xlabel('Distance along river (km)')
        ax2.set_title(f'QC filtered WSE Profiles for {self.riv_name}')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

        

