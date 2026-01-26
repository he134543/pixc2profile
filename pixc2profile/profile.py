import glob as glob
import pandas as pd 
import numpy as np
import geopandas as gpd
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt

# a helper function to get the median wse and other variables in a group
def get_median_wse_with_other_vars(df_group, keep_qual_groups):
    # if no wse values, return NaN for wse and other variables
    if df_group['wse'].isnull().all():
        # return NaN for wse and other variables
        output = pd.Series({
            'wse': np.nan,
            'dist_km': df_group['dist_km'].iloc[0],
        })
        for qual in keep_qual_groups:
            output[qual] = np.nan

        return output

    # get the median WSE and other variables
    median_wse = df_group['wse'].median()
    # find the rows where wse is closest to the median
    df_group["wse_diff"] = (df_group['wse'] - median_wse).abs()
    median_rows = df_group.loc[df_group['wse_diff'] == df_group['wse_diff'].min()]

    if median_rows.empty:
        # return NaN for wse and other variables
        output = pd.Series({
            'wse': np.nan,
            'dist_km': df_group['dist_km'].iloc[0],
        })
        for qual in keep_qual_groups:
            output[qual] = np.nan

        return output

    # select the first occurrence 
    row = median_rows.iloc[0]
    # reutrn the row with median WSE and other variables
    output = pd.Series({
        'wse': median_wse,
        'dist_km': row['dist_km'],
    })
    for qual in keep_qual_groups:
        output[qual] = int(row[qual])
    return output

class Profile:
    def __init__(self,
                 riv_name: str,
                 home_dir: str,
                 node_path: str,
                 buffer_path: str,
                 pixc_dir: str,
                 pixc_water_dir: str,
                 pixc_water_qc_filtered_dir: str,
                 output_path: str,
                 ) -> None:
        """
        Initialize the profile class for a specific river.
        Arguments:
            riv_name (str): Name of the river.
            home_dir (str): Base directory for the river data.
            node_shp_path (str): Path to the shapefile containing river nodes.
            pixc_dir (str): Directory containing raw PIXC netCDF files.
            pixc_water_dir (str): Directory containing PIXC water CSV files.
            pixc_water_qc_filtered_dir (str): Directory containing quality-controlled PIXC water CSV files.
            output_path (str): File path save the final output (csv).
        """
        self.riv_name = riv_name
        self.home_dir = home_dir
        self.node_path = node_path
        self.buffer_path = buffer_path

        # directories for PIXC data
        self.pixc_dir = pixc_dir
        self.pixc_water_dir = pixc_water_dir
        self.pixc_water_qc_filtered_dir = pixc_water_qc_filtered_dir

        # Initialize storage
        self.aval_dates = None
        self.node_gdf = None
        self.node_buffer_gdf = None

        # Output profiles
        self.output_path = output_path

    def load_available_dates(self,) -> None:
        """
        Load available dates from the PIXC water and quality-controlled directories.
        """
        pixc_water_paths = glob.glob(os.path.join(self.pixc_water_dir, "*.csv"))
        pixc_water_qc_filtered_paths = glob.glob(os.path.join(self.pixc_water_qc_filtered_dir, "*.csv"))

        # extract dates from file names

        aval_dates_water = [os.path.basename(pixc_water_path).split("_")[0] for pixc_water_path in pixc_water_paths]
        aval_dates_qc_filtered = [os.path.basename(pixc_water_qc_filtered_path).split("_")[0] for pixc_water_qc_filtered_path in pixc_water_qc_filtered_paths]

        # check if the two lists are the same
        if set(aval_dates_water) != set(aval_dates_qc_filtered):
            raise ValueError("Available dates in PIXC water and quality-controlled directories do not match.")
        self.aval_dates = sorted(list(set(aval_dates_water)))

    def load_node_buffer_gdf(self,) -> tuple:
        """
        Load the river nodes and buffers as GeoDataFrames.
        Returns:
            nodes_gdf (gpd.GeoDataFrame): GeoDataFrame of river nodes.
            buffers_gdf (gpd.GeoDataFrame): GeoDataFrame of river buffers.
        """
        nodes_gdf = gpd.read_file(self.node_path)
        nodes_gdf = nodes_gdf.to_crs(epsg=4326)  # convert to epsg 4326
        node_buffer_gdf = gpd.read_file(self.buffer_path)
        node_buffer_gdf = node_buffer_gdf.to_crs(epsg=4326) # convert to epsg 4326
        self.node_gdf = nodes_gdf
        self.node_buffer_gdf = node_buffer_gdf
        return nodes_gdf, node_buffer_gdf
    
    def cal_median_wse(self,
                          date: str,
                          profile_range: tuple = (0, np.inf),
                          agg_func: str = "median",
                          keep_qual_groups: list = ["interferogram_qual", "classification_qual", "geolocation_qual", "sig0_qual"],
                          ) -> pd.DataFrame:
        """
        Build a pair of wse profiles for a specific date: one from raw PIXC water points and one from quality-controlled PIXC water points.
        """
        # check if date is available
        if self.aval_dates is None:
            self.load_available_dates()
        if date not in self.aval_dates:
            raise ValueError(f"Date {date} not available in PIXC data.")
        # check if node_gdf is loaded
        if self.node_gdf is None or self.node_buffer_gdf is None:
            self.load_node_buffer_gdf()

        # read the csv files containing water pixel points and convert to gpd dataframe
        pixc_water_path = os.path.join(self.pixc_water_dir, f"{date}_water.csv")
        pixc_water_qc_filtered_path = os.path.join(self.pixc_water_qc_filtered_dir, f"{date}_water_qc_filtered.csv")
        df_water = pd.read_csv(pixc_water_path)
        gdf_water = gpd.GeoDataFrame(df_water, geometry=gpd.points_from_xy(df_water.longitude, df_water.latitude), crs="EPSG:4326")
        df_water_qc = pd.read_csv(pixc_water_qc_filtered_path)
        gdf_water_qc = gpd.GeoDataFrame(df_water_qc, geometry=gpd.points_from_xy(df_water_qc.longitude, df_water_qc.latitude), crs="EPSG:4326")

        # find intersected pixc points with reach node buffer
        pixc_points_in_buffer_water = gpd.sjoin(self.node_buffer_gdf, gdf_water, predicate='intersects', how = "inner")
        pixc_points_in_buffer_water_qc = gpd.sjoin(self.node_buffer_gdf, gdf_water_qc, predicate='intersects', how = "inner")

        # aggregate wse by node
        if agg_func == "median":
            agg_water = pixc_points_in_buffer_water.groupby("node_id").apply(lambda x: get_median_wse_with_other_vars(x, keep_qual_groups), include_groups=False).reset_index()
            agg_water_qc = pixc_points_in_buffer_water_qc.groupby("node_id").apply(lambda x: get_median_wse_with_other_vars(x, keep_qual_groups), include_groups=False).reset_index()
        else:
            keep_qual_groups = []
            raise ValueError(f"Aggregation function {agg_func} not supported. Currently only 'median' is supported.")
        
        # merge the two profiles
        node_wse = pd.merge(agg_water, agg_water_qc, on=["node_id", "dist_km"], suffixes=('_raw', '_qc'))

        # fill the empty nodes with NaN
        node_wse = self.node_buffer_gdf.loc[:, ["node_id", "dist_km"]].merge(node_wse, 
                                                                                on = ["node_id", "dist_km"], 
                                                                                how='left').reset_index(drop = True).sort_values(by="dist_km")
        # filter the profile based on the profile_range
        node_wse = node_wse[(node_wse.dist_km >= profile_range[0]) & (node_wse.dist_km <= profile_range[1])]

        # create a new column for the date
        node_wse["date"] = date

        return node_wse
    
    def smooth_wse_profile(
                        self,
                        node_wse, 
                        var_name = "wse_raw",
                        frac_list = [0.01, 0.05, 0.1, 0.2],
                        it = 3, 
                        seg_location = [],
                        ):
        """
        Smooth the WSE profile using LOWESS smoothing.
        Arguments:
            node_wse (pd.DataFrame): DataFrame containing the WSE profile with columns 'dist_km' and 'wse'.
            frac_list (list, optional): List of fractions for LOWESS smoothing. Defaults to [0.01, 0.05, 0.1, 0.2].
            it (int, optional): Number of robustifying iterations for LOWESS. Defaults to 3.
            seg_location (list, optional): List of segment locations (in km) to apply piecewise smoothing. 
                                           If empty, applies smoothing to the entire profile. Defaults to [].
        """
        # if seg_location is not None, we need to segment the WSE profile
        if len(seg_location) == 0:
            node_wse_list = [node_wse]
        else:
            # segment the WSE profile based on the seg_location
            node_wse_list = []
            for i in range(len(seg_location) + 1):
                if i == 0:
                    # first segment from the start to the first seg_location
                    segment = node_wse[node_wse.dist_km <= seg_location[i]]
                elif i == len(seg_location):
                    # last segment from the last seg_location to the end
                    segment = node_wse[node_wse.dist_km > seg_location[i-1]]
                else:
                    # middle segments between two seg_locations
                    segment = node_wse[(node_wse.dist_km > seg_location[i-1]) & (node_wse.dist_km <= seg_location[i])]
                node_wse_list.append(segment)

        # loop each segemented node_Wse
        for i in range(len(node_wse_list)):
            # load this segement
            df = node_wse_list[i]
            # using statsmodels to apply LOWESS to smooth the WSE
            lowess = sm.nonparametric.lowess
            for frac in frac_list:
                # apply LOWESS smoothing for each fraction in the list
                # using the dist_km as exogenous variable and wse as endogenous variable
                # it is important to use the dist_km as exogenous variable to avoid overfitting
                endog = df.dropna(subset=[var_name])[var_name].values.ravel()
                exog = df.dropna(subset=[var_name]).dist_km.values.ravel()
                xval = df.dist_km.values.ravel()

                df.loc[:, [f"wse_smooth_lowess_{frac}"]] = lowess(exog = exog, 
                                                    endog = endog, 
                                                    frac=frac, 
                                                    it=it,
                                                    xvals = xval
                                                    )

            # update the node_wse_list with the smoothed df
            node_wse_list[i] = df
        # concatenate all the smoothed segments into one dataframe
        node_wse = pd.concat(node_wse_list, ignore_index=True).sort_values(by="dist_km")

        return node_wse
    
    def build_wse_profile(self,
                          date: str,
                          profile_range: tuple = (0, np.inf),
                          agg_func: str = "median",
                          keep_qual_groups: list = ["interferogram_qual", "classification_qual", "geolocation_qual", "sig0_qual"],
                          frac_list = [0.01, 0.05, 0.1, 0.2],
                          it = 3,
                          seg_location = [],
                          ):
        """
        Build the WSE profile for a given date with smoothing.
        |date | dist_km | wse_raw | wse_qc | wse_smooth_lowess_0.01_raw | wse_smooth_lowess_0.01_qc | ...
        """
        # calculate the median WSE profile
        node_wse = self.cal_median_wse(date = date,
                                       profile_range = profile_range,
                                       agg_func = agg_func,
                                       keep_qual_groups = keep_qual_groups,
                                       )
        # smooth the WSE profile
        node_wse = self.smooth_wse_profile(node_wse = node_wse,
                                          frac_list = frac_list,
                                          it = it,
                                          seg_location = seg_location,
                                          )
        return node_wse
    
    def build_wse_profiles_over_time(self,
                                    profile_range: tuple = (0, np.inf),
                                    agg_func: str = "median",
                                    keep_qual_groups: list = ["interferogram_qual", "classification_qual", "geolocation_qual", "sig0_qual"],
                                    frac_list = [0.01, 0.05, 0.1, 0.2],
                                    it = 3,
                                    seg_location = [],
                                    ) -> pd.DataFrame:
        """
        Build WSE profiles over all available dates.
        Returns:
            pd.DataFrame: DataFrame containing WSE profiles over time.
        """
        # check if available dates are loaded
        if self.aval_dates is None:
            self.load_available_dates()
        
        # loop over all available dates to build WSE profiles
        profile_list = []
        for date in self.aval_dates:
            node_wse = self.build_wse_profile(date = date,
                                             profile_range = profile_range,
                                             agg_func = agg_func,
                                             keep_qual_groups = keep_qual_groups,
                                             frac_list = frac_list,
                                             it = it,
                                             seg_location = seg_location,
                                             )
            profile_list.append(node_wse)
        
        # concatenate all profiles into one dataframe
        all_profiles = pd.concat(profile_list, ignore_index=True).sort_values(by=["date", "dist_km"])

        # export the all_profiles to csv
        all_profiles.to_csv(self.output_path)

        return all_profiles
    
    def plot_wse_profile(self, 
                         frac: float = 0.1,
                         figsize: tuple = (12, 6)
                         ):
        """
        Plot the WSE profiles using seaborn.
        """
        wse_profiles = pd.read_csv(self.output_path)
        
        # plot use matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

        # color list with discrete colormap
        color_list = plt.cm.get_cmap('tab20', len(wse_profiles['date'].unique()))

        # left: raw wse
        for i, date in enumerate(wse_profiles['date'].unique()):
            df_date = wse_profiles[wse_profiles['date'] == date]
            ax1.scatter(df_date['dist_km'], df_date['wse_raw'], label=date, s=10, alpha=0.2, color=color_list[i])
            ax1.plot(df_date['dist_km'], df_date[f'wse_smooth_lowess_{frac}_raw'], label=f'Smoothed {date}', linewidth=2, color=color_list[i])
        ax1.set_xlabel('Distance along river (km)')
        ax1.set_ylabel('Water Surface Elevation (m)')
        ax1.set_title(f'Raw WSE Profiles for {self.riv_name}')
        ax1.legend()

        # right: qc wse
        for i, date in enumerate(wse_profiles['date'].unique()):
            df_date = wse_profiles[wse_profiles['date'] == date]
            ax2.scatter(df_date['dist_km'], df_date['wse_qc'], label=date, s=10, alpha=0.2, color=color_list[i])
            ax2.plot(df_date['dist_km'], df_date[f'wse_smooth_lowess_{frac}_qc'], label=f'Smoothed {date}', linewidth=2, color=color_list[i])

        ax2.set_xlabel('Distance along river (km)')
        ax2.set_title(f'QC filtered WSE Profiles for {self.riv_name}')
        ax2.legend()
        plt.show()
        return

        

