# Collection of helper functions for pixc2profile

import pandas as pd
import numpy as np

def aggregate_wse_with_other_vars(df_group, 
                                 keep_qual_groups,
                                 include_stats=['median', 'std', 'count', "q25", "q75"],
                                 reference_stat='median',
                                 wse_col='wse',
                                 dist_col='dist_km'
                                 ):
    """
    Comprehensive aggregate function for WSE per node with multiple statistics.
    
    Parameters:
    -----------
    df_group : pandas.DataFrame
        Grouped dataframe containing WSE and other variables
    keep_qual_groups : list
        List of quality variables to retain
    include_stats : list, default ['median', 'std', 'count']
        List of statistics to compute. Options: 'median', 'std', 'count', 'min', 'max'
    reference_stat : str, default 'median'
        Which statistic to use as reference for selecting other variables.
        Options: 'median', 'mean', or any quantile value (e.g., 0.5)
    
    Returns:
    --------
    pandas.Series
        Series containing all computed statistics and other variables
    """
    # Initialize output dictionary
    output = {}
    
    # Always include dist_km
    output['dist_km'] = df_group[dist_col].iloc[0] if not df_group[dist_col].empty else np.nan
    
    # if no wse values, return NaN for all statistics
    if df_group[wse_col].isnull().all():
        
        # Add NaN for all requested statistics
        for stat in include_stats:
            output[f'{wse_col}_{stat}'] = np.nan
        
        # Add NaN for quality variables
        for qual in keep_qual_groups:
            output[qual] = np.nan
        
        return pd.Series(output)
    
    # Compute additional statistics
    if 'mean' in include_stats:
        output[f'{wse_col}_mean'] = df_group[wse_col].mean()
    if 'std' in include_stats:
        output[f'{wse_col}_std'] = df_group[wse_col].std()
    if 'count' in include_stats:
        output[f'{wse_col}_count'] = df_group[wse_col].count()
    if 'min' in include_stats:
        output[f'{wse_col}_min'] = df_group[wse_col].min()
    if 'max' in include_stats:
        output[f'{wse_col}_max'] = df_group[wse_col].max()
    if 'median' in include_stats:
        output[f'{wse_col}_median'] = df_group[wse_col].median()
    if "q" in include_stats:
        # quantiles specified as "q25", "q75", etc.
        for stat in include_stats:
            if stat.startswith("q"):
                try:
                    # Extract quantile value
                    q_value = int(stat[1:]) / 100.0
                    output[f'{wse_col}_{stat}'] = df_group[wse_col].quantile(q_value)
                except ValueError:
                    continue
    
    # Determine reference value for selecting other variables
    if reference_stat == 'median':
        reference_value = df_group[wse_col].median()
    elif reference_stat == 'mean':
        reference_value = df_group[wse_col].mean()
    elif isinstance(reference_stat, (int, float)) and 0 <= reference_stat <= 1:
        reference_value = df_group[wse_col].quantile(reference_stat)
    else:
        # Default to median if invalid reference_stat
        print(f"Invalid reference_stat {reference_stat}. Defaulting to median.")
        reference_value = df_group[wse_col].median()
    
    # Find the rows where wse is closest to the reference value
    df_group_copy = df_group.copy()
    df_group_copy["wse_diff"] = (df_group_copy[wse_col] - reference_value).abs()
    reference_rows = df_group_copy.loc[df_group_copy['wse_diff'] == df_group_copy['wse_diff'].min()]
    
    if not reference_rows.empty:
        # Select the first occurrence and get quality variables
        row = reference_rows.iloc[0]
        for qual in keep_qual_groups:
            output[qual] = int(row[qual]) if pd.notna(row[qual]) else np.nan
    else:
        # If no reference rows found, set quality variables to NaN
        for qual in keep_qual_groups:
            output[qual] = np.nan
    
    return pd.Series(output)