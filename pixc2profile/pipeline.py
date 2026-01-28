#!/usr/bin/env python3
"""
PIXC2Profile Pipeline Script

This script runs the complete pipeline to build WSE profiles over time:
1. Download PIXC data
2. Generate river nodes and buffers
3. Extract PIXC data within buffers
4. Build WSE profiles over time

Usage:
    # As a script:
    python pipeline.py --config config.json
    
    # As an imported function:
    import pixc2profile
    pixc2profile.pipeline(config="config.json")
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Add the package to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from pixc2profile import download_pixc_data
    from pixc2profile.river import River
    from pixc2profile.pixc import PIXC
    from pixc2profile.profile import Profile
else:
    # Use relative imports when imported as a module
    from .download import download_pixc_data
    from .river import River
    from .pixc import PIXC
    from .profile import Profile


def setup_logging(log_level="INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_config(config_dict):
    """Process configuration dictionary."""
    config = config_dict.copy()
    
    # Handle inf and -inf strings in profile_range
    if 'profile_range' in config:
        profile_range = config['profile_range']
        if isinstance(profile_range, list) and len(profile_range) == 2:
            config['profile_range'] = [
                float('-inf') if str(profile_range[0]).lower() == 'inf' else float(profile_range[0]),
                float('inf') if str(profile_range[1]).lower() == 'inf' else float(profile_range[1])
            ]
    
    return config


def step1_download_pixc_data(config, logger):
    """Step 1: Download PIXC data."""
    logger.info("=== STEP 1: Downloading PIXC data ===")
    
    download_pixc_data(
        home_dir=config['home_dir'],
        riv_name=config['river_name'],
        start_date=config['start_date'],
        end_date=config['end_date'],
        pass_tile_list=config['pass_tile_list'],
    )
    
    logger.info("Step 1 completed: PIXC data downloaded")


def step2_generate_river_nodes(config, logger):
    """Step 2: Create River object and generate nodes and buffers."""
    logger.info("=== STEP 2: Generating river nodes and buffers ===")
    
    river_shp_path = os.path.join(config['home_dir'], "shps", f"{config['river_name']}.shp")
    
    # Create River object
    river = River(
        home_dir=config['home_dir'],
        riv_name=config['river_name'],
        riv_shp_path=river_shp_path,
        node_spacing=config['node_spacing'],
        riv_width=config['channel_width'],
    )
    
    # Process river (generate nodes, buffers, and export shapefiles)
    river.process_river()
    
    logger.info("Step 2 completed: River nodes and buffers generated")
    return river


def step3_extract_pixc_data(config, river, logger):
    """Step 3: Create PIXC object and extract PIXC data within buffers."""
    logger.info("=== STEP 3: Extracting PIXC data within buffers ===")
    
    # Create PIXC object
    pixc = PIXC(
        home_dir=config['home_dir'],
        riv_name=config['river_name'],
        pixc_dirname=config['pixc_version'],
        var_list=None,
        create_ref_table_on_init=True
    )
    
    # Extract water pixels within buffers
    pixc_water_paths = pixc.process_water_pixels(
        node_buffer_path=river.buffer_export_path,
        pixc_water_dir_name=config['pixc_water_dir_name'],
        n_parts=config['n_partitions'],
        classification_categories=config['classification_categories'],
        prior_water_prob_threshold=config['prior_water_prob_threshold'],
        water_frac_threshold=config['water_frac_threshold']
    )
    
    logger.info(f"Extracted water pixels: {pixc_water_paths[0] if pixc_water_paths else 'No files'}")
    
    # Filter with quality flags
    pixc_water_qc_filtered_paths = pixc.filter_with_quality_flags(
        pixc_water_paths=pixc.pixc_water_paths,
        quality_flag_dict=config['quality_flag_dict'],
        pixc_qc_dirname=config['pixc_qc_dirname']
    )
    
    logger.info(f"Quality filtered files: {pixc_water_qc_filtered_paths[0] if pixc_water_qc_filtered_paths else 'No files'}")
    logger.info("Step 3 completed: PIXC data extracted and filtered")
    return pixc


def step4_build_profiles(config, river, logger):
    """Step 4: Build WSE profiles over time."""
    logger.info("=== STEP 4: Building WSE profiles over time ===")
    
    output_path = os.path.join(config['home_dir'], config['river_name'], "profiles.csv")
    
    # Create Profile object
    profile = Profile(
        riv_name=config['river_name'],
        home_dir=config['home_dir'],
        node_path=river.node_export_path,
        buffer_path=river.buffer_export_path,
        pixc_dir=os.path.join(config['home_dir'], config['river_name'], config['pixc_version']),
        pixc_water_dir=os.path.join(config['home_dir'], config['river_name'], config['pixc_water_dir_name']),
        pixc_water_qc_filtered_dir=os.path.join(config['home_dir'], config['river_name'], config['pixc_qc_dirname']),
        output_path=output_path
    )
    
    # Set buffer_shp_path (missing from Profile.__init__)
    profile.buffer_shp_path = river.buffer_export_path
    
    # Build WSE profiles for all dates
    output_file = profile.build_wse_profiles_over_time(
        profile_range=config['profile_range'],
        agg_func=config['agg_func'],
        keep_qual_groups=config['keep_qual_groups'],
        frac_list=config['frac_list'],
        it=config['it'],
        seg_location=config['seg_location']
    )
    
    logger.info(f"Step 4 completed: WSE profiles saved to {output_file}")
    return profile, output_file

def run_pipeline(config, skip_steps=None):
    """Run the complete PIXC2Profile pipeline."""
    logger = setup_logging(config.get('log_level', 'INFO'))
    skip_steps = skip_steps or []
    
    logger.info(f"Starting PIXC2Profile pipeline for river: {config['river_name']}")
    logger.info(f"Home directory: {config['home_dir']}")
    logger.info(f"Date range: {config['start_date']} to {config['end_date']}")
    
    try:
        # Step 1: Download PIXC data
        if 1 not in skip_steps:
            step1_download_pixc_data(config, logger)
        else:
            logger.info("Skipping Step 1: Download PIXC data")
        
        # Step 2: Generate river nodes and buffers
        if 2 not in skip_steps:
            river = step2_generate_river_nodes(config, logger)
        else:
            logger.info("Skipping Step 2: Generate river nodes and buffers")
            # Still need to create river object for later steps
            river_shp_path = os.path.join(config['home_dir'], "shps", f"{config['river_name']}.shp")
            river = River(
                home_dir=config['home_dir'],
                riv_name=config['river_name'],
                riv_shp_path=river_shp_path,
                node_spacing=config['node_spacing'],
                riv_width=config['channel_width'],
            )
        
        # Step 3: Extract PIXC data
        if 3 not in skip_steps:
            pixc = step3_extract_pixc_data(config, river, logger)
        else:
            logger.info("Skipping Step 3: Extract PIXC data")
        
        # Step 4: Build WSE profiles
        if 4 not in skip_steps:
            profile, output_file = step4_build_profiles(config, river, logger)
            logger.info(f"Pipeline completed successfully!")
            logger.info(f"Final output: {output_file}")
            return output_file
        else:
            logger.info("Skipping Step 4: Build WSE profiles")
    
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise

def pipeline(config=None, skip_steps=None):
    """Main pipeline function for importable use.
    
    Args:
        config (dict): Config dictionary
        skip_steps (list): List of steps to skip (1-4)
        
    Returns:
        str: Path to output profiles CSV file
    """
    # Handle config parameter
    if config is None:
        raise ValueError("Config dictionary must be provided")
    
    processed_config = load_config(config)
    return run_pipeline(processed_config, skip_steps=skip_steps)


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='PIXC2Profile Pipeline')
    parser.add_argument('--config', '-c', default='config.json',
                        help='Path to configuration JSON file')
    parser.add_argument('--create-config', action='store_true',
                        help='Create a default configuration file')
    parser.add_argument('--skip-steps', nargs='+', type=int, choices=[1, 2, 3, 4],
                        help='Skip specified steps (1-4)')
    
    # Command line overrides
    parser.add_argument('--home-dir', help='Base directory for data')
    parser.add_argument('--river-name', help='Name of the river')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Load configuration from JSON file (for command line usage only)
    if os.path.exists(args.config):
        with open(args.config, 'r') as file:
            config = json.load(file)
    else:
        return
    
    # Apply command line overrides
    if args.home_dir:
        config['home_dir'] = args.home_dir
    if args.river_name:
        config['river_name'] = args.river_name
    if args.start_date:
        config['start_date'] = args.start_date
    if args.end_date:
        config['end_date'] = args.end_date
    
    # Process config and run pipeline
    processed_config = load_config(config)
    run_pipeline(processed_config, skip_steps=args.skip_steps)


if __name__ == "__main__":
    main()