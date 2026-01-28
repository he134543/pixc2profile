"""
pixc2profile: A Python package for extracting along-river water surface elevation (WSE) profiles directly from SWOT Pixel Cloud (PixC) data.

This package provides tools to:
- Download SWOT Pixel Cloud data
- Process river geometries and generate nodes  
- Extract water surface elevation profiles
- Apply quality filtering and smoothing
"""

__version__ = "0.1.1"
__author__ = "Xinchen He"
__email__ = "xinchen134543@gmail.com"

# Import main classes and functions
from .river import River
from .pixc import PIXC
from .profile import Profile, get_median_wse_with_other_vars
from .download import download_pixc_data
from .pipeline import pipeline

# Define what gets imported with "from pixc2profile import *"
__all__ = [
    "River",
    "PIXC", 
    "Profile",
    "download_pixc_data",
    "filter_with_quality_flags",
    "get_median_wse_with_other_vars",
    "pipeline"
]