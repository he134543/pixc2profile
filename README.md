# pixc2profile
A Python package for extracting along-river water surface elevation (WSE) profiles directly from SWOT Pixel Cloud (PixC) data.

Unlike the official SWOT River SP product, which provides WSE profiles on a predefined river network (SWORD) with a fixed 200m node spacing, pixc2profile allows users to generate WSE profiles on any user-defined river centerline with customizable node spacing (e.g., 50 m).

## Key Features:
- Download and read raw SWOT Pixel Cloud (PixC) data
- Extract along-river WSE profiles using user-supplied river shapefiles
- Flexible node spacing along the river centerline
- User-defined filtering using SWOT PixC quality flags
- Automatic removal of gappy or poorly sampled profiles
- Optional smoothing of retained profiles for analysis-ready outputs

## Why pixc2profile?

SWOT RiverSP products are tied to a global, fixed river network and uniform node spacing, which can limit analyses of fine-scale hydraulics, reservoir backwaters, or custom study reaches.
pixc2profile provides a lightweight and flexible alternative by working directly with Pixel Cloud observations and user-defined geometries.
