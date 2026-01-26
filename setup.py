from setuptools import setup, find_packages

setup(
    name="pixc2profile",
    version="0.1.0",
    author="Xinchen He",
    author_email="xinchen134543@gmail.com",
    description="A Python package for extracting along-river water surface elevation (WSE) profiles directly from SWOT Pixel Cloud (PixC) data",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "earthaccess",
        "xarray",
        "geopandas",
        "tqdm",
        "matplotlib",
        "dask-geopandas",
        "statsmodels",
        "shapely",
        "contextily",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)