import numpy as np
import pandas as pd
import os
import geopandas as gpd
from shapely import LineString, Point, box
from shapely.affinity import rotate, translate
from shapely import line_merge
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt
import contextily as cx

class River:
    """
    A class for processing river geometry and generating nodes with buffer zones.
    
    This class provides methods to load river shapefiles, generate nodes along reaches,
    create buffer zones, and export the results.
    """
    
    def __init__(self, 
                 home_dir: str,
                 riv_name: str,
                 riv_shp_path: str,
                 node_spacing: float,
                 riv_width: float,):
        """
        Initialize River processor.
        
        Arguments:
            home_dir (str): Root directory to save all data.
            riv_name (str): Name of the river.
            riv_shp_path (str): Path to the river shapefile. Note: The river gdf must follow a topology order (from upstream to downstream).
            node_spacing (float): Spacing between nodes along the reach (in meters).
            riv_width (float): Estimated river width (in meters).
        """
        self.home_dir = home_dir
        self.riv_name = riv_name
        self.riv_shp_path = riv_shp_path
        self.node_spacing = node_spacing
        self.riv_width = riv_width
        
        # Initialize data storage
        self.river_gdf = None
        self.river_gdf_utm = None
        self.utm_crs = None
        self.nodes_gdf = None
        self.nodes_buffer_gdf = None
        self.reaches_linestring = None
        
        # Paths for exported files
        self.node_export_path = None
        self.buffer_export_path = None
    
    @property
    def nodes_dir(self) -> str:
        """Get the full path to the nodes directory."""
        return os.path.join(self.home_dir, self.riv_name, "nodes")
    
    @property
    def total_river_length(self) -> float:
        """Get total length of all river reaches in meters."""
        if self.river_gdf_utm is not None:
            return self.river_gdf_utm.length.sum()
        return 0.0
    
    @property
    def n_reaches(self) -> int:
        """Get number of river reaches."""
        if self.river_gdf is not None:
            return len(self.river_gdf)
        return 0
    
    @property
    def n_nodes(self) -> int:
        """Get number of generated nodes."""
        if self.nodes_gdf is not None:
            return len(self.nodes_gdf)
        return 0
    
    def load_river_shapefile(self) -> gpd.GeoDataFrame:
        """
        Load river shapefile and reproject to UTM.
        
        Returns:
            gpd.GeoDataFrame: River GeoDataFrame in UTM projection.
        """
        # Load river shapefile
        self.river_gdf = gpd.read_file(self.riv_shp_path)
        
        # Estimate the UTM CRS for accurate distance measurements
        self.utm_crs = self.river_gdf.estimate_utm_crs()
        
        # Reproject to UTM
        self.river_gdf_utm = self.river_gdf.to_crs(self.utm_crs)
        
        return self.river_gdf_utm
    
    def generate_points_along_line(self, line: LineString, interval: float) -> List[Point]:
        """
        Generate points every `interval` meters along a reach.
        
        Arguments:
            line (LineString): The line geometry to generate points along.
            interval (float): Interval between points in meters.
            
        Returns:
            List[Point]: List of Point geometries along the line.
        """
        length = line.length  # actual reach length
        distances = list(range(0, int(length), int(interval))) + [length]
        # use shapely.line.interpolate function to sample point
        point_list = [line.interpolate(distance) for distance in distances]
        return point_list
    
    def create_buffer(self, center: Point, angle: float, length: float, width: float) -> box:
        """
        Create an oriented rectangular box centered at `center`.
        
        Arguments:
            center (Point): Center point of the buffer.
            angle (float): Rotation angle in degrees.
            length (float): Length of the buffer box.
            width (float): Width of the buffer box.
            
        Returns:
            Polygon: Oriented rectangular buffer box.
        """
        rect = box(-length/2, -width/2, length/2, width/2)  # center at origin (0,0)
        rect_rotated = rotate(rect, angle, origin=(0, 0), use_radians=False)  # rotate the box according to the flow angle
        rect_output = translate(rect_rotated, xoff=center.x, yoff=center.y)  # center at center
        return rect_output
    
    def generate_nodes(self) -> gpd.GeoDataFrame:
        """
        Generate nodes along the river reaches.
        
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing nodes with node_id and dist_m columns.
        """
        if self.river_gdf_utm is None:
            self.load_river_shapefile()
        
        # For each reach in the river gdf, generate points along the reach
        nodes_list = []
        for i in range(len(self.river_gdf_utm)):
            # append node geometry
            node_geometry = self.generate_points_along_line(
                self.river_gdf_utm.geometry.iloc[i], 
                self.node_spacing
            )
            # Create dataframe for this reach's nodes
            reach_nodes = gpd.GeoDataFrame(geometry=node_geometry, crs=self.utm_crs)
            nodes_list.append(reach_nodes)
        
        # Combine all nodes
        if nodes_list:
            self.nodes_gdf = pd.concat(nodes_list, ignore_index=True)
        else:
            self.nodes_gdf = gpd.GeoDataFrame(geometry=[], crs=self.utm_crs)
        
        # Clean the index and set node IDs
        self.nodes_gdf = self.nodes_gdf.reset_index(drop=True)
        self.nodes_gdf["node_id"] = np.arange(len(self.nodes_gdf))
        
        # Calculate distances from inlet
        self._calculate_distances()

        # Sort by distance and add distance in km
        self.nodes_gdf = self.nodes_gdf.sort_values(["dist_m"]).reset_index(drop=True)
        
        return self.nodes_gdf
    
    def _calculate_distances(self) -> None:
        """Calculate distance from inlet for each node."""
        if len(self.nodes_gdf) == 0:
            return
        
        # Set inlet as the first node of the first reach
        inlet_ind = self.nodes_gdf.index[0]
        inlet_node = self.nodes_gdf.loc[inlet_ind, "geometry"]
        
        # Merge reaches into one linestring
        reaches_coords = []
        for rch_geom in self.river_gdf_utm.geometry:
            reaches_coords.extend(list(rch_geom.coords))
        self.reaches_linestring = LineString(reaches_coords)
        
        # Calculate distance to inlet for each node
        for ind in self.nodes_gdf.index:
            node_geom = self.nodes_gdf.loc[ind, "geometry"]
            inlet_proj = self.reaches_linestring.project(inlet_node)
            node_proj = self.reaches_linestring.project(node_geom)
            
            # For node upstream, distance is set as negative
            if ind < inlet_ind:
                distance = -(node_proj - inlet_proj) if node_proj < inlet_proj else -(inlet_proj - node_proj)
                self.nodes_gdf.loc[ind, "dist_m"] = distance
            # For node downstream, distance set as positive
            elif ind > inlet_ind:
                distance = abs(node_proj - inlet_proj)
                self.nodes_gdf.loc[ind, "dist_m"] = distance
            else:
                self.nodes_gdf.loc[ind, "dist_m"] = 0
        self.nodes_gdf["dist_km"] = self.nodes_gdf["dist_m"] / 1000
    
    def generate_buffers(self) -> gpd.GeoDataFrame:
        """
        Generate buffer boxes for each node.
        
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing buffer boxes with node_id, dist_m, and dist_km columns.
        """
        if self.nodes_gdf is None:
            self.generate_nodes()
        
        buffer_list = []
        
        for i, node in self.nodes_gdf.iterrows():
            # Before the last node, use the next point to calculate the angle
            if i < len(self.nodes_gdf) - 1:
                ref_node = self.nodes_gdf.iloc[i + 1]
            else:
                # The angle of the last node would be calculated using the previous node
                ref_node = self.nodes_gdf.iloc[i - 1]
            
            dx = ref_node.geometry.x - node.geometry.x
            dy = ref_node.geometry.y - node.geometry.y
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Output box dimensions
            box_width = self.riv_width
            box_length = self.node_spacing * 2
            
            # Create buffer
            node_buffer = self.create_buffer(
                node.geometry, angle, length=box_length, width=box_width
            )
            
            # Create a geopandas dataframe for this buffer
            buffer_gdf = gpd.GeoDataFrame(
                {
                    "node_id": [node.node_id],
                    "dist_m": [node.dist_m],
                },
                geometry=[node_buffer],
                crs=self.utm_crs
            )
            buffer_list.append(buffer_gdf)
        
        # Combine all buffers
        if buffer_list:
            self.nodes_buffer_gdf = pd.concat(buffer_list, ignore_index=True)
        else:
            self.nodes_buffer_gdf = gpd.GeoDataFrame(
                columns=["node_id", "dist_m"], geometry=[], crs=self.utm_crs
            )
        
        # Sort by distance and add distance in km
        self.nodes_buffer_gdf = self.nodes_buffer_gdf.sort_values(["dist_m"]).reset_index(drop=True)
        self.nodes_buffer_gdf["dist_km"] = self.nodes_buffer_gdf["dist_m"] / 1000
        
        return self.nodes_buffer_gdf
        
    def export_shapefiles(self) -> Tuple[str, str]:
        """
        Export nodes and buffer boxes as shapefiles.
        
        Returns:
            Tuple[str, str]: Paths to exported nodes and buffer shapefiles.
        """
        if self.nodes_gdf is None or self.nodes_buffer_gdf is None:
            raise ValueError("Nodes and buffers must be generated before export. Call process_river() or generate methods first.")
        
        # Create nodes directory
        os.makedirs(self.nodes_dir, exist_ok=True)
        
        # Define export paths
        self.node_export_path = os.path.join(
            self.nodes_dir, f"nodes_{self.node_spacing}m.shp"
        )
        self.buffer_export_path = os.path.join(
            self.nodes_dir, f"buffer_{self.node_spacing}m.shp"
        )
        
        # Export shapefiles
        self.nodes_gdf.set_crs(self.utm_crs).to_file(self.node_export_path)
        self.nodes_buffer_gdf.set_crs(self.utm_crs).to_file(self.buffer_export_path)
        
        return self.node_export_path, self.buffer_export_path
    
    def process_river(self) -> Tuple[str, str]:
        """
        Complete river processing: load shapefile, generate nodes and buffers, export files.
        
        Returns:
            Tuple[str, str]: Paths to exported nodes and buffer shapefiles.
        """
        # Generate nodes and buffers
        self.generate_nodes()
        self.generate_buffers()
        
        # Export to files
        return self.export_shapefiles()
    
    def plot_flow_direction(self) -> None:
        """
        Plot the river flow direction with nodes and buffer boxes.
        """
        if self.river_gdf_utm is None or self.nodes_gdf is None or self.nodes_buffer_gdf is None:
            raise ValueError("River, nodes, and buffers must be generated before plotting. Call process_river() or generate methods first.")
        
        # Create plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        # Plot river reaches
        self.river_gdf_utm.plot(ax=ax, color='blue', linewidth=2, label='River Reaches')
        # Plot buffer boxes, alpha = 0.2 for visibility
        self.nodes_buffer_gdf.plot(ax=ax, color='green', linestyle='--', label='Buffer Boxes', alpha=0.2)
        # Plot nodes with dist_km as gradient
        self.nodes_gdf.plot(ax=ax, column='dist_km', cmap='viridis', markersize=50, label='Nodes', legend=True)
        ax.set_title(f'River Flow Direction: {self.riv_name}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        cx.add_basemap(ax, crs = self.river_gdf_utm.crs.to_string())
        ax.legend()
        plt.grid()
        plt.show()
        return

    def get_summary_stats(self) -> dict:
        """
        Get summary statistics about the river processing.
        
        Returns:
            dict: Dictionary containing summary statistics.
        """
        stats = {
            'home_dir': self.home_dir,
            'riv_name': self.riv_name,
            'riv_shp_path': self.riv_shp_path,
            'node_spacing': self.node_spacing,
            'riv_width': self.riv_width,
            'n_reaches': self.n_reaches,
            'n_nodes': self.n_nodes,
            'total_river_length': self.total_river_length,
            'nodes_dir': self.nodes_dir,
            'utm_crs': str(self.utm_crs) if self.utm_crs else None,
            'node_export_path': self.node_export_path,
            'buffer_export_path': self.buffer_export_path
        }
        
        if self.nodes_gdf is not None and len(self.nodes_gdf) > 0:
            stats['distance_range_m'] = {
                'min': float(self.nodes_gdf['dist_m'].min()),
                'max': float(self.nodes_gdf['dist_m'].max())
            }
        
        return stats
    
    def __repr__(self) -> str:
        """String representation of the River object."""
        return (f"River(home_dir='{self.home_dir}', "
                f"riv_name='{self.riv_name}', "
                f"node_spacing={self.node_spacing}m, "
                f"n_reaches={self.n_reaches}, "
                f"n_nodes={self.n_nodes})")
