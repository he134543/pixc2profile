import os
import sys

# Add parent directory to path to import pixc2profile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pixc2profile.river import River

def test_generate_nodes():
    """Test river node generation functionality."""
    # Create temporary directory
    test_dir = "./tests/data"
    
    try:
        # Set up test parameters
        home_dir = test_dir
        riv_name = "test_river"
        # Use the test shapefile from data directory
        riv_shp_path = os.path.join(home_dir, 'shps', 'test_river.shp')
        node_spacing = 100.0  # 100 meters
        riv_width = 50.0      # 50 meters
        buffer_width = 100.0  # 100 meters
        
        # Check if test shapefile exists
        assert os.path.exists(riv_shp_path), f"Test shapefile not found: {riv_shp_path}"
        
        # Create River instance
        river = River(
            home_dir=home_dir,
            riv_name=riv_name,
            riv_shp_path=riv_shp_path,
            node_spacing=node_spacing,
            riv_width=riv_width,
            buffer_width=buffer_width
        )
        
        # Test initial state
        assert river.nodes_gdf is None, "Nodes should be None initially"
        assert river.n_nodes == 0, "Number of nodes should be 0 initially"
        
        # Load river shapefile
        river_gdf_utm = river.load_river_shapefile()
        assert river_gdf_utm is not None, "River GDF should not be None after loading"
        assert len(river_gdf_utm) > 0, "River GDF should contain features"
        assert river.n_reaches > 0, "Should have at least one reach"
        
        print(f"Loaded river with {river.n_reaches} reaches")
        print(f"Total river length: {river.total_river_length:.2f} meters")
        
        # Generate nodes and buffers
        river.process_river()
        
        # Test node sequence via plotting
        river.plot_flow_direction()

        print(f"Exported nodes and buffers to {river.node_export_path, river.buffer_export_path}")

        print("✓ Node generation test passed!")

        return
        
    except Exception as e:
        print(f"Node generation test failed: {e}")
        raise e

if __name__ == "__main__":
    test_generate_nodes()
    print("Node generation test completed!")
