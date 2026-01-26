import os
import sys
import shutil
import pandas as pd
# Add parent directory to path to import pixc2profile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pixc2profile.pixc import PIXC


def test_pixc_basic_functionality():
    """Test basic PIXC class functionality."""
    # Create temporary directory
    test_dir = "./tests/data"
    
    try:
        # Copy test data to temporary directory
        test_riv_dir = os.path.join(test_dir, "test_river")
        test_pixc_dir = os.path.join(test_riv_dir, "SWOT_L2_HR_PIXC_2.0")
        os.makedirs(test_pixc_dir, exist_ok=True)
        
        # Test PIXC initialization with real data
        pixc = PIXC(
            home_dir=test_dir,
            riv_name="test_river",
            create_ref_table_on_init=True
        )
        
        # Test basic properties
        assert pixc.home_dir == test_dir
        assert pixc.riv_name == "test_river"
        assert pixc.pixc_dir == test_pixc_dir
        
        # Test reference table creation
        if pixc.ref_table is not None:
            assert len(pixc.ref_table) > 0, "Reference table should contain entries"
            
            # Check required columns exist
            required_cols = ['PIXC_paths', 'date', 'tile', 'pass_id']
            for col in required_cols:
                if col in pixc.ref_table.columns:
                    print(f"✓ Reference table contains '{col}' column")
                else:
                    print(f"⚠ Reference table missing '{col}' column")
            
            print(f"Reference table created with {len(pixc.ref_table)} entries")
            
            # Test available dates
            dates = pixc.available_dates
            if dates:
                print(f"Available dates: {dates}")
                assert len(dates) > 0, "Should have available dates"
        else:
            print("No reference table created (no PIXC files found)")
        
        # Test string representation
        repr_str = str(pixc)
        assert "PIXC" in repr_str
        assert pixc.riv_name in repr_str
        
        print("✓ PIXC functionality test passed!")
        return
        
    except Exception as e:
        print(f"PIXC test failed: {e}")
        raise e
    
def test_pixc_process_water_qcfilter():
    # test processing water pixels
    test_dir = "./tests/data"
    node_buffer_path = f"{test_dir}/test_river/nodes/buffer_100.0m.shp"
    pixc = PIXC(
            home_dir=test_dir,
            riv_name="test_river",
            create_ref_table_on_init=True
        )
    
    try:

        pixc.process_water_pixels(node_buffer_path=node_buffer_path)
        # read one of the water pixel files to check format
        if pixc.pixc_water_paths:
            sample_file = pixc.pixc_water_paths[0]
            df = pd.read_csv(sample_file)
            print(df.head())
            print(df.describe())


        print("✓ PIXC water pixel processing test passed!")

        # test filtering with quality flags
        pixc.filter_with_quality_flags(pixc.pixc_water_paths, 
                                       quality_flag_dict={"geolocation_qual":[4]})
        # read one of the filtered files to check format
        if pixc.pixc_water_qc_filtered_paths:
            sample_file = pixc.pixc_water_qc_filtered_paths[0]
            df = pd.read_csv(sample_file)
            print(df.head())
            print(df.describe())

        print("✓ PIXC quality flag filtering test passed!")

        return
    except Exception as e:
        print(f"PIXC water pixel processing test failed: {e}")
        raise e
    

if __name__ == "__main__":
    print("Running PIXC tests...")
    test_pixc_basic_functionality()
    test_pixc_process_water_qcfilter()
    print("All PIXC tests completed!")
