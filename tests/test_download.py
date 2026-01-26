import os
import sys
import shutil

# Add parent directory to path to import pixc2profile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pixc2profile.download import download_pixc_data


def test_download_pixc_data():
    """Test if download function can run without errors."""
    # Use a temporary directory
    test_dir = "./tests/data"
    
    try:
        # Test parameters
        riv_name = "test_river"
        start_date = "2024-01-01"
        end_date = "2024-03-01"  # Short date range to minimize download
        pass_tile_list = ["454_082L"]  # Single pass to test
        
        # Call the download function
        # This will test if the function runs without crashing
        # Note: This requires valid earthdata credentials
        downloaded_files = download_pixc_data(
            home_dir=test_dir,
            riv_name=riv_name,
            start_date=start_date,
            end_date=end_date,
            pass_tile_list=pass_tile_list
        )
        
        # Basic assertions
        assert isinstance(downloaded_files, list), "Downloaded files should be a list"
        print(f"Download completed. Found {len(downloaded_files)} files.")
        
        # Check if files actually exist (if any were downloaded)
        for file_path in downloaded_files:
            if file_path and os.path.exists(file_path):
                print(f"Downloaded file exists: {file_path}")
                assert os.path.isfile(file_path), f"Downloaded path should be a file: {file_path}"
        
        return
        
    except Exception as e:
        print(f"Download test failed with error: {e}")
        # Don't fail the test for authentication or network issues
        if "authentication" in str(e).lower() or "login" in str(e).lower():
            print("Test skipped: Authentication required")
            return
        else:
            raise e


if __name__ == "__main__":
    test_download_pixc_data()
    print("Download test completed!")