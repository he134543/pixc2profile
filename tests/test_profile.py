import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path to import pixc2profile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pixc2profile.profile import Profile

def test_profile_initialization():
    home_dir = "./tests/data"
    riv_name = "test_river"
    node_path = os.path.join(home_dir, riv_name, "nodes", "nodes_100.0m.shp")
    buffer_path = os.path.join(home_dir, riv_name, "nodes", "buffer_100.0m.shp")
    pixc_dir = os.path.join(home_dir, riv_name, "SWOT_L2_HR_PIXC_2.0")
    pixc_water_dir = os.path.join(home_dir, riv_name, "pixc_water")
    pixc_water_qc_filtered_dir = os.path.join(home_dir, riv_name, "pixc_water_qc_filtered")
    output_path = os.path.join(home_dir, riv_name, "test_river_wse_profiles.csv")

    profile = Profile(
        riv_name=riv_name,
        home_dir=home_dir,
        node_path=node_path,
        buffer_path=buffer_path,
        pixc_dir=pixc_dir,
        pixc_water_dir=pixc_water_dir,
        pixc_water_qc_filtered_dir=pixc_water_qc_filtered_dir,
        output_path=output_path
    )

    assert profile.riv_name == riv_name
    assert profile.home_dir == home_dir
    assert profile.pixc_dir == pixc_dir
    assert profile.pixc_water_dir == pixc_water_dir
    assert profile.pixc_water_qc_filtered_dir == pixc_water_qc_filtered_dir
    assert profile.output_path == output_path

    print("Profile initialization test passed.")
    return

def test_generate_profiles():
    home_dir = "./tests/data"
    riv_name = "test_river"
    node_path = os.path.join(home_dir, riv_name, "nodes", "nodes_100.0m.shp")
    buffer_path = os.path.join(home_dir, riv_name, "nodes", "buffer_100.0m.shp")
    pixc_dir = os.path.join(home_dir, riv_name, "SWOT_L2_HR_PIXC_2.0")
    pixc_water_dir = os.path.join(home_dir, riv_name, "pixc_water")
    pixc_water_qc_filtered_dir = os.path.join(home_dir, riv_name, "pixc_water_qc_filtered")
    output_path = os.path.join(home_dir, riv_name, "test_river_wse_profiles.csv")

    profile = Profile(
        riv_name=riv_name,
        home_dir=home_dir,
        node_path=node_path,
        buffer_path=buffer_path,
        pixc_dir=pixc_dir,
        pixc_water_dir=pixc_water_dir,
        pixc_water_qc_filtered_dir=pixc_water_qc_filtered_dir,
        output_path=output_path
    )

    # profile.load_available_dates()
    all_profiles = profile.build_wse_profiles_over_time(profile_range = (0, np.inf),
                                    agg_func = "median",
                                    keep_qual_groups = ["interferogram_qual", "classification_qual", "geolocation_qual", "sig0_qual"],
                                    frac_list = [0.01, 0.05, 0.1, 0.2],
                                    it = 3,
                                    seg_location = [-42],
                                    )

    print(all_profiles.head())
    assert isinstance(all_profiles, pd.DataFrame)
    assert not all_profiles.empty
    assert "date" in all_profiles.columns
    assert "dist_km" in all_profiles.columns

    # check via plotting wse profiles
    profile.plot_wse_profile()

    print("Generate profiles test passed.")

    return

if __name__ == "__main__":
    test_profile_initialization()
    test_generate_profiles()
    print("All tests passed.")
