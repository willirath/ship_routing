from pathlib import Path
import pytest

from ship_routing.core.data import (
    load_currents,
    load_winds,
    load_waves,
)
from ship_routing.core.geodesics import compute_ellipse_bbox
from conftest import TEST_DATA_DIR


def get_currents_data_files():
    return sorted(TEST_DATA_DIR.glob("currents/cmems_*.nc"))


def get_waves_data_files():
    return sorted(TEST_DATA_DIR.glob("waves/cmems_*.nc"))


def get_winds_data_files():
    return sorted(TEST_DATA_DIR.glob("winds/cmems_*.nc"))


@pytest.mark.parametrize("currents_data_file", get_currents_data_files())
def test_currents_simple_loading(currents_data_file):
    load_currents(currents_data_file)


@pytest.mark.parametrize("winds_data_file", get_winds_data_files())
def test_winds_simple_loading(winds_data_file):
    load_winds(winds_data_file)


@pytest.mark.parametrize("waves_data_file", get_waves_data_files())
def test_waves_simple_loading(waves_data_file):
    load_waves(waves_data_file)


@pytest.mark.parametrize("currents_data_file", get_currents_data_files())
def test_currents_names(currents_data_file):
    ds = load_currents(
        data_file=currents_data_file,
        lon_name="longitude",
        lat_name="latitude",
        time_name="time",
        uo_name="uo",
        vo_name="vo",
    )
    assert "time" in ds
    assert "lat" in ds
    assert "lon" in ds
    assert "vo" in ds
    assert "uo" in ds


@pytest.mark.parametrize("winds_data_file", get_winds_data_files())
def test_winds_names(winds_data_file):
    ds = load_winds(
        data_file=winds_data_file,
        lon_name="longitude",
        lat_name="latitude",
        time_name="time",
    )
    assert "time" in ds
    assert "lat" in ds
    assert "lon" in ds
    assert "vw" in ds
    assert "uw" in ds


@pytest.mark.parametrize("waves_data_file", get_waves_data_files())
def test_waves_names(waves_data_file):
    ds = load_waves(
        data_file=waves_data_file,
        lon_name="longitude",
        lat_name="latitude",
        time_name="time",
    )
    assert "time" in ds
    assert "lat" in ds
    assert "lon" in ds
    assert "wh" in ds


@pytest.mark.parametrize("currents_data_file", get_currents_data_files())
def test_spatial_cropping_currents(currents_data_file):
    """Test that spatial cropping reduces dataset size."""
    # Load full dataset
    ds_full = load_currents(data_file=currents_data_file)

    # Define spatial bounds
    spatial_bounds = (-80.0, -60.0, 30.0, 40.0)

    # Load cropped dataset
    ds_cropped = load_currents(
        data_file=currents_data_file,
        spatial_bounds=spatial_bounds,
    )

    # Cropped dataset should be smaller
    assert ds_cropped["lon"].size < ds_full["lon"].size
    assert ds_cropped["lat"].size < ds_full["lat"].size

    # Cropped dataset should have correct bounds
    assert ds_cropped["lon"].min() >= spatial_bounds[0]
    assert ds_cropped["lon"].max() <= spatial_bounds[1]
    assert ds_cropped["lat"].min() >= spatial_bounds[2]
    assert ds_cropped["lat"].max() <= spatial_bounds[3]


@pytest.mark.parametrize("winds_data_file", get_winds_data_files())
def test_spatial_cropping_winds(winds_data_file):
    """Test spatial cropping for winds dataset."""
    # Load full dataset
    ds_full = load_winds(data_file=winds_data_file)

    # Define spatial bounds
    spatial_bounds = (-80.0, -60.0, 30.0, 40.0)

    # Load cropped dataset
    ds_cropped = load_winds(
        data_file=winds_data_file,
        spatial_bounds=spatial_bounds,
    )

    # Cropped dataset should be smaller or equal
    assert ds_cropped["lon"].size <= ds_full["lon"].size
    assert ds_cropped["lat"].size <= ds_full["lat"].size


@pytest.mark.parametrize("waves_data_file", get_waves_data_files())
def test_spatial_cropping_waves(waves_data_file):
    """Test spatial cropping for waves dataset."""
    # Load full dataset
    ds_full = load_waves(data_file=waves_data_file)

    # Define spatial bounds
    spatial_bounds = (-80.0, -60.0, 30.0, 40.0)

    # Load cropped dataset
    ds_cropped = load_waves(
        data_file=waves_data_file,
        spatial_bounds=spatial_bounds,
    )

    # Cropped dataset should be smaller or equal
    assert ds_cropped["lon"].size <= ds_full["lon"].size
    assert ds_cropped["lat"].size <= ds_full["lat"].size


@pytest.mark.parametrize("currents_data_file", get_currents_data_files())
def test_spatial_cropping_disabled(currents_data_file):
    """Test that cropping is disabled when spatial_bounds=None."""
    # Load with bounds=None
    ds_no_bounds = load_currents(
        data_file=currents_data_file,
        spatial_bounds=None,
    )

    # Load without bounds parameter
    ds_default = load_currents(data_file=currents_data_file)

    # Should be the same size
    assert ds_no_bounds["lon"].size == ds_default["lon"].size
    assert ds_no_bounds["lat"].size == ds_default["lat"].size


def test_ellipse_bbox_integration():
    """Test that ellipse bbox is reasonable for a known route."""
    # Atlantic crossing
    lon_start, lat_start = -80.0, 30.0
    lon_end, lat_end = -10.0, 40.0

    spatial_bounds = compute_ellipse_bbox(
        lon_start=lon_start,
        lat_start=lat_start,
        lon_end=lon_end,
        lat_end=lat_end,
        length_multiplier=4.0,
        buffer_degrees=1.0,
    )

    lon_min, lon_max, lat_min, lat_max = spatial_bounds

    # Bbox should be reasonable: contains start point and roughly extends beyond
    # due to buffer and grid discretization
    # Both endpoints should be close to the bbox (within buffer distance)
    assert lon_min <= lon_start
    assert lon_end <= lon_max
    assert lon_min < lon_max

    # Bbox should extend in latitude
    assert lat_min < lat_max

    # Bbox should extend beyond direct distance due to multiplier > 1
    direct_lon_extent = abs(lon_end - lon_start)
    bbox_lon_extent = lon_max - lon_min
    assert bbox_lon_extent > direct_lon_extent
