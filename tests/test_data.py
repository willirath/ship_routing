from pathlib import Path
import pytest

from ship_routing.data import (
    load_currents,
    load_winds,
    load_waves,
)

TEST_DATA_DIR = Path(__file__).parent.resolve() / "test_data"


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
