from pathlib import Path
import pytest

from ship_routing.currents import (
    load_currents,
)

TEST_DATA_DIR = Path(__file__).parent.resolve() / "test_data"


def get_current_data_files():
    return sorted(TEST_DATA_DIR.glob("currents/cmems_*.nc"))


@pytest.mark.parametrize("current_data_file", get_current_data_files())
def test_currents_simple_loading(current_data_file):
    load_currents(current_data_file)


@pytest.mark.parametrize("current_data_file", get_current_data_files())
def test_currents_names(current_data_file):
    ds = load_currents(
        data_file=current_data_file,
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
