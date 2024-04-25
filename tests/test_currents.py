from pathlib import Path
import pytest

from ship_routing.currents import (
    load_currents,
    load_currents_time_average,
    select_currents_along_traj,
)
from ship_routing import Trajectory

TEST_DATA_DIR = Path(__file__).parent.resolve() / "test_data"


def get_current_data_files():
    return sorted(TEST_DATA_DIR.glob("currents/cmems_*.nc"))


@pytest.mark.parametrize("current_data_file", get_current_data_files())
def test_currents_simple_loading(current_data_file):
    load_currents(current_data_file)


@pytest.mark.parametrize("current_data_file", get_current_data_files())
def test_currents_time_average_simple_loading(current_data_file):
    load_currents_time_average(
        data_file=current_data_file,
        lon_name="longitude",
        lat_name="latitude",
        time_name="time",
        uo_name="uo",
        vo_name="vo",
    )


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


@pytest.mark.parametrize("current_data_file", get_current_data_files())
def test_currents_time_average_names(current_data_file):
    ds = load_currents_time_average(
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


@pytest.mark.parametrize("current_data_file", get_current_data_files())
def test_current_selection_along_traj(current_data_file):
    ds = load_currents_time_average(
        data_file=current_data_file,
        lon_name="longitude",
        lat_name="latitude",
        time_name="time",
        uo_name="uo",
        vo_name="vo",
    )
    ship_positions = (
        Trajectory(lon=[-80, 0], lat=[25, 50], duration_seconds=10 * 24 * 3600)
        .refine(new_dist=10_000)
        .data_frame
    )
    currents = select_currents_along_traj(ds=ds, ship_positions=ship_positions)

    assert "lon" in currents.coords
    assert "lat" in currents.coords
    assert "uo" in currents
    assert "vo" in currents
