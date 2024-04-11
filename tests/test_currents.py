from pathlib import Path
import pytest

from ship_routing.currents import (
    load_currents,
    load_currents_time_average,
    select_currents_along_traj,
)
from ship_routing import Trajectory

FIXTURE_DIR = Path(__file__).parent.resolve() / "test_data"


def test_currents_simple_loading():
    load_currents(
        FIXTURE_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_1deg_5day.nc",
        lon_name="longitude",
        lat_name="latitude",
        time_name="time",
        uo_name="uo",
        vo_name="vo",
    )


def test_currents_time_average_simple_loading():
    load_currents_time_average(
        FIXTURE_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_1deg_5day.nc",
        lon_name="longitude",
        lat_name="latitude",
        time_name="time",
        uo_name="uo",
        vo_name="vo",
    )


def test_currents_names():
    ds = load_currents(
        FIXTURE_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_1deg_5day.nc",
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


def test_currents_time_average_names():
    ds = load_currents_time_average(
        FIXTURE_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_1deg_5day.nc",
        lon_name="longitude",
        lat_name="latitude",
        time_name="time",
        uo_name="uo",
        vo_name="vo",
    )
    assert "time" not in ds
    assert "lat" in ds
    assert "lon" in ds
    assert "vo" in ds
    assert "uo" in ds


def test_current_selection_along_traj():
    ds = load_currents_time_average(
        FIXTURE_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_1deg_5day.nc",
        lon_name="longitude",
        lat_name="latitude",
        time_name="time",
        uo_name="uo",
        vo_name="vo",
    )
    traj = Trajectory(lon=[-80, 0], lat=[25, 50]).refine(new_dist=10_000)
    currents = select_currents_along_traj(ds=ds, trajectory=traj)

    assert "lon" in currents.coords
    assert "lat" in currents.coords
    assert "uo" in currents
    assert "vo" in currents
