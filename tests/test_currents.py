from pathlib import Path
import pytest

from ship_routing.currents import load_currents

FIXTURE_DIR = Path(__file__).parent.resolve() / "test_data"


def test_currents_simple_loading():
    load_currents(
        FIXTURE_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_1deg_5day.nc",
        lon_name="longitude",
        lat_name="latitude",
        time_name="time",
        u_name="uo",
        v_name="vo",
    )


def test_currents_names():
    ds = load_currents(
        FIXTURE_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_1deg_5day.nc",
        lon_name="longitude",
        lat_name="latitude",
        time_name="time",
        u_name="uo",
        v_name="vo",
    )
    assert "lon" in ds
    assert "time" in ds
    assert "u" in ds
    assert "v" in ds
