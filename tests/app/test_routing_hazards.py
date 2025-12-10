import numpy as np

from ship_routing.app.config import HyperParams, RoutingConfig, ForcingData
from ship_routing.app.routing import RoutingApp
from ship_routing.core import Route, WayPoint
from ship_routing.core.data import load_currents, load_waves, load_winds, make_hashable

from pathlib import Path
from conftest import TEST_DATA_DIR


def _hazard_forcing():
    currents = load_currents(
        data_file=TEST_DATA_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_1deg_5day.nc"
    )
    winds = load_winds(
        data_file=TEST_DATA_DIR
        / "winds/cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_2021-01_6hours_0.5deg_100W-020E_10N-65N.nc"
    )
    waves = load_waves(
        data_file=TEST_DATA_DIR
        / "waves/cmems_mod_glo_wav_my_0.2deg_PT3H-i_VHM0_2021-01_1d-max_100W-020E_10N-65N.nc"
    )
    waves["wh"] = 50.0 + 0.0 * waves["wh"].fillna(0.0)
    return ForcingData(
        currents=make_hashable(currents),
        winds=make_hashable(winds),
        waves=make_hashable(waves),
    )


def _simple_route():
    return Route(
        way_points=(
            WayPoint(lon=0, lat=-1 / 60.0, time=np.datetime64("2001-01-01T00:00:00")),
            WayPoint(lon=0, lat=0.0, time=np.datetime64("2001-01-01T06:00:00")),
            WayPoint(lon=0, lat=1 / 60.0, time=np.datetime64("2001-01-01T12:00:00")),
        )
    )


def test_route_cost_respects_hazards_flag_enabled():
    config = RoutingConfig(hyper=HyperParams())
    app = RoutingApp(config=config)
    forcing = _hazard_forcing()
    cost = app._route_cost(route=_simple_route(), forcing=forcing)

    # Get baseline cost with hazards ignored
    config_baseline = RoutingConfig(hyper=HyperParams(hazard_penalty_multiplier=0))
    app_baseline = RoutingApp(config=config_baseline)
    cost_baseline = app_baseline._route_cost(route=_simple_route(), forcing=forcing)

    # Check hazardous cost is penalized correctly
    assert (
        cost > cost_baseline * 50
    )  # Should be at least 50x baseline (with default multiplier 100)
    assert np.isfinite(cost)  # But still finite


def test_route_cost_respects_hazards_flag_disabled():
    config = RoutingConfig(hyper=HyperParams(hazard_penalty_multiplier=0))
    app = RoutingApp(config=config)
    forcing = _hazard_forcing()
    cost = app._route_cost(route=_simple_route(), forcing=forcing)
    assert np.isfinite(cost)


def test_hazard_penalty_is_multiplicative():
    """Verify that hazard penalty scales with base cost."""
    # Get baseline cost first (no hazards)
    config_baseline = RoutingConfig(hyper=HyperParams(hazard_penalty_multiplier=0))
    app_baseline = RoutingApp(config=config_baseline)
    forcing = _hazard_forcing()
    cost_baseline = app_baseline._route_cost(route=_simple_route(), forcing=forcing)

    # Test with different penalty multipliers
    for multiplier in [10.0, 100.0, 1000.0]:
        config = RoutingConfig(hyper=HyperParams(hazard_penalty_multiplier=multiplier))
        app = RoutingApp(config=config)

        cost_hazard = app._route_cost(route=_simple_route(), forcing=forcing)

        # Check multiplicative relationship
        expected = cost_baseline * (1 + multiplier)
        assert abs(cost_hazard - expected) / expected < 0.01  # Within 1%
