"""Minimal routing script for experimentation.

Adapt the placeholders below to point to your own datasets and journey.
The RoutingApp currently only contains scaffolding, so this file is a
starting point for future integrations.
"""

import logging
from pathlib import Path

from ship_routing.app import RoutingApp, RoutingResult
from ship_routing.config import ForcingConfig, JourneyConfig, RoutingConfig


def run_example() -> RoutingResult:
    """Configure and run a routing experiment."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    journey = JourneyConfig(
        lon_waypoints=(-80.5, -12.0),
        lat_waypoints=(30.0, 45.0),
        time_start="2021-01-01T00:00",
        speed_knots=10.0,
        time_resolution_hours=12.0,
    )
    base = Path(__file__).resolve().parent / "data_large"
    forcing = ForcingConfig(
        currents_path=str(
            base
            / "cmems_mod_glo_phy_my_0.083deg_P1D-m_time_2021_lat_+10_+65_lon_-100_+010_uo-vo.zarr"
        ),
        waves_path=str(
            base
            / "cmems_mod_glo_wav_my_0.2deg_PT3H-i_time_2021_lat_+10_+65_lon_-100_+010_VHM0-VMDR.zarr"
        ),
        winds_path=str(
            base
            / "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_time_2021_lat_+10_+65_lon_-100_+010_eastward_wind-northward_wind.zarr"
        ),
    )
    config = RoutingConfig(journey=journey, forcing=forcing)
    app = RoutingApp(config=config)
    return app.run()


if __name__ == "__main__":
    result = run_example()
    print(result)
