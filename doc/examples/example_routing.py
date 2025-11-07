"""Minimal routing script for experimentation.

Adapt the placeholders below to point to your own datasets and journey.
The RoutingApp currently only contains scaffolding, so this file is a
starting point for future integrations.
"""

from ship_routing.app import RoutingApp, RoutingResult
from ship_routing.config import (
    EnvironmentConfig,
    JourneyConfig,
    RoutingConfig,
)


def run_example() -> RoutingResult:
    """Configure and run a routing experiment."""
    journey = JourneyConfig(
        lon_waypoints=(-80.5, -12.0),
        lat_waypoints=(30.0, 45.0),
        time_start="2021-01-01T00:00",
        speed_knots=10.0,
        time_resolution_hours=12.0,
    )
    environment = EnvironmentConfig(
        currents_path="PATH/TO/CURRENTS.zarr",
        waves_path="PATH/TO/WAVES.zarr",
        winds_path="PATH/TO/WINDS.zarr",
    )
    config = RoutingConfig(journey=journey, environment=environment)
    app = RoutingApp(config=config)
    return app.run()


if __name__ == "__main__":
    result = run_example()
    print(result)
