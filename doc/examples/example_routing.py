"""Example routing script for experimentation.

Provides a complete JourneyConfig and ForcingConfig so it can run unchanged
once the matching datasets exist under ``doc/examples/data_large``.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ship_routing.app import (
    RoutingApp,
    RoutingResult,
    HyperParams,
    ForcingConfig,
    JourneyConfig,
    RoutingConfig,
)


def run_example() -> RoutingResult:
    """Configure and run a routing experiment."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    journey = JourneyConfig(
        lon_waypoints=(-80.5, -62.0),
        lat_waypoints=(30.0, 35.0),
        time_start="2021-01-01T00:00",
        speed_knots=7.0,
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
    config = RoutingConfig(
        journey=journey,
        forcing=forcing,
        hyper=HyperParams(
            population_size=6,
            random_seed=345,
            generations=2,
            mutation_iterations=2,
            crossover_rounds=2,
            selection_quantile=0.5,
            selection_acceptance_rate_warmup=0.3,
            selection_acceptance_rate=0.0,
            num_elites=2,
            crossover_strategy="minimal_cost",
        ),
    )
    app = RoutingApp(config=config)
    return app.run()


if __name__ == "__main__":
    result = run_example()
    print(result)
    runs_dir = Path(__file__).resolve().parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    latest_path = runs_dir / "example_routing_result.json"
    result.dump_json(latest_path)
    print(f"Dumped result to {latest_path}")
    if result.logs:
        ga_stages = result.logs.stages_named("ga_generation")
        if ga_stages and "cost_mean" in ga_stages[0].metrics:
            cost_series = pd.Series(
                data=[stage.metrics["cost_mean"] for stage in ga_stages],
                index=[
                    stage.metrics.get("generation", idx)
                    for idx, stage in enumerate(ga_stages)
                ],
                name="Mean cost",
            )
            cost_series.index.name = "Generation"
            cost_series.plot(marker="o", figsize=(8, 4))
            plt.show()
