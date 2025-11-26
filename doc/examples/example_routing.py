"""Example routing script for experimentation.

Provides a complete JourneyConfig and ForcingConfig so it can run unchanged
once the matching datasets exist under ``doc/examples/data_large``.
"""

import json
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

from ship_routing.app import (
    RoutingApp,
    RoutingResult,
    StageLog,
    RoutingLog,
    HyperParams,
    ForcingConfig,
    JourneyConfig,
    RoutingConfig,
)
from ship_routing.core.config import Physics, Ship
from ship_routing.core.population import Population, PopulationMember


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
        engine="zarr",
        chunks="auto",
        load_eagerly=True,
    )
    config = RoutingConfig(
        journey=journey,
        forcing=forcing,
        ship=Ship(),
        physics=Physics(),
        hyper=HyperParams(
            population_size=4,
            random_seed=345,
            generations=2,
            selection_quantile=0.5,
            selection_acceptance_rate_warmup=0.4,
            selection_acceptance_rate=0.0,
            mutation_width_fraction=0.9,
            mutation_displacement_fraction=0.25,
            mutation_iterations=3,
            crossover_rounds=2,
            num_elites=2,
            gd_iterations=2,
            learning_rate_time=0.5,
            learning_rate_space=0.5,
            time_increment=1_200.0,
            distance_increment=10_000.0,
            crossover_strategy="random",
        ),
    )
    app = RoutingApp(config=config)
    return app.run()


def visualize_result(result: RoutingResult) -> None:
    """Plot GA mean cost if available."""
    ga_stages = result.logs.stages_named("ga_generation")
    if not ga_stages or "cost_min" not in ga_stages[0].metrics:
        return
    cost_series = pd.Series(
        data=[stage.metrics["cost_min"] for stage in ga_stages],
        index=[
            stage.metrics.get("generation", idx) for idx, stage in enumerate(ga_stages)
        ],
        name="Mean cost",
    )
    cost_series.index.name = "Generation"
    gradient_steps = result.logs.stages_named("gradient_step")
    gradient_costs = [
        step.metrics["post_cost"]
        for step in gradient_steps
        if "post_cost" in step.metrics
    ]
    final_costs = (
        [member.cost for member in result.elite_population.members]
        if result.elite_population
        else []
    )
    ax = cost_series.plot(marker="o", figsize=(8, 4))
    if gradient_costs:
        ax.axhline(
            min(gradient_costs),
            color="tab:orange",
            linestyle="--",
            label="Gradient descent min cost",
        )
    if final_costs:
        ax.axhline(
            min(final_costs),
            color="tab:green",
            linestyle=":",
            label="Final route min cost",
        )
    ax.legend()
    plt.show()


def load_result_json(path: Path) -> RoutingResult:
    """Load a RoutingResult from disk."""
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    seed_member = (
        PopulationMember.from_dict(data["seed_member"])
        if data.get("seed_member")
        else None
    )
    elite_population = (
        Population.from_dict(data["elite_population"])
        if data.get("elite_population")
        else None
    )
    log_data = data.get("log")
    logs = (
        RoutingLog(
            config=log_data.get("config", {}),
            stages=[
                StageLog(
                    name=stage["name"],
                    metrics=stage.get("metrics", {}),
                    timestamp=stage.get("timestamp", ""),
                )
                for stage in log_data.get("stages", [])
            ],
        )
        if log_data
        else None
    )
    return RoutingResult(
        seed_member=seed_member, elite_population=elite_population, logs=logs
    )


if __name__ == "__main__":
    runs_dir = Path(__file__).resolve().parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    latest_path = runs_dir / "example_routing_result.json"

    if "--visualize-only" in sys.argv:
        visualize_result(load_result_json(latest_path))
        raise SystemExit(0)

    result = run_example()
    print(result)
    result.dump_json(latest_path)
    print(f"Dumped result to {latest_path}")
    visualize_result(load_result_json(latest_path))
