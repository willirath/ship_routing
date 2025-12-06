#!/usr/bin/env python

from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import uuid

import matplotlib.pyplot as plt
import pandas as pd

import click

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


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    default=".",
    help="Base directory containing forcing data.",
)
# Population parameters
@click.option(
    "--population-size",
    type=int,
    default=4,
    help="Population size for genetic algorithm.",
)
@click.option(
    "--random-seed", type=int, default=None, help="Random seed for reproducibility."
)
# Stage 2: Warmup
@click.option(
    "--selection-acceptance-rate-warmup",
    type=float,
    default=0.3,
    help="Acceptance rate during warmup (p_w).",
)
@click.option(
    "--mutation-width-fraction-warmup",
    type=float,
    default=0.9,
    help="Width fraction for warmup mutation (W_w).",
)
@click.option(
    "--mutation-displacement-fraction-warmup",
    type=float,
    default=0.2,
    help="Max displacement fraction for warmup (D_w).",
)
# Stage 3: Genetic evolution
@click.option("--generations", type=int, default=5, help="Number of generations (N_G).")
@click.option(
    "--offspring-size", type=int, default=4, help="Offspring size (M_offspring)."
)
@click.option(
    "--selection-quantile", type=float, default=0.2, help="Selection quantile (q)."
)
@click.option(
    "--selection-acceptance-rate",
    type=float,
    default=0.0,
    help="Acceptance rate after warmup (p).",
)
@click.option(
    "--mutation-width-fraction",
    type=float,
    default=0.9,
    help="Width fraction for mutation (W).",
)
@click.option(
    "--mutation-displacement-fraction",
    type=float,
    default=0.1,
    help="Max displacement fraction (D).",
)
@click.option(
    "--mutation-iterations",
    type=int,
    default=2,
    help="Number of mutation iterations (N_mut).",
)
@click.option(
    "--crossover-strategy",
    type=click.Choice(["minimal_cost", "random"], case_sensitive=False),
    default="minimal_cost",
    help="Crossover strategy: minimal_cost or random.",
)
@click.option(
    "--crossover-rounds",
    type=int,
    default=1,
    help="Number of crossover rounds (N_crossover). Set to 0 to skip crossover.",
)
@click.option(
    "--ignore-hazards",
    type=bool,
    default=False,
    help="Ignore hazard conditions during routing.",
)
# Stage 4: Post-processing (Gradient descent)
@click.option("--num-elites", type=int, default=2, help="Number of elite members (k).")
@click.option(
    "--gd-iterations",
    type=int,
    default=2,
    help="Number of gradient descent iterations (N_GD).",
)
@click.option(
    "--learning-rate-time",
    type=float,
    default=0.5,
    help="Learning rate for time (gamma_t).",
)
@click.option(
    "--learning-rate-space",
    type=float,
    default=0.5,
    help="Learning rate for space (gamma_s).",
)
@click.option(
    "--time-increment",
    type=float,
    default=1200.0,
    help="Time increment in seconds (delta t).",
)
@click.option(
    "--distance-increment",
    type=float,
    default=10000.0,
    help="Distance increment in meters (delta d).",
)
@click.option(
    "--log-dir",
    type=click.Path(),
    default="runs",
    help="Directory to save experiment results.",
)
# Redis options
@click.option(
    "--redis-host",
    type=str,
    default=None,
    help="Redis host for result collection (if specified, skips file output).",
)
@click.option(
    "--redis-port",
    type=int,
    default=6379,
    help="Redis port.",
)
@click.option(
    "--redis-password",
    type=str,
    default=None,
    help="Redis password (optional).",
)
# Journey parameters
@click.option(
    "--journey-name",
    type=str,
    default="Journey",
    help="Human-readable name for this journey (e.g., 'NYC-London-Winter').",
)
@click.option(
    "--lon-wp",
    "lon_waypoints",
    type=float,
    multiple=True,
    default=(-80.5, -11.0),
    help="Longitude waypoints (e.g., --lon-wp -80.5 --lon-wp -11.0).",
)
@click.option(
    "--lat-wp",
    "lat_waypoints",
    type=float,
    multiple=True,
    default=(30.0, 50.0),
    help="Latitude waypoints (e.g., --lat-wp 30.0 --lat-wp 50.0).",
)
@click.option(
    "--time-start",
    type=str,
    default="2021-01-01T00:00",
    help="Start time in ISO format (e.g., 2021-01-01T00:00).",
)
@click.option(
    "--time-end",
    type=str,
    default=None,
    help="End time in ISO format (optional).",
)
@click.option(
    "--speed-knots",
    type=float,
    default=10.0,
    help="Ship speed in knots (optional).",
)
@click.option(
    "--time-resolution-hours",
    type=float,
    default=6.0,
    help="Time resolution in hours.",
)
def run_experiment(
    data_dir,
    population_size,
    random_seed,
    selection_acceptance_rate_warmup,
    mutation_width_fraction_warmup,
    mutation_displacement_fraction_warmup,
    generations,
    offspring_size,
    selection_quantile,
    selection_acceptance_rate,
    mutation_width_fraction,
    mutation_displacement_fraction,
    mutation_iterations,
    crossover_strategy,
    crossover_rounds,
    ignore_hazards,
    num_elites,
    gd_iterations,
    learning_rate_time,
    learning_rate_space,
    time_increment,
    distance_increment,
    log_dir,
    redis_host,
    redis_port,
    redis_password,
    journey_name,
    lon_waypoints,
    lat_waypoints,
    time_start,
    time_end,
    speed_knots,
    time_resolution_hours,
) -> RoutingResult:
    """Configure and run a routing experiment."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    journey = JourneyConfig(
        name=journey_name,
        lon_waypoints=lon_waypoints,
        lat_waypoints=lat_waypoints,
        time_start=time_start,
        time_end=time_end,
        speed_knots=speed_knots,
        time_resolution_hours=time_resolution_hours,
    )
    base = Path(data_dir) / "data_large"
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
            # Population
            population_size=population_size,
            random_seed=random_seed,
            # Stage 2: Warmup
            selection_acceptance_rate_warmup=selection_acceptance_rate_warmup,
            mutation_width_fraction_warmup=mutation_width_fraction_warmup,
            mutation_displacement_fraction_warmup=mutation_displacement_fraction_warmup,
            # Stage 3: Genetic evolution
            generations=generations,
            offspring_size=offspring_size,
            crossover_rounds=crossover_rounds,
            selection_quantile=selection_quantile,
            selection_acceptance_rate=selection_acceptance_rate,
            mutation_width_fraction=mutation_width_fraction,
            mutation_displacement_fraction=mutation_displacement_fraction,
            mutation_iterations=mutation_iterations,
            crossover_strategy=crossover_strategy,
            ignore_hazards=ignore_hazards,
            # Stage 4: Post-processing (Gradient descent)
            num_elites=num_elites,
            gd_iterations=gd_iterations,
            learning_rate_time=learning_rate_time,
            learning_rate_space=learning_rate_space,
            time_increment=time_increment,
            distance_increment=distance_increment,
        ),
    )
    app = RoutingApp(config=config)
    result = app.run()

    # Write logs
    run_id = datetime.now().isoformat(timespec="milliseconds").replace(":", "-")
    run_id = f"{run_id}_{uuid.uuid4()}"

    if redis_host:
        # Write to Redis using msgpack
        import redis

        r = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=False,
        )
        key = f"result:{run_id}"
        r.set(key, result.to_msgpack())
        click.echo(f"Result stored in Redis: {key}")
    else:
        # Original file-based output
        output_file = Path(log_dir) / f"run_{run_id}.json"
        result.dump_json(output_file)
        click.echo(f"Results saved to {output_file}")

    return result


if __name__ == "__main__":
    run_experiment()
