#!/usr/bin/env python
"""
Debug script for crossover failures.

Run with: pixi run python debug/debug_crossover.py

Configuration:
- 32 member population
- Sequential execution (no multiprocessing)
- Many generations to increase chance of failure
- No hazards
- No gradient descent (gd_iterations=0)
"""

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
    help="Base directory containing forcing data (should have data/large/ subdir).",
)
@click.option(
    "--log-dir",
    type=click.Path(),
    default="debug/runs",
    help="Directory to save debug results.",
)
def run_debug_experiment(data_dir, log_dir):
    """Run crossover debugging experiment.

    Fixed parameters for debugging:
    - population_size: 32
    - executor_type: sequential
    - generations: 50 (many generations to trigger failures)
    - hazard_penalty_multiplier: 0.0 (no hazards)
    - gd_iterations: 0 (no gradient descent)
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Fixed debug configuration
    population_size = 128
    generations = 50
    offspring_size = 32
    crossover_rounds = 1
    gd_iterations = 0  # No GD
    hazard_penalty_multiplier = 10.0  # No hazards
    executor_type = "process"
    num_workers = 6

    journey = JourneyConfig(
        name="Debug-Crossover-Test",
        lon_waypoints=(-80.5, -11.0),
        lat_waypoints=(30.0, 50.0),
        time_start="2021-01-01T00:00",
        time_end=None,
        speed_knots=10.0,
        time_resolution_hours=6.0,
    )

    # Look for data in data/large/ subdirectory
    data_base = Path(data_dir)
    if (data_base / "data" / "large").exists():
        base = data_base / "data" / "large"
    elif (data_base / "data_large").exists():
        base = data_base / "data_large"
    else:
        raise FileNotFoundError(
            f"Cannot find data/large/ or data_large/ in {data_dir}. "
            "Please ensure data is linked or use --data-dir to specify location."
        )

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
            random_seed=42,  # Fixed seed for reproducibility
            # Stage 2: Warmup
            selection_acceptance_rate_warmup=0.3,
            mutation_width_fraction_warmup=0.9,
            mutation_displacement_fraction_warmup=0.2,
            # Stage 3: Genetic evolution
            generations=generations,
            offspring_size=offspring_size,
            crossover_rounds=crossover_rounds,
            selection_quantile=0.2,
            selection_acceptance_rate=0.0,
            mutation_width_fraction=0.9,
            mutation_displacement_fraction=0.1,
            mutation_iterations=1,
            crossover_strategy="minimal_cost",
            hazard_penalty_multiplier=hazard_penalty_multiplier,
            # Stage 4: Post-processing (Gradient descent)
            num_elites=2,
            gd_iterations=gd_iterations,
            learning_rate_time=0.5,
            learning_rate_space=0.5,
            time_increment=1200.0,
            distance_increment=10000.0,
            # Parallelization
            executor_type=executor_type,
            num_workers=num_workers,
        ),
    )

    click.echo("=" * 80)
    click.echo("CROSSOVER DEBUG EXPERIMENT")
    click.echo("=" * 80)
    click.echo(f"Population size: {population_size}")
    click.echo(f"Generations: {generations}")
    click.echo(f"Offspring size: {offspring_size}")
    click.echo(f"Crossover rounds: {crossover_rounds}")
    click.echo(f"Executor: {executor_type}")
    click.echo(
        f"Hazards: {'disabled' if hazard_penalty_multiplier == 0 else 'enabled'}"
    )
    click.echo(f"Gradient descent: {'disabled' if gd_iterations == 0 else 'enabled'}")
    click.echo("=" * 80)
    click.echo("")
    click.echo("*** SET BREAKPOINT AT routing.py:754 IN YOUR DEBUGGER ***")
    click.echo("")

    app = RoutingApp(config=config)
    result = app.run()

    # Write logs
    run_id = datetime.now().isoformat(timespec="milliseconds").replace(":", "-")
    run_id = f"{run_id}_{uuid.uuid4()}"

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(log_dir) / f"debug_run_{run_id}.json"
    result.dump_json(output_file)
    click.echo(f"Results saved to {output_file}")

    return result


if __name__ == "__main__":
    run_debug_experiment()
