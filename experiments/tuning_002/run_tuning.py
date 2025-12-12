#!/usr/bin/env python
"""Parsl-based hyperparameter tuning for ship routing.

This script orchestrates large-scale hyperparameter tuning experiments using
Parsl for task distribution. It replaces the previous bash-based job scripts
with a cleaner Python-based approach.

Usage:
    # Local testing
    python run_tuning.py --profile test --executor local

    # Production on SLURM
    python run_tuning.py --profile production --executor slurm

    # Quick smoke test
    python run_tuning.py --profile quick --executor local
"""

from __future__ import annotations

import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import msgpack
import parsl
from parsl import python_app
from parsl.app.futures import DataFuture
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from parsl_config import get_parsl_config
from profiles import PROFILES, TuningProfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Parsl app - runs in worker process
# Note: imports inside function to ensure they're available in worker
@python_app
def run_single_experiment(
    journey_config: dict[str, Any],
    forcing_config: dict[str, Any],
    hyper_params: dict[str, Any],
) -> bytes:
    """Run one routing optimization, return msgpack bytes.

    This function executes in a Parsl worker process. All imports must be
    inside the function body to ensure they're available in the worker.

    Parameters
    ----------
    journey_config : dict
        Journey configuration (waypoints, times, speed)
    forcing_config : dict
        Forcing data configuration (paths, engine)
    hyper_params : dict
        Hyperparameters for the optimization

    Returns
    -------
    bytes
        Msgpack-serialized RoutingResult
    """
    from ship_routing.app.config import (
        ForcingConfig,
        HyperParams,
        JourneyConfig,
        RoutingConfig,
    )
    from ship_routing.app.routing import RoutingApp

    config = RoutingConfig(
        journey=JourneyConfig(**journey_config),
        forcing=ForcingConfig(**forcing_config),
        hyper=HyperParams(**hyper_params),
    )
    result = RoutingApp(config).run()
    return result.to_msgpack()


def sample_value(options: tuple | list) -> Any:
    """Sample a value from options (returns single value if only one option)."""
    if len(options) == 1:
        return options[0]
    return random.choice(options)


def generate_experiment_configs(
    profile: TuningProfile,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate all experiment configurations for a tuning run.

    Parameters
    ----------
    profile : TuningProfile
        Profile defining parameter space and experiment scale
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    list[dict]
        List of experiment configurations, each containing:
        - journey_config: dict for JourneyConfig
        - forcing_config: dict for ForcingConfig
        - hyper_params: dict for HyperParams
        - metadata: dict with experiment identifiers
    """
    if seed is not None:
        random.seed(seed)

    configs = []

    # Outer loops: realisations x months x speeds
    for realisation in range(profile.n_realisations):
        for month in profile.months:
            for speed in profile.speeds_knots:
                time_start = f"2021-{month:02d}-01T00:00:00"

                # Generate n_runs experiments for forward and backward routes
                for direction in ["forward", "backward"]:
                    if direction == "forward":
                        lon_wp = (profile.lon_start, profile.lon_end)
                        lat_wp = (profile.lat_start, profile.lat_end)
                    else:
                        lon_wp = (profile.lon_end, profile.lon_start)
                        lat_wp = (profile.lat_end, profile.lat_start)

                    journey_name = f"{profile.journey_name}_{direction}"

                    for run_idx in range(profile.n_runs):
                        # Sample random hyperparameters
                        experiment_seed = random.randint(0, 2**31 - 1)

                        config = {
                            "journey_config": {
                                "name": journey_name,
                                "lon_waypoints": lon_wp,
                                "lat_waypoints": lat_wp,
                                "time_start": time_start,
                                "speed_knots": speed,
                                "time_resolution_hours": profile.time_resolution_hours,
                            },
                            "forcing_config": {
                                "currents_path": profile.currents_path,
                                "waves_path": profile.waves_path,
                                "winds_path": profile.winds_path,
                                "engine": profile.engine,
                            },
                            "hyper_params": {
                                "random_seed": experiment_seed,
                                "population_size": sample_value(profile.population_sizes),
                                "generations": sample_value(profile.generations),
                                "offspring_size": sample_value(profile.population_sizes),
                                "selection_quantile": sample_value(profile.selection_quantiles),
                                "selection_acceptance_rate": sample_value(profile.selection_acceptance_rates),
                                "selection_acceptance_rate_warmup": profile.selection_acceptance_rate_warmup,
                                "mutation_width_fraction": profile.mutation_width_fraction,
                                "mutation_displacement_fraction": profile.mutation_displacement_fraction,
                                "mutation_width_fraction_warmup": sample_value(profile.mutation_width_fractions_warmup),
                                "mutation_displacement_fraction_warmup": sample_value(profile.mutation_displacement_fractions_warmup),
                                "mutation_iterations": sample_value(profile.mutation_iterations),
                                "crossover_strategy": sample_value(profile.crossover_strategies),
                                "crossover_rounds": sample_value(profile.crossover_rounds),
                                "hazard_penalty_multiplier": sample_value(profile.hazard_penalty_multipliers),
                                "num_elites": profile.num_elites,
                                "gd_iterations": sample_value(profile.gd_iterations_options),
                                "learning_rate_time": profile.learning_rate_time,
                                "learning_rate_space": profile.learning_rate_space,
                                "time_increment": profile.time_increment,
                                "distance_increment": profile.distance_increment,
                                "enable_adaptation": sample_value(profile.enable_adaptation_options),
                                "adaptation_scale_W": sample_value(profile.adaptation_scale_W_options),
                                "adaptation_scale_D": sample_value(profile.adaptation_scale_D_options),
                                "executor_type": profile.executor_type,
                                "num_workers": profile.num_workers,
                            },
                            "timeout": profile.task_timeout,
                            "metadata": {
                                "realisation": realisation,
                                "month": month,
                                "speed_knots": speed,
                                "direction": direction,
                                "run_idx": run_idx,
                            },
                        }
                        configs.append(config)

    return configs


def make_result_key(config: dict[str, Any]) -> str:
    """Generate a unique key for a result.

    Format matches the previous Redis-based approach for compatibility
    with existing analysis scripts.
    """
    meta = config["metadata"]
    journey = config["journey_config"]
    hyper = config["hyper_params"]

    return (
        f"result:{journey['name']}"
        f":r{meta['realisation']}"
        f":m{meta['month']:02d}"
        f":s{meta['speed_knots']}"
        f":run{meta['run_idx']}"
        f":seed{hyper['random_seed']}"
    )


def save_results(results: dict[str, bytes], output_path: Path) -> None:
    """Save results to msgpack file.

    Parameters
    ----------
    results : dict[str, bytes]
        Mapping from result keys to msgpack-serialized results
    output_path : Path
        Path to save the results file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        msgpack.pack(results, f)
    logger.info(f"Saved {len(results)} results to {output_path}")


@click.command()
@click.option(
    "--profile",
    type=click.Choice(list(PROFILES.keys())),
    default="test",
    help="Profile to use for the tuning run",
)
@click.option(
    "--executor",
    type=click.Choice(["local", "slurm"]),
    default="local",
    help="Executor to use (local for testing, slurm for production)",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducible experiment generation",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Generate configs but don't run experiments",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Override output file path",
)
def main(
    profile: str,
    executor: str,
    seed: int | None,
    dry_run: bool,
    output: str | None,
) -> None:
    """Run hyperparameter tuning experiments."""
    # Load profile
    prof = PROFILES[profile]
    logger.info(f"Using profile: {profile}")
    logger.info(f"Executor: {executor}")

    # Generate experiment configurations
    logger.info("Generating experiment configurations...")
    configs = generate_experiment_configs(prof, seed=seed)
    logger.info(f"Generated {len(configs)} experiment configurations")

    if dry_run:
        logger.info("Dry run - not submitting experiments")
        logger.info(f"First config: {configs[0]}")
        return

    # Configure output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output:
        output_path = Path(output)
    else:
        output_path = Path(prof.output_dir) / f"{prof.output_prefix}_{timestamp}.msgpack"

    # Configure and load Parsl
    logger.info("Configuring Parsl...")
    parsl_config = get_parsl_config(prof, executor=executor)
    parsl.load(parsl_config)

    try:
        # Submit all tasks
        logger.info("Submitting experiments...")
        futures: list[tuple[str, DataFuture]] = []
        for config in tqdm(configs, desc="Submitting"):
            key = make_result_key(config)
            future = run_single_experiment(
                journey_config=config["journey_config"],
                forcing_config=config["forcing_config"],
                hyper_params=config["hyper_params"],
            )
            futures.append((key, future))

        # Collect results
        logger.info("Collecting results...")
        results: dict[str, bytes] = {}
        failed = 0
        for key, future in tqdm(futures, desc="Collecting"):
            try:
                result = future.result()
                results[key] = result
            except Exception as e:
                logger.warning(f"Task {key} failed: {e}")
                failed += 1

        logger.info(f"Completed: {len(results)} successful, {failed} failed")

        # Save results
        save_results(results, output_path)

    finally:
        # Clean up Parsl
        parsl.clear()


if __name__ == "__main__":
    main()
