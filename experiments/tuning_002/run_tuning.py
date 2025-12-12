#!/usr/bin/env python
"""Parsl-based hyperparameter tuning for ship routing.

This script orchestrates large-scale hyperparameter tuning experiments using
Parsl for task distribution. It replaces the previous bash-based job scripts
with a cleaner Python-based approach.

Usage:
    # Local testing with small resources
    python run_tuning.py --experiment test --execution local-small

    # Production on SLURM
    python run_tuning.py --experiment production --execution nesh-prod

    # Quick smoke test
    python run_tuning.py --experiment quick --execution local-small

    # Dry run to see experiment count
    python run_tuning.py --experiment test --execution local-small --dry-run
"""

from __future__ import annotations

import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import msgpack
import parsl
from parsl.app.futures import DataFuture
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from parsl_config import get_parsl_config
from experiment_params import EXPERIMENT_PARAMS, ExperimentParams
from execution_config import EXECUTION_CONFIGS
from ship_routing.app.config import (
    ForcingConfig,
    HyperParams,
    JourneyConfig,
    RoutingConfig,
)
from ship_routing.app.parsl import run_single_experiment
from ship_routing.app.routing import RoutingResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Container for experiment configuration and metadata."""

    config: RoutingConfig
    metadata: dict[str, Any]


def sample_value(options: tuple | list) -> Any:
    """Sample a value from options (returns single value if only one option)."""
    if len(options) == 1:
        return options[0]
    return random.choice(options)


def generate_experiment_configs(
    params: ExperimentParams,
    seed: int | None = None,
) -> list[ExperimentConfig]:
    """Generate all experiment configurations for a tuning run.

    Parameters
    ----------
    params : ExperimentParams
        Experiment parameters defining parameter space and experiment scale
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    list[ExperimentConfig]
        List of experiment configurations with RoutingConfig and metadata.
    """
    if seed is not None:
        random.seed(seed)

    configs = []

    # Outer loops: realisations x start_times x speeds
    for realisation in range(params.n_realisations):
        for time_start in params.start_times:
            for speed in params.speeds_knots:
                # Generate n_runs experiments for forward and backward routes
                for direction in ["forward", "backward"]:
                    if direction == "forward":
                        lon_wp = (params.lon_start, params.lon_end)
                        lat_wp = (params.lat_start, params.lat_end)
                    else:
                        lon_wp = (params.lon_end, params.lon_start)
                        lat_wp = (params.lat_end, params.lat_start)

                    journey_name = f"{params.journey_name}_{direction}"

                    for run_idx in range(params.n_runs):
                        # Sample random hyperparameters
                        experiment_seed = random.randint(0, 2**31 - 1)

                        routing_config = RoutingConfig(
                            journey=JourneyConfig(
                                name=journey_name,
                                lon_waypoints=lon_wp,
                                lat_waypoints=lat_wp,
                                time_start=time_start,
                                speed_knots=speed,
                                time_resolution_hours=params.time_resolution_hours,
                            ),
                            forcing=ForcingConfig(
                                currents_path=params.currents_path,
                                waves_path=params.waves_path,
                                winds_path=params.winds_path,
                                engine=params.engine,
                            ),
                            hyper=HyperParams(
                                random_seed=experiment_seed,
                                population_size=sample_value(params.population_sizes),
                                generations=sample_value(params.generations),
                                offspring_size=sample_value(params.population_sizes),
                                selection_quantile=sample_value(
                                    params.selection_quantiles
                                ),
                                selection_acceptance_rate=sample_value(
                                    params.selection_acceptance_rates
                                ),
                                selection_acceptance_rate_warmup=params.selection_acceptance_rate_warmup,
                                mutation_width_fraction=params.mutation_width_fraction,
                                mutation_displacement_fraction=params.mutation_displacement_fraction,
                                mutation_width_fraction_warmup=sample_value(
                                    params.mutation_width_fractions_warmup
                                ),
                                mutation_displacement_fraction_warmup=sample_value(
                                    params.mutation_displacement_fractions_warmup
                                ),
                                mutation_iterations=sample_value(
                                    params.mutation_iterations
                                ),
                                crossover_strategy=sample_value(
                                    params.crossover_strategies
                                ),
                                crossover_rounds=sample_value(params.crossover_rounds),
                                hazard_penalty_multiplier=sample_value(
                                    params.hazard_penalty_multipliers
                                ),
                                num_elites=params.num_elites,
                                gd_iterations=sample_value(
                                    params.gd_iterations_options
                                ),
                                learning_rate_time=params.learning_rate_time,
                                learning_rate_space=params.learning_rate_space,
                                time_increment=params.time_increment,
                                distance_increment=params.distance_increment,
                                enable_adaptation=sample_value(
                                    params.enable_adaptation_options
                                ),
                                adaptation_scale_W=sample_value(
                                    params.adaptation_scale_W_options
                                ),
                                adaptation_scale_D=sample_value(
                                    params.adaptation_scale_D_options
                                ),
                                executor_type=params.executor_type,
                                num_workers=params.num_workers,
                            ),
                        )

                        metadata = {
                            "realisation": realisation,
                            "time_start": time_start,
                            "speed_knots": speed,
                            "direction": direction,
                            "run_idx": run_idx,
                        }

                        configs.append(
                            ExperimentConfig(config=routing_config, metadata=metadata)
                        )

    return configs


def make_result_key(exp_config: ExperimentConfig) -> str:
    """Generate a unique key for a result.

    New format with explicit time and clearer labels for better readability.
    """
    meta = exp_config.metadata
    journey = exp_config.config.journey
    hyper = exp_config.config.hyper

    return (
        f"result:{journey.name}"
        f":{meta['time_start']}"
        f":spd{meta['speed_knots']}"
        f":run{meta['run_idx']}"
        f":real{meta['realisation']}"
        f":seed{hyper.random_seed}"
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
    "--experiment",
    type=click.Choice(list(EXPERIMENT_PARAMS.keys())),
    default="test",
    help="Experiment configuration to use",
)
@click.option(
    "--execution",
    type=click.Choice(list(EXECUTION_CONFIGS.keys())),
    default="local-small",
    help="Execution environment (local-small, local-large, nesh-test, nesh-prod)",
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
    experiment: str,
    execution: str,
    seed: int | None,
    dry_run: bool,
    output: str | None,
) -> None:
    """Run hyperparameter tuning experiments."""
    # Load experiment and execution configs
    exp_params = EXPERIMENT_PARAMS[experiment]

    # Generate experiment configurations
    configs = generate_experiment_configs(exp_params, seed=seed)

    if dry_run:
        return

    # Configure output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output:
        output_path = Path(output)
    else:
        output_path = (
            Path(exp_params.output_dir)
            / f"{exp_params.output_prefix}_{timestamp}.msgpack"
        )

    # Configure and load Parsl
    parsl_config = get_parsl_config(execution)
    parsl.load(parsl_config)

    try:
        # Submit all tasks
        futures: list[tuple[str, DataFuture]] = []
        for exp_config in tqdm(configs, desc="Submitting"):
            key = make_result_key(exp_config)
            future = run_single_experiment(config=exp_config.config)
            futures.append((key, future))

        # Collect results and serialize to msgpack
        results: dict[str, bytes] = {}
        failed = 0
        for key, future in tqdm(futures, desc="Collecting"):
            try:
                result: RoutingResult = future.result()
                # Serialize to msgpack for disk storage (notebook compatibility)
                results[key] = result.to_msgpack()
            except Exception as e:
                failed += 1

        # Save results
        save_results(results, output_path)

    finally:
        # Clean up Parsl
        parsl.clear()


if __name__ == "__main__":
    main()
