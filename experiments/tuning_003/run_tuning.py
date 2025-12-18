#!/usr/bin/env python
"""Parsl-based hyperparameter tuning for ship routing (tuning_003).

This script uses the ship_routing.htc infrastructure for parameter sweeps.
Key modification from tuning_002: offspring_size is computed as a ratio of population_size.

Usage:
    # Local testing with small resources
    python run_tuning.py --experiment quick --execution local-small

    # Production on SLURM
    python run_tuning.py --experiment ablation_baseline --execution nesh-prod-40

    # Dry run to see experiment count
    python run_tuning.py --experiment quick --execution local-small --dry-run
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import click

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from experiment_params import EXPERIMENTS
from ship_routing.htc import (
    EXECUTION_CONFIGS,
    run_tuning_sweep,
    sample_routing_configs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configure Parsl logging to reduce verbosity (suppress DEBUG messages)
logging.getLogger("parsl").setLevel(logging.INFO)


# TODO: Put into experiment_params?
def transform_offspring_size(sampled: dict) -> dict:
    """Compute offspring_size from offspring_ratio * population_size.

    This transform function is called after parameter sampling to compute
    derived parameters. For tuning_003, we compute offspring_size as:
        offspring_size = int(population_size * offspring_ratio)

    Parameters
    ----------
    sampled : dict
        Sampled parameters dict with 'hyper' containing offspring_ratio

    Returns
    -------
    dict
        Modified dict with offspring_size computed and offspring_ratio removed
    """
    hyper = sampled.get("hyper", {})
    if "offspring_ratio" in hyper:
        ratio = hyper.pop("offspring_ratio")
        population_size = hyper.get("population_size", 4)
        hyper["offspring_size"] = int(population_size * ratio)
    return sampled


@click.command()
@click.option(
    "--experiment",
    type=click.Choice(list(EXPERIMENTS.keys())),
    default="quick",
    help="Experiment configuration to use",
)
@click.option(
    "--execution",
    type=click.Choice(list(EXECUTION_CONFIGS.keys())),
    default="local-small",
    help="Execution environment (local-small, local-large, nesh-test, nesh-prod-40)",
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
    # Load experiment config
    exp_config = EXPERIMENTS[experiment]

    # Generate routing configurations with custom transform
    configs = sample_routing_configs(
        param_space=exp_config["param_space"],
        n_samples=exp_config["n_samples"],
        seed=seed,
        transform_fn=transform_offspring_size,
    )

    logger.info(f"Generated {len(configs)} configs for experiment '{experiment}'")

    if dry_run:
        logger.info("Dry run mode - not executing experiments")
        return

    # Configure output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output:
        output_path = Path(output)
    else:
        output_path = (
            Path("results") / f"{exp_config['output_prefix']}_{timestamp}.msgpack"
        )

    # Generate unique run directory to prevent race conditions (use absolute path)
    run_dir = (Path("runinfo") / f"{experiment}_{timestamp}").resolve()

    # Run the parameter sweep
    logger.info(f"Running sweep with execution profile: {execution}")
    logger.info(f"Parsl run directory: {run_dir}")
    run_tuning_sweep(configs, execution, output_path, run_dir=run_dir)


if __name__ == "__main__":
    main()
