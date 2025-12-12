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
import sys
from datetime import datetime
from pathlib import Path

import click
import msgpack
import parsl
from parsl.app.futures import DataFuture
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from parsl_config import get_parsl_config
from experiment_params import EXPERIMENTS
from execution_config import EXECUTION_CONFIGS
from ship_routing.app.config import RoutingConfig
from ship_routing.app.config_factory import sample_routing_configs
from ship_routing.app.parsl import run_single_experiment
from ship_routing.app.routing import RoutingResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_routing_configs(
    experiment_name: str,
    seed: int | None = None,
) -> list[RoutingConfig]:
    """Generate routing configurations using the factory.

    Parameters
    ----------
    experiment_name : str
        Name of experiment (e.g., 'test', 'production', 'quick')
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    list[RoutingConfig]
        Randomly sampled routing configurations
    """
    exp_config = EXPERIMENTS[experiment_name]
    return sample_routing_configs(
        param_space=exp_config["param_space"],
        n_samples=exp_config["n_samples"],
        seed=seed,
    )


def make_result_key(config_idx: int, config: RoutingConfig) -> str:
    """Generate a unique key for a result.

    Parameters
    ----------
    config_idx : int
        Sequential index of this config
    config : RoutingConfig
        The routing configuration

    Returns
    -------
    str
        Unique result key
    """
    return f"result:{config_idx:04d}:seed{config.hyper.random_seed}"


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
    type=click.Choice(list(EXPERIMENTS.keys())),
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
    # Load experiment config
    exp_config = EXPERIMENTS[experiment]

    # Generate routing configurations
    configs = generate_routing_configs(experiment, seed=seed)

    if dry_run:
        logger.info(f"Would generate {len(configs)} configs")
        return

    # Configure output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output:
        output_path = Path(output)
    else:
        output_path = (
            Path("results") / f"{exp_config['output_prefix']}_{timestamp}.msgpack"
        )

    # Configure and load Parsl
    parsl_config = get_parsl_config(execution)
    parsl.load(parsl_config)

    try:
        # Submit all tasks
        futures: list[tuple[str, DataFuture]] = []
        for i, config in enumerate(tqdm(configs, desc="Submitting")):
            key = make_result_key(i, config)
            future = run_single_experiment(config=config)
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
                logger.error(f"Failed {key}: {e}")
                failed += 1

        # Save results
        logger.info(f"Completed: {len(results)} successful, {failed} failed")
        save_results(results, output_path)

    finally:
        # Clean up Parsl
        parsl.clear()


if __name__ == "__main__":
    main()
