#!/usr/bin/env python
"""Parsl-based cost analysis runs for cost_001.

Runs all 48 journey configurations with default hyperparameters.
Each submission produces 5 replicas with independent seeds.

Usage:
    # Local smoke test (2 journeys, 1 replica)
    python run_cost.py --submission-id 1 --execution local-small --quick

    # Full production on SLURM
    python run_cost.py --submission-id 1 --execution nesh-prod-40

    # Second submission (independent seeds)
    python run_cost.py --submission-id 2 --execution nesh-prod-40

    # Dry run
    python run_cost.py --submission-id 1 --execution local-small --dry-run
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))

from experiment_params import (
    ALL_JOURNEYS,
    make_production_configs,
)
from ship_routing.htc import (
    EXECUTION_CONFIGS,
    run_tuning_sweep as run_sweep,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

N_REPLICAS = 5


@click.command()
@click.option(
    "--submission-id",
    type=int,
    required=True,
    help="Submission ID (1, 2, ...). Different IDs produce independent seed streams.",
)
@click.option(
    "--execution",
    type=click.Choice(list(EXECUTION_CONFIGS.keys())),
    default="local-small",
    help="Execution environment",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Generate configs but don't run",
)
@click.option(
    "--quick",
    is_flag=True,
    help="Smoke test: 2 journeys, 1 replica",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Override output file path",
)
def main(
    submission_id: int,
    execution: str,
    dry_run: bool,
    quick: bool,
    output: str | None,
) -> None:
    """Run cost analysis experiments with fixed hyperparameters."""
    journeys = ALL_JOURNEYS[:2] if quick else ALL_JOURNEYS
    n_replicas = 1 if quick else N_REPLICAS

    configs = make_production_configs(
        journeys=journeys,
        n_replicas=n_replicas,
        submission_id=submission_id,
    )

    logger.info(
        f"Generated {len(configs)} configs "
        f"({len(journeys)} journeys x {n_replicas} replicas, "
        f"submission {submission_id})"
    )

    if dry_run:
        logger.info("Dry run — not executing")
        for i, c in enumerate(configs[:5]):
            logger.info(
                f"  [{i}] {c.journey.name} {c.journey.time_start} "
                f"{c.journey.speed_knots}kn seed={c.hyper.random_seed}"
            )
        if len(configs) > 5:
            logger.info(f"  ... ({len(configs) - 5} more)")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output:
        output_path = Path(output)
    else:
        output_path = (
            Path("results") / f"cost_sub{submission_id:02d}_{timestamp}.msgpack"
        )

    run_dir = (Path("runinfo") / f"cost_sub{submission_id:02d}_{timestamp}").resolve()

    logger.info(f"Running with execution profile: {execution}")
    logger.info(f"Output: {output_path}")
    run_sweep(configs, execution, output_path, run_dir=run_dir)


if __name__ == "__main__":
    main()
