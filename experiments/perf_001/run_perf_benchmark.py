#!/usr/bin/env python
"""Performance benchmark for perf_001.

Runs a single (case, replica) combination and saves the result to msgpack.
Designed to be dispatched in parallel by SLURM job files via xargs/srun.

Usage:
    # Single run (dispatched by job file)
    python run_perf_benchmark.py --num-workers 4 --case-index 2 --replica-index 0

    # Quick smoke test (all cases, 1 replica, sequential)
    python run_perf_benchmark.py --num-workers 1 --all

    # List all (case_index, replica_index) pairs
    python run_perf_benchmark.py --list
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import click
import msgpack

# Import shared experiment params from cost_001
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cost_001"))

from experiment_params import (
    REPRESENTATIVE_JOURNEYS,
    make_production_configs,
)
from ship_routing.app.routing import RoutingApp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

N_REPLICAS = 3
N_CASES = len(REPRESENTATIVE_JOURNEYS)  # 8

# Seed offset to avoid overlap with cost run seeds
PERF_SUBMISSION_OFFSET = 1000


def _get_config(case_index, replica_index, num_workers, submission_id=1):
    """Build a single RoutingConfig for the given case and replica."""
    journey = REPRESENTATIVE_JOURNEYS[case_index]
    configs = make_production_configs(
        journeys=[journey],
        n_replicas=N_REPLICAS,
        submission_id=PERF_SUBMISSION_OFFSET + submission_id,
        executor_type="process" if num_workers > 1 else "sequential",
        num_workers=num_workers,
    )
    return configs[replica_index]


def _run_single(config, num_workers, case_index, replica_index):
    """Run one benchmark and return (key, msgpack_bytes) or None on failure."""
    journey = config.journey
    label = (
        f"c{case_index}r{replica_index} "
        f"{journey.name} {journey.time_start[:7]} "
        f"{journey.speed_knots}kn w{num_workers}"
    )
    logger.info(f"START {label} seed={config.hyper.random_seed}")

    t0 = time.perf_counter()
    try:
        result = RoutingApp(config).run()
        elapsed = time.perf_counter() - t0
        best_cost = result.elite_population.members[0].cost
        logger.info(f"DONE  {label} {elapsed:.1f}s cost={best_cost:.3e}")

        key = (
            f"perf:c{case_index:02d}:r{replica_index:02d}"
            f":w{num_workers:02d}:seed{config.hyper.random_seed}"
        )
        return key, result.to_msgpack()
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error(f"FAIL  {label} {elapsed:.1f}s: {e}")
        return None


def _save_result(key, data_bytes, output_path):
    """Save a single result to msgpack file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        msgpack.pack({key: data_bytes}, f)
    logger.info(f"Saved to {output_path}")


@click.command()
@click.option("--num-workers", type=int, default=1, help="Workers for process executor")
@click.option("--case-index", type=int, default=None, help="Case index (0..7)")
@click.option("--replica-index", type=int, default=None, help="Replica index (0..2)")
@click.option("--submission-id", type=int, default=1, help="Submission ID (default: 1)")
@click.option("--output", type=click.Path(), default=None, help="Override output path")
@click.option(
    "--all", "run_all", is_flag=True, help="Run all cases and replicas sequentially"
)
@click.option(
    "--list", "list_tasks", is_flag=True, help="List all (case, replica) pairs and exit"
)
def main(
    num_workers: int,
    case_index: int | None,
    replica_index: int | None,
    submission_id: int,
    output: str | None,
    run_all: bool,
    list_tasks: bool,
) -> None:
    """Run performance benchmark."""
    if list_tasks:
        for ci in range(N_CASES):
            j = REPRESENTATIVE_JOURNEYS[ci]
            for ri in range(N_REPLICAS):
                print(f"{ci} {ri}  # {j.name} {j.time_start[:7]} {j.speed_knots}kn")
        return

    if run_all:
        # Sequential mode: run everything in one process (for testing)
        results = {}
        for ci in range(N_CASES):
            for ri in range(N_REPLICAS):
                config = _get_config(ci, ri, num_workers, submission_id)
                out = _run_single(config, num_workers, ci, ri)
                if out:
                    results[out[0]] = out[1]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = (
            Path(output)
            if output
            else (
                Path("results")
                / f"perf_w{num_workers:02d}_sub{submission_id:02d}_{timestamp}.msgpack"
            )
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            msgpack.pack(results, f)
        logger.info(f"Saved {len(results)} results to {out_path}")
        return

    # Single-run mode (dispatched by job file)
    if case_index is None or replica_index is None:
        raise click.UsageError("Provide --case-index and --replica-index, or use --all")

    config = _get_config(case_index, replica_index, num_workers, submission_id)
    result = _run_single(config, num_workers, case_index, replica_index)

    if result:
        key, data_bytes = result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = (
            Path(output)
            if output
            else (
                Path("results")
                / f"perf_w{num_workers:02d}_c{case_index:02d}_r{replica_index:02d}_{timestamp}.msgpack"
            )
        )
        _save_result(key, data_bytes, out_path)


if __name__ == "__main__":
    main()
