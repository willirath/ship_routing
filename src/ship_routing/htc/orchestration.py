"""Shared orchestration logic for HTC parameter sweep experiments.

This module provides common functions for running parameter sweeps using Parsl,
collecting results, and saving them to msgpack files.
"""

from __future__ import annotations

from concurrent.futures import as_completed
import hashlib
import logging
from datetime import datetime
from pathlib import Path

import msgpack
import parsl
from parsl.app.errors import AppTimeout
from tqdm import tqdm

from ship_routing.app.config import RoutingConfig
from ship_routing.app.parsl import run_single_experiment
from ship_routing.app.routing import RoutingResult
from ship_routing.htc.parsl_configs import get_execution_and_parsl_config

logger = logging.getLogger(__name__)


def make_result_key(config_idx: int, config: RoutingConfig) -> str:
    """Generate a unique key for a result.

    Includes a hash of config content to ensure uniqueness when combining
    results from different sweep runs.

    Parameters
    ----------
    config_idx : int
        Sequential index of this config
    config : RoutingConfig
        The routing configuration

    Returns
    -------
    str
        Unique result key with format: result:{idx}:seed{seed}:hash{hash}
    """
    # Hash config just to be on the safe side for uniquenes of keys
    config_hash = hashlib.sha256(repr(config).encode()).hexdigest()[:8]
    return f"result:{config_idx:04d}:seed{config.hyper.random_seed}:hash{config_hash}"


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


def run_tuning_sweep(
    configs: list[RoutingConfig],
    execution_name: str,
    output_path: Path | str,
    run_dir: str | Path | None = None,
) -> dict[str, bytes]:
    """Run parameter sweep and collect results.

    Parameters
    ----------
    configs : list[RoutingConfig]
        List of routing configurations to evaluate
    execution_name : str
        Name of execution config (e.g., "local-small", "nesh-prod")
    output_path : Path | str
        Path to save results msgpack file
    run_dir : str | Path | None, optional
        Path to Parsl run directory (default: "runinfo")

    Returns
    -------
    dict[str, bytes]
        Mapping from result keys to msgpack-serialized results

    Raises
    ------
    ValueError
        If execution_name is not found in EXECUTION_CONFIGS
    """
    output_path = Path(output_path)

    # Get both configs in one call (eliminates redundant lookup)
    execution_config, parsl_config = get_execution_and_parsl_config(
        execution_name, run_dir=run_dir
    )

    # Configure and load Parsl
    parsl.load(parsl_config)

    try:
        # Submit all tasks
        futures = []
        for i, config in enumerate(tqdm(configs, desc="Submitting")):
            key = make_result_key(i, config)
            future = run_single_experiment(
                config=config, walltime=execution_config.task_timeout
            )
            futures.append((key, future))

        # Collect results and serialize to msgpack
        results: dict[str, bytes] = {}
        failed = 0
        timeouts = 0

        # Process results as they complete (not in submission order)
        futures_map = {future: key for key, future in futures}
        for future in tqdm(
            as_completed(futures_map.keys()), total=len(futures), desc="Collecting"
        ):
            key = futures_map[future]
            try:
                result: RoutingResult = future.result()
                # Serialize to msgpack for disk storage (notebook compatibility)
                results[key] = result.to_msgpack()
            except AppTimeout as e:
                logger.error(
                    f"Timeout {key}: exceeded {execution_config.task_timeout}s ({type(e).__name__})"
                )
                timeouts += 1
                failed += 1
            except Exception as e:
                logger.error(f"Failed {key}: {e}")
                failed += 1

        # Save results
        logger.info(
            f"Completed: {len(results)} successful, {failed} failed "
            f"({timeouts} timeouts)"
        )
        save_results(results, output_path)

        return results

    finally:
        # Clean up Parsl (wait for cleanup to complete)
        parsl.dfk().cleanup()
        parsl.clear()
