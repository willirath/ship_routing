#!/usr/bin/env python
"""Helper to load results from extracted msgpack file."""

import msgpack
import pandas as pd
from pathlib import Path
from ship_routing.app import RoutingResult


def load_results(msgpack_file: str, first_n: int = None) -> dict[str, RoutingResult]:
    """Load all results from msgpack file.

    Parameters
    ----------
    msgpack_file : str
        Path to msgpack file containing serialized results.
    first_n: int, optional
        Only extract first_n results.

    Returns
    -------
    dict[str, RoutingResult]
        Dictionary mapping result keys to RoutingResult objects.
    """
    with open(msgpack_file, "rb") as f:
        raw_results = msgpack.unpack(f, raw=False)

    return {
        key: RoutingResult.from_msgpack(value) for key, value in list(raw_results.items())[:first_n]
    }


def results_to_dataframe(msgpack_file: str) -> pd.DataFrame:
    """Convert results to analysis-ready DataFrame.

    Extracts hyperparameter configuration and key metrics from each result
    into a single DataFrame row for easy analysis.

    Parameters
    ----------
    msgpack_file : str
        Path to msgpack file containing serialized results.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns for hyperparameters and metrics.
    """
    results = load_results(msgpack_file)

    records = []
    for key, result in results.items():
        if result.logs is None or result.elite_population is None:
            continue

        record = {"filename": key}

        # Extract hyperparameters from config
        if "hyper" in result.logs.config:
            hyper = result.logs.config["hyper"]
            for param_name, param_value in hyper.items():
                record[f"hyper_{param_name}"] = param_value

        # Extract elite metrics
        for n, member in enumerate(result.elite_population.members):
            record[f"elite_{n}_cost"] = member.cost
            if result.seed_member:
                record[f"elite_{n}_cost_relative"] = (
                    member.cost / result.seed_member.cost
                )

        # Extract runtime from logs
        df_logs = result.logs.to_dataframe()
        if not df_logs.empty:
            record["runtime"] = (
                df_logs["timestamp"].max() - df_logs["timestamp"].min()
            ).total_seconds()

        records.append(record)

    df = pd.DataFrame(records)
    return df
