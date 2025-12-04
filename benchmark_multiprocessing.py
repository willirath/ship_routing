#!/usr/bin/env python
"""Benchmark multiprocessing performance for ship routing.

Quick performance test comparing sequential (num_workers=0) vs parallel
(num_workers=2) execution using simplified configuration without ocean
forcing data. Runs 3 repetitions of each configuration and reports average
timing, speedup, and parallel efficiency.

Each repetition is run as a fresh Python process to avoid JIT/caching effects.

Usage:
    pixi run python benchmark_multiprocessing.py
"""

import sys
import time
import subprocess
import pandas as pd
from pathlib import Path

from src.ship_routing.app.routing import RoutingApp
from src.ship_routing.app.config import (
    RoutingConfig,
    HyperParams,
    JourneyConfig,
    ForcingConfig,
)

# Benchmark configuration
WORKER_COUNTS = [0, 2, 4, 8]  # Test scaling
REPETITIONS = 2

BENCHMARK_CONFIG = {
    "population_size": 16,
    "generations": 2,
    "crossover_rounds": 1,
    "mutation_iterations": 2,
    "gd_iterations": 0,
    "num_elites": 2,
    "random_seed": 42,
}

JOURNEY_CONFIG = {
    "lon_waypoints": (-80.5, -62.0),
    "lat_waypoints": (30.0, 35.0),
    "time_start": "2021-01-01T00:00",
    "speed_knots": 7.0,
    "time_resolution_hours": 12.0,
}

# Real forcing data for CPU-bound benchmark
base = Path(__file__).resolve().parent / "doc" / "examples" / "data_large"

FORCING_CONFIG = {
    "currents_path": str(
        base
        / "cmems_mod_glo_phy_my_0.083deg_P1D-m_time_2021_lat_+10_+65_lon_-100_+010_uo-vo.zarr"
    ),
    "waves_path": str(
        base
        / "cmems_mod_glo_wav_my_0.2deg_PT3H-i_time_2021_lat_+10_+65_lon_-100_+010_VHM0-VMDR.zarr"
    ),
    "winds_path": str(
        base
        / "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_time_2021_lat_+10_+65_lon_-100_+010_eastward_wind-northward_wind.zarr"
    ),
    "engine": "zarr",
    "chunks": "auto",
    "load_eagerly": True,
}


def benchmark_single_run(num_workers: int) -> float:
    """Run a single routing optimization and return elapsed time.

    Parameters
    ----------
    num_workers : int
        Number of worker processes (0 = sequential)

    Returns
    -------
    float
        Elapsed time in seconds
    """
    # Create configuration
    config = RoutingConfig(
        journey=JourneyConfig(**JOURNEY_CONFIG),
        forcing=ForcingConfig(**FORCING_CONFIG),
        hyper=HyperParams(num_workers=num_workers, **BENCHMARK_CONFIG),
    )

    # Run routing and time it
    app = RoutingApp(config)
    start = time.perf_counter()
    result = app.run()
    elapsed = time.perf_counter() - start

    return elapsed


def run_benchmark() -> pd.DataFrame:
    """Run full benchmark with multiple configurations and repetitions.

    Each repetition is run as a fresh subprocess to avoid JIT/caching effects.

    Returns
    -------
    pd.DataFrame
        Results with columns: num_workers, repetition, elapsed_time
    """
    print("=" * 60)
    print("Multiprocessing Benchmark")
    print("=" * 60)
    print(
        f"Configuration: population_size={BENCHMARK_CONFIG['population_size']}, "
        f"generations={BENCHMARK_CONFIG['generations']}"
    )
    print(f"Repetitions: {REPETITIONS}")
    print()

    results = []

    for num_workers in WORKER_COUNTS:
        for rep in range(1, REPETITIONS + 1):
            print(
                f"Running: num_workers={num_workers}, rep {rep}/{REPETITIONS}...",
                end=" ",
                flush=True,
            )

            try:
                # Run as fresh subprocess to avoid JIT/caching effects
                result = subprocess.run(
                    [sys.executable, __file__, "--single-run", str(num_workers)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                elapsed = float(result.stdout.strip())
                print(f"{elapsed:.2f}s")
                results.append(
                    {
                        "num_workers": num_workers,
                        "repetition": rep,
                        "elapsed_time": elapsed,
                    }
                )
            except subprocess.CalledProcessError as e:
                print(f"FAILED: {e.stderr}")
                raise
            except ValueError as e:
                print(f"FAILED to parse output: {result.stdout}")
                raise

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compute statistics and speedup metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Raw benchmark results

    Returns
    -------
    pd.DataFrame
        Summary statistics with speedup and efficiency
    """
    # Group by num_workers and compute statistics
    summary = (
        df.groupby("num_workers")["elapsed_time"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary.columns = ["num_workers", "mean_time", "std_time", "n_runs"]

    # Calculate speedup (relative to num_workers=0 baseline)
    baseline_time = summary[summary["num_workers"] == 0]["mean_time"].iloc[0]

    summary["speedup"] = baseline_time / summary["mean_time"]

    # Calculate parallel efficiency (speedup / num_workers)
    # For num_workers=0, efficiency is defined as 1.0
    summary["efficiency"] = summary.apply(
        lambda row: (
            1.0 if row["num_workers"] == 0 else row["speedup"] / row["num_workers"]
        ),
        axis=1,
    )

    return summary


def display_results(summary: pd.DataFrame, raw_df: pd.DataFrame):
    """Display formatted benchmark results.

    Parameters
    ----------
    summary : pd.DataFrame
        Summary statistics
    raw_df : pd.DataFrame
        Raw timing data
    """
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print()

    # Format summary table
    print(
        summary[
            ["num_workers", "mean_time", "std_time", "speedup", "efficiency"]
        ].to_string(index=False, float_format="%.2f")
    )
    print()

    # Highlight key findings (use the non-zero worker count)
    parallel_workers = summary[summary["num_workers"] != 0]["num_workers"].iloc[0]
    speedup = summary[summary["num_workers"] == parallel_workers]["speedup"].iloc[0]
    efficiency = summary[summary["num_workers"] == parallel_workers]["efficiency"].iloc[
        0
    ]

    print(f"Parallel speedup ({int(parallel_workers)} workers): {speedup:.2f}x")
    print(f"Parallel efficiency: {efficiency:.1%}")
    print()

    # Save results
    output_file = "benchmark_results.csv"
    raw_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")

    summary_file = "benchmark_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    # Handle single-run mode (called by subprocess)
    if len(sys.argv) == 3 and sys.argv[1] == "--single-run":
        num_workers = int(sys.argv[2])
        elapsed = benchmark_single_run(num_workers)
        # Print only the timing (will be captured by parent process)
        print(elapsed)
        sys.exit(0)

    # Run full benchmark suite
    raw_results = run_benchmark()

    # Analyze results
    summary = analyze_results(raw_results)

    # Display and save
    display_results(summary, raw_results)
