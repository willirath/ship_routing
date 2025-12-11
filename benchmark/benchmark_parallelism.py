#!/usr/bin/env python
"""Benchmark multithreading vs multiprocessing performance for ship routing.

Quick performance test comparing executor types (process vs thread) with the
same worker count. Runs repetitions of each configuration and reports average
timing, speedup, and comparison between executor types.

Each repetition is run as a fresh Python process to avoid JIT/caching effects.

Usage:
    pixi run python benchmark/benchmark_parallelism.py
"""

import sys
import time
import subprocess
import pandas as pd
from pathlib import Path

from ship_routing.app import RoutingApp, build_config

# Benchmark configuration
# Test configurations: (num_workers, executor_type)
TEST_CONFIGS = [
    (0, "sequential"),  # Sequential baseline
    (1, "process"),  # Single process worker
    (1, "thread"),  # Single thread worker
    (4, "process"),  # Multi-process
    (4, "thread"),  # Multi-thread
]
REPETITIONS = 1  # Reduced for quick testing

BENCHMARK_CONFIG = {
    "population_size": 8,
    "generations": 1,
    "offspring_size": 8,
    "crossover_rounds": 1,
    "mutation_iterations": 2,
    "gd_iterations": 0,
    "num_elites": 2,
    "random_seed": 42,
}

# Use same config as working example_routing.py
JOURNEY_CONFIG = {
    "name": "Benchmark-Journey",
    "lon_waypoints": (-80.5, -62.0),
    "lat_waypoints": (30.0, 35.0),
    "time_start": "2021-01-01T00:00",
    "speed_knots": 7.0,
    "time_resolution_hours": 4.0,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
base = PROJECT_ROOT / "data" / "large"

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


def benchmark_single_run(num_workers: int, executor_type: str) -> float:
    """Run a single routing optimization and return elapsed time.

    Parameters
    ----------
    num_workers : int
        Number of workers (0 = sequential)
    executor_type : str
        Executor type ("process" or "thread")

    Returns
    -------
    float
        Elapsed time in seconds
    """
    # Create configuration using build_config
    config = build_config(
        **JOURNEY_CONFIG,
        **FORCING_CONFIG,
        num_workers=num_workers,
        executor_type=executor_type,
        **BENCHMARK_CONFIG,
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
        Results with columns: num_workers, executor_type, repetition, elapsed_time
    """
    print("=" * 60)
    print("Multithreading vs Multiprocessing Benchmark")
    print("=" * 60)
    print(
        f"Configuration: population_size={BENCHMARK_CONFIG['population_size']}, "
        f"generations={BENCHMARK_CONFIG['generations']}"
    )
    print(f"Repetitions: {REPETITIONS}")
    print()

    results = []

    for num_workers, executor_type in TEST_CONFIGS:
        for rep in range(1, REPETITIONS + 1):
            config_desc = (
                f"sequential"
                if num_workers == 0
                else f"{executor_type} workers={num_workers}"
            )
            print(
                f"Running: {config_desc}, rep {rep}/{REPETITIONS}...",
                end=" ",
                flush=True,
            )

            try:
                # Run as fresh subprocess to avoid JIT/caching effects
                result = subprocess.run(
                    [
                        sys.executable,
                        __file__,
                        "--single-run",
                        str(num_workers),
                        executor_type,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                elapsed = float(result.stdout.strip())
                print(f"{elapsed:.2f}s")
                results.append(
                    {
                        "num_workers": num_workers,
                        "executor_type": executor_type,
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
        Summary statistics with speedup comparison
    """
    # Group by num_workers and executor_type, compute statistics
    summary = (
        df.groupby(["num_workers", "executor_type"])["elapsed_time"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary.columns = [
        "num_workers",
        "executor_type",
        "mean_time",
        "std_time",
        "n_runs",
    ]

    # Calculate speedup relative to sequential (num_workers=0)
    sequential_time = summary[summary["num_workers"] == 0]["mean_time"].iloc[0]
    summary["speedup_vs_sequential"] = sequential_time / summary["mean_time"]

    # Create config label for display
    summary["config"] = summary.apply(
        lambda row: (
            "sequential"
            if row["num_workers"] == 0
            else f"{row['executor_type']}(n={row['num_workers']})"
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
        summary[["config", "mean_time", "std_time", "speedup_vs_sequential"]].to_string(
            index=False, float_format="%.2f"
        )
    )
    print()

    # Highlight key findings
    seq_time = summary[summary["num_workers"] == 0]["mean_time"].iloc[0]

    print(f"Sequential baseline: {seq_time:.2f}s")
    print()

    # Compare num_workers=1
    proc1 = summary[
        (summary["num_workers"] == 1) & (summary["executor_type"] == "process")
    ]
    thread1 = summary[
        (summary["num_workers"] == 1) & (summary["executor_type"] == "thread")
    ]

    if not proc1.empty and not thread1.empty:
        print("Single worker (n=1):")
        print(
            f"  Process: {proc1['mean_time'].iloc[0]:.2f}s (speedup: {proc1['speedup_vs_sequential'].iloc[0]:.2f}x)"
        )
        print(
            f"  Thread:  {thread1['mean_time'].iloc[0]:.2f}s (speedup: {thread1['speedup_vs_sequential'].iloc[0]:.2f}x)"
        )
        if thread1["mean_time"].iloc[0] < proc1["mean_time"].iloc[0]:
            print(
                f"  -> Thread is {proc1['mean_time'].iloc[0] / thread1['mean_time'].iloc[0]:.2f}x faster"
            )
        else:
            print(
                f"  -> Process is {thread1['mean_time'].iloc[0] / proc1['mean_time'].iloc[0]:.2f}x faster"
            )
        print()

    # Compare num_workers=4
    proc4 = summary[
        (summary["num_workers"] == 4) & (summary["executor_type"] == "process")
    ]
    thread4 = summary[
        (summary["num_workers"] == 4) & (summary["executor_type"] == "thread")
    ]

    if not proc4.empty and not thread4.empty:
        print("Multiple workers (n=4):")
        print(
            f"  Process: {proc4['mean_time'].iloc[0]:.2f}s (speedup: {proc4['speedup_vs_sequential'].iloc[0]:.2f}x)"
        )
        print(
            f"  Thread:  {thread4['mean_time'].iloc[0]:.2f}s (speedup: {thread4['speedup_vs_sequential'].iloc[0]:.2f}x)"
        )
        if thread4["mean_time"].iloc[0] < proc4["mean_time"].iloc[0]:
            print(
                f"  -> Thread is {proc4['mean_time'].iloc[0] / thread4['mean_time'].iloc[0]:.2f}x faster"
            )
        else:
            print(
                f"  -> Process is {thread4['mean_time'].iloc[0] / proc4['mean_time'].iloc[0]:.2f}x faster"
            )
        print()

    # Save results
    output_file = "benchmark_parallelism_results.csv"
    raw_df.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")

    summary_file = "benchmark_parallelism_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    # Handle single-run mode (called by subprocess)
    if len(sys.argv) == 4 and sys.argv[1] == "--single-run":
        num_workers = int(sys.argv[2])
        executor_type = sys.argv[3]
        elapsed = benchmark_single_run(num_workers, executor_type)
        # Print only the timing (will be captured by parent process)
        print(elapsed)
        sys.exit(0)

    # Run full benchmark suite
    raw_results = run_benchmark()

    # Analyze results
    summary = analyze_results(raw_results)

    # Display and save
    display_results(summary, raw_results)
