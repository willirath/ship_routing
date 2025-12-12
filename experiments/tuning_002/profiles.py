"""Profile configurations for hyperparameter tuning experiments.

Each profile defines the parameter space and SLURM configuration for a tuning run.
Fixed parameters (single-element lists) are used as-is; multi-element lists are
sampled randomly for each experiment.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class TuningProfile:
    """Configuration for a hyperparameter tuning run."""

    # Experiment scale
    n_runs: int  # Number of experiments per route direction
    n_realisations: int = 1  # Repetitions of the full sweep
    start_times: tuple[str, ...] = ("2021-01-01T00:00:00",)  # Journey start times
    speeds_knots: tuple[float, ...] = (10.0,)  # Ship speeds to test

    # Route definition (defaults: Atlantic crossing)
    lon_start: float = -80.5
    lon_end: float = -11.0
    lat_start: float = 30.0
    lat_end: float = 50.0
    journey_name: str = "Atlantic"

    # Forcing data paths (relative to experiment dir)
    currents_path: str = (
        "data_large/cmems_mod_glo_phy_my_0.083deg_P1D-m_time_2021_lat_+10_+65_lon_-100_+010_uo-vo.zarr"
    )
    waves_path: str = (
        "data_large/cmems_mod_glo_wav_my_0.2deg_PT3H-i_time_2021_lat_+10_+65_lon_-100_+010_VHM0-VMDR.zarr"
    )
    winds_path: str = (
        "data_large/cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_time_2021_lat_+10_+65_lon_-100_+010_eastward_wind-northward_wind.zarr"
    )
    engine: str = "zarr"

    # Fixed hyperparameters (single values)
    selection_acceptance_rate_warmup: float = 0.3
    mutation_width_fraction_warmup_single: float = 0.9
    mutation_displacement_fraction_warmup_single: float = 0.2
    mutation_width_fraction: float = 0.9
    mutation_displacement_fraction: float = 0.1
    num_elites: int = 2
    gd_iterations: int = 2
    learning_rate_time: float = 0.5
    learning_rate_space: float = 0.5
    time_increment: float = 1200.0
    distance_increment: float = 10000.0
    time_resolution_hours: float = 6.0

    # Random-sampled hyperparameters (lists)
    hazard_penalty_multipliers: tuple[float, ...] = (0.0, 100.0)
    selection_quantiles: tuple[float, ...] = (0.1, 0.25)
    selection_acceptance_rates: tuple[float, ...] = (0.0, 0.25)
    population_sizes: tuple[int, ...] = (128, 256)
    generations: tuple[int, ...] = (1, 2, 4)
    mutation_iterations: tuple[int, ...] = (1, 3)
    gd_iterations_options: tuple[int, ...] = (1, 2)
    crossover_strategies: tuple[str, ...] = ("minimal_cost", "random")
    crossover_rounds: tuple[int, ...] = (0, 1, 2)
    mutation_width_fractions_warmup: tuple[float, ...] = (0.99,)
    mutation_displacement_fractions_warmup: tuple[float, ...] = (0.1, 0.25)

    # Adaptation parameters
    enable_adaptation_options: tuple[bool, ...] = (True, False)
    adaptation_scale_W_options: tuple[float, ...] = (0.5, 0.8)
    adaptation_scale_D_options: tuple[float, ...] = (0.707, 0.894)

    # Parallelization within each experiment (passed to RoutingApp)
    # Use sequential execution - Parsl handles parallelism across experiments
    executor_type: Literal["process", "thread", "sequential"] = "sequential"
    num_workers: int = 1

    # SLURM configuration
    nodes_per_block: int = 10
    max_blocks: int = 100
    workers_per_node: int = 8
    walltime: str = "04:00:00"
    partition: str = "base"
    qos: str = "express"

    # Per-task timeout (seconds)
    task_timeout: int = 1000

    # Output
    output_dir: str = "results"
    output_prefix: str = "results"


# Pre-defined profiles
PROFILES: dict[str, TuningProfile] = {
    "test": TuningProfile(
        n_runs=20,
        n_realisations=1,
        start_times=("2021-01-01T00:00:00",),
        speeds_knots=(10.0,),
        # Smaller parameter space for testing
        population_sizes=(4, 8),
        generations=(1, 2),
        mutation_iterations=(1, 2),
        gd_iterations_options=(0, 1),
        crossover_rounds=(0, 1),
        mutation_width_fractions_warmup=(0.5, 0.9),
        # Smaller SLURM allocation
        nodes_per_block=2,
        max_blocks=5,
        workers_per_node=4,
        walltime="01:00:00",
        task_timeout=300,
        output_prefix="results_test",
    ),
    "production": TuningProfile(
        n_runs=1000,
        n_realisations=10,
        start_times=(
            "2021-01-01T00:00:00",
            "2021-02-01T00:00:00",
            "2021-03-01T00:00:00",
            "2021-04-01T00:00:00",
            "2021-05-01T00:00:00",
            "2021-06-01T00:00:00",
            "2021-07-01T00:00:00",
            "2021-08-01T00:00:00",
            "2021-09-01T00:00:00",
            "2021-10-01T00:00:00",
            "2021-11-01T00:00:00",
            "2021-12-01T00:00:00",
        ),
        speeds_knots=(8.0, 10.0, 12.0),
        # Full parameter space (defaults from dataclass)
        nodes_per_block=10,
        max_blocks=100,
        workers_per_node=8,
        walltime="04:00:00",
        task_timeout=1000,
        output_prefix="results",
    ),
    "quick": TuningProfile(
        n_runs=5,
        n_realisations=1,
        start_times=("2021-01-01T00:00:00",),
        speeds_knots=(10.0,),
        population_sizes=(4,),
        generations=(1,),
        mutation_iterations=(1,),
        gd_iterations_options=(1,),
        crossover_rounds=(0,),
        crossover_strategies=("random",),
        enable_adaptation_options=(False,),
        nodes_per_block=1,
        max_blocks=1,
        workers_per_node=2,
        walltime="00:30:00",
        task_timeout=120,
        output_prefix="results_quick",
    ),
}
