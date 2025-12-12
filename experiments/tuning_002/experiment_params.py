"""Experiment parameter configurations for hyperparameter tuning.

Each experiment configuration defines the parameter space and experiment scale.
Fixed parameters (single-element tuples) are used as-is; multi-element tuples are
sampled randomly for each experiment.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ExperimentParams:
    """Configuration for a hyperparameter tuning experiment."""

    # Experiment scale
    n_runs: int  # Number of experiments per route direction
    n_realisations: int = 1  # Repetitions of the full sweep
    start_times: tuple[str, ...] = ("2021-01-01T00:00:00",)  # Journey start times
    speeds_knots: tuple[float, ...] = (10.0,)  # Ship speeds to test

    # Route definition (defaults: Atlantic crossing)
    # TODO: Use journey config here and then randomly sample from this one as well?
    # journeys = [JourneyConfig(), JourneyConfig(), ...]
    lon_start: float = -80.5
    lon_end: float = -11.0
    lat_start: float = 30.0
    lat_end: float = 50.0
    journey_name: str = "Atlantic"

    # Forcing data paths (relative to experiment dir)
    # TODO: Pathlib objects?
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

    # Random-sampled hyperparameters
    selection_acceptance_rate_warmup: float = (0.3,)
    mutation_width_fraction_warmup_single: float = (0.9,)
    mutation_displacement_fraction_warmup_single: float = (0.2,)
    mutation_width_fraction: float = (0.9,)
    mutation_displacement_fraction: float = (0.1,)
    num_elites: int = (2,)
    gd_iterations: int = (2,)
    learning_rate_time: float = (0.5,)
    learning_rate_space: float = (0.5,)
    time_increment: float = (1200.0,)
    distance_increment: float = (10000.0,)
    time_resolution_hours: float = (6.0,)
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

    # Random-sampled adaptation parameters
    enable_adaptation_options: tuple[bool, ...] = (True, False)
    adaptation_scale_W_options: tuple[float, ...] = (0.5, 0.8)
    adaptation_scale_D_options: tuple[float, ...] = (0.707, 0.894)

    # Intra-experiment parallelization (within each routing task)
    # Best practice: use "sequential" and let Parsl handle parallelism across experiments
    executor_type: Literal["process", "thread", "sequential"] = "sequential"
    num_workers: int = 1

    # Output
    # TODO: Pathlib objects?
    output_dir: str = "results"
    output_prefix: str = "results"


# Pre-defined experiment configurations
EXPERIMENT_PARAMS: dict[str, ExperimentParams] = {
    "test": ExperimentParams(
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
        output_prefix="results_test",
    ),
    "production": ExperimentParams(
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
        # Full parameter space (defaults from dataclass)  # TODO: What's this comment?
        output_prefix="results",
    ),
    "quick": ExperimentParams(
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
        output_prefix="results_quick",
    ),
}
