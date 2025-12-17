"""Unified CLI for RoutingApp.

Provides Click-based command-line interface and a programmatic build_config() function
for creating RoutingConfig objects from individual parameters.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import json
import logging
import uuid

import click

from .config import (
    ForcingConfig,
    HyperParams,
    JourneyConfig,
    RoutingConfig,
)
from .routing import RoutingApp, RoutingResult
from ..core.config import Physics, Ship

# Calculate default data directory relative to this file's location
# cli.py is at: ship_routing/src/ship_routing/app/cli.py
# Data is at: ship_routing/data/test/
# From cli.py, go up 4 levels to reach ship_routing/, then add data/test/
_CLI_FILE = Path(__file__).resolve()
_SHIP_ROUTING_DIR = _CLI_FILE.parent.parent.parent.parent
_DEFAULT_DATA_DIR = _SHIP_ROUTING_DIR / "data" / "test"


def build_config(
    # Journey parameters
    lon_waypoints: Optional[tuple[float, ...]] = None,
    lat_waypoints: Optional[tuple[float, ...]] = None,
    journey_name: str = "Journey",
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    speed_knots: Optional[float] = None,
    time_resolution_hours: Optional[float] = None,
    # Forcing parameters
    currents_path: Optional[str] = None,
    waves_path: Optional[str] = None,
    winds_path: Optional[str] = None,
    engine: str = "zarr",
    chunks: str = "auto",
    load_eagerly: bool = True,
    enable_spatial_cropping: bool = True,
    route_length_multiplier: float = 1.5,
    spatial_buffer_degrees: float = 0.5,
    # Population parameters
    population_size: int = 4,
    random_seed: Optional[int] = None,
    # Warmup parameters
    selection_acceptance_rate_warmup: float = 0.3,
    mutation_width_fraction_warmup: float = 0.9,
    mutation_displacement_fraction_warmup: float = 0.2,
    # Genetic algorithm parameters
    generations: int = 2,
    offspring_size: int = 4,
    selection_quantile: float = 0.2,
    selection_acceptance_rate: float = 0.0,
    mutation_width_fraction: float = 0.9,
    mutation_displacement_fraction: float = 0.1,
    mutation_iterations: int = 2,
    crossover_strategy: str = "minimal_cost",
    crossover_rounds: int = 1,
    hazard_penalty_multiplier: float = 100.0,
    # Post-processing (Gradient descent) parameters
    num_elites: int = 2,
    gd_iterations: int = 2,
    learning_rate_time: float = 0.5,
    learning_rate_space: float = 0.5,
    time_increment: float = 1_200.0,
    distance_increment: float = 10_000.0,
    # Adaptation parameters
    enable_adaptation: bool = True,
    target_relative_improvement: float = 0.01,
    adaptation_scale_W: float = 0.8,
    adaptation_scale_D: float = 0.894427191,
    W_min: float = 0.1,
    W_max: float = 1.0,
    D_min: float = 0.01,
    D_max: float = 0.5,
    # Parallelization parameters
    executor_type: str = "sequential",
    num_workers: int = 2,
    # Config file override
    config_dict: Optional[dict[str, Any]] = None,
) -> RoutingConfig:
    """Build RoutingConfig from individual parameters.

    Parameters are merged with defaults from config classes.
    If config_dict is provided, it overrides individual parameters.

    Parameters
    ----------
    lon_waypoints : tuple[float, ...], optional
        Longitude waypoints for the journey
    lat_waypoints : tuple[float, ...], optional
        Latitude waypoints for the journey
    journey_name : str
        Human-readable name for the journey
    time_start : str, optional
        Start time in ISO format
    time_end : str, optional
        End time in ISO format
    speed_knots : float, optional
        Ship speed in knots
    time_resolution_hours : float, optional
        Time resolution in hours
    currents_path : str, optional
        Path to currents data
    waves_path : str, optional
        Path to waves data
    winds_path : str, optional
        Path to winds data
    engine : str
        Data engine (zarr, netcdf, etc.)
    chunks : str
        Chunk strategy for data loading
    load_eagerly : bool
        Whether to load data eagerly
    enable_spatial_cropping : bool
        Whether to enable spatial cropping
    route_length_multiplier : float
        Multiplier for route length bounding box
    spatial_buffer_degrees : float
        Buffer around route in degrees
    population_size : int
        Population size for GA
    random_seed : int, optional
        Random seed for reproducibility
    selection_acceptance_rate_warmup : float
        Acceptance rate during warmup (p_w)
    mutation_width_fraction_warmup : float
        Width fraction for warmup mutation (W_w)
    mutation_displacement_fraction_warmup : float
        Max displacement fraction for warmup (D_w)
    generations : int
        Number of GA generations (N_G)
    offspring_size : int
        Offspring size (M_offspring)
    selection_quantile : float
        Selection quantile (q)
    selection_acceptance_rate : float
        Acceptance rate after warmup (p)
    mutation_width_fraction : float
        Width fraction for mutation (W)
    mutation_displacement_fraction : float
        Max displacement fraction (D)
    mutation_iterations : int
        Number of mutation iterations (N_mut)
    crossover_strategy : str
        Crossover strategy: minimal_cost or random
    crossover_rounds : int
        Number of crossover rounds (N_crossover)
    hazard_penalty_multiplier : float
        Hazard penalty multiplier
    num_elites : int
        Number of elite members (k)
    gd_iterations : int
        Number of gradient descent iterations (N_GD)
    learning_rate_time : float
        Learning rate for time (gamma_t)
    learning_rate_space : float
        Learning rate for space (gamma_s)
    time_increment : float
        Time increment in seconds (delta t)
    distance_increment : float
        Distance increment in meters (delta d)
    enable_adaptation : bool
        Enable W, D adaptation
    target_relative_improvement : float
        Target relative cost improvement
    adaptation_scale_W : float
        Scale factor for W when improvement < target
    adaptation_scale_D : float
        Scale factor for D
    W_min : float
        Minimum mutation width fraction
    W_max : float
        Maximum mutation width fraction
    D_min : float
        Minimum mutation displacement fraction
    D_max : float
        Maximum mutation displacement fraction
    executor_type : str
        Executor type: process, thread, or sequential
    num_workers : int
        Number of worker processes/threads
    config_dict : dict, optional
        Configuration dictionary that overrides individual parameters

    Returns
    -------
    RoutingConfig
        Configured routing configuration object
    """
    # If config_dict is provided, use it as base and override with individual params
    if config_dict:
        # Start with config_dict
        params = config_dict.copy()
        # Overlay non-None individual parameters
        if lon_waypoints is not None:
            if "journey" not in params:
                params["journey"] = {}
            params["journey"]["lon_waypoints"] = lon_waypoints
        if lat_waypoints is not None:
            if "journey" not in params:
                params["journey"] = {}
            params["journey"]["lat_waypoints"] = lat_waypoints
        # ... (similar for other params if needed for config_dict merging)
    else:
        params = {}

    # Build journey config
    journey_kwargs = {}
    if lon_waypoints is not None:
        journey_kwargs["lon_waypoints"] = lon_waypoints
    if lat_waypoints is not None:
        journey_kwargs["lat_waypoints"] = lat_waypoints
    if journey_name:
        journey_kwargs["name"] = journey_name
    if time_start is not None:
        journey_kwargs["time_start"] = time_start
    if time_end is not None:
        journey_kwargs["time_end"] = time_end
    if speed_knots is not None:
        journey_kwargs["speed_knots"] = speed_knots
    if time_resolution_hours is not None:
        journey_kwargs["time_resolution_hours"] = time_resolution_hours

    journey = JourneyConfig(**journey_kwargs) if journey_kwargs else JourneyConfig()

    # Build forcing config
    forcing = ForcingConfig(
        currents_path=currents_path,
        waves_path=waves_path,
        winds_path=winds_path,
        engine=engine,
        chunks=chunks,
        load_eagerly=load_eagerly,
        enable_spatial_cropping=enable_spatial_cropping,
        route_length_multiplier=route_length_multiplier,
        spatial_buffer_degrees=spatial_buffer_degrees,
    )

    # Build hyperparams
    hyper = HyperParams(
        # Population
        population_size=population_size,
        random_seed=random_seed,
        # Warmup
        selection_acceptance_rate_warmup=selection_acceptance_rate_warmup,
        mutation_width_fraction_warmup=mutation_width_fraction_warmup,
        mutation_displacement_fraction_warmup=mutation_displacement_fraction_warmup,
        # Genetic algorithm
        generations=generations,
        offspring_size=offspring_size,
        selection_quantile=selection_quantile,
        selection_acceptance_rate=selection_acceptance_rate,
        mutation_width_fraction=mutation_width_fraction,
        mutation_displacement_fraction=mutation_displacement_fraction,
        mutation_iterations=mutation_iterations,
        crossover_strategy=crossover_strategy,
        crossover_rounds=crossover_rounds,
        hazard_penalty_multiplier=hazard_penalty_multiplier,
        # Post-processing
        num_elites=num_elites,
        gd_iterations=gd_iterations,
        learning_rate_time=learning_rate_time,
        learning_rate_space=learning_rate_space,
        time_increment=time_increment,
        distance_increment=distance_increment,
        # Adaptation
        enable_adaptation=enable_adaptation,
        target_relative_improvement=target_relative_improvement,
        adaptation_scale_W=adaptation_scale_W,
        adaptation_scale_D=adaptation_scale_D,
        W_min=W_min,
        W_max=W_max,
        D_min=D_min,
        D_max=D_max,
        # Parallelization
        executor_type=executor_type,
        num_workers=num_workers,
    )

    return RoutingConfig(
        journey=journey,
        forcing=forcing,
        ship=Ship(),
        physics=Physics(),
        hyper=hyper,
    )


@click.command()
# Journey parameters
@click.option(
    "--journey-name",
    type=str,
    default="Journey",
    help="Human-readable name for this journey (e.g., 'Jacksonville-Irish_Sea-Winter').",
)
@click.option(
    "--lon-wp",
    "lon_waypoints",
    type=float,
    multiple=True,
    default=(-80.5, -11.0),
    help="Longitude waypoints (e.g., --lon-wp -80.5 --lon-wp -11.0).",
)
@click.option(
    "--lat-wp",
    "lat_waypoints",
    type=float,
    multiple=True,
    default=(30.0, 50.0),
    help="Latitude waypoints (e.g., --lat-wp 30.0 --lat-wp 50.0).",
)
@click.option(
    "--time-start",
    type=str,
    default="2024-01-01T00:00",
    help="Start time in ISO format (e.g., 2024-01-01T00:00).",
)
@click.option(
    "--time-end",
    type=str,
    default=None,
    help="End time in ISO format (optional).",
)
@click.option(
    "--speed-knots",
    type=float,
    default=10.0,
    help="Ship speed in knots (optional).",
)
@click.option(
    "--time-resolution-hours",
    type=float,
    default=6.0,
    help="Time resolution in hours.",
)
# Forcing parameters
@click.option(
    "--data-dir",
    type=click.Path(exists=True),
    default=str(_DEFAULT_DATA_DIR),
    help="Base directory containing forcing data.",
)
@click.option(
    "--currents-path",
    type=str,
    default=None,
    help="Path to currents data (overrides data-dir).",
)
@click.option(
    "--waves-path",
    type=str,
    default=None,
    help="Path to waves data (overrides data-dir).",
)
@click.option(
    "--winds-path",
    type=str,
    default=None,
    help="Path to winds data (overrides data-dir).",
)
@click.option(
    "--engine",
    type=str,
    default="netcdf4",
    help="Data engine (netcdf4, zarr, etc.).",
)
@click.option(
    "--chunks",
    type=str,
    default="auto",
    help="Chunk strategy for data loading.",
)
@click.option(
    "--load-eagerly/--no-load-eagerly",
    default=True,
    help="Whether to load data eagerly.",
)
@click.option(
    "--enable-spatial-cropping/--no-spatial-cropping",
    default=True,
    help="Whether to enable spatial cropping.",
)
# Population parameters
@click.option(
    "--population-size",
    type=int,
    default=4,
    help="Population size for genetic algorithm.",
)
@click.option(
    "--random-seed", type=int, default=None, help="Random seed for reproducibility."
)
# Warmup parameters
@click.option(
    "--selection-acceptance-rate-warmup",
    type=float,
    default=0.3,
    help="Acceptance rate during warmup (p_w).",
)
@click.option(
    "--mutation-width-fraction-warmup",
    type=float,
    default=0.9,
    help="Width fraction for warmup mutation (W_w).",
)
@click.option(
    "--mutation-displacement-fraction-warmup",
    type=float,
    default=0.2,
    help="Max displacement fraction for warmup (D_w).",
)
# Genetic algorithm parameters
@click.option("--generations", type=int, default=2, help="Number of generations (N_G).")
@click.option(
    "--offspring-size", type=int, default=4, help="Offspring size (M_offspring)."
)
@click.option(
    "--selection-quantile", type=float, default=0.2, help="Selection quantile (q)."
)
@click.option(
    "--selection-acceptance-rate",
    type=float,
    default=0.0,
    help="Acceptance rate after warmup (p).",
)
@click.option(
    "--mutation-width-fraction",
    type=float,
    default=0.9,
    help="Width fraction for mutation (W).",
)
@click.option(
    "--mutation-displacement-fraction",
    type=float,
    default=0.1,
    help="Max displacement fraction (D).",
)
@click.option(
    "--mutation-iterations",
    type=int,
    default=2,
    help="Number of mutation iterations (N_mut).",
)
@click.option(
    "--crossover-strategy",
    type=click.Choice(["minimal_cost", "random"], case_sensitive=False),
    default="minimal_cost",
    help="Crossover strategy: minimal_cost or random.",
)
@click.option(
    "--crossover-rounds",
    type=int,
    default=1,
    help="Number of crossover rounds (N_crossover). Set to 0 to skip crossover.",
)
@click.option(
    "--hazard-penalty-multiplier",
    type=float,
    default=100.0,
    help="Hazard penalty multiplier (0 to ignore hazards, >0 to apply penalty).",
)
# Post-processing (Gradient descent) parameters
@click.option("--num-elites", type=int, default=2, help="Number of elite members (k).")
@click.option(
    "--gd-iterations",
    type=int,
    default=2,
    help="Number of gradient descent iterations (N_GD).",
)
@click.option(
    "--learning-rate-time",
    type=float,
    default=0.5,
    help="Learning rate for time (gamma_t).",
)
@click.option(
    "--learning-rate-space",
    type=float,
    default=0.5,
    help="Learning rate for space (gamma_s).",
)
@click.option(
    "--time-increment",
    type=float,
    default=1200.0,
    help="Time increment in seconds (delta t).",
)
@click.option(
    "--distance-increment",
    type=float,
    default=10000.0,
    help="Distance increment in meters (delta d).",
)
# Adaptation parameters
@click.option(
    "--enable-adaptation/--no-adaptation",
    default=True,
    help="Enable W, D adaptation.",
)
@click.option(
    "--target-relative-improvement",
    type=float,
    default=0.01,
    help="Target relative cost improvement (1%).",
)
@click.option(
    "--adaptation-scale-W",
    type=float,
    default=0.8,
    help="Adaptation scaling factor for W (mutation width fraction). Default: 0.8",
)
@click.option(
    "--adaptation-scale-D",
    type=float,
    default=0.8**0.5,
    help="Adaptation scaling factor for D (mutation displacement fraction). Default: sqrt(0.8)",
)
@click.option(
    "--W-min",
    type=float,
    default=0.1,
    help="Minimum allowed value for W (mutation width fraction). Default: 0.1",
)
@click.option(
    "--W-max",
    type=float,
    default=1.0,
    help="Maximum allowed value for W (mutation width fraction). Default: 1.0",
)
@click.option(
    "--D-min",
    type=float,
    default=0.01,
    help="Minimum allowed value for D (mutation displacement fraction). Default: 0.01",
)
@click.option(
    "--D-max",
    type=float,
    default=0.5,
    help="Maximum allowed value for D (mutation displacement fraction). Default: 0.5",
)
# Parallelization parameters
@click.option(
    "--executor-type",
    type=click.Choice(["process", "thread", "sequential"], case_sensitive=False),
    default="sequential",
    help="Executor type for parallelization.",
)
@click.option(
    "--num-workers",
    type=int,
    default=2,
    help="Number of worker processes/threads (ignored if executor-type=sequential).",
)
# Output parameters
@click.option(
    "--log-dir",
    type=click.Path(),
    default="runs",
    help="Directory to save experiment results.",
)
@click.option(
    "--redis-host",
    type=str,
    default=None,
    help="Redis host for result collection (if specified, skips file output).",
)
@click.option(
    "--redis-port",
    type=int,
    default=6379,
    help="Redis port.",
)
@click.option(
    "--redis-password",
    type=str,
    default=None,
    help="Redis password (optional).",
)
def main(
    # Journey
    journey_name,
    lon_waypoints,
    lat_waypoints,
    time_start,
    time_end,
    speed_knots,
    time_resolution_hours,
    # Forcing
    data_dir,
    currents_path,
    waves_path,
    winds_path,
    engine,
    chunks,
    load_eagerly,
    enable_spatial_cropping,
    # Population
    population_size,
    random_seed,
    # Warmup
    selection_acceptance_rate_warmup,
    mutation_width_fraction_warmup,
    mutation_displacement_fraction_warmup,
    # GA
    generations,
    offspring_size,
    selection_quantile,
    selection_acceptance_rate,
    mutation_width_fraction,
    mutation_displacement_fraction,
    mutation_iterations,
    crossover_strategy,
    crossover_rounds,
    hazard_penalty_multiplier,
    # GD
    num_elites,
    gd_iterations,
    learning_rate_time,
    learning_rate_space,
    time_increment,
    distance_increment,
    # Adaptation
    enable_adaptation,
    target_relative_improvement,
    adaptation_scale_w,
    adaptation_scale_d,
    w_min,
    w_max,
    d_min,
    d_max,
    # Parallelization
    executor_type,
    num_workers,
    # Output
    log_dir,
    redis_host,
    redis_port,
    redis_password,
) -> RoutingResult:
    """Configure and run a routing experiment."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # NOTE: Defaults use data/test files for easy testing without large downloads.
    # For production runs with large data, use --data-dir option or specify paths explicitly.
    # Use explicit test file names for deterministic behavior
    if not currents_path:
        currents_path = str(
            Path(data_dir)
            / "currents"
            / "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2024-01_100W-020E_10N-65N.nc"
        )
    if not waves_path:
        waves_path = str(
            Path(data_dir)
            / "waves"
            / "cmems_mod_glo_wav_my_0.2deg_PT3H-i_VHM0_2024-01_1d-max_100W-020E_10N-65N.nc"
        )
    if not winds_path:
        winds_path = str(
            Path(data_dir)
            / "winds"
            / "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_2024-01_6hours_0.5deg_100W-020E_10N-65N.nc"
        )

    config = build_config(
        # Journey
        journey_name=journey_name,
        lon_waypoints=lon_waypoints,
        lat_waypoints=lat_waypoints,
        time_start=time_start,
        time_end=time_end,
        speed_knots=speed_knots,
        time_resolution_hours=time_resolution_hours,
        # Forcing
        currents_path=currents_path,
        waves_path=waves_path,
        winds_path=winds_path,
        engine=engine,
        chunks=chunks,
        load_eagerly=load_eagerly,
        enable_spatial_cropping=enable_spatial_cropping,
        # Population
        population_size=population_size,
        random_seed=random_seed,
        # Warmup
        selection_acceptance_rate_warmup=selection_acceptance_rate_warmup,
        mutation_width_fraction_warmup=mutation_width_fraction_warmup,
        mutation_displacement_fraction_warmup=mutation_displacement_fraction_warmup,
        # GA
        generations=generations,
        offspring_size=offspring_size,
        selection_quantile=selection_quantile,
        selection_acceptance_rate=selection_acceptance_rate,
        mutation_width_fraction=mutation_width_fraction,
        mutation_displacement_fraction=mutation_displacement_fraction,
        mutation_iterations=mutation_iterations,
        crossover_strategy=crossover_strategy,
        crossover_rounds=crossover_rounds,
        hazard_penalty_multiplier=hazard_penalty_multiplier,
        # GD
        num_elites=num_elites,
        gd_iterations=gd_iterations,
        learning_rate_time=learning_rate_time,
        learning_rate_space=learning_rate_space,
        time_increment=time_increment,
        distance_increment=distance_increment,
        # Adaptation
        enable_adaptation=enable_adaptation,
        target_relative_improvement=target_relative_improvement,
        adaptation_scale_W=adaptation_scale_w,
        adaptation_scale_D=adaptation_scale_d,
        W_min=w_min,
        W_max=w_max,
        D_min=d_min,
        D_max=d_max,
        # Parallelization
        executor_type=executor_type,
        num_workers=num_workers,
    )

    app = RoutingApp(config=config)
    result = app.run()

    # Write logs
    run_id = datetime.now().isoformat(timespec="milliseconds").replace(":", "-")
    run_id = f"{run_id}_{uuid.uuid4()}"

    if redis_host:
        # Write to Redis using msgpack
        import redis

        r = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            decode_responses=False,
        )
        key = f"result:{run_id}"
        r.set(key, result.to_msgpack())
        click.echo(f"Result stored in Redis: {key}")
    else:
        # File-based output
        output_file = Path(log_dir) / f"run_{run_id}.json"
        result.dump_json(output_file)
        click.echo(f"Results saved to {output_file}")

    return result


if __name__ == "__main__":
    main()
