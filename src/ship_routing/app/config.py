from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ..core.hashable_dataset import HashableDataset
from ..core.config import Physics, Ship


@dataclass(frozen=True)
class JourneyConfig:
    """Definition of the trip that needs to be routed."""

    name: str = "Journey"
    lon_waypoints: Tuple[float, ...] = (-80.5, -75.5)
    lat_waypoints: Tuple[float, ...] = (30.0, 30.0)
    time_start: str = "2024-01-01T00:00"
    time_end: str | None = None
    speed_knots: float | None = 10.0
    time_resolution_hours: float = 12.0

    def __post_init__(self):
        """Calculate time_end from route length and speed if not provided."""
        if self.time_end is None and self.speed_knots is not None:
            import numpy as np
            from shapely.geometry import LineString
            from ..core.geodesics import get_length_meters, knots_to_ms

            # Get distance in meters along way points
            line_string = LineString(zip(self.lon_waypoints, self.lat_waypoints))
            length_meters = get_length_meters(line_string)

            # Calculate end time
            speed_ms = knots_to_ms(self.speed_knots)
            duration_seconds = length_meters / speed_ms
            start_dt = np.datetime64(self.time_start)
            end_dt = start_dt + np.timedelta64(int(duration_seconds), "s")

            # Update end time
            time_end_str = str(np.datetime_as_string(end_dt, unit="s"))
            object.__setattr__(self, "time_end", time_end_str)


@dataclass(frozen=True)
class ForcingConfig:
    """Paths and IO settings for currents, winds, and waves."""

    currents_path: str | None = None
    waves_path: str | None = None
    winds_path: str | None = None
    engine: str = "zarr"
    chunks: str = "auto"
    load_eagerly: bool = True
    enable_spatial_cropping: bool = True
    route_length_multiplier: float = 1.5
    spatial_buffer_degrees: float = 0.5
    scenario_name: str | None = None


@dataclass
class ForcingData:
    """Loaded forcing datasets."""

    currents: HashableDataset | None = None
    waves: HashableDataset | None = None
    winds: HashableDataset | None = None


@dataclass(frozen=True)
class HyperParams:
    """Unified hyperparameter configuration (matches Table~\\ref{tab:hyperparams})."""

    # Population
    population_size: int = 4
    random_seed: int | None = 345

    # Stage 2: Warmup
    selection_acceptance_rate_warmup: float = 0.3  # p_w
    mutation_width_fraction_warmup: float = 0.9  # W_w
    mutation_displacement_fraction_warmup: float = 0.2  # D_w

    # Stage 3: Genetic evolution
    generations: int = 5  # N_G
    offspring_size: int = 4  # M_offspring
    crossover_rounds: int = 1  # N_crossover
    selection_quantile: float = 0.2  # q
    selection_acceptance_rate: float = 0.0  # p
    mutation_width_fraction: float = 0.9  # W
    mutation_displacement_fraction: float = 0.1  # D
    mutation_iterations: int = 2  # N_mut (max; actual sampled uniformly from 1..N_mut)
    crossover_strategy: Literal["minimal_cost", "random"] = "minimal_cost"  # C_e or C_r
    hazard_penalty_multiplier: float = 100.0  # Penalty multiplier for hazardous routes

    # Stage 4: Post-processing (Gradient descent)
    num_elites: int = 2  # k
    gd_iterations: int = 2  # N_GD
    learning_rate_time: float = 0.5  # gamma_t
    learning_rate_space: float = 0.5  # gamma_perp (applied to along/across)
    time_increment: float = 1_200.0  # delta t
    distance_increment: float = 10_000.0  # delta d

    # Stage 5: Parameter adaptation
    enable_adaptation: bool = False  # Enable W, D adaptation
    target_relative_improvement: float = 0.01  # Target relative cost improvement (1%)
    adaptation_scale_W: float = 0.8  # Scale factor for W when improvement < target
    adaptation_scale_D: float = 0.894427191  # Scale factor for D (0.8**0.5)
    W_min: float = 0.1  # Minimum mutation width fraction
    W_max: float = 1.0  # Maximum mutation width fraction
    D_min: float = 0.01  # Minimum mutation displacement fraction
    D_max: float = 0.5  # Maximum mutation displacement fraction

    # Parallelization
    num_workers: int = (
        2  # Number of worker processes/threads (ignored if executor_type="sequential")
    )
    executor_type: Literal["process", "thread", "sequential"] = (
        "sequential"  # Executor type
    )


@dataclass(frozen=True)
class RoutingConfig:
    """Top-level configuration consumed by the routing application."""

    journey: JourneyConfig = JourneyConfig()
    forcing: ForcingConfig = ForcingConfig()
    ship: Ship = Ship()
    physics: Physics = Physics()
    hyper: HyperParams = HyperParams()
