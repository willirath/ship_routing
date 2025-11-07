from dataclasses import dataclass
from typing import Any, Tuple


MAX_CACHE_SIZE = 10_000


@dataclass(frozen=True)
class Physics:
    """Physical constants used in power estimation."""

    gravity_acceleration_ms2: float = 9.80665
    sea_water_density_kgm3: float = 1029.0
    air_density_kgm3: float = 1.225


@dataclass(frozen=True)
class Ship:
    """Ship dimensions, resistance coefficients, engine characteristics."""

    waterline_width_m: float = 30.0
    waterline_length_m: float = 210.0
    total_propulsive_efficiency: float = 0.7
    reference_engine_power_W: float = 14296344.0
    reference_speed_calm_water_ms: float = 9.259
    draught_m: float = 11.5
    projected_frontal_area_above_waterline_m2: float = 690.0
    wind_resistance_coefficient: float = 0.4


@dataclass(frozen=True)
class JourneyConfig:
    """Definition of the trip that needs to be routed."""

    lon_waypoints: Tuple[float, ...] = (-80.5, -12.0)
    lat_waypoints: Tuple[float, ...] = (30.0, 45.0)
    time_start: str = "2021-01-01T00:00"
    time_end: str | None = None
    speed_knots: float | None = 10.0
    time_resolution_hours: float = 12.0


@dataclass(frozen=True)
class ForcingConfig:
    """Paths and IO settings for currents, winds, and waves."""

    currents_path: str | None = None
    waves_path: str | None = None
    winds_path: str | None = None
    engine: str = "zarr"
    chunks: str = "auto"


@dataclass
class ForcingData:
    """Loaded forcing datasets."""

    currents: Any | None = None
    waves: Any | None = None
    winds: Any | None = None


@dataclass(frozen=True)
class PopulationConfig:
    """Population-level optimisation settings."""

    size: int = 16
    random_seed: int | None = 345
    mix_seed_route_each_generation: bool = True


@dataclass(frozen=True)
class StochasticStageConfig:
    """Parameters for the stochastic (mutation) stage."""

    num_generations: int = 4
    num_iterations: int = 3
    acceptance_rate_target: float = 0.3
    acceptance_rate_for_increase_cost: float = 0.0
    refinement_factor: float = 0.7
    warmup_iterations: int = 1
    warmup_acceptance_rate_target: float = 0.1
    warmup_acceptance_for_increase_cost: float = 1.0
    warmup_mod_width_fraction: float = 0.9
    warmup_max_move_fraction: float = 0.1


@dataclass(frozen=True)
class CrossoverConfig:
    """Crossover strategy settings."""

    strategy: str = "minimal_cost"
    generations: int = 1
    resample_with_replacement: bool = True


@dataclass(frozen=True)
class SelectionConfig:
    """Selection operator settings."""

    quantile: float = 0.2
    with_replacement: bool = True
    elite_fraction: float = 0.0


@dataclass(frozen=True)
class GradientConfig:
    """Gradient-descent refinement settings."""

    enabled: bool = True
    num_iterations: int = 3
    learning_rate_percent_time: float = 0.5
    time_increment: float = 1_200.0
    learning_rate_percent_along: float = 0.5
    dist_shift_along: float = 10_000.0
    learning_rate_percent_across: float = 0.5
    dist_shift_across: float = 10_000.0
    num_elites: int = 1


@dataclass(frozen=True)
class ConcurrencyConfig:
    """Execution backend options."""

    process_pool_workers: int = 2


@dataclass(frozen=True)
class RoutingConfig:
    """Top-level configuration consumed by the routing application."""

    journey: JourneyConfig = JourneyConfig()
    forcing: ForcingConfig = ForcingConfig()
    ship: Ship = Ship()
    physics: Physics = Physics()
    population: PopulationConfig = PopulationConfig()
    stochastic: StochasticStageConfig = StochasticStageConfig()
    crossover: CrossoverConfig = CrossoverConfig()
    selection: SelectionConfig = SelectionConfig()
    gradient: GradientConfig = GradientConfig()
    concurrency: ConcurrencyConfig = ConcurrencyConfig()
