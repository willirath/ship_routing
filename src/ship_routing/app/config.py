from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..core.hashable_dataset import HashableDataset
from ..core.config import Physics, Ship


@dataclass(frozen=True)
class JourneyConfig:
    """Definition of the trip that needs to be routed."""

    lon_waypoints: Tuple[float, ...] = (-80.5, -75.5)
    lat_waypoints: Tuple[float, ...] = (30.0, 30.0)
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
    load_eagerly: bool = True


@dataclass
class ForcingData:
    """Loaded forcing datasets."""

    currents: HashableDataset | None = None
    waves: HashableDataset | None = None
    winds: HashableDataset | None = None


@dataclass(frozen=True)
class PopulationConfig:
    """Population-level optimisation settings."""

    size: int = 4
    random_seed: int | None = 345


@dataclass(frozen=True)
class WarmupConfig:
    """Configuration for warmup stage (initial diversification).

    Attributes
    ----------
    num_iterations : int
        Number of mutation iterations per member during warmup
    acceptance_rate_for_increase_cost : float
        Probability of accepting cost increases (for escaping local minima)
    mod_width_fraction : float
        Mutation window width as fraction of route length
    max_move_fraction : float
        Maximum waypoint move distance as fraction of route length
    refinement_factor : float
        Factor to reduce mutation parameters when acceptance drops
    acceptance_rate_target : float
        Target acceptance rate for adaptive refinement
    """

    num_iterations: int = 50
    acceptance_rate_for_increase_cost: float = 0.3
    mod_width_fraction: float = 0.3
    max_move_fraction: float = 0.15
    refinement_factor: float = 0.8
    acceptance_rate_target: float = 0.5


@dataclass(frozen=True)
class StochasticStageConfig:
    """Parameters for the stochastic (mutation) stage."""

    num_generations: int = 5
    num_iterations: int = 2
    acceptance_rate_target: float = 0.3
    acceptance_rate_for_increase_cost: float = 0.0
    refinement_factor: float = 0.7
    mod_width_fraction: float = 0.9
    max_move_fraction: float = 0.1


@dataclass(frozen=True)
class CrossoverConfig:
    """Crossover strategy settings."""

    strategy: str = "minimal_cost"
    offspring_size: int = 4
    generations: int = 1
    resample_with_replacement: bool = True


@dataclass(frozen=True)
class SelectionConfig:
    """Selection operator settings."""

    quantile: float = 0.2


@dataclass(frozen=True)
class GradientConfig:
    """Gradient-descent polishing settings."""

    enabled: bool = True
    num_iterations: int = 2
    learning_rate_percent_time: float = 0.5
    time_increment: float = 1_200.0
    learning_rate_percent_along: float = 0.5
    dist_shift_along: float = 10_000.0
    learning_rate_percent_across: float = 0.5
    dist_shift_across: float = 10_000.0
    num_elites: int = 2


@dataclass(frozen=True)
class RoutingConfig:
    """Top-level configuration consumed by the routing application."""

    journey: JourneyConfig = JourneyConfig()
    forcing: ForcingConfig = ForcingConfig()
    ship: Ship = Ship()
    physics: Physics = Physics()
    population: PopulationConfig = PopulationConfig()
    warmup: WarmupConfig = WarmupConfig()
    mix_seed_route_each_generation: bool = True
    stochastic: StochasticStageConfig = StochasticStageConfig()
    crossover: CrossoverConfig = CrossoverConfig()
    selection: SelectionConfig = SelectionConfig()
    gradient: GradientConfig = GradientConfig()
