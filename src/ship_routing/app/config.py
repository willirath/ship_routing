from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

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
class HyperParams:
    """Unified hyperparameter configuration (matches Table~\\ref{tab:hyperparams})."""

    # Population
    population_size: int = 4
    random_seed: int | None = 345

    # Stage 2: Genetic evolution
    generations: int = 5  # N_G
    selection_quantile: float = 0.2  # q
    selection_acceptance_rate_warmup: float = 0.3  # p_w
    selection_acceptance_rate: float = 0.0  # p
    mutation_width_fraction: float = 0.9  # W
    mutation_displacement_fraction: float = 0.1  # D_max
    mutation_iterations: int = 2  # N_mut
    crossover_strategy: Literal["minimal_cost", "random"] = "minimal_cost"  # C_e or C_r
    crossover_rounds: int = 1

    # Stage 3: Gradient descent
    num_elites: int = 2  # k
    gd_iterations: int = 2  # N_GD
    learning_rate_time: float = 0.5  # gamma_t
    learning_rate_space: float = 0.5  # gamma_s (applied to along/across)
    time_increment: float = 1_200.0  # delta t
    distance_increment: float = 10_000.0  # delta d


@dataclass(frozen=True)
class RoutingConfig:
    """Top-level configuration consumed by the routing application."""

    journey: JourneyConfig = JourneyConfig()
    forcing: ForcingConfig = ForcingConfig()
    ship: Ship = Ship()
    physics: Physics = Physics()
    hyper: HyperParams = HyperParams()
