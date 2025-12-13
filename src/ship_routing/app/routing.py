from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from statistics import mean
import threading
from typing import Any, Sequence

import numpy as np
from . import parallel
from .parallel import (
    WorkerState,
    SequentialExecutor,
    _initialize_worker_process,
    _initialize_worker_thread,
    _initialize_sequential,
    _get_state,
)
from matplotlib import pyplot as plt
import cartopy
import pandas as pd

from ..algorithms import (
    crossover_routes_minimal_cost,
    crossover_routes_random,
    gradient_descent,
    select_from_pair,
    select_from_population,
    stochastic_mutation,
)
from .config import ForcingConfig, ForcingData, RoutingConfig
from ..core.routes import Route
from ..core.data import load_currents, load_waves, load_winds
from ..core.geodesics import compute_ellipse_bbox
from ..core.population import Population, PopulationMember

np.seterr(divide="ignore", invalid="ignore")

# Fallback for @profile decorator when not using line_profiler
try:
    profile
except NameError:

    def profile(func):
        return func


@dataclass
class CostImprovementStats:
    """Track cost improvement statistics for a generation."""

    cost_before: float
    cost_after: float

    @property
    def absolute_improvement(self) -> float:
        """Absolute cost reduction (positive = improvement)."""
        return self.cost_before - self.cost_after

    @property
    def relative_improvement(self) -> float:
        """Relative cost reduction as fraction of original cost."""
        return (
            (self.cost_before - self.cost_after) / self.cost_before
            if self.cost_before > 0
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cost_before": self.cost_before,
            "cost_after": self.cost_after,
            "absolute_improvement": self.absolute_improvement,
            "relative_improvement": self.relative_improvement,
        }


@dataclass
class RoutingResult:
    """Container returned by RoutingApp.run."""

    seed_member: PopulationMember | None = None
    elite_population: Population | None = None
    logs: "RoutingLog | None" = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "seed_member": self.seed_member.to_dict() if self.seed_member else None,
            "elite_population": (
                self.elite_population.to_dict() if self.elite_population else None
            ),
            "log": self.logs.to_dict() if self.logs else None,
        }

    def to_msgpack(self) -> bytes:
        """Serialize to MessagePack binary format."""
        import msgpack

        def _default(obj: Any):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Path):
                return str(obj)
            # Handle pandas Timestamp
            try:
                if hasattr(obj, "isoformat"):
                    return obj.isoformat()
            except Exception:
                pass
            raise TypeError(f"Object of type {type(obj)!r} is not serialisable")

        return msgpack.packb(self.to_dict(), use_bin_type=True, default=_default)

    @classmethod
    def from_dict(cls, data: dict) -> "RoutingResult":
        """Reconstruct RoutingResult from dictionary."""
        seed_member = (
            PopulationMember.from_dict(data["seed_member"])
            if data.get("seed_member")
            else None
        )
        elite_population = (
            Population.from_dict(data["elite_population"])
            if data.get("elite_population")
            else None
        )
        log_data = data.get("log")
        logs = (
            RoutingLog(
                config=log_data.get("config", {}),
                stages=[
                    StageLog(
                        name=stage["name"],
                        metrics=stage.get("metrics", {}),
                        timestamp=stage.get("timestamp", ""),
                    )
                    for stage in log_data.get("stages", [])
                ],
            )
            if log_data
            else None
        )
        return cls(
            seed_member=seed_member, elite_population=elite_population, logs=logs
        )

    @classmethod
    def from_msgpack(cls, data: bytes) -> "RoutingResult":
        """Deserialize from MessagePack binary format."""
        import msgpack

        dict_data = msgpack.unpackb(data, raw=False)
        return cls.from_dict(dict_data)

    def dump_json(self, path: Path | str, *, indent: int = 2) -> None:
        """Write routes and logs to JSON."""

        def _default(obj: Any):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)!r} is not JSON serialisable")

        data = self.to_dict()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=indent, default=_default)

    @classmethod
    def load_json(cls, path: Path) -> RoutingResult:
        """Load a RoutingResult from disk."""
        with Path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    def plot_cost_evolution(self, ax=None):
        """Plot cost evolution across optimization stages."""
        df = self.logs.to_dataframe()
        df = df.filter(like="cost_")
        df = df.drop(["cost_std", "cost_mean"], axis=1)
        ax = df.dropna().plot(ax=ax)
        ax.set_title("cost evolution")
        ax.grid()
        return ax

    def plot_routes(self, ax=None):
        """Plot seed and elite routes."""
        plt_add_kwargs = {}
        if ax is None:
            _lonlat = self.seed_member.route.data_frame[["lon", "lat"]]
            central_lon, central_lat = _lonlat.mean()
            ax_extent = [
                _lonlat.lon.min() - 5.0,
                _lonlat.lon.max() + 5.0,
                _lonlat.lat.min() - 5.0,
                _lonlat.lat.max() + 5.0,
            ]
            _, ax = plt.subplots(
                1,
                1,
                subplot_kw={
                    "projection": cartopy.crs.Gnomonic(
                        central_latitude=central_lat, central_longitude=central_lon
                    )
                },
            )
        if isinstance(ax, cartopy.mpl.geoaxes.GeoAxes):
            plt_add_kwargs.update({"transform": cartopy.crs.PlateCarree()})

        seed_member = self.seed_member
        elite_members = self.elite_population.members

        for em in elite_members:
            ax.plot(*em.route.line_string.xy, "black", **plt_add_kwargs)
        ax.plot(*seed_member.route.line_string.xy, "orange", **plt_add_kwargs)
        try:
            ax.gridlines(draw_labels=False)
            ax.coastlines()
            ax.set_extent(ax_extent)
            ax.set_title("routes, Gnomonic proj")
        except:
            ax.grid()
            ax.set_title("routes")
        return ax

    def plot_elite_cost(self, ax=None):
        """Plot elite costs as percentage of seed cost."""
        elite_cost = pd.Series(
            data=[m.cost for m in self.elite_population.members],
            name="cost",
            index=[f"elite_{n:02d}" for n in range(len(self.elite_population.members))],
        )
        seed_cost = self.seed_member.cost
        ax = (100 * elite_cost / seed_cost).plot(ax=ax)
        ax.set_title("elite cost (rel)")
        ax.grid()
        return ax

    def plot_full_routing_result(self):
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))

        self.plot_cost_evolution(ax=ax[0])
        self.plot_elite_cost(ax=ax[1])
        self.plot_routes(ax=ax[2])

        fig.tight_layout()

        return fig, ax


@dataclass
class StageLog:
    """Record of a single optimisation stage event."""

    name: str
    metrics: dict[str, Any]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_record(self) -> dict[str, Any]:
        """Return a flat record with stage, timestamp, and metrics."""
        return {
            "stage": self.name,
            "timestamp": self.timestamp,
            **self.metrics,
        }


@dataclass
class RoutingLog:
    """Structured information needed for paper figures / reproducibility."""

    config: dict[str, Any]
    stages: list[StageLog] = field(default_factory=list)

    def add_stage(self, name: str, **metrics: Any) -> None:
        """Append a stage log entry."""
        self.stages.append(StageLog(name=name, metrics=dict(metrics)))

    def stages_named(self, name: str) -> list[StageLog]:
        """Return all stage logs matching the provided name."""
        return [stage for stage in self.stages if stage.name == name]

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert logs to a pandas DataFrame.

        Includes all stages with columns: stage, timestamp, and metric keys.
        """
        records = [s.to_record() for s in self.stages]
        if not records:
            empty = pd.DataFrame(columns=["stage", "timestamp"])
            return empty

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    def to_dict(self) -> dict[str, Any]:
        """Return log contents as plain dict."""
        return {
            "config": self.config,
            "stages": [
                {
                    "name": stage.name,
                    "metrics": stage.metrics,
                    "timestamp": stage.timestamp,
                }
                for stage in self.stages
            ],
        }


class RoutingApp:
    """High-level orchestrator wrapping the routing workflow."""

    def __init__(self, config: RoutingConfig):
        self.config = config
        self.log = RoutingLog(config=asdict(config))
        self._rng: np.random.Generator | None = None

    def _create_executor(
        self, *, forcing: ForcingData, params: HyperParams
    ) -> SequentialExecutor | ThreadPoolExecutor | ProcessPoolExecutor:
        """Create and initialize the appropriate executor based on configuration.

        Parameters
        ----------
        forcing : ForcingData
            Ocean forcing data to pass to workers
        params : HyperParams
            Configuration parameters including executor type and num_workers

        Returns
        -------
        SequentialExecutor | ThreadPoolExecutor | ProcessPoolExecutor
            Configured executor ready for use

        Raises
        ------
        ValueError
            If executor configuration is invalid
        """
        # Initialize worker state globals in parallel module
        parallel._WORKER_STATE = None
        parallel._THREAD_LOCAL_STATE = threading.local()
        parallel._SHARED_FORCING = None

        # Generate seed for worker initialization
        worker_seed = int(self._rng.integers(0, 2**31 - 1))

        if params.executor_type == "sequential":
            # Sequential mode: use SequentialExecutor for unified interface
            if params.num_workers > 1:
                logging.warning(
                    f"Sequential executor requested but num_workers={params.num_workers} > 1. "
                    "Sequential execution will use single thread regardless."
                )
            return SequentialExecutor(
                initializer=_initialize_sequential,
                initargs=(forcing, worker_seed, params),
            )
        elif params.executor_type == "thread":
            # Thread pool executor: shared memory, GIL-released NumPy operations
            if params.num_workers == 0:
                raise ValueError("Thread executor requested but num_workers=0")
            parallel._SHARED_FORCING = forcing
            return ThreadPoolExecutor(
                max_workers=params.num_workers,
                initializer=_initialize_worker_thread,
                initargs=(worker_seed, params),
            )
        elif params.executor_type == "process":
            # Process pool executor: true parallelism, serialization overhead
            if params.num_workers == 0:
                raise ValueError("Process executor requested but num_workers=0")
            return ProcessPoolExecutor(
                max_workers=params.num_workers,
                initializer=_initialize_worker_process,
                initargs=(forcing, worker_seed, params),
            )
        else:
            raise ValueError(f"Unknown executor_type: {params.executor_type}")

    @profile
    def run(self) -> RoutingResult:
        """Execute the optimisation pipeline."""
        self._log_stage_metrics("run", message="starting routing run")
        self._rng = np.random.default_rng(self.config.hyper.random_seed)

        forcing = self._load_forcing(self.config.journey)

        # Create executor for parallel/sequential processing
        executor = self._create_executor(forcing=forcing, params=self.config.hyper)

        # Stage 0 to 4:
        seed_member, population = self._stage_initialization(forcing)
        assert len(population.members) > 0

        population = self._stage_warmup(population, seed_member, forcing, executor)
        assert len(population.members) > 0

        # Initialize adaptive parameters for genetic algorithm
        W = self.config.hyper.mutation_width_fraction
        D = self.config.hyper.mutation_displacement_fraction
        q = self.config.hyper.selection_quantile

        # Genetic algorithm generation loop
        for gen_idx in range(self.config.hyper.generations):
            # Mutation stage now returns cost improvement statistics
            population, cost_improvement_stats = self._stage_ga_mutation(
                population, seed_member, forcing, W, D, q, executor
            )
            assert len(population.members) > 0
            # Crossover (unchanged)
            population = self._stage_ga_crossover(
                population, seed_member, forcing, executor
            )
            assert len(population.members) > 0
            # Selection (unchanged)
            population = self._stage_ga_selection(population, seed_member, q)
            assert len(population.members) > 0
            # Adaptation: pass cost improvement stats and population stats
            pop_stats = self._population_stats(population.members)
            W, D, q = self._stage_ga_adaptation(
                W, D, q, cost_improvement_stats, pop_stats
            )

        elite_population = self._stage_post_processing(population, forcing, executor)
        assert len(population.members) > 0

        # Clean up executor (GC would handle this, but be explicit)
        executor.shutdown()

        return RoutingResult(
            seed_member=seed_member,
            elite_population=elite_population,
            logs=self.log,
        )

    @profile
    def _load_forcing(self, journey_config) -> ForcingData:
        """Load wind, wave, and current fields according to the config."""
        config = self.config.forcing
        time_start = np.datetime64(journey_config.time_start)
        time_end = np.datetime64(journey_config.time_end)

        # Compute spatial bounds if cropping is enabled
        spatial_bounds = None
        if config.enable_spatial_cropping:
            spatial_bounds = compute_ellipse_bbox(
                lon_start=journey_config.lon_waypoints[0],
                lat_start=journey_config.lat_waypoints[0],
                lon_end=journey_config.lon_waypoints[-1],
                lat_end=journey_config.lat_waypoints[-1],
                length_multiplier=config.route_length_multiplier,
                buffer_degrees=config.spatial_buffer_degrees,
            )

        forcing = ForcingData(
            currents=load_currents(
                data_file=config.currents_path,
                time_start=time_start,
                time_end=time_end,
                load_eagerly=config.load_eagerly,
                engine=config.engine,
                chunks=config.chunks,
                spatial_bounds=spatial_bounds,
            ),
            waves=load_waves(
                data_file=config.waves_path,
                time_start=time_start,
                time_end=time_end,
                load_eagerly=config.load_eagerly,
                engine=config.engine,
                chunks=config.chunks,
                spatial_bounds=spatial_bounds,
            ),
            winds=load_winds(
                data_file=config.winds_path,
                time_start=time_start,
                time_end=time_end,
                load_eagerly=config.load_eagerly,
                engine=config.engine,
                chunks=config.chunks,
                spatial_bounds=spatial_bounds,
            ),
        )
        self._log_stage_metrics(
            "load_forcing",
            currents=forcing.currents is not None,
            currents_shape=(
                str(forcing.currents.sizes) if forcing.currents is not None else "{}"
            ),
            waves=forcing.waves is not None,
            waves_shape=str(forcing.waves.sizes) if forcing.waves is not None else "{}",
            winds=forcing.winds is not None,
            winds_shape=str(forcing.winds.sizes) if forcing.winds is not None else "{}",
        )
        return forcing

    @profile
    def _stage_initialization(
        self, forcing: ForcingData
    ) -> tuple[PopulationMember, Population]:
        """Stage 0: Initialize seed route and population.

        Creates the seed route from journey configuration,
        evaluates its cost, and initializes the population
        as M copies of the seed member.

        Returns
        -------
        tuple[PopulationMember, Population]
            The seed member and initial population
        """
        # Create seed route from journey configuration
        # Note: JourneyConfig.__post_init__ ensures time_end is set,
        # so we don't pass speed_knots to avoid over-constraining
        seed_route = Route.create_route(
            lon_waypoints=self.config.journey.lon_waypoints,
            lat_waypoints=self.config.journey.lat_waypoints,
            time_start=self.config.journey.time_start,
            time_end=self.config.journey.time_end,
            speed_knots=None,
            time_resolution_hours=self.config.journey.time_resolution_hours,
        )

        # Evaluate seed route cost
        seed_member = PopulationMember(
            route=seed_route, cost=self._route_cost(seed_route, forcing)
        )

        # Initialize population as M copies of seed
        population = Population.from_members(
            [seed_member] * self.config.hyper.population_size
        )

        self._log_stage_metrics(
            "initialization",
            population_size=population.size,
            seed_route_cost=seed_member.cost,
        )

        return seed_member, population

    @staticmethod
    def _task_warmup(member: PopulationMember) -> PopulationMember:
        """Task function for parallel warmup stage.

        Parameters
        ----------
        member : PopulationMember
            Population member to mutate and select

        Returns
        -------
        PopulationMember
            The processed member after mutation and selection
        """
        state = _get_state()
        length = member.route.length_meters

        mutated_route = stochastic_mutation(
            route=member.route,
            max_iterations=state.params.mutation_iterations,
            mod_width=state.params.mutation_width_fraction_warmup * length,
            max_move_meters=state.params.mutation_displacement_fraction_warmup * length,
            rng=state.rng,
        )
        mutated_cost = mutated_route.cost_through(
            current_data_set=state.forcing.currents,
            wave_data_set=state.forcing.waves,
            wind_data_set=state.forcing.winds,
        )

        selected_route, selected_cost = select_from_pair(
            p=state.params.selection_acceptance_rate_warmup,
            route_a=member.route,
            route_b=mutated_route,
            cost_a=member.cost,
            cost_b=mutated_cost,
            rng=state.rng,
        )
        return PopulationMember(route=selected_route, cost=selected_cost)

    @profile
    def _stage_warmup(
        self,
        population: Population,
        seed_member: PopulationMember,
        forcing: ForcingData,
        executor: ProcessPoolExecutor | None = None,
    ) -> Population:
        """Stage 1: Warmup - diversify initial population.

        Mutates all members with warmup parameters,
        then adds seed back.
        """
        params = self.config.hyper
        rng = self._ensure_rng()

        # Mutate M-1 members with warmup parameters
        members_to_process = population.members[:-1]

        warmed_members = list(executor.map(RoutingApp._task_warmup, members_to_process))

        # Add seed back: P ← P ∪ {r_seed}
        population = Population.from_members(warmed_members).add_member(seed_member)
        population.remove_invalid()

        self._log_stage_metrics(
            "warmup",
            **self._population_stats(population.members),
        )

        return population

    @staticmethod
    def _task_mutation(
        member: PopulationMember, W: float, D: float
    ) -> PopulationMember:
        """Task function for parallel GA mutation stage.

        Parameters
        ----------
        member : PopulationMember
            Population member to mutate and select
        W : float
            Mutation width fraction
        D : float
            Mutation displacement fraction

        Returns
        -------
        PopulationMember
            The processed member after mutation and selection
        """
        state = _get_state()
        length = member.route.length_meters

        mutated_route = stochastic_mutation(
            route=member.route,
            max_iterations=state.params.mutation_iterations,
            mod_width=W * length,
            max_move_meters=D * length,
            rng=state.rng,
        )
        mutated_cost = mutated_route.cost_through(
            current_data_set=state.forcing.currents,
            wave_data_set=state.forcing.waves,
            wind_data_set=state.forcing.winds,
        )
        selected_route, selected_cost = select_from_pair(
            p=state.params.selection_acceptance_rate,
            route_a=member.route,
            route_b=mutated_route,
            cost_a=member.cost,
            cost_b=mutated_cost,
            rng=state.rng,
        )
        return PopulationMember(route=selected_route, cost=selected_cost)

    @profile
    def _stage_ga_mutation(
        self,
        population: Population,
        seed_member: PopulationMember,
        forcing: ForcingData,
        W: float,
        D: float,
        q: float,
        executor: ProcessPoolExecutor | None = None,
    ) -> tuple[Population, CostImprovementStats]:
        """GA sub-stage 1: Directed mutation of population members."""
        params = self.config.hyper
        rng = self._ensure_rng()
        members = population.members

        # Track elite cost before mutation (q/2 quantile)
        elite_quantile = max(1, int(np.ceil(q / 2 * len(members))))
        costs_before = sorted([m.cost for m in members])
        elite_cost_before = costs_before[elite_quantile - 1]  # 0-indexed

        # Directed mutation
        members_to_process = members[:-1]

        # Prepare arguments for workers (member, W, D)
        worker_args = [(member, W, D) for member in members_to_process]

        mutated_members = list(
            executor.map(RoutingApp._task_mutation, *zip(*worker_args))
        )

        # New population incl. seed member again
        population = Population.from_members(mutated_members).add_member(seed_member)

        # Track elite cost after mutation (q/2 quantile)
        costs_after = sorted([m.cost for m in population.members])
        elite_cost_after = costs_after[elite_quantile - 1]

        # Compute improvement statistics
        stats = CostImprovementStats(
            cost_before=elite_cost_before,
            cost_after=elite_cost_after,
        )

        self._log_stage_metrics(
            "ga_mutation",
            **self._population_stats(population.members),
            **stats.to_dict(),
            elite_quantile=elite_quantile,
        )

        return population, stats

    @staticmethod
    def _task_crossover(
        parent_indices: tuple[int, int], population_members: list[PopulationMember]
    ) -> PopulationMember:
        """Task function for parallel GA crossover stage.

        Parameters
        ----------
        parent_indices : tuple[int, int]
            Indices of parents in population_members
        population_members : list[PopulationMember]
            Population to select parents from

        Returns
        -------
        PopulationMember
            The offspring member after crossover and cost evaluation
        """
        state = _get_state()
        parent_a = population_members[parent_indices[0]]
        parent_b = population_members[parent_indices[1]]

        try:
            if state.params.crossover_strategy == "minimal_cost":
                child_member = crossover_routes_minimal_cost(
                    parent_a,
                    parent_b,
                    current_data_set=state.forcing.currents,
                    wind_data_set=state.forcing.winds,
                    wave_data_set=state.forcing.waves,
                    hazard_penalty_multiplier=state.params.hazard_penalty_multiplier,
                )
            else:  # "random"
                child_member = crossover_routes_random(
                    parent_a,
                    parent_b,
                    current_data_set=state.forcing.currents,
                    wind_data_set=state.forcing.winds,
                    wave_data_set=state.forcing.waves,
                    hazard_penalty_multiplier=state.params.hazard_penalty_multiplier,
                )
        except Exception:
            logging.warning("crossover failed; using parent_a")
            return parent_a

        # Cost already computed in crossover function (benefits from caching)
        return child_member

    @profile
    def _stage_ga_crossover(
        self,
        population: Population,
        seed_member: PopulationMember,
        forcing: ForcingData,
        executor: ProcessPoolExecutor | None = None,
    ) -> Population:
        """GA sub-stage 2: Crossover to generate offspring through accumulating rounds.

        Implements N_crossover rounds where each round:
        - Creates offspring_size offspring from the previous round's offspring
        - Accumulates all offspring: P -> (P + P') -> (P + P' + P'') -> ...

        If crossover_rounds = 0, skips crossover entirely.
        """
        params = self.config.hyper
        M = params.population_size
        rng = self._ensure_rng()

        # Handle crossover_rounds = 0: skip crossover entirely
        if params.crossover_rounds == 0:
            return population.add_member(seed_member)

        # Accumulating rounds: each round creates offspring from previous round
        current_source = list(population.members)
        accumulated_offspring = []

        for round_idx in range(params.crossover_rounds):
            # Select parent indices for all offspring in this round
            parent_indices_list = []
            for _ in range(params.offspring_size):
                indices = rng.choice(len(current_source), size=2, replace=True)
                parent_indices_list.append(tuple(indices))

            # Prepare arguments for workers (bug fix: moved outside offspring loop)
            worker_args = [
                (parent_indices, current_source)
                for parent_indices in parent_indices_list
            ]

            # Create offspring for this round in parallel
            round_offspring = list(
                executor.map(RoutingApp._task_crossover, *zip(*worker_args))
            )

            # Accumulate offspring
            accumulated_offspring.extend(round_offspring)

            # For next round, use only this round's offspring as parents
            current_source = round_offspring

        # Combine: P (original population) + all accumulated offspring + seed
        combined_members = list(population.members) + accumulated_offspring
        result_population = Population.from_members(combined_members).add_member(
            seed_member
        )

        self._log_stage_metrics(
            "ga_crossover",
            **self._population_stats(result_population.members),
        )

        return result_population

    @profile
    def _stage_ga_selection(
        self,
        population: Population,
        seed_member: PopulationMember,
        q: float,
    ) -> Population:
        """GA sub-stage 3: Selection from population."""
        params = self.config.hyper
        M = params.population_size

        # Selection from offspring: Add seed route, select, add seed route.
        population_with_seed = population.add_member(seed_member)
        selected_members = select_from_population(
            members=population_with_seed.members,
            quantile=q,
            target_size=M - 1,
            rng=self._rng,
        )
        population = Population.from_members(selected_members).add_member(seed_member)

        self._log_stage_metrics(
            "ga_selection",
            **self._population_stats(population.members),
        )

        return population

    @profile
    def _stage_ga_adaptation(
        self,
        W: float,
        D: float,
        q: float,
        cost_improvement_stats: CostImprovementStats,
        population_stats: dict[str, Any],
    ) -> tuple[float, float, float]:
        """GA sub-stage 4: Adapt mutation and selection parameters.

        Parameters
        ----------
        W : float
            Current mutation width fraction
        D : float
            Current mutation displacement fraction
        q : float
            Current selection quantile
        cost_improvement_stats : CostImprovementStats
            Cost improvement statistics from the mutation stage
        population_stats : dict
            Population statistics (including cost_std, cost_mean)

        Returns
        -------
        tuple[float, float, float]
            Updated (W_new, D_new, q_new)
        """
        params = self.config.hyper

        if not params.enable_adaptation:
            # Adaptation disabled, log and return unchanged
            self._log_stage_metrics(
                "ga_adaptation",
                W=W,
                D=D,
                q=q,
                adaptation_enabled=False,
            )
            return W, D, q

        # Adapt W and D based on cost improvement
        improvement = cost_improvement_stats.relative_improvement
        target = params.target_relative_improvement

        if improvement < target:
            W_new = W * params.adaptation_scale_W
            D_new = D * params.adaptation_scale_D
        else:
            W_new = W
            D_new = D

        # q remains unchanged
        q_new = q

        # Enforce bounds on W and D only
        W_new = np.clip(W_new, params.W_min, params.W_max)
        D_new = np.clip(D_new, params.D_min, params.D_max)

        # Log adaptation decisions
        self._log_stage_metrics(
            "ga_adaptation",
            W=W_new,
            D=D_new,
            q=q_new,
            W_delta=W_new / W,
            D_delta=D_new / D,
            relative_improvement=improvement,
            target_relative_improvement=target,
            adaptation_enabled=True,
        )

        return W_new, D_new, q_new

    @staticmethod
    def _task_gradient_descent(member: PopulationMember) -> PopulationMember:
        """Task function for parallel gradient descent stage.

        Parameters
        ----------
        member : PopulationMember
            Elite member to optimize with gradient descent

        Returns
        -------
        PopulationMember
            The processed member after gradient descent
        """
        state = _get_state()

        route = gradient_descent(
            route=member.route,
            learning_rate_percent_time=state.params.learning_rate_time,
            time_increment=state.params.time_increment,
            learning_rate_percent_along=state.params.learning_rate_space,
            dist_shift_along=state.params.distance_increment,
            learning_rate_percent_across=state.params.learning_rate_space,
            dist_shift_across=state.params.distance_increment,
            current_data_set=state.forcing.currents,
            wave_data_set=state.forcing.waves,
            wind_data_set=state.forcing.winds,
        )
        cost = route.cost_through(
            current_data_set=state.forcing.currents,
            wave_data_set=state.forcing.waves,
            wind_data_set=state.forcing.winds,
            hazard_penalty_multiplier=state.params.hazard_penalty_multiplier,
        )
        return PopulationMember(route=route, cost=cost)

    @profile
    def _stage_post_processing(
        self,
        population: Population,
        forcing: ForcingData,
        executor: ProcessPoolExecutor | None = None,
    ) -> Population:
        """Stage 3: Gradient descent polishing of elite members."""
        params = self.config.hyper
        if not population.members:
            return Population(members=[])

        # Extract elite members
        sorted_population = population.sort()
        elite_members = sorted_population.members[: params.num_elites]

        # Log initial state
        self._log_stage_metrics(
            "gradient_polishing",
            population_size=len(elite_members),
            elites=len(elite_members),
        )

        # Initialize worker state (for both parallel and sequential modes)
        rng = self._ensure_rng()

        # Outer loop: GD iterations (like GA generations)
        for gd_iter in range(params.gd_iterations):
            updated_members = list(
                executor.map(RoutingApp._task_gradient_descent, elite_members)
            )

            # Update elite population for next iteration
            elite_members = updated_members

            # Log population statistics (same schema as GA)
            self._log_stage_metrics(
                "gd_iteration",
                **self._population_stats(elite_members),
            )

        return Population(members=elite_members)

    def _log_stage_metrics(
        self,
        name: str,
        **metrics: Any,
    ) -> None:
        """Convenience wrapper for stage-level logging."""
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        logging.info("%s [%s] %s", name, timestamp, metrics)
        self.log.add_stage(name=name, **metrics)

    def _route_cost(self, route: Route, forcing: ForcingData) -> float:
        params = self.config.hyper
        return route.cost_through(
            current_data_set=forcing.currents,
            wave_data_set=forcing.waves,
            wind_data_set=forcing.winds,
            ship=self.config.ship,
            physics=self.config.physics,
            hazard_penalty_multiplier=params.hazard_penalty_multiplier,
        )

    def _population_stats(
        self, population: Sequence[PopulationMember]
    ) -> dict[str, Any]:
        if not population:
            return {
                "population_size": 0,
                "cost_min": np.nan,
                "cost_max": np.nan,
                "cost_mean": np.nan,
                "cost_median": np.nan,
                "cost_std": np.nan,
                "cost_q25": np.nan,
                "cost_q75": np.nan,
            }
        costs = np.array([member.cost for member in population])
        return {
            "population_size": len(population),
            "cost_min": float(np.nanmin(costs)),
            "cost_max": float(np.nanmax(costs)),
            "cost_mean": float(np.nanmean(costs)),
            "cost_median": float(np.nanmedian(costs)),
            "cost_std": float(np.nanstd(costs)),
            "cost_q25": float(np.nanquantile(costs, 0.25)),
            "cost_q75": float(np.nanquantile(costs, 0.75)),
        }

    def _ensure_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(self.config.hyper.random_seed)
        return self._rng
