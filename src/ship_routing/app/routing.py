from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np
from matplotlib import pyplot as plt
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
        if ax is None:
            _, ax = plt.subplots(1, 1)

        seed_member = self.seed_member
        elite_members = self.elite_population.members

        for em in elite_members:
            ax.plot(*em.route.line_string.xy, "black")
        ax.plot(*seed_member.route.line_string.xy, "orange")
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

    def run(self) -> RoutingResult:
        """Execute the optimisation pipeline."""
        self._log_stage_metrics("run", message="starting routing run")
        self._rng = np.random.default_rng(self.config.hyper.random_seed)

        forcing = self._load_forcing(self.config.journey)

        # Stage 0 to 4:
        seed_member, population = self._stage_initialization(forcing)
        population = self._stage_warmup(population, seed_member, forcing)

        # Initialize adaptive parameters for genetic algorithm
        W = self.config.hyper.mutation_width_fraction
        D = self.config.hyper.mutation_displacement_fraction
        q = self.config.hyper.selection_quantile

        # Genetic algorithm generation loop
        for _ in range(self.config.hyper.generations):
            population = self._stage_ga_mutation(
                population, seed_member, forcing, W, D, q
            )
            population = self._stage_ga_crossover(population, seed_member, forcing)
            population = self._stage_ga_selection(population, seed_member, q)
            W, D, q = self._stage_ga_adaptation(W, D, q)

        elite_population = self._stage_post_processing(population, forcing)

        return RoutingResult(
            seed_member=seed_member,
            elite_population=elite_population,
            logs=self.log,
        )

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
            waves=forcing.waves is not None,
            winds=forcing.winds is not None,
        )
        return forcing

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

    def _stage_warmup(
        self,
        population: Population,
        seed_member: PopulationMember,
        forcing: ForcingData,
    ) -> Population:
        """Stage 1: Warmup - diversify initial population.

        Mutates all members with warmup parameters,
        then adds seed back.
        """
        params = self.config.hyper
        rng = self._ensure_rng()

        # Mutate M-1 members with warmup parameters
        warmed_members = []
        for member in population.members[:-1]:
            length = member.route.length_meters

            # Apply mutation: M_{W_w,D_w}(r_m)
            mutated_route = stochastic_mutation(
                route=member.route,
                number_of_iterations=params.mutation_iterations,
                mod_width=params.mutation_width_fraction * length,
                max_move_meters=params.mutation_displacement_fraction * length,
                rng=rng,
            )
            mutated_cost = self._route_cost(mutated_route, forcing)

            selected_route, selected_cost = select_from_pair(
                p=params.selection_acceptance_rate_warmup,
                route_a=member.route,
                route_b=mutated_route,
                cost_a=member.cost,
                cost_b=mutated_cost,
                rng=rng,
            )
            warmed_members.append(
                PopulationMember(route=selected_route, cost=selected_cost)
            )

        # Add seed back: P ← P ∪ {r_seed}
        population = Population.from_members(warmed_members).add_member(seed_member)
        population.remove_invalid()

        self._log_stage_metrics(
            "warmup",
            **self._population_stats(population.members),
        )

        return population

    def _stage_ga_mutation(
        self,
        population: Population,
        seed_member: PopulationMember,
        forcing: ForcingData,
        W: float,
        D: float,
        q: float,
    ) -> Population:
        """GA sub-stage 1: Directed mutation of population members."""
        params = self.config.hyper
        rng = self._ensure_rng()
        members = population.members

        # Directed mutation
        mutated_members = []
        for member in members[:-1]:
            length = member.route.length_meters
            mutated_route = stochastic_mutation(
                route=member.route,
                number_of_iterations=params.mutation_iterations,
                mod_width=W * length,
                max_move_meters=D * length,
                rng=rng,
            )
            mutated_cost = self._route_cost(mutated_route, forcing)
            selected_route, selected_cost = select_from_pair(
                p=params.selection_acceptance_rate,
                route_a=member.route,
                route_b=mutated_route,
                cost_a=member.cost,
                cost_b=mutated_cost,
                rng=rng,
            )
            mutated_members.append(
                PopulationMember(route=selected_route, cost=selected_cost)
            )

        # New population incl. seed member again
        population = Population.from_members(mutated_members).add_member(seed_member)

        self._log_stage_metrics(
            "ga_mutation",
            **self._population_stats(population.members),
        )

        return population

    def _stage_ga_crossover(
        self,
        population: Population,
        seed_member: PopulationMember,
        forcing: ForcingData,
    ) -> Population:
        """GA sub-stage 2: Crossover to generate offspring."""
        params = self.config.hyper
        M = params.population_size

        # Crossover
        if params.crossover_rounds == 0:
            # No crossover: use mutated population directly
            # Get members without seed (last member is always seed)
            offspring_members = (
                population.members[:-1].copy()
                if hasattr(population.members[:-1], "copy")
                else list(population.members[:-1])
            )
        else:
            offspring_members = []
            for _ in range(params.crossover_rounds):
                for _ in range(M):
                    # Select two parents from current population
                    parent_a, parent_b = self._rng.choice(
                        population.members, size=2, replace=False
                    )

                    # Apply crossover operator C_s
                    try:
                        if params.crossover_strategy == "minimal_cost":
                            child_route = crossover_routes_minimal_cost(
                                parent_a.route,
                                parent_b.route,
                                current_data_set=forcing.currents,
                                wind_data_set=forcing.winds,
                                wave_data_set=forcing.waves,
                            )
                        else:  # "random"
                            child_route = crossover_routes_random(
                                parent_a.route, parent_b.route
                            )
                    except Exception:
                        logging.warning(
                            "crossover_routes_minimal_cost failed; using parent_a"
                        )
                        child_route = parent_a.route

                    offspring_members.append(
                        PopulationMember(
                            route=child_route,
                            cost=self._route_cost(child_route, forcing),
                        )
                    )

        # Add back seed member
        offspring = Population.from_members(offspring_members).add_member(seed_member)

        self._log_stage_metrics(
            "ga_crossover",
            **self._population_stats(offspring.members),
        )

        return offspring

    def _stage_ga_selection(
        self,
        population: Population,
        seed_member: PopulationMember,
        q: float,
    ) -> Population:
        """GA sub-stage 3: Selection from population."""
        params = self.config.hyper
        M = params.population_size

        # Selection from offspring
        selected_members = select_from_population(
            members=population.members,
            quantile=q,
            target_size=M - 1,
            rng=self._rng,
        )

        # Add back seed route
        population = Population.from_members(selected_members).add_member(seed_member)

        self._log_stage_metrics(
            "ga_selection",
            **self._population_stats(population.members),
        )

        return population

    def _stage_ga_adaptation(
        self,
        W: float,
        D: float,
        q: float,
    ) -> tuple[float, float, float]:
        """GA sub-stage 4: Adapt mutation and selection parameters."""
        # TODO: Implement adaptive W, D, q
        W_new, D_new, q_new = W, D, q

        self._log_stage_metrics(
            "ga_adaptation",
            W=W_new,
            D=D_new,
            q=q_new,
        )

        return W_new, D_new, q_new

    def _stage_post_processing(
        self,
        population: Population,
        forcing: ForcingData,
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

        # Outer loop: GD iterations (like GA generations)
        for gd_iter in range(params.gd_iterations):
            # Inner loop: Apply GD to each elite member
            updated_members = []
            for member in elite_members:
                route = gradient_descent(
                    route=member.route,
                    learning_rate_percent_time=params.learning_rate_time,
                    time_increment=params.time_increment,
                    learning_rate_percent_along=params.learning_rate_space,
                    dist_shift_along=params.distance_increment,
                    learning_rate_percent_across=params.learning_rate_space,
                    dist_shift_across=params.distance_increment,
                    current_data_set=forcing.currents,
                    wave_data_set=forcing.waves,
                    wind_data_set=forcing.winds,
                )
                cost = self._route_cost(route, forcing)
                updated_members.append(PopulationMember(route=route, cost=cost))

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
        return route.cost_through(
            current_data_set=forcing.currents,
            wave_data_set=forcing.waves,
            wind_data_set=forcing.winds,
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
