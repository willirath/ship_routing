from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np

from ..algorithms.optimization import (
    crossover_routes_minimal_cost,
    crossover_routes_random,
    gradient_descent,
    stochastic_mutation,
)
from .config import ForcingConfig, ForcingData, RoutingConfig
from ..core.routes import Route
from ..core.data import load_currents, load_waves, load_winds, load_and_filter_forcing
from ..core.population import Population, PopulationMember

np.seterr(divide="ignore", invalid="ignore")


# TODO: Refactor to contain seed_member instead of seed_route and
# elite_population instead of best_routes.  Note that this will
# also include creating a PopulationMember .to_dict() and a
# Population.to_dict() method.
@dataclass
class RoutingResult:
    """Container returned by RoutingApp.run."""

    seed_route: Route | None = None
    best_routes: Sequence[Route] | None = None
    logs: "RoutingLog | None" = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "seed_route": self.seed_route.to_dict() if self.seed_route else None,
            "best_routes": [route.to_dict() for route in (self.best_routes or [])],
            "log": self.logs.to_dict() if self.logs else None,
        }

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


@dataclass
class StageLog:
    """Record of a single optimisation stage event."""

    name: str
    metrics: dict[str, Any]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


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
        self._rng = np.random.default_rng(self.config.population.random_seed)
        forcing = self._load_forcing(self.config.journey)
        seed_route = Route.create_route(
            lon_waypoints=self.config.journey.lon_waypoints,
            lat_waypoints=self.config.journey.lat_waypoints,
            time_start=self.config.journey.time_start,
            time_end=self.config.journey.time_end,
            speed_knots=self.config.journey.speed_knots,
            time_resolution_hours=self.config.journey.time_resolution_hours,
        )
        seed_member = PopulationMember(
            route=seed_route, cost=self._route_cost(seed_route, forcing)
        )
        population = self._initialize_population(
            forcing, seed_member, self.config.population
        )
        population = self._run_ga_generations(
            population,
            seed_member,
            forcing,
            self.config.population,
            self.config.stochastic,
            self.config.crossover,
            self.config.selection,
        )
        best_routes = self._refine_with_gradient(
            population, forcing, self.config.gradient
        )
        return RoutingResult(
            seed_route=seed_route,
            best_routes=best_routes,
            logs=self.log,
        )

    def _load_forcing(self, journey_config) -> ForcingData:
        """Load wind, wave, and current fields according to the config."""
        config = self.config.forcing
        time_start = np.datetime64(journey_config.time_start)
        time_end = np.datetime64(journey_config.time_end)

        forcing = ForcingData(
            currents=load_and_filter_forcing(
                path=config.currents_path,
                loader=load_currents,
                time_start=time_start,
                time_end=time_end,
                engine=config.engine,
                chunks=config.chunks,
                load_eagerly=config.load_eagerly,
            ),
            waves=load_and_filter_forcing(
                path=config.waves_path,
                loader=load_waves,
                time_start=time_start,
                time_end=time_end,
                engine=config.engine,
                chunks=config.chunks,
                load_eagerly=config.load_eagerly,
            ),
            winds=load_and_filter_forcing(
                path=config.winds_path,
                loader=load_winds,
                time_start=time_start,
                time_end=time_end,
                engine=config.engine,
                chunks=config.chunks,
                load_eagerly=config.load_eagerly,
            ),
        )
        self._log_stage_metrics(
            "load_forcing",
            currents=bool(forcing.currents),
            waves=bool(forcing.waves),
            winds=bool(forcing.winds),
        )
        return forcing

    def _initialize_population(
        self, forcing: ForcingData, seed_member: PopulationMember, population_config
    ) -> Population:
        """Seed the initial population using the configured journey."""
        self._log_stage_metrics(
            "initialize_population",
            population_size=population_config.size,
            seed_route_cost=seed_member.cost,
        )

        population = Population.from_seed_member(
            seed_member=seed_member,
            size=population_config.size,
        )

        return population

    # TODO: Needs to become a thin wrapper around a population method.
    def _run_ga_generations(
        self,
        population: Population,
        seed_member: PopulationMember,
        forcing: ForcingData,
        population_config,
        stochastic_config,
        crossover_config,
        selection_config,
    ) -> Population:
        """Apply stochastic search, crossover, and selection loops."""
        members = population.members
        target_size = population_config.size
        num_generations = max(stochastic_config.num_generations or 0, 0)
        for generation in range(num_generations):
            mutated = self._mutate_population(members, forcing, stochastic_config)
            offspring = mutated
            crossover_rounds = max(crossover_config.generations or 0, 0)
            for _ in range(crossover_rounds):
                offspring = self._crossover_population(
                    offspring,
                    forcing,
                    crossover_config,
                )
            combined = members + mutated + offspring
            if self.config.mix_seed_route_each_generation:
                combined.append(
                    PopulationMember(
                        route=deepcopy(seed_member.route),
                        cost=seed_member.cost,
                    )
                )
            members = self._select_population(
                combined,
                selection_config,
                target_size=target_size,
            )
            self._log_stage_metrics(
                "ga_generation",
                generation=generation,
                **self._population_stats(members),
            )
        return Population(members=members)

    def _refine_with_gradient(
        self,
        population: Population,
        forcing: ForcingData,
        gradient_config,
    ) -> Sequence[Route]:
        """Run gradient descent on the elites and return them."""
        if not population.members:
            return []
        sorted_population = population.sort()
        elites = sorted_population.members[: gradient_config.num_elites]
        if not gradient_config.enabled:
            self._log_stage_metrics(
                "gradient_refinement",
                population_size=population.size,
                elites=len(elites),
                skipped=True,
            )
            return [member.route for member in elites]
        self._log_stage_metrics(
            "gradient_refinement",
            population_size=population.size,
            elites=len(elites),
        )
        refined_routes = []
        for idx, member in enumerate(elites):
            route = gradient_descent(
                route=member.route,
                num_iterations=gradient_config.num_iterations,
                learning_rate_percent_time=gradient_config.learning_rate_percent_time,
                time_increment=gradient_config.time_increment,
                learning_rate_percent_along=gradient_config.learning_rate_percent_along,
                dist_shift_along=gradient_config.dist_shift_along,
                learning_rate_percent_across=gradient_config.learning_rate_percent_across,
                dist_shift_across=gradient_config.dist_shift_across,
                current_data_set=forcing.currents,
                wave_data_set=forcing.waves,
                wind_data_set=forcing.winds,
            )
            cost = self._route_cost(route, forcing)
            refined_routes.append(route)
            self._log_stage_metrics(
                "gradient_step",
                pre_cost=member.cost,
                post_cost=cost,
                elite_index=idx,
                member_seed_cost=member.cost,
            )
        return refined_routes

    # TODO: Put largely into core with a Population class.
    def _mutate_population(
        self,
        population: Sequence[PopulationMember],
        forcing: ForcingData,
        stochastic_config,
    ) -> list[PopulationMember]:
        mutated = []
        for member in population:
            length = member.route.length_meters
            # TODO: we want to more explicitly expose refinement to the user / optimizer.
            mod_width = stochastic_config.warmup_mod_width_fraction * length
            max_move = stochastic_config.warmup_max_move_fraction * length
            route = stochastic_mutation(
                route=member.route,
                number_of_iterations=stochastic_config.num_iterations,
                acceptance_rate_target=stochastic_config.acceptance_rate_target,
                acceptance_rate_for_increase_cost=stochastic_config.acceptance_rate_for_increase_cost,
                refinement_factor=stochastic_config.refinement_factor,
                mod_width=mod_width,
                max_move_meters=max_move,
                current_data_set=forcing.currents,
                wave_data_set=forcing.waves,
                wind_data_set=forcing.winds,
            )
            mutated.append(
                PopulationMember(route=route, cost=self._route_cost(route, forcing))
            )
        return mutated

    # TODO: Large parts of this logic should go into the core module.
    def _crossover_population(
        self,
        population: Sequence[PopulationMember],
        forcing: ForcingData,
        crossover_config,
    ) -> list[PopulationMember]:
        if len(population) < 2:
            return list(population)
        strategy = crossover_config.strategy
        offspring = []
        rng = self._ensure_rng()
        for _ in range(len(population)):
            idx_a, idx_b = rng.choice(len(population), size=2, replace=False)
            parent_a = population[idx_a]
            parent_b = population[idx_b]
            if strategy == "minimal_cost":
                # TODO: remove check when https://github.com/willirath/ship_routing/issues/53 is fixed
                try:
                    child = crossover_routes_minimal_cost(
                        parent_a.route,
                        parent_b.route,
                        current_data_set=forcing.currents,
                        wind_data_set=forcing.winds,
                        wave_data_set=forcing.waves,
                    )
                except UnboundLocalError:
                    logging.warning(
                        "crossover_routes_minimal_cost failed; falling back to first parent route",
                    )
                    child = parent_a.route
            else:
                child = crossover_routes_random(parent_a.route, parent_b.route)
            offspring.append(
                PopulationMember(route=child, cost=self._route_cost(child, forcing))
            )
        return offspring

    # TODO: This should go into the core submodule.
    # We'll probably need a population class there.
    def _select_population(
        self,
        population: Sequence[PopulationMember],
        selection_config,
        *,
        target_size: int,
    ) -> list[PopulationMember]:
        if not population:
            return []
        sorted_pop = sorted(population, key=lambda member: member.cost)
        quantile = selection_config.quantile
        elite_count = int(np.ceil(len(sorted_pop) * quantile))
        elite_pool = sorted_pop[:elite_count]
        rng = self._ensure_rng()
        indices = rng.integers(0, len(elite_pool), size=target_size)
        return [elite_pool[idx] for idx in indices]

    def _log_stage_metrics(
        self,
        name: str,
        **metrics: Any,
    ) -> None:
        """Convenience wrapper for stage-level logging."""
        logging.info("%s %s", name, metrics)
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
            }
        costs = [member.cost for member in population]
        return {
            "population_size": len(population),
            "cost_min": min(costs),
            "cost_max": max(costs),
            "cost_mean": mean(costs),
        }

    def _ensure_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(self.config.population.random_seed)
        return self._rng
