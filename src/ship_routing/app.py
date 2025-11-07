from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import logging
import random
from statistics import mean
from typing import Any, Sequence

import numpy as np

from .algorithms import (
    crossover_routes_minimal_cost,
    crossover_routes_random,
)
from .config import ForcingConfig, ForcingData, RoutingConfig
from .convenience import create_route, gradient_descent, stochastic_search
from .core import Route
from .data import load_currents, load_waves, load_winds


@dataclass
class RoutingResult:
    """Container returned by RoutingApp.run."""

    best_routes: Sequence[Route] | None
    logs: "RoutingLog | None" = None


@dataclass
class StageLog:
    """Record of a single optimisation stage (per generation/iteration)."""

    name: str
    iteration: int
    metrics: dict[str, Any]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


@dataclass
class RoutingLog:
    """Structured information needed for paper figures / reproducibility."""

    config: dict[str, Any]
    stages: list[StageLog] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add_stage(self, name: str, iteration: int, **metrics: Any) -> None:
        """Append a stage log entry."""
        self.stages.append(StageLog(name=name, iteration=iteration, metrics=metrics))

    def add_note(self, message: str) -> None:
        """Attach a free-form note (e.g., warnings, manual tweaks)."""
        self.notes.append(message)

    def set_metadata(self, **metadata: Any) -> None:
        """Store high-level metadata such as dataset descriptions."""
        self.metadata.update(metadata)


@dataclass
class PopulationMember:
    route: Route
    cost: float


class RoutingApp:
    """High-level orchestrator wrapping the routing workflow."""

    def __init__(self, config: RoutingConfig):
        self.config = config
        self.log = RoutingLog(config=asdict(config))

    def run(self) -> RoutingResult:
        """Execute the optimisation pipeline."""
        self._log_stage_metrics("run", 0, message="starting routing run")
        forcing = self._load_forcing()
        population, seed_member = self._initialize_population(
            forcing, self.config.journey, self.config.population
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
        return RoutingResult(best_routes=best_routes, logs=self.log)

    def _load_forcing(self) -> ForcingData:
        """Load wind, wave, and current fields according to the config."""
        forcing_config = self.config.forcing
        forcing = ForcingData(
            currents=self._load_single_forcing(
                forcing_config,
                forcing_config.currents_path,
                load_currents,
            ),
            waves=self._load_single_forcing(
                forcing_config,
                forcing_config.waves_path,
                load_waves,
            ),
            winds=self._load_single_forcing(
                forcing_config,
                forcing_config.winds_path,
                load_winds,
            ),
        )
        self._log_stage_metrics(
            "load_forcing",
            0,
            currents=bool(forcing.currents),
            waves=bool(forcing.waves),
            winds=bool(forcing.winds),
        )
        return forcing

    def _load_single_forcing(self, config: ForcingConfig, path: str | None, loader):
        if not path:
            return None
        ds = loader(
            data_file=path,
            engine=config.engine,
            chunks=config.chunks,
        )
        if config.time_steps is not None and "time" in ds.dims:
            ds = ds.isel(time=slice(None, config.time_steps))
        if config.load_eagerly:
            ds = ds.load()
        return ds

    def _initialize_population(
        self, forcing: ForcingData, journey_config, population_config
    ) -> tuple[list[PopulationMember], PopulationMember]:
        """Seed the initial population using the configured journey."""
        journey = journey_config
        route = create_route(
            lon_waypoints=journey.lon_waypoints,
            lat_waypoints=journey.lat_waypoints,
            time_start=journey.time_start,
            time_end=journey.time_end,
            speed_knots=journey.speed_knots,
            time_resolution_hours=journey.time_resolution_hours,
        )
        seed_cost = self._route_cost(route, forcing)
        self._log_stage_metrics(
            "initialize_population",
            0,
            population_size=population_config.size,
            seed_route_cost=seed_cost,
        )
        seed_member = PopulationMember(route=route, cost=seed_cost)
        members = [seed_member]
        for _ in range(1, population_config.size):
            member_route = deepcopy(route)
            members.append(
                PopulationMember(
                    route=member_route,
                    cost=self._route_cost(member_route, forcing),
                )
            )
        return members, seed_member

    def _run_ga_generations(
        self,
        population: Sequence[PopulationMember],
        seed_member: PopulationMember,
        forcing: ForcingData,
        population_config,
        stochastic_config,
        crossover_config,
        selection_config,
    ) -> Sequence[PopulationMember]:
        """Apply stochastic search, crossover, and selection loops."""
        members = list(population)
        target_size = population_config.size
        if population_config.random_seed is not None:
            random.seed(population_config.random_seed)
            np.random.seed(population_config.random_seed)
        num_generations = stochastic_config.num_generations or 1
        for generation in range(num_generations):
            mutated = self._mutate_population(members, forcing, stochastic_config)
            offspring = mutated
            crossover_rounds = crossover_config.generations or 1
            for _ in range(crossover_rounds):
                offspring = self._crossover_population(
                    offspring, forcing, crossover_config
                )
            combined = members + mutated + offspring
            if population_config.mix_seed_route_each_generation:
                combined.append(
                    PopulationMember(
                        route=deepcopy(seed_member.route),
                        cost=self._route_cost(seed_member.route, forcing),
                    )
                )
            members = self._select_population(
                combined, selection_config, target_size=target_size
            )
            self._log_stage_metrics(
                "ga_generation",
                generation,
                **self._population_stats(members),
            )
        return members

    def _refine_with_gradient(
        self,
        population: Sequence[PopulationMember],
        forcing: ForcingData,
        gradient_config,
    ) -> Sequence[Route]:
        """Run gradient descent on the elites and return them."""
        num_elites = gradient_config.num_elites or 1
        sorted_population = sorted(population, key=lambda m: m.cost)
        elite_count = min(max(1, num_elites), len(sorted_population) or 1)
        elites = sorted_population[:elite_count]
        if not gradient_config.enabled:
            self._log_stage_metrics(
                "gradient_refinement",
                0,
                population_size=len(population),
                elites=elite_count,
                skipped=True,
            )
            return [member.route for member in elites]
        self._log_stage_metrics(
            "gradient_refinement",
            0,
            population_size=len(population),
            elites=elite_count,
        )
        refined_routes = []
        for idx, member in enumerate(elites):
            route, _ = gradient_descent(
                route=member.route,
                num_iterations=gradient_config.num_iterations,
                learning_rate_percent_time=gradient_config.learning_rate_percent_time,
                time_increment=gradient_config.time_increment,
                learning_rate_percent_along=gradient_config.learning_rate_percent_along,
                dist_shift_along=gradient_config.dist_shift_along,
                learning_rate_percent_across=gradient_config.learning_rate_percent_across,
                dist_shift_across=gradient_config.dist_shift_across,
                include_logs_routes=False,
                current_data_set=forcing.currents,
                wave_data_set=forcing.waves,
                wind_data_set=forcing.winds,
            )
            cost = self._route_cost(route, forcing)
            refined_routes.append(route)
            self._log_stage_metrics(
                "gradient_step",
                idx,
                pre_cost=member.cost,
                post_cost=cost,
            )
        return refined_routes

    def _mutate_population(
        self,
        population: Sequence[PopulationMember],
        forcing: ForcingData,
        stochastic_config,
    ) -> list[PopulationMember]:
        mutated = []
        for member in population:
            length = getattr(member.route, "length_meters", None) or 1.0
            mod_width = max(
                stochastic_config.warmup_mod_width_fraction * length,
                1.0,
            )
            max_move = max(
                stochastic_config.warmup_max_move_fraction * length,
                1.0,
            )
            route, _ = stochastic_search(
                route=member.route,
                number_of_iterations=stochastic_config.num_iterations,
                acceptance_rate_target=stochastic_config.acceptance_rate_target,
                acceptance_rate_for_increase_cost=stochastic_config.acceptance_rate_for_increase_cost,
                refinement_factor=stochastic_config.refinement_factor,
                mod_width=mod_width,
                max_move_meters=max_move,
                include_logs_routes=False,
                current_data_set=forcing.currents,
                wave_data_set=forcing.waves,
                wind_data_set=forcing.winds,
            )
            mutated.append(
                PopulationMember(route=route, cost=self._route_cost(route, forcing))
            )
        return mutated

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
        for _ in range(len(population)):
            parent_a, parent_b = random.sample(population, 2)
            if strategy == "minimal_cost":
                child = crossover_routes_minimal_cost(
                    parent_a.route,
                    parent_b.route,
                    current_data_set=forcing.currents,
                    wind_data_set=forcing.winds,
                    wave_data_set=forcing.waves,
                )
            else:
                child = crossover_routes_random(parent_a.route, parent_b.route)
            offspring.append(
                PopulationMember(route=child, cost=self._route_cost(child, forcing))
            )
        return offspring

    def _select_population(
        self,
        population: Sequence[PopulationMember],
        selection_config,
        target_size: int,
    ) -> list[PopulationMember]:
        if not population:
            return []
        size = target_size or len(population)
        sorted_pop = sorted(population, key=lambda member: member.cost)
        elite_fraction = selection_config.elite_fraction or 0.0
        elite_count = min(size, int(size * elite_fraction))
        elites = sorted_pop[:elite_count]
        remaining = sorted_pop[elite_count:]
        quantile = selection_config.quantile
        pool_count = max(1, int(size * quantile)) if quantile else len(remaining)
        pool = remaining[:pool_count] if remaining else []
        survivors = elites.copy()
        needed = size - len(survivors)
        if needed <= 0:
            return survivors[:size]
        if not pool:
            pool = remaining if remaining else sorted_pop
        if selection_config.with_replacement:
            survivors.extend(random.choices(pool, k=needed))
        else:
            survivors.extend(pool[:needed])
        return survivors[:size]

    def _log_stage_metrics(
        self,
        name: str,
        iteration: int,
        **metrics: Any,
    ) -> None:
        """Convenience wrapper for stage-level logging."""
        logging.info("%s[%s] %s", name, iteration, metrics)
        self.log.add_stage(name=name, iteration=iteration, **metrics)

    def _route_cost(self, route: Route, forcing: ForcingData) -> float:
        return route.cost_through(
            current_data_set=forcing.currents,
            wave_data_set=forcing.waves,
            wind_data_set=forcing.winds,
        )

    def _population_stats(self, population: Sequence[PopulationMember]) -> dict[str, Any]:
        if not population:
            return {"population_size": 0}
        costs = [member.cost for member in population]
        return {
            "population_size": len(population),
            "cost_min": min(costs),
            "cost_max": max(costs),
            "cost_mean": mean(costs),
        }
