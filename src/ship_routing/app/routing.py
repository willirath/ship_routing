from __future__ import annotations

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
    select_from_pair,
    select_from_population,
    stochastic_mutation,
)
from .config import ForcingConfig, ForcingData, RoutingConfig
from ..core.routes import Route
from ..core.data import load_currents, load_waves, load_winds
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

        # Stage 0 to 4:
        seed_member, population = self._stage_initialization(forcing)
        population = self._stage_warmup(population, seed_member, forcing)
        population = self._stage_genetic_evolution(population, seed_member, forcing)
        elite_population = self._stage_gradient_descent(population, forcing)

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

        forcing = ForcingData(
            currents=load_currents(
                path=config.currents_path,
                time_start=time_start,
                time_end=time_end,
                load_eagerly=config.load_eagerly,
                engine=config.engine,
                chunks=config.chunks,
            ),
            waves=load_waves(
                path=config.waves_path,
                time_start=time_start,
                time_end=time_end,
                load_eagerly=config.load_eagerly,
                engine=config.engine,
                chunks=config.chunks,
            ),
            winds=load_winds(
                path=config.winds_path,
                time_start=time_start,
                time_end=time_end,
                load_eagerly=config.load_eagerly,
                engine=config.engine,
                chunks=config.chunks,
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
        seed_route = Route.create_route(
            lon_waypoints=self.config.journey.lon_waypoints,
            lat_waypoints=self.config.journey.lat_waypoints,
            time_start=self.config.journey.time_start,
            time_end=self.config.journey.time_end,
            speed_knots=self.config.journey.speed_knots,
            time_resolution_hours=self.config.journey.time_resolution_hours,
        )

        # Evaluate seed route cost
        seed_member = PopulationMember(
            route=seed_route,
            cost=self._route_cost(seed_route, forcing)
        )

        # Initialize population as M copies of seed
        population = Population.from_members(
            [seed_member] * self.config.population.size
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

        Pseudocode:
            P ← {S_2^(p_w)(r_m, M_{W_w,D_w}(r_m)) | m=1...M-1}
            P ← P ∪ {r_seed}

        Mutates all members except last with warmup parameters,
        then adds seed back.
        """
        warmup_cfg = self.config.warmup

        # Mutate M-1 members with warmup parameters
        warmed_members = []
        for member in population.members[:-1]:
            length = member.route.length_meters

            # Apply mutation: M_{W_w,D_w}(r_m)
            mutated_route = stochastic_mutation(
                route=member.route,
                number_of_iterations=warmup_cfg.num_iterations,
                acceptance_rate_target=warmup_cfg.acceptance_rate_target,
                acceptance_rate_for_increase_cost=warmup_cfg.acceptance_rate_for_increase_cost,
                refinement_factor=warmup_cfg.refinement_factor,
                mod_width=warmup_cfg.mod_width_fraction * length,
                max_move_meters=warmup_cfg.max_move_fraction * length,
                current_data_set=forcing.currents,
                wave_data_set=forcing.waves,
                wind_data_set=forcing.winds,
            )
            mutated_cost = self._route_cost(mutated_route, forcing)

            warmed_members.append(
                PopulationMember(route=mutated_route, cost=mutated_cost)
            )

        # Add seed back: P ← P ∪ {r_seed}
        population = Population.from_members(warmed_members).add_member(seed_member)

        self._log_stage_metrics(
            "warmup",
            population_size=population.size,
            **self._population_stats(population.members),
        )

        return population

    def _stage_genetic_evolution(
        self,
        population: Population,
        seed_member: PopulationMember,
        forcing: ForcingData,
    ) -> Population:
        """Stage 2: Genetic evolution via mutation, crossover, selection.

        Pseudocode (per generation):
            1. Directed mutation (line 399):
               P ← {S_2^(p)(r_m, M_{W,D}(r_m)) | m=1...M-1}
               P ← P ∪ {r_seed}

            2. Crossover (lines 402-406):
               For m = 1 to M_offspring:
                   Select two routes r_a, r_b ∈ P
                   P_offspring ← P_offspring ∪ {C_s(r_a, r_b)}
               P_offspring ← P_offspring ∪ {r_seed}

            3. Selection (line 407):
               P ← S_{M_offspring,q,M-1}(P_offspring)
               P ← P ∪ {r_seed}

            4. Parameter adaptation (line 409):
               Adapt W, D, q (TODO)
        """
        stochastic_cfg = self.config.stochastic
        crossover_cfg = self.config.crossover
        selection_cfg = self.config.selection
        M = self.config.population.size

        # Extract adaptive parameters (placeholder for now)
        W = stochastic_cfg.mod_width_fraction
        D = stochastic_cfg.max_move_fraction
        q = selection_cfg.quantile

        for generation in range(stochastic_cfg.num_generations):
            members = population.members

            # 3.1: Directed mutation (line 399)
            # P ← {S_2^(p)(r_m, M_{W,D}(r_m)) | m=1...M-1}
            mutated_members = []
            for member in members[:-1]:
                length = member.route.length_meters
                mutated_route = stochastic_mutation(
                    route=member.route,
                    number_of_iterations=stochastic_cfg.num_iterations,
                    acceptance_rate_target=stochastic_cfg.acceptance_rate_target,
                    acceptance_rate_for_increase_cost=stochastic_cfg.acceptance_rate_for_increase_cost,
                    refinement_factor=stochastic_cfg.refinement_factor,
                    mod_width=W * length,
                    max_move_meters=D * length,
                    current_data_set=forcing.currents,
                    wave_data_set=forcing.waves,
                    wind_data_set=forcing.winds,
                )
                mutated_members.append(
                    PopulationMember(
                        route=mutated_route,
                        cost=self._route_cost(mutated_route, forcing),
                    )
                )

            # P ← P ∪ {r_seed}
            population = Population.from_members(mutated_members).add_member(
                seed_member
            )

            # 3.2: Crossover (lines 402-406)
            # Create M_offspring children
            M_offspring = crossover_cfg.offspring_size
            offspring_members = []

            for _ in range(M_offspring):
                # Select two parents from current population
                parent_a, parent_b = self._rng.choice(
                    population.members, size=2, replace=False
                )

                # Apply crossover operator C_s
                if crossover_cfg.strategy == "minimal_cost":
                    try:
                        child_route = crossover_routes_minimal_cost(
                            parent_a.route,
                            parent_b.route,
                            current_data_set=forcing.currents,
                            wind_data_set=forcing.winds,
                            wave_data_set=forcing.waves,
                        )
                    except UnboundLocalError:
                        logging.warning(
                            "crossover_routes_minimal_cost failed; using parent_a"
                        )
                        child_route = parent_a.route
                else:  # "random"
                    child_route = crossover_routes_random(
                        parent_a.route, parent_b.route
                    )

                offspring_members.append(
                    PopulationMember(
                        route=child_route,
                        cost=self._route_cost(child_route, forcing),
                    )
                )

            # P_offspring ← P_offspring ∪ {r_seed}
            offspring = Population.from_members(offspring_members).add_member(
                seed_member
            )

            # 3.3: Selection (line 407)
            # KEY FIX: Select from offspring ONLY, not from combined!
            # P ← S_{M_offspring,q,M-1}(P_offspring)
            selected_members = select_from_population(
                members=offspring.members,
                quantile=q,
                target_size=M - 1,
                rng=self._rng,
            )

            # P ← P ∪ {r_seed}
            population = Population.from_members(selected_members).add_member(
                seed_member
            )

            # 3.4: Parameter adaptation (line 409)
            # TODO: Implement adaptive W, D, q
            W, D, q = W, D, q  # Placeholder

            self._log_stage_metrics(
                "ga_generation",
                generation=generation,
                **self._population_stats(population.members),
            )

        return population

    def _stage_gradient_descent(
        self,
        population: Population,
        forcing: ForcingData,
    ) -> Population:
        """Stage 3: Gradient descent polishing of elite members.

        Pseudocode (lines 413-416):
            For n = 1 to N_gd:
                P_elite ← S_{M,k/M,k-1}(P_elite)
                P ← P ∪ {r_seed}
                P_elite ← {G_t^(γ) ∘ G_⊥^(γ)(r_m) | m=1...k}

        Note: N_gd is gradient descent iterations *per elite*, not outer loop.
        """
        gradient_config = self.config.gradient
        if not population.members:
            return Population(members=[])
        sorted_population = population.sort()
        elites = sorted_population.members[: gradient_config.num_elites]
        if not gradient_config.enabled:
            self._log_stage_metrics(
                "gradient_polishing",
                population_size=population.size,
                elites=len(elites),
                skipped=True,
            )
            return Population(members=list(elites))
        self._log_stage_metrics(
            "gradient_polishing",
            population_size=population.size,
            elites=len(elites),
        )
        polished_members = []
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
            polished_members.append(PopulationMember(route=route, cost=cost))
            self._log_stage_metrics(
                "gradient_step",
                pre_cost=member.cost,
                post_cost=cost,
                elite_index=idx,
                member_seed_cost=member.cost,
            )
        return Population(members=polished_members)

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
