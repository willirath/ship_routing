from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from .config import ForcingConfig, ForcingData, RoutingConfig
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


class RoutingApp:
    """High-level orchestrator wrapping the routing workflow."""

    def __init__(self, config: RoutingConfig):
        self.config = config
        self.log = RoutingLog(config=asdict(config))

    def run(self) -> RoutingResult:
        """Execute the optimisation pipeline."""
        self._log_stage_metrics("run", 0, message="starting routing run")
        forcing = self._load_forcing()
        population = self._initialize_population(forcing)
        population = self._run_ga_generations(population, forcing)
        best_routes = self._refine_with_gradient(population, forcing)
        return RoutingResult(best_routes=best_routes, logs=self.log)

    def _load_forcing(self) -> ForcingData:
        """Load wind, wave, and current fields according to the config."""
        forcing_config = self.config.forcing
        forcing = ForcingData(
            currents=(
                load_currents(
                    data_file=forcing_config.currents_path,
                    engine=forcing_config.engine,
                    chunks=forcing_config.chunks,
                )
                if forcing_config.currents_path
                else None
            ),
            waves=(
                load_waves(
                    data_file=forcing_config.waves_path,
                    engine=forcing_config.engine,
                    chunks=forcing_config.chunks,
                )
                if forcing_config.waves_path
                else None
            ),
            winds=(
                load_winds(
                    data_file=forcing_config.winds_path,
                    engine=forcing_config.engine,
                    chunks=forcing_config.chunks,
                )
                if forcing_config.winds_path
                else None
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


    def _initialize_population(self, forcing: Any) -> Sequence[Route]:
        """Seed the initial population using the configured journey."""
        raise NotImplementedError

    def _run_ga_generations(
        self,
        population: Sequence[Route],
        forcing: Any,
    ) -> Sequence[Route]:
        """Apply stochastic search, crossover, and selection loops."""
        raise NotImplementedError

    def _refine_with_gradient(
        self,
        population: Sequence[Route],
        forcing: Any,
    ) -> Sequence[Route]:
        """Run gradient descent on the elites and return them."""
        raise NotImplementedError

    def _log_stage_metrics(
        self,
        name: str,
        iteration: int,
        **metrics: Any,
    ) -> None:
        """Convenience wrapper for stage-level logging."""
        self.log.add_stage(name=name, iteration=iteration, **metrics)
