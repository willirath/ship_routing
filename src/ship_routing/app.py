from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .config import RoutingConfig
from .core import Route


@dataclass
class RoutingResult:
    """Container returned by RoutingApp.run."""

    best_routes: Sequence[Route] | None
    logs: dict[str, Any] | None = None


class RoutingApp:
    """High-level orchestrator wrapping the routing workflow."""

    def __init__(self, config: RoutingConfig):
        self.config = config

    def run(self) -> RoutingResult:
        """Execute the optimisation pipeline."""
        raise NotImplementedError("Routing execution not implemented yet.")

    def _load_environment(self) -> Any:
        """Load wind, wave, and current fields according to the config."""
        raise NotImplementedError

    def _initialize_population(self, environment: Any) -> Sequence[Route]:
        """Seed the initial population using the configured journey."""
        raise NotImplementedError

    def _run_ga_generations(
        self,
        population: Sequence[Route],
        environment: Any,
    ) -> Sequence[Route]:
        """Apply stochastic search, crossover, and selection loops."""
        raise NotImplementedError

    def _refine_with_gradient(
        self,
        population: Sequence[Route],
        environment: Any,
    ) -> RoutingResult:
        """Run gradient descent on the elites and package the result."""
        raise NotImplementedError
