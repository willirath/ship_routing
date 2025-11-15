from ..core.routes import Route

import pandas as pd
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Logs:
    """Logging dataclass for optimization iterations.

    Note: This is temporary and will be removed in future refactoring.
    """

    iteration: int = 0
    cost: float = 0.0
    stoch_acceptance_rate_target: float = None
    stoch_acceptance_rate_for_increase_cost: float = None
    stoch_refinement_factor: float = None
    stoch_mod_width: float = None
    stoch_max_move_meters: float = None
    grad_learning_rate_percent_time: float = None
    grad_time_increment: float = None
    grad_learning_rate_percent_along: float = None
    grad_dist_shift_along: float = None
    grad_learning_rate_percent_across: float = None
    grad_dist_shift_across: float = None

    method: str = "initial"

    @property
    def data_frame(self):
        return pd.DataFrame(
            asdict(self),
            index=[
                0,
            ],
        )


@dataclass(frozen=True)
class LogsRoute:
    """Combined logs and route for tracking optimization progress.

    Note: This is temporary and will be removed in future refactoring.
    """

    logs: Logs
    route: Route

    @property
    def data_frame(self):
        return self.logs.data_frame.join(self.route.data_frame, how="cross")
