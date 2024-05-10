from .core import Route

import xarray as xr
import numpy as np


def gradient_descent_time_shift(
    route: Route = None,
    current_data_set: xr.Dataset = None,
    time_shift_seconds: float = None,
    learning_rate_percent: float = None,
):
    gradients = np.array(
        [
            route.cost_gradient_time_shift(
                n=n,
                current_data_set=current_data_set,
                time_shift_seconds=time_shift_seconds,
            )
            for n in range(1, len(route) - 1)
        ]
    )
    # TODO: Think about the normalization here...
    cost_before = route.cost_through(current_data_set=current_data_set)
    desired_cost_reduction = learning_rate_percent / 100 * cost_before
    gradients_squared_sum = (gradients**2).sum()
    if gradients_squared_sum != 0:
        time_shifts = -desired_cost_reduction * gradients / gradients_squared_sum
    else:
        time_shifts = np.zeros_like(gradients)
    for n in range(1, len(route) - 1):
        ts = time_shifts[n - 1]
        if np.isinf(ts):
            continue
        route = route.replace_waypoint(
            n=n,
            new_way_point=route.way_points[n].move_time(
                time_diff=ts * 1000 * np.timedelta64(1, "ms")
            ),
        )
    return route
