from __future__ import annotations

import numpy as np
import xarray as xr

from ..core.config import PHYSICS_DEFAULT, SHIP_DEFAULT, Physics, Ship
from ..core.routes import Route


class InvalidGradientError(RuntimeError):
    """Raised if gradients are invalid, e.g., because modified route on land."""

    pass


class ZeroGradientsError(RuntimeError):
    """Raised if gradients are zero, indicating that route is an extremum."""

    pass


class LargeIncrementError(RuntimeError):
    """Raised if shifts adjusted for learning rate are larger than the test increment."""

    pass


def gradient_descent_time_shift(
    route: Route = None,
    current_data_set: xr.Dataset = None,
    wind_data_set: xr.Dataset = None,
    wave_data_set: xr.Dataset = None,
    time_shift_seconds: float = None,
    learning_rate_percent: float = None,
    ship=SHIP_DEFAULT,
    physics=PHYSICS_DEFAULT,
):
    """Do one step of gradient descent by shifting times.

    Computes cost gradients with respect to waypoint times and adjusts
    times to reduce total route cost.

    Parameters
    ----------
    route : Route
        Route to optimize
    current_data_set : xr.Dataset
        Ocean current forcing data
    wind_data_set : xr.Dataset
        Wind forcing data
    wave_data_set : xr.Dataset
        Wave forcing data
    time_shift_seconds : float
        Test increment for gradient computation in seconds
    learning_rate_percent : float
        Learning rate as percentage of current cost
    ship : Ship, default=SHIP_DEFAULT
        Ship characteristics
    physics : Physics, default=PHYSICS_DEFAULT
        Physics parameters        Whether to ignore hazard conditions when computing costs

    Returns
    -------
    Route
        Route with adjusted waypoint times

    Raises
    ------
    ZeroGradientsError
        If all gradients are zero (route at extremum)
    LargeIncrementError
        If computed shifts exceed test increment
    """
    gradients = np.array(
        [
            route.cost_gradient_time_shift(
                n=n,
                current_data_set=current_data_set,
                wind_data_set=wind_data_set,
                wave_data_set=wave_data_set,
                time_shift_seconds=time_shift_seconds,
                ship=ship,
                physics=physics,
            )
            for n in range(1, len(route) - 1)
        ]
    )
    cost_before = route.cost_through(
        current_data_set=current_data_set,
        wind_data_set=wind_data_set,
        wave_data_set=wave_data_set,
        ship=ship,
        physics=physics,
    )
    desired_cost_reduction = learning_rate_percent / 100 * cost_before
    gradients_squared_sum = (gradients**2).sum()
    time_shifts = -desired_cost_reduction * gradients / gradients_squared_sum
    if gradients_squared_sum == 0:
        raise ZeroGradientsError
    if np.any(abs(time_shifts) > time_shift_seconds):
        raise LargeIncrementError
    for n in range(1, len(route) - 1):
        ts = time_shifts[n - 1]
        route = route.replace_waypoint(
            n=n,
            new_way_point=route.way_points[n].move_time(
                time_diff=ts * 1000 * np.timedelta64(1, "ms")
            ),
        )
    return route


def gradient_descent_along_track(
    route: Route = None,
    current_data_set: xr.Dataset = None,
    wind_data_set: xr.Dataset = None,
    wave_data_set: xr.Dataset = None,
    distance_meters: float = None,
    learning_rate_percent: float = None,
    ship=SHIP_DEFAULT,
    physics=PHYSICS_DEFAULT,
):
    """Do one step of gradient descent with along-track shifts.

    Computes cost gradients with respect to along-track waypoint positions
    and adjusts positions to reduce total route cost.

    Parameters
    ----------
    route : Route
        Route to optimize
    current_data_set : xr.Dataset
        Ocean current forcing data
    wind_data_set : xr.Dataset
        Wind forcing data
    wave_data_set : xr.Dataset
        Wave forcing data
    distance_meters : float
        Test increment for gradient computation in meters
    learning_rate_percent : float
        Learning rate as percentage of current cost
    ship : Ship, default=SHIP_DEFAULT
        Ship characteristics
    physics : Physics, default=PHYSICS_DEFAULT
        Physics parameters        Whether to ignore hazard conditions when computing costs

    Returns
    -------
    Route
        Route with adjusted waypoint positions

    Raises
    ------
    ZeroGradientsError
        If all gradients are zero (route at extremum)
    InvalidGradientError
        If gradients are invalid (e.g., waypoint on land)
    LargeIncrementError
        If computed shifts exceed test increment
    """
    gradients = np.array(
        [
            route.cost_gradient_along_track(
                n=n,
                current_data_set=current_data_set,
                wind_data_set=wind_data_set,
                wave_data_set=wave_data_set,
                distance_meters=distance_meters,
                ship=ship,
                physics=physics,
            )
            for n in range(1, len(route) - 1)
        ]
    )
    cost_before = route.cost_through(
        current_data_set=current_data_set,
        wind_data_set=wind_data_set,
        wave_data_set=wave_data_set,
        ship=ship,
        physics=physics,
    )
    desired_cost_reduction = learning_rate_percent / 100 * cost_before
    gradients_squared_sum = (gradients**2).sum()
    dist_shifts = -desired_cost_reduction * gradients / gradients_squared_sum
    if gradients_squared_sum == 0:
        raise ZeroGradientsError
    if np.isnan(gradients_squared_sum):
        raise InvalidGradientError
    if np.any(abs(dist_shifts) > distance_meters):
        raise LargeIncrementError
    _route = route
    for n in range(1, len(route) - 1):
        ds = dist_shifts[n - 1]
        _route = _route.move_waypoint(
            n=n,
            distance_meters=ds,
            azimuth_degrees=route.waypoint_azimuth(n=n),
        )
    return _route


def gradient_descent_across_track_left(
    route: Route = None,
    current_data_set: xr.Dataset = None,
    wind_data_set: xr.Dataset = None,
    wave_data_set: xr.Dataset = None,
    distance_meters: float = None,
    learning_rate_percent: float = None,
    ship=SHIP_DEFAULT,
    physics=PHYSICS_DEFAULT,
):
    """Do one step of gradient descent with across-track shifts.

    Computes cost gradients with respect to across-track waypoint positions
    and adjusts positions perpendicular to route to reduce total cost.

    Parameters
    ----------
    route : Route
        Route to optimize
    current_data_set : xr.Dataset
        Ocean current forcing data
    wind_data_set : xr.Dataset
        Wind forcing data
    wave_data_set : xr.Dataset
        Wave forcing data
    distance_meters : float
        Test increment for gradient computation in meters
    learning_rate_percent : float
        Learning rate as percentage of current cost
    ship : Ship, default=SHIP_DEFAULT
        Ship characteristics
    physics : Physics, default=PHYSICS_DEFAULT
        Physics parameters        Whether to ignore hazard conditions when computing costs

    Returns
    -------
    Route
        Route with adjusted waypoint positions

    Raises
    ------
    ZeroGradientsError
        If all gradients are zero (route at extremum)
    InvalidGradientError
        If gradients are invalid (e.g., waypoint on land)
    LargeIncrementError
        If computed shifts exceed test increment
    """
    gradients = np.array(
        [
            route.cost_gradient_across_track_left(
                n=n,
                current_data_set=current_data_set,
                wind_data_set=wind_data_set,
                wave_data_set=wave_data_set,
                distance_meters=distance_meters,
                ship=ship,
                physics=physics,
            )
            for n in range(1, len(route) - 1)
        ]
    )
    cost_before = route.cost_through(
        current_data_set=current_data_set,
        wind_data_set=wind_data_set,
        wave_data_set=wave_data_set,
        ship=ship,
        physics=physics,
    )
    desired_cost_reduction = learning_rate_percent / 100 * cost_before
    gradients_squared_sum = (gradients**2).sum()
    dist_shifts = -desired_cost_reduction * gradients / gradients_squared_sum
    if gradients_squared_sum == 0:
        raise ZeroGradientsError
    if np.isnan(gradients_squared_sum):
        raise InvalidGradientError
    if np.any(abs(dist_shifts) > distance_meters):
        raise LargeIncrementError
    _route = route
    for n in range(1, len(route) - 1):
        _route = _route.move_waypoint(
            n=n,
            distance_meters=dist_shifts[n - 1],
            azimuth_degrees=route.waypoint_azimuth(n=n) - 90.0,
        )
    return _route


def gradient_descent(
    route: Route = None,
    learning_rate_percent_time: float = 0.5,
    time_increment: float = 1_200,
    learning_rate_percent_along: float = 0.5,
    dist_shift_along: float = 10_000,
    learning_rate_percent_across: float = 0.5,
    dist_shift_across: float = 10_000,
    current_data_set: xr.Dataset = None,
    wave_data_set: xr.Dataset = None,
    wind_data_set: xr.Dataset = None,
) -> Route:
    """Execute a single iteration of gradient descent (3 steps: time, across, along).

    Performs gradient descent in three sequential steps: time shifts, across-track
    shifts, and along-track shifts. Handles exceptions by adjusting parameters.

    Parameters
    ----------
    route : Route
        Route to optimize
    learning_rate_percent_time : float, default=0.5
        Learning rate for time shifts as percentage of cost
    time_increment : float, default=1200
        Test increment for time gradient computation in seconds
    learning_rate_percent_along : float, default=0.5
        Learning rate for along-track shifts as percentage of cost
    dist_shift_along : float, default=10000
        Test increment for along-track gradient computation in meters
    learning_rate_percent_across : float, default=0.5
        Learning rate for across-track shifts as percentage of cost
    dist_shift_across : float, default=10000
        Test increment for across-track gradient computation in meters
    current_data_set : xr.Dataset
        Ocean current forcing data
    wave_data_set : xr.Dataset
        Wave forcing data
    wind_data_set : xr.Dataset
        Wind forcing data        Whether to ignore hazard conditions when computing costs

    Returns
    -------
    Route
        Optimized route after gradient descent iteration
    """
    try:
        route = gradient_descent_time_shift(
            route=route,
            current_data_set=current_data_set,
            wave_data_set=wave_data_set,
            wind_data_set=wind_data_set,
            time_shift_seconds=time_increment,
            learning_rate_percent=learning_rate_percent_time,
        )
    except ZeroGradientsError:
        pass
    except InvalidGradientError:
        time_increment /= 2.0
        learning_rate_percent_time /= 2.0
    except LargeIncrementError:
        learning_rate_percent_time /= 2.0

    try:
        route = gradient_descent_across_track_left(
            route=route,
            current_data_set=current_data_set,
            wave_data_set=wave_data_set,
            wind_data_set=wind_data_set,
            distance_meters=dist_shift_across,
            learning_rate_percent=learning_rate_percent_across,
        )
    except ZeroGradientsError:
        pass
    except InvalidGradientError:
        dist_shift_across /= 2.0
        learning_rate_percent_across /= 2.0
    except LargeIncrementError:
        learning_rate_percent_across /= 2.0

    try:
        route = gradient_descent_along_track(
            route=route,
            current_data_set=current_data_set,
            wave_data_set=wave_data_set,
            wind_data_set=wind_data_set,
            distance_meters=dist_shift_along,
            learning_rate_percent=learning_rate_percent_along,
        )
    except ZeroGradientsError:
        pass
    except InvalidGradientError:
        dist_shift_along /= 2.0
        learning_rate_percent_along /= 2.0
    except LargeIncrementError:
        learning_rate_percent_along /= 2.0

    return route


__all__ = [
    "InvalidGradientError",
    "ZeroGradientsError",
    "LargeIncrementError",
    "gradient_descent_time_shift",
    "gradient_descent_along_track",
    "gradient_descent_across_track_left",
    "gradient_descent",
]
