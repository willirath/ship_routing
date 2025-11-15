from ..core.routes import Route
from ..core.config import (
    SHIP_DEFAULT,
    PHYSICS_DEFAULT,
    Ship,
    Physics,
)  # TODO: replace globals with explicit config wiring

import xarray as xr
import numpy as np


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

    Parameters
    ----------
    route: Route
        Initial route.
    current_data_set: xr.Dataset
        Contains currents.
    wind_data_set: xr.Dataset
        Contains winds.
    wave_data_set: xr.Dataset
        Contains waves.
    time_shift_seconds: float
        Time shift used for estimating gradients.
    learning_rate_percent: float
        Desired learning rate in percent.
    ship: Ship
        Ship parameters.
    physics: Physics
        Physical constants.

    Returns
    -------
    Route:
        Route with updated time stamps but unchanged locations of way points.
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
    # Note that there's no invalid gradients here, because time shifts won't
    # make a formerly valid route touch land.
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

    Parameters
    ----------
    route: Route
        Initial route.
    current_data_set: xr.Dataset
        Contains currents.
    wind_data_set: xr.Dataset
        Contains winds.
    wave_data_set: xr.Dataset
        Contains waves.
    distance_meters: float
        Spatial shift used for estimating gradients.
    learning_rate_percent: float
        Desired learning rate in percent.
    ship: Ship
        Ship parameters.
    physics: Physics
        Physical constants.

    Returns
    -------
    Route:
        Route with updated locations but unchanged times of way points.
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
    # keep a copy of the original route, because we need the original way point azimuths
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

    Parameters
    ----------
    route: Route
        Initial route.
    current_data_set: xr.Dataset
        Contains currents.
    wind_data_set: xr.Dataset
        Contains winds.
    wave_data_set: xr.Dataset
        Contains waves.
    distance_meters: float
        Spatial shift used for estimating gradients.
    learning_rate_percent: float
        Desired learning rate in percent.
    ship: Ship
        Ship parameters.
    physics: Physics
        Physical constants.

    Returns
    -------
    Route:
        Route with updated locations but unchanged times of way points.
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
    # keep a copy of the original route, because we need the original way point azimuths
    _route = route
    for n in range(1, len(route) - 1):
        _route = _route.move_waypoint(
            n=n,
            distance_meters=dist_shifts[n - 1],
            azimuth_degrees=route.waypoint_azimuth(n=n) - 90.0,
        )
    return _route


def crossover_routes_random(
    route_0: Route = None,
    route_1: Route = None,
) -> Route:
    """Randomly cross over routes.

    Parameters
    ----------
    route_0: Route
        First route.
    route_1: Route
        Second Route.

    Returns
    -------
    Route:
        Route of randomly selected segments.
    """
    segments_0, segments_1 = route_0.segment_at(route_1)
    segments_mix = [
        s0s1[np.random.randint(0, 2)] for s0s1 in zip(segments_0, segments_1)
    ]
    route_mix = segments_mix[0]
    for s in segments_mix[1:]:
        route_mix = route_mix + s
    ref_timestep_seconds = route_0.data_frame.time.diff().mean() / np.timedelta64(
        1, "s"
    )
    route_mix = route_mix.remove_consecutive_duplicate_timesteps(
        min_time_diff_seconds=ref_timestep_seconds / 5
    )
    return route_mix


def crossover_routes_minimal_cost(
    route_0: Route = None,
    route_1: Route = None,
    current_data_set: xr.Dataset = None,
    wind_data_set: xr.Dataset = None,
    wave_data_set: xr.Dataset = None,
    ship: Ship = SHIP_DEFAULT,
    physics: Physics = PHYSICS_DEFAULT,
) -> Route:
    """Cross over routes to minimy cost.

    Parameters
    ----------
    route_0: Route
        First route.
    route_1: Route
        Second Route.
    current_data_set: xr.Dataset
        Current data set.
    wind_data_set: xr.Dataset
        Wind data set.
    wave_data_set: xr.Dataset
        Wave data set.
    ship: Ship
        Ship parameters. Defaults to: SHIP_DEFAULT
    physics: Physics
        Physics parameters. Defaults to: PHYSICS_DEFAULT

    Returns
    -------
    Route:
        Route with segments selected such that the cost is minimised.

    """
    segments_0, segments_1 = route_0.segment_at(route_1)
    cost_0 = [
        s.cost_through(
            current_data_set=current_data_set,
            wind_data_set=wind_data_set,
            wave_data_set=wave_data_set,
            ship=ship,
            physics=physics,
        )
        for s in segments_0
    ]
    cost_1 = [
        s.cost_through(
            current_data_set=current_data_set,
            wind_data_set=wind_data_set,
            wave_data_set=wave_data_set,
            ship=ship,
            physics=physics,
        )
        for s in segments_1
    ]
    segments_mix = [
        s0s1c0c1[int(s0s1c0c1[-1] < s0s1c0c1[-2])]
        for s0s1c0c1 in zip(segments_0, segments_1, cost_0, cost_1)
    ]
    route_mix = segments_mix[0]
    for s in segments_mix[1:]:
        route_mix = route_mix + s
    route_mix = route_mix.remove_consecutive_duplicate_timesteps()
    return route_mix


def stochastic_search(
    route: Route = None,
    number_of_iterations: int = 10,
    acceptance_rate_target: float = 0.05,
    acceptance_rate_for_increase_cost: float = 0.0,
    refinement_factor: float = 0.5,
    mod_width: float = None,
    max_move_meters: float = None,
    include_logs_routes: bool = True,
    current_data_set: xr.Dataset = None,
    wave_data_set: xr.Dataset = None,
    wind_data_set: xr.Dataset = None,
) -> Route:
    """Stochastic search optimization using non-local route mutations.

    Parameters
    ----------
    route : Route
        Initial route.
    number_of_iterations : int
        Number of iterations to run. Defaults to: 10
    acceptance_rate_target : float
        Target acceptance rate. Defaults to: 0.05
    acceptance_rate_for_increase_cost : float
        Probability of accepting cost increase. Defaults to: 0.0
    refinement_factor : float
        Factor to reduce mutation width when acceptance rate too low. Defaults to: 0.5
    mod_width : float
        Width of modification window in meters.
    max_move_meters : float
        Maximum movement distance in meters.
    include_logs_routes : bool
        Whether to include logs. Defaults to: True
    current_data_set : xr.Dataset
        Current data set.
    wave_data_set : xr.Dataset
        Wave data set.
    wind_data_set : xr.Dataset
        Wind data set.

    Returns
    -------
    tuple
        (route, logs_routes) where route is the optimized Route and logs_routes
        is a list of LogsRoute objects if include_logs_routes is True, else None.
    """
    # Import here to avoid circular dependency
    from .logging import Logs, LogsRoute

    if include_logs_routes:
        logs_routes = []
    else:
        logs_routes = None

    cost = route.cost_through(
        current_data_set=current_data_set,
        wave_data_set=wave_data_set,
        wind_data_set=wind_data_set,
    )
    if include_logs_routes:
        logs_routes.append(
            LogsRoute(
                logs=Logs(
                    iteration=0,
                    cost=cost,
                    stoch_acceptance_rate_target=acceptance_rate_target,
                    stoch_acceptance_rate_for_increase_cost=acceptance_rate_for_increase_cost,
                    stoch_refinement_factor=refinement_factor,
                    stoch_mod_width=mod_width,
                    stoch_max_move_meters=max_move_meters,
                    method="stochastic_search_initial",
                ),
                route=route,
            )
        )

    accepted = 0
    n_reset = 0
    for iteration in range(1, number_of_iterations + 1):
        route_ = route.move_waypoints_left_nonlocal(
            center_distance_meters=np.random.uniform(
                mod_width / 2.0, route.length_meters - mod_width / 2.0
            ),
            width_meters=mod_width,
            max_move_meters=max_move_meters * np.random.uniform(-1, 1),
        )
        cost_ = route_.cost_through(
            current_data_set=current_data_set,
            wave_data_set=wave_data_set,
            wind_data_set=wind_data_set,
        )
        if not np.isnan(cost_) and (
            (cost_ < cost)
            or (np.random.uniform(0, 1) < acceptance_rate_for_increase_cost)
        ):
            route = route_
            cost = cost_
            accepted += 1
            if include_logs_routes:
                logs_routes.append(
                    LogsRoute(
                        logs=Logs(
                            iteration=iteration,
                            cost=cost,
                            stoch_acceptance_rate_target=acceptance_rate_target,
                            stoch_acceptance_rate_for_increase_cost=acceptance_rate_for_increase_cost,
                            stoch_refinement_factor=refinement_factor,
                            stoch_mod_width=mod_width,
                            stoch_max_move_meters=max_move_meters,
                            method="stochastic_search",
                        ),
                        route=route,
                    )
                )
        if (accepted + 1) / (n_reset + 1) < acceptance_rate_target:
            n_reset = 0
            accepted = 0
            mod_width *= refinement_factor
            max_move_meters *= refinement_factor

        n_reset += 1

    return route, logs_routes


def gradient_descent(
    route: Route = None,
    num_iterations: int = 1,
    learning_rate_percent_time: float = 0.5,
    time_increment: float = 1_200,
    learning_rate_percent_along: float = 0.5,
    dist_shift_along: float = 10_000,
    learning_rate_percent_across: float = 0.5,
    dist_shift_across: float = 10_000,
    include_logs_routes: bool = True,
    current_data_set: xr.Dataset = None,
    wave_data_set: xr.Dataset = None,
    wind_data_set: xr.Dataset = None,
) -> tuple[Route, list]:
    """Execute multiple iterations of gradient descent optimization.

    Combines time_shift, along_track, and across_track gradient descent methods.

    Parameters
    ----------
    route : Route
        Initial route.
    num_iterations : int
        Number of iterations. Defaults to: 1
    learning_rate_percent_time : float
        Learning rate for time shifts. Defaults to: 0.5
    time_increment : float
        Time increment for gradient estimation. Defaults to: 1_200
    learning_rate_percent_along : float
        Learning rate for along-track shifts. Defaults to: 0.5
    dist_shift_along : float
        Distance for along-track gradient estimation. Defaults to: 10_000
    learning_rate_percent_across : float
        Learning rate for across-track shifts. Defaults to: 0.5
    dist_shift_across : float
        Distance for across-track gradient estimation. Defaults to: 10_000
    include_logs_routes : bool
        Whether to include logs. Defaults to: True
    current_data_set : xr.Dataset
        Current data set.
    wave_data_set : xr.Dataset
        Wave data set.
    wind_data_set : xr.Dataset
        Wind data set.

    Returns
    -------
    tuple
        (route, logs_routes) where route is the optimized Route and logs_routes
        is a list of LogsRoute objects if include_logs_routes is True, else None.
    """
    # Import here to avoid circular dependency
    from .logging import Logs, LogsRoute

    if include_logs_routes:
        iteration = 0
        logs_routes = [
            LogsRoute(
                logs=Logs(
                    iteration=iteration,
                    cost=route.cost_through(
                        current_data_set=current_data_set,
                        wave_data_set=wave_data_set,
                        wind_data_set=wind_data_set,
                    ),
                    grad_learning_rate_percent_time=learning_rate_percent_time,
                    grad_time_increment=time_increment,
                    grad_learning_rate_percent_along=learning_rate_percent_along,
                    grad_dist_shift_along=dist_shift_along,
                    grad_learning_rate_percent_across=learning_rate_percent_across,
                    grad_dist_shift_across=dist_shift_across,
                    method="gradient_descent_initial",
                ),
                route=route,
            )
        ]
    else:
        logs_routes = None

    for _ in range(num_iterations):
        try:
            route = gradient_descent_time_shift(
                route=route,
                current_data_set=current_data_set,
                wave_data_set=wave_data_set,
                wind_data_set=wind_data_set,
                time_shift_seconds=time_increment,
                learning_rate_percent=learning_rate_percent_time,
            )
            if include_logs_routes:
                iteration += 1
                cost = route.cost_through(
                    current_data_set=current_data_set,
                    wave_data_set=wave_data_set,
                    wind_data_set=wind_data_set,
                )
                logs_routes.append(
                    LogsRoute(
                        logs=Logs(
                            iteration=iteration,
                            cost=cost,
                            grad_learning_rate_percent_time=learning_rate_percent_time,
                            grad_time_increment=time_increment,
                            grad_learning_rate_percent_along=learning_rate_percent_along,
                            grad_dist_shift_along=dist_shift_along,
                            grad_learning_rate_percent_across=learning_rate_percent_across,
                            grad_dist_shift_across=dist_shift_across,
                            method="gradient_descent_time_shift",
                        ),
                        route=route,
                    )
                )
        except ZeroGradientsError:
            # converged, just pass
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
            if include_logs_routes:
                iteration += 1
                cost = route.cost_through(
                    current_data_set=current_data_set,
                    wave_data_set=wave_data_set,
                    wind_data_set=wind_data_set,
                )
                logs_routes.append(
                    LogsRoute(
                        logs=Logs(
                            iteration=iteration,
                            cost=cost,
                            grad_learning_rate_percent_time=learning_rate_percent_time,
                            grad_time_increment=time_increment,
                            grad_learning_rate_percent_along=learning_rate_percent_along,
                            grad_dist_shift_along=dist_shift_along,
                            grad_learning_rate_percent_across=learning_rate_percent_across,
                            grad_dist_shift_across=dist_shift_across,
                            method="gradient_descent_across_track_left",
                        ),
                        route=route,
                    )
                )
        except ZeroGradientsError:
            # converged, just pass
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
            if include_logs_routes:
                iteration += 1
                cost = route.cost_through(
                    current_data_set=current_data_set,
                    wave_data_set=wave_data_set,
                    wind_data_set=wind_data_set,
                )
                logs_routes.append(
                    LogsRoute(
                        logs=Logs(
                            iteration=iteration,
                            cost=cost,
                            grad_learning_rate_percent_time=learning_rate_percent_time,
                            grad_time_increment=time_increment,
                            grad_learning_rate_percent_along=learning_rate_percent_along,
                            grad_dist_shift_along=dist_shift_along,
                            grad_learning_rate_percent_across=learning_rate_percent_across,
                            grad_dist_shift_across=dist_shift_across,
                            method="gradient_descent_along_track",
                        ),
                        route=route,
                    )
                )
        except ZeroGradientsError:
            # converged, just pass
            pass
        except InvalidGradientError:
            dist_shift_along /= 2.0
            learning_rate_percent_along /= 2.0
        except LargeIncrementError:
            learning_rate_percent_along /= 2.0

    return route, logs_routes
