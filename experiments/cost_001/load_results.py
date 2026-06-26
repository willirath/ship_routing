#!/usr/bin/env python
"""Helper to load and process results from extracted msgpack files.

This module provides ETL (Extract, Transform, Load) utilities for loading
routing optimization results from msgpack files and converting them to
analysis-ready DataFrames with hyperparameters, journey configurations,
elite populations, and runtime metrics.
"""

import msgpack
import pandas as pd
import geopandas as gpd
import numpy as np
import warnings
from pathlib import Path
from typing import Union, List
from tqdm.auto import tqdm

from ship_routing.app import RoutingResult


# =============================================================================
# Shared figure styling
# =============================================================================

# One consistent visual encoding for the four production speeds, reused across
# all figures so 16 kn (etc.) reads the same everywhere. Okabe-Ito colours,
# ordered cool -> warm with increasing speed.
SPEED_COLORS = {
    8.0: "#0072B2",   # blue
    12.0: "#009E73",  # green
    16.0: "#E69F00",  # orange
    20.0: "#D55E00",  # vermillion
}

# Direction is encoded by line style (eastward solid, westward dashed). Keyed by
# both the route name and the short E/W label used across notebooks.
DIRECTION_LINESTYLES = {
    "Atlantic_forward": "-",
    "Atlantic_backward": "--",
    "E": "-",
    "W": "--",
}


# =============================================================================
# Core Loading Functions
# =============================================================================


def load_results_raw(msgpack_file_list: Union[List[Path], List[str]]) -> dict:
    """Load raw msgpack data from multiple files without deserialization.

    Parameters
    ----------
    msgpack_file_list : list of Path or str
        List of paths to msgpack files containing serialized results.

    Returns
    -------
    dict
        Dictionary mapping result keys to raw msgpack data.
    """
    raw_results = {}
    for mf in msgpack_file_list:
        with open(mf, "rb") as f:
            raw_results.update(msgpack.unpack(f, raw=False))
    return raw_results


def load_result_for_key(
    key: str, msgpack_file_list: Union[List[Path], List[str]]
) -> RoutingResult:
    """Load a single RoutingResult by key from msgpack file list.

    Parameters
    ----------
    key : str
        The result key to load.
    msgpack_file_list : list of Path or str
        List of paths to msgpack files.

    Returns
    -------
    RoutingResult
        Deserialized routing result for the specified key.
    """
    return RoutingResult.from_msgpack(load_results_raw(msgpack_file_list)[key])


def load_results(
    msgpack_file_list: Union[List[Path], List[str]]
) -> dict[str, RoutingResult]:
    """Load all results from msgpack file list with progress bar.

    This function loads raw msgpack data from multiple files and deserializes
    each result into a RoutingResult object with a tqdm progress bar.

    Parameters
    ----------
    msgpack_file_list : list of Path or str
        List of paths to msgpack files containing serialized results.

    Returns
    -------
    dict[str, RoutingResult]
        Dictionary mapping result keys to RoutingResult objects.

    Notes
    -----
    This replaces the old single-file load_results(msgpack_file, first_n) function.
    The old function is available as load_results_single_file() for backward compatibility.
    """
    raw_results = load_results_raw(msgpack_file_list)
    return {
        key: RoutingResult.from_msgpack(value)
        for key, value in tqdm(list(raw_results.items()), desc="records")
    }


def _get_cost_spread_genetic_q75_25(routing_result: RoutingResult = None):
    df = routing_result.logs.to_dataframe()
    df = df.where(df.stage.str.startswith("ga_")).dropna(how="all")
    df = df.set_index(["timestamp", "stage"])
    q75_25 = df.cost_q75 - df.cost_q25
    q75_25 = q75_25.dropna()
    q75_25 = q75_25.rename("q75_25")
    return q75_25


def _get_cost_spread_genetic_q50_00(routing_result: RoutingResult = None):
    df = routing_result.logs.to_dataframe()
    df = df.where(df.stage.str.startswith("ga_")).dropna(how="all")
    df = df.set_index(["timestamp", "stage"])
    q50_00 = df.cost_median - df.cost_min
    q50_00 = q50_00.dropna()
    q50_00 = q50_00.rename("q50_00")
    return q50_00


# =============================================================================
# DataFrame Extraction Functions
# =============================================================================


def get_diversity_df(routing_results_dict: dict[str, RoutingResult]) -> pd.Series:
    df = pd.Series(
        {
            f: (_get_cost_spread_genetic_q75_25(rr) ** 2).mean() ** 0.5
            for f, rr in tqdm(routing_results_dict.items(), desc="diversity")
        },
        name="cost_q75_25_rms",
    ).to_frame()
    df = df.assign(
        cost_q50_00_rms=pd.Series(
            {
                f: (_get_cost_spread_genetic_q50_00(rr) ** 2).mean() ** 0.5
                for f, rr in tqdm(routing_results_dict.items(), desc="diversity")
            },
            name="cost_q50_00_rms",
        )
    )
    return df.add_prefix("diversity_")


def get_journey_params_df(
    routing_results_dict: dict[str, RoutingResult]
) -> pd.DataFrame:
    """Extract journey configuration parameters as DataFrame.

    Extracts journey configuration from each result's log and converts
    waypoint lists to string representations with categorical typing.

    Parameters
    ----------
    routing_results_dict : dict[str, RoutingResult]
        Dictionary of routing results from load_results().

    Returns
    -------
    pd.DataFrame
        DataFrame with journey parameters, indexed by 'filename'.
        Columns are prefixed with 'journey_' and include:
        - journey_name (category)
        - journey_lon_waypoints (category, stringified list)
        - journey_lat_waypoints (category, stringified list)
        - journey_time_start (category)
        - journey_time_end
        - journey_speed_knots (category)
        - journey_time_resolution_hours
    """

    def _fix_waypoints(dct):
        """Convert waypoint lists to strings for categorical comparison."""
        dct["lon_waypoints"] = str(dct["lon_waypoints"])
        dct["lat_waypoints"] = str(dct["lat_waypoints"])
        return dct

    df = pd.concat(
        [
            pd.DataFrame(
                _fix_waypoints(rr.logs.config["journey"]),
                index=[f],
            )
            for f, rr in routing_results_dict.items()
        ]
    ).add_prefix("journey_")

    df = df.assign(
        journey_lon_waypoints=df["journey_lon_waypoints"].astype("category"),
        journey_lat_waypoints=df["journey_lat_waypoints"].astype("category"),
        journey_name=df["journey_name"].astype("category"),
        journey_time_start=df["journey_time_start"].astype("category"),
        journey_speed_knots=df["journey_speed_knots"].astype("category"),
    )
    df.index = df.index.rename("filename")
    return df


def get_hyper_params_df(routing_results_dict: dict[str, RoutingResult]) -> pd.DataFrame:
    """Extract hyperparameter configuration as DataFrame.

    Parameters
    ----------
    routing_results_dict : dict[str, RoutingResult]
        Dictionary of routing results from load_results().

    Returns
    -------
    pd.DataFrame
        DataFrame with hyperparameters, indexed by 'filename'.
        Columns are prefixed with 'hyper_' and include algorithm parameters
        like population_size, generations, mutation_iterations, etc.
    """
    df = pd.concat(
        [
            pd.DataFrame(
                rr.logs.config["hyper"],
                index=[f],
            )
            for f, rr in routing_results_dict.items()
        ]
    ).add_prefix("hyper_")

    df = df.assign(
        hyper_crossover_strategy=df["hyper_crossover_strategy"].astype("category")
    )
    df.index = df.index.rename("filename")
    return df


def get_runtime_df(routing_results_dict: dict[str, RoutingResult]) -> pd.DataFrame:
    """Calculate runtime from log timestamps.

    Parameters
    ----------
    routing_results_dict : dict[str, RoutingResult]
        Dictionary of routing results from load_results().

    Returns
    -------
    pd.DataFrame
        DataFrame with index 'filename' and columns:
        - runtime (timedelta)
        - runtime_seconds (float)
    """
    _records = []
    for f, rr in routing_results_dict.items():
        _records.append(
            {
                "filename": f,
                "runtime": rr.logs.to_dataframe().timestamp.max()
                - rr.logs.to_dataframe().timestamp.min(),
            }
        )
    df = pd.DataFrame.from_records(_records).set_index("filename")
    df = df.assign(runtime_seconds=df.runtime.dt.total_seconds())
    return df


def get_elite_df(routing_results_dict: dict[str, RoutingResult]) -> pd.DataFrame:
    """Extract elite population members with geometry.

    Creates one row per elite member with metrics relative to seed member.
    Includes LineString geometry for GeoDataFrame compatibility.

    Parameters
    ----------
    routing_results_dict : dict[str, RoutingResult]
        Dictionary of routing results from load_results().

    Returns
    -------
    pd.DataFrame
        DataFrame with index 'filename' and columns:
        - n_elite (int, elite member index)
        - elite_length_meters (float)
        - elite_length_relative (float, relative to seed)
        - elite_cost_absolute (float)
        - elite_cost_relative (float, relative to seed)
        - elite_has_negative_time (bool, True if any leg has backwards time)
        - geometry (LineString, from route.line_string)

    Notes
    -----
    This function shows a tqdm progress bar during processing.
    Returns one row per elite member, so results may have multiple rows
    per filename if elite_population has multiple members.
    """
    _records = []
    for f, rr in tqdm(routing_results_dict.items(), desc="elite"):
        seed_member = rr.seed_member
        _records.extend(
            [
                {
                    "filename": f,
                    "n_elite": n,
                    "elite_length_meters": m.route.length_meters,
                    "elite_length_relative": m.route.length_meters
                    / seed_member.route.length_meters,
                    "elite_cost_absolute": m.cost,
                    "elite_cost_relative": m.cost / seed_member.cost,
                    "elite_has_negative_time": m.route.has_negative_time_increment,
                    "geometry": m.route.line_string,
                }
                for n, m in enumerate(rr.elite_population.members)
            ]
        )
    return pd.DataFrame.from_records(_records).set_index("filename")


def get_seed_routes_gdf(
    routing_results_dict: dict[str, RoutingResult]
) -> gpd.GeoDataFrame:
    """Extract seed routes with geometry for all results.

    Creates one row per routing result with the seed (initial great-circle) route
    as LineString geometry, along with cost and length metrics.

    Parameters
    ----------
    routing_results_dict : dict[str, RoutingResult]
        Dictionary of routing results from load_results().

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with index 'filename' and columns:
        - seed_cost_absolute (float)
        - seed_length_meters (float)
        - geometry (LineString, from seed route)

    Notes
    -----
    This function shows a tqdm progress bar during processing.
    Skips results where seed_member is None.
    """
    _records = []
    for f, rr in tqdm(routing_results_dict.items(), desc="seed routes"):
        if rr.seed_member is not None:
            _records.append(
                {
                    "filename": f,
                    "seed_length_meters": rr.seed_member.route.length_meters,
                    "geometry": rr.seed_member.route.line_string,
                }
            )
    return gpd.GeoDataFrame.from_records(_records).set_index("filename")


def get_forcing_df(routing_results_dict: dict[str, RoutingResult]) -> pd.DataFrame:
    """Extract forcing data file paths for all results.

    Extracts the file paths to ocean currents, waves, and winds data from
    the routing configuration for each result.

    Parameters
    ----------
    routing_results_dict : dict[str, RoutingResult]
        Dictionary of routing results from load_results().

    Returns
    -------
    pd.DataFrame
        DataFrame with index 'filename' and columns:
        - forcing_currents_path (category)
        - forcing_waves_path (category)
        - forcing_winds_path (category)

    Notes
    -----
    Path columns are converted to categorical dtype since many results
    typically reference the same forcing data files.
    Missing paths are represented as None.
    """
    _records = []
    for f, rr in routing_results_dict.items():
        forcing_config = rr.logs.config.get("forcing", {})
        _records.append(
            {
                "filename": f,
                "scenario_name": forcing_config.get("scenario_name"),
                "currents_path": forcing_config.get("currents_path"),
                "waves_path": forcing_config.get("waves_path"),
                "winds_path": forcing_config.get("winds_path"),
            }
        )
    df = pd.DataFrame.from_records(_records).set_index("filename")

    # Convert to categorical since many results share the same forcing data
    df = df.assign(
        currents_path=df["currents_path"].astype("category"),
        waves_path=df["waves_path"].astype("category"),
        winds_path=df["winds_path"].astype("category"),
    )
    df.columns = df.columns.map(lambda x: f"forcing_{x}")

    return df


# =============================================================================
# Feature Engineering and Data Quality Functions
# =============================================================================


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for analysis.

    Computes the following derived columns from existing data:
    - hyper_num_individuals: Total number of individuals (population_size × generations)
    - journey_duration: Journey duration as timedelta (time_end - time_start)
    - seed_cost: Cost of the seed route (elite_cost_absolute / elite_cost_relative)
    - elite_speed_og_mps_average: Average speed over ground (elite_length_meters / journey_duration_seconds)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns from merged journey/hyper/elite/runtime tables.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional derived columns.
    """
    df = df.assign(
        hyper_num_individuals=df.hyper_population_size * df.hyper_generations
    )

    df = df.assign(
        journey_duration=pd.to_datetime(df.journey_time_end.astype(str))
        - pd.to_datetime(df.journey_time_start.astype(str))
    )

    df = df.assign(
        seed_cost=(df.elite_cost_absolute / df.elite_cost_relative).fillna(np.inf)
    )

    df = df.assign(
        elite_speed_og_mps_average=df.elite_length_meters
        / df.journey_duration.dt.total_seconds()
    )

    return df


def max_turning_angle_deg(rr) -> float:
    """Compute maximum turning angle between consecutive legs of the best elite route.

    Parameters
    ----------
    rr : RoutingResult
        A routing result containing an elite population.

    Returns
    -------
    float
        Maximum turning angle in degrees (0 = straight, 180 = full reversal).
    """
    route = rr.elite_population.members[0].route
    wps = route.way_points
    lons = np.array([wp.lon for wp in wps])
    lats = np.array([wp.lat for wp in wps])
    dlons = np.diff(lons)
    dlats = np.diff(lats)
    max_angle = 0.0
    for i in range(len(dlons) - 1):
        v1 = np.array([dlons[i], dlats[i]])
        v2 = np.array([dlons[i + 1], dlats[i + 1]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            continue
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
        max_angle = max(max_angle, np.degrees(np.arccos(cos_a)))
    return max_angle


def filter_kinked_routes(
    rr_dict: dict, threshold_deg: float = 60.0
) -> dict:
    """Remove routes with sharp turning angles (crossover artifacts).

    Parameters
    ----------
    rr_dict : dict
        Mapping of key -> RoutingResult.
    threshold_deg : float
        Maximum allowed turning angle in degrees. Routes exceeding this
        are removed. Default 60 deg.

    Returns
    -------
    dict
        Filtered dict with kinked routes removed.
    """
    clean = {}
    n_kinked = 0
    for key, rr in rr_dict.items():
        if max_turning_angle_deg(rr) <= threshold_deg:
            clean[key] = rr
        else:
            n_kinked += 1
    total = len(rr_dict)
    print(f"{n_kinked}/{total} routes removed (max turning angle > {threshold_deg}°)")
    return clean


def identify_suspicious_routes(df: pd.DataFrame) -> pd.Series:
    """Identify routes with data quality issues.

    Marks routes as suspicious if any of the following conditions are true:
    - elite_cost_absolute is infinite
    - seed_cost is infinite
    - elite_cost_relative > 1.0 (elite route is worse than seed route)
    - elite_has_negative_time is True (route has backwards time increments)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with elite and seed cost columns.

    Returns
    -------
    pd.Series
        Boolean series with True for suspicious routes.
    """
    return (
        np.isinf(df.elite_cost_absolute)
        | np.isinf(df.seed_cost)
        | (df.elite_cost_relative > 1.0)
        | df.elite_has_negative_time
    )


def filter_suspicious_routes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove suspicious routes from DataFrame.

    Identifies suspicious routes using identify_suspicious_routes() and removes them,
    then drops any remaining rows with NaN values. Prints the percentage of routes removed.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with routes to filter.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with suspicious routes removed.

    Notes
    -----
    Prints percentage of suspicious routes found.
    """
    suspicious = identify_suspicious_routes(df)
    print(f"{suspicious.mean() * 100:.2f}% suspicious routes")
    return df.where(~suspicious).dropna(how="all")
