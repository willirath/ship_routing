#!/usr/bin/env python
"""Compute cross-evaluation ablation costs for tuning_003 results.

This script augments existing optimization results by evaluating each elite
route under all forcing scenarios (baseline, no_currents, no_waves, no_winds, calm).

Usage:
    python compute_ablation_costs.py results/results_ablation_baseline_*.msgpack
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import msgpack
import numpy as np
from tqdm import tqdm

# Add parent directory and notebooks/ for local imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "notebooks"))

from experiment_params import FORCING_SCENARIOS, FORCING_SCENARIO_CALM
from ship_routing.app.routing import RoutingResult
from ship_routing.app.config import ForcingData
from ship_routing.core.config import SHIP_DEFAULT, PHYSICS_DEFAULT
from ship_routing.core.data import load_currents, load_waves, load_winds
from ship_routing.core.routes import Route, WayPoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def fix_route_times(route: Route) -> Route:
    """Reconstruct route with proper datetime64 times.

    When routes are deserialized from msgpack, waypoint times become strings.
    This function reconstructs the route with proper np.datetime64 objects.
    """
    # Reconstruct waypoints with proper datetime64 times
    fixed_waypoints = []
    for wp in route.way_points:
        fixed_wp = WayPoint(
            lon=wp.lon,
            lat=wp.lat,
            time=np.datetime64(wp.time),  # Convert string to datetime64
        )
        fixed_waypoints.append(fixed_wp)

    return Route(way_points=tuple(fixed_waypoints))


def load_forcing_for_scenario(
    scenario_name: str,
    time_start: str,
    time_end: str,
) -> ForcingData:
    """Load forcing data for a specific ablation scenario.

    Parameters
    ----------
    scenario_name : str
        One of: baseline, no_currents, no_waves, no_winds, calm
    time_start : str
        Journey start time
    time_end : str
        Journey end time

    Returns
    -------
    ForcingData
        Forcing data with appropriate components enabled/disabled
    """
    # Get scenario config (calm is separate from FORCING_SCENARIOS)
    if scenario_name == "calm":
        scenario = FORCING_SCENARIO_CALM
    else:
        scenario = FORCING_SCENARIOS[scenario_name]

    time_start_dt = np.datetime64(time_start)
    time_end_dt = np.datetime64(time_end)

    return ForcingData(
        currents=(
            load_currents(
                scenario["currents_path"],
                time_start=time_start_dt,
                time_end=time_end_dt,
                engine=scenario["engine"],
            )
            if scenario["currents_path"]
            else None
        ),
        waves=(
            load_waves(
                scenario["waves_path"],
                time_start=time_start_dt,
                time_end=time_end_dt,
                engine=scenario["engine"],
            )
            if scenario["waves_path"]
            else None
        ),
        winds=(
            load_winds(
                scenario["winds_path"],
                time_start=time_start_dt,
                time_end=time_end_dt,
                engine=scenario["engine"],
            )
            if scenario["winds_path"]
            else None
        ),
    )


def compute_ablation_costs_for_result(
    result: RoutingResult,
    forcings: dict[str, ForcingData],
) -> dict[str, dict[str, float]]:
    """Compute cost evaluation matrix for all elites in a result.

    Parameters
    ----------
    result : RoutingResult
        Routing result with elite population
    forcings : dict[str, ForcingData]
        Forcing data for each scenario (keys: baseline, no_currents, no_waves, no_winds, calm)

    Returns
    -------
    dict[str, dict[str, float]]
        Nested dict: {elite_0: {cost_baseline: ..., cost_no_currents: ..., cost_calm: ..., ...}, ...}
    """
    if not result.elite_population or not result.elite_population.members:
        return {}

    ablation_costs = {}

    for i, member in enumerate(result.elite_population.members):
        # Fix route times (msgpack deserialization converts datetime64 to strings)
        fixed_route = fix_route_times(member.route)

        costs = {}
        for scenario_name, forcing in forcings.items():
            try:
                cost = fixed_route.cost_through(
                    current_data_set=forcing.currents,
                    wind_data_set=forcing.winds,
                    wave_data_set=forcing.waves,
                    ship=SHIP_DEFAULT,
                    physics=PHYSICS_DEFAULT,
                )
                costs[f"cost_{scenario_name}"] = float(cost)
            except Exception as e:
                logger.warning(
                    f"Failed cost for elite {i}, scenario {scenario_name}: {e}"
                )
                costs[f"cost_{scenario_name}"] = float("nan")

        ablation_costs[f"elite_{i}"] = costs

    return ablation_costs


def process_result_file(
    input_path: Path,
    output_path: Path,
) -> None:
    """Process one result file and save augmented version.

    Parameters
    ----------
    input_path : Path
        Input msgpack file
    output_path : Path
        Output msgpack file with augmented results
    """
    logger.info(f"Processing {input_path.name}")

    # Load raw results (dict of msgpack bytes)
    with open(input_path, "rb") as f:
        raw_results = msgpack.unpack(f, raw=False)

    logger.info(f"Loaded {len(raw_results)} results")

    # Get time range from first result for loading forcing data
    first_result = RoutingResult.from_msgpack(list(raw_results.values())[0])
    journey_config = first_result.logs.config["journey"]
    time_start = journey_config["time_start"]
    time_end = journey_config["time_end"]

    # Load all forcing scenarios once
    logger.info("Loading forcing scenarios...")
    forcings = {}
    for scenario_name in ["baseline", "no_currents", "no_waves", "no_winds", "calm"]:
        logger.info(f"  Loading {scenario_name}...")
        forcings[scenario_name] = load_forcing_for_scenario(
            scenario_name, time_start, time_end
        )

    # Process each result
    logger.info("Computing ablation costs...")
    augmented_results = {}

    for key, result_bytes in tqdm(raw_results.items(), desc="Results"):
        try:
            # Deserialize
            result = RoutingResult.from_msgpack(result_bytes)

            # Compute ablation costs
            ablation_costs = compute_ablation_costs_for_result(result, forcings)

            # Convert to dict and add ablation_costs field
            result_dict = result.to_dict()
            result_dict["ablation_costs"] = ablation_costs

            # Re-serialize
            augmented_results[key] = msgpack.packb(result_dict)

        except Exception as e:
            logger.error(f"Failed to process {key}: {e}")
            # Keep original on failure
            augmented_results[key] = result_bytes

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        msgpack.pack(augmented_results, f)

    logger.info(f"Saved {len(augmented_results)} results to {output_path.name}")


@click.command()
@click.argument("input_files", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--output-suffix",
    default="_with_crosseval",
    help="Suffix to add to output filenames",
)
def main(input_files: tuple[str, ...], output_suffix: str):
    """Compute ablation costs for result files."""
    if not input_files:
        logger.error("No input files specified")
        sys.exit(1)

    for input_file in input_files:
        input_path = Path(input_file)

        # Generate output path
        output_name = input_path.stem + output_suffix + input_path.suffix
        output_path = input_path.parent / output_name

        try:
            process_result_file(input_path, output_path)
        except Exception as e:
            logger.error(f"Failed to process {input_path.name}: {e}", exc_info=True)
            continue

    logger.info("Done!")


if __name__ == "__main__":
    main()
