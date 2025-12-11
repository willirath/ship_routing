"""Example routing script for experimentation.

Demonstrates how to programmatically create a RoutingConfig and run the optimizer.
Uses the unified build_config() function instead of manually constructing config objects.
"""

import logging
from pathlib import Path
import sys

from ship_routing.app import RoutingApp, build_config, RoutingResult


def run_example() -> RoutingResult:
    """Configure and run a routing experiment."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Find project root and data directory
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    LARGE_DATA_DIR = PROJECT_ROOT / "data" / "large"

    if not LARGE_DATA_DIR.exists() or not any(LARGE_DATA_DIR.glob("*.zarr")):
        print(f"ERROR: Large data not found at {LARGE_DATA_DIR}")
        print("\nTo download the required data, run:")
        print("  pixi run download-data")
        sys.exit(1)

    base = LARGE_DATA_DIR

    # Use build_config to create the routing configuration
    config = build_config(
        # Journey parameters
        journey_name="Example-Atlantic-Journey",
        lon_waypoints=(-80.5, -62.0),
        lat_waypoints=(30.0, 35.0),
        time_start="2021-01-01T00:00",
        speed_knots=7.0,
        time_resolution_hours=12.0,
        # Forcing paths
        currents_path=str(
            base
            / "cmems_mod_glo_phy_my_0.083deg_P1D-m_time_2021_lat_+10_+65_lon_-100_+010_uo-vo.zarr"
        ),
        waves_path=str(
            base
            / "cmems_mod_glo_wav_my_0.2deg_PT3H-i_time_2021_lat_+10_+65_lon_-100_+010_VHM0-VMDR.zarr"
        ),
        winds_path=str(
            base
            / "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_time_2021_lat_+10_+65_lon_-100_+010_eastward_wind-northward_wind.zarr"
        ),
        # Forcing options
        engine="zarr",
        chunks="auto",
        load_eagerly=True,
        # Population parameters
        population_size=4,
        random_seed=345,
        # Warmup parameters
        selection_acceptance_rate_warmup=0.1,
        mutation_width_fraction_warmup=0.5,
        mutation_displacement_fraction_warmup=0.15,
        # Genetic algorithm parameters
        generations=2,
        offspring_size=4,
        crossover_rounds=1,
        selection_quantile=0.5,
        selection_acceptance_rate=0.1,
        mutation_width_fraction=0.5,
        mutation_displacement_fraction=0.1,
        mutation_iterations=1,
        crossover_strategy="minimal_cost",
        hazard_penalty_multiplier=100.0,
        # Post-processing parameters
        num_elites=2,
        gd_iterations=1,
        learning_rate_time=0.5,
        learning_rate_space=0.5,
        time_increment=1_200.0,
        distance_increment=10_000.0,
        # Parallelization parameters
        num_workers=2,
        executor_type="sequential",
    )

    app = RoutingApp(config=config)
    return app.run()


if __name__ == "__main__":
    runs_dir = Path(__file__).resolve().parent / "runs"
    runs_dir.mkdir(exist_ok=True)
    latest_path = runs_dir / "example_routing_result.json"

    result = run_example()

    result.dump_json(latest_path)
    print(f"Dumped result to {latest_path}")
