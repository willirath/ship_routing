"""Experiment parameters for cost_001.

Defines 96 journey configurations (2 directions x 4 speeds x 12 months)
using default hyperparameters. Only journey definition and random seed
vary across runs.

Seed management: Each submission uses a different base_seed derived from the
submission_id, producing independent RNG streams. Resubmitting with a new
submission_id appends independent replicas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from ship_routing.app.config import (
    ForcingConfig,
    HyperParams,
    JourneyConfig,
    RoutingConfig,
)


# --- Routes ---

ROUTES = {
    "Atlantic_forward": {
        "lon_waypoints": (-80.5, -11.0),
        "lat_waypoints": (30.0, 50.0),
    },
    "Atlantic_backward": {
        "lon_waypoints": (-11.0, -80.5),
        "lat_waypoints": (50.0, 30.0),
    },
}


# --- Forcing ---

FORCING_BASELINE = ForcingConfig(
    currents_path=Path(
        "data_large/cmems_mod_glo_phy_my_0.083deg_P1D-m_time_2021_lat_+10_+65_lon_-100_+010_uo-vo.zarr"
    ),
    waves_path=Path(
        "data_large/cmems_mod_glo_wav_my_0.2deg_PT3H-i_time_2021_lat_+10_+65_lon_-100_+010_VHM0-VMDR.zarr"
    ),
    winds_path=Path(
        "data_large/cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_time_2021_lat_+10_+65_lon_-100_+010_eastward_wind-northward_wind.zarr"
    ),
    engine="zarr",
)

FORCING_NO_CURRENTS = ForcingConfig(
    currents_path=None,
    waves_path=FORCING_BASELINE.waves_path,
    winds_path=FORCING_BASELINE.winds_path,
    engine="zarr",
)


# --- Journeys ---

SPEEDS_KNOTS = (8.0, 12.0, 16.0, 20.0)
MONTHS = tuple(range(1, 13))
TIME_RESOLUTION_HOURS = 6.0


def _make_journeys() -> list[JourneyConfig]:
    """Enumerate all 96 journey configurations."""
    journeys = []
    for route_name, route in ROUTES.items():
        for month in MONTHS:
            for speed in SPEEDS_KNOTS:
                journeys.append(
                    JourneyConfig(
                        lon_waypoints=route["lon_waypoints"],
                        lat_waypoints=route["lat_waypoints"],
                        time_start=f"2021-{month:02d}-01T00:00:00",
                        speed_knots=speed,
                        time_resolution_hours=TIME_RESOLUTION_HOURS,
                        name=route_name,
                    )
                )
    return journeys


ALL_JOURNEYS = _make_journeys()

# 16 representative cases (Table 7): 2 directions x 2 months x 4 speeds
REPRESENTATIVE_CASES = [
    {"route": "Atlantic_forward", "month": 8, "speed": 8.0},
    {"route": "Atlantic_forward", "month": 8, "speed": 12.0},
    {"route": "Atlantic_forward", "month": 8, "speed": 16.0},
    {"route": "Atlantic_forward", "month": 8, "speed": 20.0},
    {"route": "Atlantic_backward", "month": 8, "speed": 8.0},
    {"route": "Atlantic_backward", "month": 8, "speed": 12.0},
    {"route": "Atlantic_backward", "month": 8, "speed": 16.0},
    {"route": "Atlantic_backward", "month": 8, "speed": 20.0},
    {"route": "Atlantic_forward", "month": 1, "speed": 8.0},
    {"route": "Atlantic_forward", "month": 1, "speed": 12.0},
    {"route": "Atlantic_forward", "month": 1, "speed": 16.0},
    {"route": "Atlantic_forward", "month": 1, "speed": 20.0},
    {"route": "Atlantic_backward", "month": 1, "speed": 8.0},
    {"route": "Atlantic_backward", "month": 1, "speed": 12.0},
    {"route": "Atlantic_backward", "month": 1, "speed": 16.0},
    {"route": "Atlantic_backward", "month": 1, "speed": 20.0},
]

REPRESENTATIVE_JOURNEYS = [
    j
    for j in ALL_JOURNEYS
    if any(
        j.name == c["route"]
        and f"2021-{c['month']:02d}-01" in j.time_start
        and j.speed_knots == c["speed"]
        for c in REPRESENTATIVE_CASES
    )
]


# --- Config factory ---

# Master base seed. submission_id offsets this to produce independent streams.
MASTER_SEED = 2026_02_11


def make_production_configs(
    journeys: list[JourneyConfig],
    n_replicas: int,
    submission_id: int = 1,
    forcing: ForcingConfig = FORCING_BASELINE,
    executor_type: Literal["sequential", "process", "thread"] = "sequential",
    num_workers: int = 1,
) -> list[RoutingConfig]:
    """Generate production RoutingConfigs with deterministic independent seeds.

    Parameters
    ----------
    journeys : list[JourneyConfig]
        Journey configurations to run.
    n_replicas : int
        Number of replicas per journey.
    submission_id : int
        Submission identifier (1, 2, ...). Different IDs produce independent
        seed streams, so resubmitting with a new ID appends independent runs.
    forcing : ForcingConfig
        Forcing configuration to use. Default: FORCING_BASELINE.
    executor_type : str, default="sequential"
        Executor type for the routing app.
    num_workers : int, default=1
        Number of workers (ignored for sequential).

    Returns
    -------
    list[RoutingConfig]
        One config per (journey, replica) combination.
    """
    # Each submission_id gets its own RNG stream via SeedSequence spawning
    master_rng = np.random.default_rng(MASTER_SEED)
    # Spawn one child per possible submission_id (1-indexed)
    submission_rngs = master_rng.spawn(submission_id)
    rng = submission_rngs[submission_id - 1]

    configs = []
    for journey in journeys:
        replica_rngs = rng.spawn(n_replicas)
        for rep_rng in replica_rngs:
            experiment_seed = int(rep_rng.bit_generator.seed_seq.generate_state(1)[0])
            configs.append(
                RoutingConfig(
                    journey=journey,
                    forcing=forcing,
                    hyper=HyperParams(
                        random_seed=experiment_seed,
                        executor_type=executor_type,
                        num_workers=num_workers,
                    ),
                )
            )
    return configs
