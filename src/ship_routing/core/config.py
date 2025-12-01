from __future__ import annotations

from dataclasses import dataclass

MAX_CACHE_SIZE = 10_000


@dataclass(frozen=True)
class Physics:
    """Physical constants used in power estimation."""

    gravity_acceleration_ms2: float = 9.80665
    sea_water_density_kgm3: float = 1029.0
    air_density_kgm3: float = 1.225


@dataclass(frozen=True)
class Ship:
    """Ship dimensions, resistance coefficients, engine characteristics."""

    waterline_width_m: float = 30.0
    waterline_length_m: float = 210.0
    total_propulsive_efficiency: float = 0.7
    reference_engine_power_W: float = 14296344.0
    reference_speed_calm_water_ms: float = 9.259
    draught_m: float = 11.5
    projected_frontal_area_above_waterline_m2: float = 690.0
    wind_resistance_coefficient: float = 0.4


PHYSICS_DEFAULT = Physics()  # TODO: drop once legacy imports are cleaned up

SHIP_DEFAULT = Ship()  # TODO: drop once legacy imports are cleaned up
