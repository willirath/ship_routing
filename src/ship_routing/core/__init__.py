"""Core layer: Routes, waypoints, legs, data structures, and cost calculation."""

from .routes import WayPoint, Leg, Route
from .config import Physics, Ship, MAX_CACHE_SIZE, PHYSICS_DEFAULT, SHIP_DEFAULT
from .geodesics import (
    move_fwd,
    get_distance_meters,
    get_length_meters,
    get_refinement_factor,
    refine_along_great_circle,
    get_leg_azimuth,
)
from .cost import power_maintain_speed, hazard_conditions_wave_height
from .cost_ufuncs import power_maintain_speed_ufunc, hazard_conditions_wave_height_ufunc
from .data import load_currents, load_winds, load_waves, select_data_for_leg
from .hashable_dataset import HashableDataset, make_hashable
from .remix import segment_lines_with_each_other, SHAPELY_RESOLUTION

__all__ = [
    "WayPoint",
    "Leg",
    "Route",
    "Physics",
    "Ship",
    "MAX_CACHE_SIZE",
    "PHYSICS_DEFAULT",
    "SHIP_DEFAULT",
    "move_fwd",
    "get_distance_meters",
    "get_length_meters",
    "get_refinement_factor",
    "refine_along_great_circle",
    "get_leg_azimuth",
    "power_maintain_speed",
    "hazard_conditions_wave_height",
    "power_maintain_speed_ufunc",
    "hazard_conditions_wave_height_ufunc",
    "load_currents",
    "load_winds",
    "load_waves",
    "select_data_for_leg",
    "HashableDataset",
    "make_hashable",
    "segment_lines_with_each_other",
    "SHAPELY_RESOLUTION",
]
