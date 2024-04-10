from ship_routing.geodesics import (
    get_refinement_factor,
    refine_along_great_circle,
    move_first_point_left,
)
from ship_routing import Trajectory

from shapely.geometry import LineString


import numpy as np


def test_refinement_factor():
    assert 10 == get_refinement_factor(original_dist=1000, new_dist=100)
    assert 34 == get_refinement_factor(original_dist=100, new_dist=3)
    assert 1 == get_refinement_factor(original_dist=1234, new_dist=1234)


def test_traj_refinement():
    # simple 2-point traj
    traj = Trajectory(lon=[0, 10], lat=[0, 0])
    dist = 10 * 111e3  # good enough

    # simples traj
    lon, _ = refine_along_great_circle(lon=traj.lon, lat=traj.lat, new_dist=dist / 2.1)
    assert 3 == len(lon)
    lon, _ = refine_along_great_circle(lon=traj.lon, lat=traj.lat, new_dist=dist / 4.1)
    assert 5 == len(lon)

    # three point traj
    traj = Trajectory(lon=[0, 10, 20], lat=[0, 0, 0])
    dist = 10 * 111e3  # good enough
    lon, _ = refine_along_great_circle(lon=traj.lon, lat=traj.lat, new_dist=dist / 2.1)
    assert 5 == len(lon)
    lon, _ = refine_along_great_circle(lon=traj.lon, lat=traj.lat, new_dist=dist / 4.1)
    assert 9 == len(lon)


def test_move_first_point_left_by_zero():
    line_string = LineString([[50, 40], [51, 40]])
    moved_line_string = move_first_point_left(lstr=line_string, move_by_meters=0)
    np.testing.assert_array_almost_equal(line_string.xy[0], moved_line_string.xy[0])
    np.testing.assert_array_almost_equal(line_string.xy[1], moved_line_string.xy[1])
