from ship_routing.geodesics import (
    get_refinement_factor,
    refine_along_great_circle,
    move_first_point_left,
    move_second_point_left,
    move_middle_point_left,
)
from ship_routing import Trajectory

import numpy as np


def test_refinement_factor():
    assert 10 == get_refinement_factor(original_dist=1000, new_dist=100)
    assert 34 == get_refinement_factor(original_dist=100, new_dist=3)
    assert 1 == get_refinement_factor(original_dist=1234, new_dist=1234)


def test_traj_refinement():
    # simple 2-point traj
    traj = Trajectory(lon=[0, 10], lat=[0, 0])
    dist = 10 * 111e3  # good enough

    # simplest traj
    lon, _ = refine_along_great_circle(lon=traj.lon, lat=traj.lat, new_dist=dist / 2.1)
    assert 4 == len(lon)  # 2 + 2x2
    lon, _ = refine_along_great_circle(lon=traj.lon, lat=traj.lat, new_dist=dist / 4.1)
    assert 6 == len(lon)  # 2 + 2x3

    # three point traj
    traj = Trajectory(lon=[0, 10, 20], lat=[0, 0, 0])
    dist = 10 * 111e3  # good enough
    lon, _ = refine_along_great_circle(lon=traj.lon, lat=traj.lat, new_dist=dist / 2.1)
    assert 7 == len(lon)  # 3 + 2x2
    lon, _ = refine_along_great_circle(lon=traj.lon, lat=traj.lat, new_dist=dist / 4.1)
    assert 11 == len(lon)  # 3 + 2x4


def test_move_first_point_left_by_zero():
    lon1_moved, lat1_moved = move_first_point_left(
        lon1=50, lat1=40, lon2=51, lat2=40, move_by_meters=0
    )
    np.testing.assert_almost_equal(lon1_moved, 50)
    np.testing.assert_almost_equal(lat1_moved, 40)


def test_move_second_point_left_by_zero():
    lon2_moved, lat2_moved = move_second_point_left(
        lon1=50, lat1=40, lon2=51, lat2=40, move_by_meters=0
    )
    np.testing.assert_almost_equal(lon2_moved, 51)
    np.testing.assert_almost_equal(lat2_moved, 40)


def test_move_first_point_small_steps():
    lon1_moved, lat1_moved = move_first_point_left(
        lon1=50,
        lat1=0,
        lon2=50,
        lat2=10,
        move_by_meters=111e3 / 100,
    )
    np.testing.assert_almost_equal(lon1_moved, 49.99, decimal=2)
    np.testing.assert_almost_equal(lat1_moved, 0, decimal=2)


def test_move_second_point_small_steps():
    lon2_moved, lat2_moved = move_second_point_left(
        lon1=50,
        lat1=-10,
        lon2=50,
        lat2=0,
        move_by_meters=111e3 / 100,
    )
    np.testing.assert_almost_equal(lon2_moved, 49.99, decimal=2)
    np.testing.assert_almost_equal(lat2_moved, 0, decimal=2)


def test_move_middle_point_left_by_zero():
    lon2_moved, lat2_moved = move_middle_point_left(
        lon1=50,
        lat1=-10,
        lon2=50,
        lat2=0,
        lon3=50,
        lat3=10,
        move_by_meters=0,
    )
    np.testing.assert_almost_equal(lon2_moved, 50)
    np.testing.assert_almost_equal(lat2_moved, 0)


def test_move_middle_point_left_by_small_step():
    lon2_moved, lat2_moved = move_middle_point_left(
        lon1=50,
        lat1=-10,
        lon2=50,
        lat2=0,
        lon3=50,
        lat3=10,
        move_by_meters=111e3 / 100,
    )
    np.testing.assert_almost_equal(lon2_moved, 49.99, decimal=2)
    np.testing.assert_almost_equal(lat2_moved, 0, decimal=2)
