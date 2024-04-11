from ship_routing import Trajectory

import pint

import numpy as np


def test_trajectory_from_line_string_idempotency():
    traj_0 = Trajectory(lon=[1, 2, 3], lat=[-3, -4, -5])
    traj_1 = Trajectory.from_line_string(traj_0.line_string)

    np.testing.assert_array_equal(traj_0.lon, traj_1.lon)
    np.testing.assert_array_equal(traj_0.lat, traj_1.lat)


def test_trajectory_from_data_frame_idempotency():
    traj_0 = Trajectory(lon=[1, 2, 3], lat=[-3, -4, -5])
    traj_1 = Trajectory.from_data_frame(traj_0.data_frame)

    np.testing.assert_array_equal(traj_0.lon, traj_1.lon)
    np.testing.assert_array_equal(traj_0.lat, traj_1.lat)


def test_trajectory_move_node():
    traj_0 = Trajectory(lon=[0, 0, 0], lat=[-1, 0, 1])

    # don't move, each node
    traj_1 = traj_0.move_node_left(node_num=0, move_by_meters=0)
    np.testing.assert_array_almost_equal(traj_0.data_frame, traj_1.data_frame)
    traj_1 = traj_0.move_node_left(node_num=1, move_by_meters=0)
    np.testing.assert_array_almost_equal(traj_0.data_frame, traj_1.data_frame)
    traj_1 = traj_0.move_node_left(node_num=2, move_by_meters=0)
    np.testing.assert_array_almost_equal(traj_0.data_frame, traj_1.data_frame)

    # move each node by approx one hundreth degree
    traj_2 = traj_0.move_node_left(node_num=0, move_by_meters=111_139 / 100)
    np.testing.assert_array_almost_equal(traj_2.lon[0], -1 / 100, decimal=3)
    traj_2 = traj_0.move_node_left(node_num=1, move_by_meters=111_139 / 100)
    np.testing.assert_array_almost_equal(traj_2.lon[1], -1 / 100, decimal=3)
    traj_2 = traj_0.move_node_left(node_num=2, move_by_meters=111_139 / 100)
    np.testing.assert_array_almost_equal(traj_2.lon[2], -1 / 100, decimal=3)


def test_traj_refinement():
    traj_0 = Trajectory(lon=[0, 0], lat=[-1, 1])
    traj_1 = traj_0.refine(new_dist=111e3)
    assert len(traj_1) == 3


def test_traj_refinement_no_downsampling():
    # simple traj no refinement nowhere
    traj_0 = Trajectory(lon=[0, 0], lat=[-1, 1])
    traj_1 = traj_0.refine(new_dist=333e3)
    assert len(traj_0) == len(traj_1)

    # 3pt traj no refinement in first segment
    traj_0 = Trajectory(lon=[0, 0, 0], lat=[-1, 0, 3])
    traj_1 = traj_0.refine(new_dist=222e3)
    assert len(traj_1) == len(traj_0) + 1


def test_traj_repr():
    traj = Trajectory(lon=[123, 45], lat=[67, 89])
    rpr = repr(traj)
    assert "123" in rpr
    assert "45" in rpr
    assert "67" in rpr
    assert "89" in rpr
    assert "lon" in rpr
    assert "lat" in rpr


def test_traj_speed():
    traj = Trajectory(lon=[0, 1], lat=[0, 0], duration_seconds=3600)
    ureg = pint.UnitRegistry()
    speed = (traj.length_meters * ureg.meter / traj.duration_seconds / ureg.second).to(
        "knots"
    )
    speed_expected = (60 * ureg.nautical_mile / ureg.hour).to("knots")
    np.testing.assert_almost_equal(float(speed / speed_expected), 1.0, decimal=2)
