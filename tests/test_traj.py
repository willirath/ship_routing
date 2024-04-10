from ship_routing import Trajectory

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
    traj_0 = Trajectory(lon=[0, 0], lat=[0, 1])

    # don't move
    traj_1 = traj_0.move_node_left(node_num=0, move_by_meters=0)
    np.testing.assert_array_almost_equal(traj_0.data_frame, traj_1.data_frame)

    # move by approx one hundreth degree
    traj_2 = traj_0.move_node_left(node_num=0, move_by_meters=111_139 / 100)
    np.testing.assert_array_almost_equal(traj_2.lon[0], 1 / 100, decimal=3)
