from ship_routing.mc import move_random_node
from ship_routing import Trajectory

import numpy as np
import operator

import pytest


@pytest.mark.parametrize("intermediate_flag", [True, False])
def test_random_movement_zero_step(intermediate_flag):
    traj_0 = Trajectory(lon=[0, 0, 0], lat=[-1, 0, 1])
    traj_1 = move_random_node(
        trajectory=traj_0,
        max_dist_meters=0,
        only_move_intermediate_nodes=intermediate_flag,
    )
    np.testing.assert_array_almost_equal(
        traj_0.data_frame.to_numpy(), traj_1.data_frame.to_numpy()
    )


@pytest.mark.parametrize("intermediate_flag", [True, False])
def test_random_movement_nonvanishing_single_step(intermediate_flag):
    traj_0 = Trajectory(lon=[0, 0, 0], lat=[-1, 0, 1])
    traj_1 = move_random_node(
        trajectory=traj_0,
        max_dist_meters=1e6,
        only_move_intermediate_nodes=intermediate_flag,
    )
    # assert arrays compare _not equal_
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        traj_0.data_frame.to_numpy(),
        traj_1.data_frame.to_numpy(),
    )
