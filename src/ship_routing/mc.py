import numpy as np

from .traj import Trajectory


def move_random_node(
    trajectory: Trajectory = None,
    max_dist_meters: float = 1_000,
    only_move_intermediate_nodes=True,
):
    num_nodes = len(trajectory)
    if only_move_intermediate_nodes:
        random_node = np.random.randint(1, num_nodes - 1)
    else:
        random_node = np.random.randint(0, num_nodes)
    random_dist = np.random.uniform(-max_dist_meters, max_dist_meters)
    return trajectory.move_node_left(node_num=random_node, move_by_meters=random_dist)
