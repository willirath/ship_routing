import numpy as np

from .traj import Trajectory

def move_random_intermediate_node(
        trajectory: Trajectory = None,
        max_dist_meters: float = 1_000,
):
    num_nodes = len(trajectory)
    random_node = np.random.randint(1, num_nodes-1)
    random_dist = np.random.uniform(- max_dist_meters, max_dist_meters)
    return trajectory.move_node_left(node_num=random_node, move_by_meters=random_dist)