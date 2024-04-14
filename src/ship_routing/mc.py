import numpy as np

from .traj import Trajectory

from .config import MINIMAL_NODE_MOVE_DIST


def move_random_node(
    trajectory: Trajectory = None,
    max_dist_meters: float = 1_000,
    only_move_intermediate_nodes=True,
):
    if max_dist_meters <= MINIMAL_NODE_MOVE_DIST:
        raise ValueError(
            f"max_dist_meters must be larger than {MINIMAL_NODE_MOVE_DIST}"
        )
    num_nodes = len(trajectory)
    if only_move_intermediate_nodes:
        random_node = np.random.randint(1, num_nodes - 1)
    else:
        random_node = np.random.randint(0, num_nodes)
    random_dist = np.random.uniform(MINIMAL_NODE_MOVE_DIST, max_dist_meters) * (
        (-1) ** np.random.randint(0, 2)
    )
    return trajectory.move_node_left(node_num=random_node, move_by_meters=random_dist)
