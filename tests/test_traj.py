from ship_routing import Trajectory
from ship_routing.currents import load_currents_time_average
from ship_routing.mc import move_random_node

from pathlib import Path

import pint

import numpy as np

import pytest

FIXTURE_DIR = Path(__file__).parent.resolve() / "test_data"


def test_trajectory_from_line_string_idempotency():
    traj_0 = Trajectory(lon=[1, 2, 3], lat=[-3, -4, -5])
    traj_1 = Trajectory.from_line_string(traj_0.line_string)

    np.testing.assert_array_equal(traj_0.lon, traj_1.lon)
    np.testing.assert_array_equal(traj_0.lat, traj_1.lat)


def test_traj_from_scalar_position():
    traj = Trajectory(lon=1, lat=2)


def test_trajectory_from_data_frame_idempotency():
    traj_0 = Trajectory(lon=[1, 2, 3], lat=[-3, -4, -5])
    traj_1 = Trajectory.from_data_frame(traj_0.data_frame)

    np.testing.assert_array_equal(traj_0.lon, traj_1.lon)
    np.testing.assert_array_equal(traj_0.lat, traj_1.lat)


def test_trajectory_from_data_frame_types():
    traj_0 = Trajectory(lon=[1, 2, 3], lat=[-3, -4, -5])
    traj_1 = Trajectory.from_data_frame(traj_0.data_frame)

    assert isinstance(traj_1.lon, list)
    assert isinstance(traj_1.lat, list)


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


def test_traj_refinement_retains_duration():
    traj_0 = Trajectory(lon=[0, 0], lat=[-1, 1], duration_seconds=1_234_567)
    traj_1 = traj_0.refine(new_dist=111e3)
    np.testing.assert_almost_equal(traj_0.duration_seconds, traj_1.duration_seconds)


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
    np.testing.assert_almost_equal(
        1.0,
        float(traj.speed_ms * ureg.meter / ureg.second / speed),
    )


def test_traj_cost():
    traj = Trajectory(lon=[0, 1], lat=[0, 0], duration_seconds=3600).refine(
        new_dist=1_000
    )
    data_set = load_currents_time_average(
        data_file=FIXTURE_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_1deg_5day.nc"
    )
    cost = traj.estimate_cost_through(data_set=data_set)


def test_traj_cost_power_law():
    traj_fast = Trajectory(lon=[0, 1], lat=[0, 0], duration_seconds=3_600).refine(
        new_dist=1_000
    )
    traj_slow = Trajectory(lon=[0, 1], lat=[0, 0], duration_seconds=36_000).refine(
        new_dist=1_000
    )
    data_set = load_currents_time_average(
        data_file=FIXTURE_DIR
        / "currents/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_2021-01_1deg_5day.nc"
    )
    cost_fast = traj_fast.estimate_cost_through(data_set=data_set)
    cost_slow = traj_slow.estimate_cost_through(data_set=data_set)

    np.testing.assert_array_less(cost_slow, cost_fast)


def test_traj_slicing():
    traj_0 = Trajectory(lon=[0, 1, 2, 3], lat=[0, 1, 2, 4])
    traj_1 = traj_0[:3]
    traj_2 = traj_0[1]
    assert traj_1.lon[0] == traj_0.lon[0]
    assert traj_1.lat[0] == traj_0.lat[0]
    assert traj_1.lon[2] == traj_0.lon[2]
    assert traj_1.lat[2] == traj_0.lat[2]
    assert len(traj_1) == 3
    assert traj_2.lon[0] == traj_0.lon[1]
    assert traj_2.lat[0] == traj_0.lat[1]
    assert len(traj_2) == 1


def test_traj_cumulative_distance():
    traj_0 = Trajectory(lon=[0, 1, 2, 3], lat=[0, 0, 0, 0])
    ureg = pint.UnitRegistry()
    true_dist = (np.array([0, 60, 120, 180]) * ureg["nautical_mile"]).to("meter")
    test_dist = np.array(traj_0.dist) * ureg.meter
    assert test_dist[0] == true_dist[0]
    np.testing.assert_array_almost_equal(1, true_dist[1:] / test_dist[1:], decimal=2)


def test_traj_cumulative_time():
    traj_0 = Trajectory(lon=[0, 1, 2, 3], lat=[0, 0, 0, 0], duration_seconds=1_000)
    np.testing.assert_array_almost_equal(
        [0, 1000 / 3, 2000 / 3, 1000], traj_0.time_since_start
    )


def test_traj_add_waypoint():
    ureg = pint.UnitRegistry()
    traj_0 = Trajectory(lon=[0, 0], lat=[-1 / 60, 1 / 60])
    traj_1 = traj_0.add_waypoint(dist=float(1 * ureg.nautical_mile / ureg.meter))
    np.testing.assert_almost_equal(traj_1.lon[1], 0, decimal=3)
    np.testing.assert_almost_equal(traj_1.lat[1], 0, decimal=3)
    assert len(traj_1) == 3


def test_traj_slice_with_distance_large_distances():
    traj = Trajectory(
        lon=[0, 1, 2, 3, 4, 5],
        lat=[0, 0, 0, 0, 0, 0],
    )
    distances = traj.dist
    d0 = (distances[1] + distances[2]) / 2.0
    d1 = (distances[3] + distances[4]) / 2.0
    traj_sliced = traj.slice_with_dist(d0=d0, d1=d1)
    np.testing.assert_almost_equal(
        traj_sliced.dist[-1],
        (distances[3] + distances[4] - distances[1] - distances[2]) / 2.0,
    )


@pytest.mark.parametrize("offset", [100, 10, 1, 0.1, 0.01, 0.001])
def test_traj_slice_with_distance_small_distances(offset):
    traj = Trajectory(
        lon=[0, 1, 2, 3, 4, 5],
        lat=[0, 0, 0, 0, 0, 0],
    )
    distances = traj.dist
    d0 = distances[1] - offset
    d1 = distances[1] + offset
    traj_sliced = traj.slice_with_dist(d0=d0, d1=d1)
    np.testing.assert_almost_equal(traj_sliced.dist[-1], 2.0 * offset)


def test_traj_simple_slicing_handles_duration():
    traj_0 = Trajectory(lon=[0, 1, 2], lat=[0, 0, 0], duration_seconds=2000)
    traj_1 = traj_0[:2]
    np.testing.assert_almost_equal(
        traj_1.duration_seconds, 0.5 * traj_0.duration_seconds
    )


def test_traj_dist_slicing_handles_duration():
    traj_0 = Trajectory(lon=[0, 1, 2], lat=[0, 0, 0], duration_seconds=2000)
    length = traj_0.length_meters
    d0 = length / 3
    d1 = 2 * length / 3
    traj_1 = traj_0.slice_with_dist(d0=d0, d1=d1)
    np.testing.assert_almost_equal(
        traj_1.duration_seconds, 1 / 3 * traj_0.duration_seconds
    )


def test_traj_segmentation_equal_num_seg():
    traj_0 = Trajectory(
        lon=[0, 1, 2, 3, 4, 5],
        lat=[0, -1, 0, 0, 1, 0],
    ).refine(new_dist=20_000)
    traj_1 = Trajectory(
        lon=[0, 5],
        lat=[0, 0],
    ).refine(new_dist=30_000)
    seg_0, seg_1 = traj_0.segment_at_other_traj(traj_1)
    assert len(seg_0) == len(seg_1)


@pytest.mark.xfail(
    reason="May fail due to roundoff error.",
    strict=False,
)
def test_traj_segmentation_equal_num_seg_fuzzy():
    for nfuzz in range(7):
        traj_0 = Trajectory(lon=[0, 1, 2, 3], lat=[0, -1, 0, 0]).refine(new_dist=10_000)
        traj_1 = Trajectory(lon=[0, 3], lat=[0, 0]).refine(new_dist=10_000)
        for nmod in range(33):
            traj_0 = move_random_node(trajectory=traj_0, max_dist_meters=200)
            traj_1 = move_random_node(trajectory=traj_1, max_dist_meters=200)
        seg_0, seg_1 = traj_0.segment_at_other_traj(traj_1)
        assert len(seg_0) == len(seg_1)


@pytest.mark.xfail(
    reason="May fail due to roundoff error.",
    strict=False,
)
def test_traj_segmentation_no_singleton_segments_fuzzy():
    for nfuzz in range(7):
        traj_0 = Trajectory(lon=[0, 1, 2, 3], lat=[0, -1, 0, 0]).refine(new_dist=10_000)
        traj_1 = Trajectory(lon=[0, 3], lat=[0, 0]).refine(new_dist=10_000)
        for nmod in range(33):
            traj_0 = move_random_node(trajectory=traj_0, max_dist_meters=200)
            traj_1 = move_random_node(trajectory=traj_1, max_dist_meters=200)
        seg_0, seg_1 = traj_0.segment_at_other_traj(traj_1)
        assert all([len(s) > 1 for s in seg_0])
        assert all([len(s) > 1 for s in seg_1])


def test_traj_segmentation_handles_duration():
    traj_0 = Trajectory(
        lon=[0, 1, 2, 3], lat=[0, -1, 0, 0], duration_seconds=24 * 3600
    ).refine(new_dist=10_000)
    traj_1 = Trajectory(lon=[0, 3], lat=[0, 0], duration_seconds=24 * 3600).refine(
        new_dist=10_000
    )
    seg_0, seg_1 = traj_0.segment_at_other_traj(traj_1)
    for s in seg_0:
        np.testing.assert_almost_equal(
            s.duration_seconds / s.length_meters,
            traj_0.duration_seconds / traj_0.length_meters,
        )
    for s in seg_1:
        np.testing.assert_almost_equal(
            s.duration_seconds / s.length_meters,
            traj_1.duration_seconds / traj_1.length_meters,
        )


def test_traj_concat_preserves_lengths():
    traj_0 = Trajectory(lon=[0, 1], lat=[1, 2]).refine(new_dist=20_000)
    traj_1 = Trajectory(lon=[1, 5], lat=[2, 3]).refine(new_dist=10_000)
    traj_2 = traj_0 + traj_1
    np.testing.assert_almost_equal(
        traj_0.length_meters + traj_1.length_meters, traj_2.length_meters
    )


def test_traj_concatenation_handles_duration():
    traj_0 = Trajectory(lon=[0, 1], lat=[1, 2], duration_seconds=24 * 3600 * 2).refine(
        new_dist=20_000
    )
    traj_1 = Trajectory(lon=[1, 5], lat=[2, 3], duration_seconds=24 * 3600 * 2).refine(
        new_dist=10_000
    )
    traj_2 = traj_0 + traj_1
    np.testing.assert_almost_equal(
        traj_0.duration_seconds + traj_1.duration_seconds, traj_2.duration_seconds
    )


def test_traj_copying():
    traj_0 = Trajectory(lon=[1, 2, 3], lat=[-1, 2, 4], duration_seconds=100_000)
    traj_1 = traj_0.copy()

    # ensure DIFFERENT object identities
    assert traj_0 is not traj_1
    assert traj_0.lon is not traj_1.lon
    assert traj_0.lat is not traj_1.lat
    assert traj_0.data_frame is not traj_0.data_frame
    assert traj_0.line_string is not traj_0.line_string

    # ensure IDENTICAL values
    np.testing.assert_almost_equal(traj_0.lon, traj_1.lon)
    np.testing.assert_almost_equal(traj_0.lat, traj_1.lat)
    np.testing.assert_almost_equal(traj_0.length_meters, traj_1.length_meters)
    np.testing.assert_almost_equal(traj_0.duration_seconds, traj_1.duration_seconds)


def test_traj_legs_pos():
    traj_0 = Trajectory(lon=[1, 2, 3], lat=[-1, 2, 4], duration_seconds=100_000)
    legs_pos_truth = (((1, -1), (2, 2)), ((2, 2), (3, 4)))
    legs_pos_test = traj_0.legs_pos
    np.testing.assert_array_almost_equal(legs_pos_truth, legs_pos_test)


def test_traj_legs_duration():
    traj_0 = Trajectory(lon=[1, 2, 3], lat=[-1, 2, 4], duration_seconds=100_000)
    legs_duration = traj_0.legs_duration

    np.testing.assert_almost_equal(sum(legs_duration), 100_000)


def test_traj_legs_length():
    traj_0 = Trajectory(lon=[1, 2, 3], lat=[-1, 2, 4], duration_seconds=100_000)
    legs_length_meters = traj_0.legs_length_meters
    np.testing.assert_almost_equal(sum(legs_length_meters), traj_0.length_meters)


def test_traj_legs_speed():
    traj_0 = Trajectory(lon=[1, 2, 3], lat=[-1, 2, 4], duration_seconds=100_000)
    legs_speed = traj_0.legs_speed
    np.testing.assert_array_almost_equal(legs_speed, traj_0.speed_ms)