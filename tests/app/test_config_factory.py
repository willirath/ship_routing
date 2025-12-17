"""Tests for config_factory sampling logic."""

import random

import pytest

from ship_routing.app.config_factory import (
    _is_named_options_dict,
    _sample_dict,
    sample_routing_configs,
    sample_value,
)


class TestSampleValue:
    """Tests for sample_value function."""

    def test_single_value_passthrough(self):
        """Single values should be returned as-is."""
        assert sample_value(42) == 42
        assert sample_value("hello") == "hello"
        assert sample_value(None) is None

    def test_single_element_tuple(self):
        """Single-element tuples should return the element."""
        assert sample_value((42,)) == 42

    def test_single_element_list(self):
        """Single-element lists should return the element."""
        assert sample_value([42]) == 42

    def test_multi_element_choice(self):
        """Multi-element sequences should sample uniformly."""
        random.seed(42)
        options = [1, 2, 3, 4, 5]
        # Sample multiple times to check it's actually random
        samples = [sample_value(options) for _ in range(100)]
        # All sampled values should be in options
        assert all(s in options for s in samples)
        # Should get multiple different values (probabilistic but very likely)
        assert len(set(samples)) > 1


class TestIsNamedOptionsDict:
    """Tests for _is_named_options_dict helper."""

    def test_empty_dict_returns_false(self):
        """Empty dict should return False."""
        assert not _is_named_options_dict({})

    def test_all_dict_values_returns_true(self):
        """Dict with all dict values should return True."""
        d = {
            "route1": {"lon": 1, "lat": 2},
            "route2": {"lon": 3, "lat": 4},
        }
        assert _is_named_options_dict(d)

    def test_mixed_values_returns_false(self):
        """Dict with mixed value types should return False."""
        d = {
            "route1": {"lon": 1, "lat": 2},
            "speed": 10.0,  # Not a dict
        }
        assert not _is_named_options_dict(d)

    def test_all_non_dict_values_returns_false(self):
        """Dict with no dict values should return False."""
        d = {"speed": 10.0, "time": "2021-01-01"}
        assert not _is_named_options_dict(d)


class TestSampleDict:
    """Tests for _sample_dict function."""

    def test_simple_dict(self):
        """Simple dict with no nested structure."""
        random.seed(42)
        param = {"a": 1, "b": (2, 3), "c": "fixed"}
        result = _sample_dict(param)
        assert result["a"] == 1
        assert result["b"] in (2, 3)
        assert result["c"] == "fixed"

    def test_nested_dict(self):
        """Nested dict structure."""
        random.seed(42)
        param = {"outer": {"inner": (1, 2, 3), "fixed": "value"}}
        result = _sample_dict(param)
        assert result["outer"]["inner"] in (1, 2, 3)
        assert result["outer"]["fixed"] == "value"

    def test_named_routes_pairing(self):
        """Named routes dict should sample one route and merge contents."""
        param = {
            "route": {
                "Atlantic_forward": {
                    "lon_waypoints": (-80.5, -11.0),
                    "lat_waypoints": (30.0, 50.0),
                },
                "Atlantic_backward": {
                    "lon_waypoints": (-11.0, -80.5),
                    "lat_waypoints": (50.0, 30.0),
                },
            },
            "speed": (8.0, 10.0),
        }

        # Sample multiple times to verify pairing
        random.seed(100)
        for _ in range(20):
            result = _sample_dict(param)
            # Verify pairing: forward route has specific coordinates
            if result["name"] == "Atlantic_forward":
                assert result["lon_waypoints"] == (-80.5, -11.0)
                assert result["lat_waypoints"] == (30.0, 50.0)
            else:
                assert result["name"] == "Atlantic_backward"
                assert result["lon_waypoints"] == (-11.0, -80.5)
                assert result["lat_waypoints"] == (50.0, 30.0)
            # Speed should still be independent
            assert result["speed"] in (8.0, 10.0)
            # route key should not appear in result (merged into parent)
            assert "route" not in result

    def test_named_routes_nested_in_journey(self):
        """Named routes can be nested inside journey config."""
        param = {
            "journey": {
                "route": {
                    "forward": {"lon": (-80, -11), "lat": (30, 50)},
                    "backward": {"lon": (-11, -80), "lat": (50, 30)},
                },
                "speed": 10.0,
            },
        }

        random.seed(42)
        result = _sample_dict(param)

        # Journey should have lon, lat, name, speed (no nested route key)
        assert "journey" in result
        journey = result["journey"]
        assert "name" in journey
        assert "lon" in journey
        assert "lat" in journey
        assert journey["speed"] == 10.0
        assert "route" not in journey

    def test_multiple_named_options_dicts(self):
        """Multiple named options dicts can be in same parent dict.

        Note: When multiple named options dicts are present, the last one's
        "name" will overwrite previous ones. This is a limitation of the
        current implementation, but not an issue for our use case where
        we only have one named options dict per config level.
        """
        param = {
            "route": {
                "r1": {"lon": 1, "lat": 2},
                "r2": {"lon": 3, "lat": 4},
            },
            "scenario": {
                "calm": {"wind": 5.0, "wave": 1.0},
                "stormy": {"wind": 20.0, "wave": 5.0},
            },
        }

        random.seed(42)
        result = _sample_dict(param)

        # Both dicts are sampled and merged
        assert "name" in result
        # Last named options dict wins for "name" key
        assert result["name"] in ("calm", "stormy")
        assert "lon" in result
        assert "lat" in result
        assert "wind" in result
        assert "wave" in result


class TestSampleRoutingConfigs:
    """Integration tests for sample_routing_configs."""

    def test_atlantic_routes_pairing(self):
        """Atlantic routes should maintain waypoint pairing."""
        param_space = {
            "journey": {
                "route": {
                    "Atlantic_forward": {
                        "lon_waypoints": (-80.5, -11.0),
                        "lat_waypoints": (30.0, 50.0),
                    },
                    "Atlantic_backward": {
                        "lon_waypoints": (-11.0, -80.5),
                        "lat_waypoints": (50.0, 30.0),
                    },
                },
                "time_start": "2021-01-01T00:00:00",
                "speed_knots": 10.0,
                "time_resolution_hours": 6.0,
            },
            "forcing": {
                "currents_path": "data/currents.zarr",
                "waves_path": "data/waves.zarr",
                "winds_path": "data/winds.zarr",
                "engine": "zarr",
            },
            "hyper": {
                "population_size": 4,
                "generations": 2,
            },
        }

        configs = sample_routing_configs(param_space, n_samples=50, seed=42)

        # Verify all configs have valid pairings
        for config in configs:
            journey = config.journey
            if journey.name == "Atlantic_forward":
                assert journey.lon_waypoints == (-80.5, -11.0)
                assert journey.lat_waypoints == (30.0, 50.0)
            elif journey.name == "Atlantic_backward":
                assert journey.lon_waypoints == (-11.0, -80.5)
                assert journey.lat_waypoints == (50.0, 30.0)
            else:
                pytest.fail(f"Unexpected route name: {journey.name}")

        # Should get both routes in the sample (probabilistic but very likely with 50 samples)
        names = [c.journey.name for c in configs]
        assert "Atlantic_forward" in names
        assert "Atlantic_backward" in names

    def test_independent_random_seeds(self):
        """Each config should get a unique random seed."""
        param_space = {
            "journey": {
                "route": {
                    "test_route": {
                        "lon_waypoints": (-80.5, -11.0),
                        "lat_waypoints": (30.0, 50.0),
                    },
                },
                "time_start": "2021-01-01T00:00:00",
                "speed_knots": 10.0,
                "time_resolution_hours": 6.0,
            },
            "forcing": {
                "currents_path": "data/currents.zarr",
                "waves_path": "data/waves.zarr",
                "winds_path": "data/winds.zarr",
                "engine": "zarr",
            },
            "hyper": {"population_size": 4, "generations": 2},
        }

        configs = sample_routing_configs(param_space, n_samples=10, seed=42)
        seeds = [c.hyper.random_seed for c in configs]

        # All seeds should be different
        assert len(set(seeds)) == len(seeds)

    def test_no_invalid_waypoint_combinations(self):
        """Regression test: ensure no invalid waypoint pairings occur."""
        param_space = {
            "journey": {
                "route": {
                    "Atlantic_forward": {
                        "lon_waypoints": (-80.5, -11.0),
                        "lat_waypoints": (30.0, 50.0),
                    },
                    "Atlantic_backward": {
                        "lon_waypoints": (-11.0, -80.5),
                        "lat_waypoints": (50.0, 30.0),
                    },
                },
                "time_start": "2021-01-01T00:00:00",
                "speed_knots": 10.0,
                "time_resolution_hours": 6.0,
            },
            "forcing": {
                "currents_path": "data/currents.zarr",
                "waves_path": "data/waves.zarr",
                "winds_path": "data/winds.zarr",
                "engine": "zarr",
            },
            "hyper": {
                "population_size": 4,
                "generations": 2,
            },
        }

        # Generate many configs to ensure no invalid combinations slip through
        configs = sample_routing_configs(param_space, n_samples=200, seed=123)

        # Define valid combinations
        valid_combos = {
            ("Atlantic_forward", (-80.5, -11.0), (30.0, 50.0)),
            ("Atlantic_backward", (-11.0, -80.5), (50.0, 30.0)),
        }

        # Check all configs
        for config in configs:
            combo = (
                config.journey.name,
                config.journey.lon_waypoints,
                config.journey.lat_waypoints,
            )
            assert combo in valid_combos, f"Invalid combination found: {combo}"
