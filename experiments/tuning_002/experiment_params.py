"""Parameter spaces for hyperparameter tuning experiments.

Each experiment defines a parameter space (ranges to sample from) and the number
of random samples to generate. Parameter values can be:
- Single values (fixed across all samples)
- Tuples/lists (sampled uniformly for each sample)
"""

# Atlantic crossing route options (Florida <-> Irish Sea)
# Key = route name, Value = route waypoints
ROUTES = {
    "Atlantic_forward": {
        "lon_waypoints": (-80.5, -11.0),
        "lat_waypoints": (30.0, 50.0),
    },
    "Atlantic_backward": {
        "lon_waypoints": (-11.0, -80.5),
        "lat_waypoints": (50.0, 30.0),
    },
}

# Standard forcing data paths for 2021
FORCING_PATHS = {
    "currents_path": (
        "data_large/cmems_mod_glo_phy_my_0.083deg_P1D-m_time_2021_lat_+10_+65_lon_-100_+010_uo-vo.zarr"
    ),
    "waves_path": (
        "data_large/cmems_mod_glo_wav_my_0.2deg_PT3H-i_time_2021_lat_+10_+65_lon_-100_+010_VHM0-VMDR.zarr"
    ),
    "winds_path": (
        "data_large/cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_time_2021_lat_+10_+65_lon_-100_+010_eastward_wind-northward_wind.zarr"
    ),
    "engine": "zarr",
}

# Forcing scenarios for ablation studies
FORCING_SCENARIOS = {
    "baseline": {
        "currents_path": "data_large/cmems_mod_glo_phy_my_0.083deg_P1D-m_time_2021_lat_+10_+65_lon_-100_+010_uo-vo.zarr",
        "waves_path": "data_large/cmems_mod_glo_wav_my_0.2deg_PT3H-i_time_2021_lat_+10_+65_lon_-100_+010_VHM0-VMDR.zarr",
        "winds_path": "data_large/cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_time_2021_lat_+10_+65_lon_-100_+010_eastward_wind-northward_wind.zarr",
        "engine": "zarr",
    },
    "no_currents": {
        "currents_path": None,
        "waves_path": "data_large/cmems_mod_glo_wav_my_0.2deg_PT3H-i_time_2021_lat_+10_+65_lon_-100_+010_VHM0-VMDR.zarr",
        "winds_path": "data_large/cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_time_2021_lat_+10_+65_lon_-100_+010_eastward_wind-northward_wind.zarr",
        "engine": "zarr",
    },
    "no_winds": {
        "currents_path": "data_large/cmems_mod_glo_phy_my_0.083deg_P1D-m_time_2021_lat_+10_+65_lon_-100_+010_uo-vo.zarr",
        "waves_path": "data_large/cmems_mod_glo_wav_my_0.2deg_PT3H-i_time_2021_lat_+10_+65_lon_-100_+010_VHM0-VMDR.zarr",
        "winds_path": None,
        "engine": "zarr",
    },
    "no_waves": {
        "currents_path": "data_large/cmems_mod_glo_phy_my_0.083deg_P1D-m_time_2021_lat_+10_+65_lon_-100_+010_uo-vo.zarr",
        "waves_path": None,
        "winds_path": "data_large/cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_time_2021_lat_+10_+65_lon_-100_+010_eastward_wind-northward_wind.zarr",
        "engine": "zarr",
    },
}

# Quick hyperparameters for smoke testing
QUICK_HYPER = {
    "population_size": (4,),
    "generations": (1,),
    "mutation_iterations": (1,),
    "gd_iterations": (1,),
    "crossover_rounds": (0,),
    "crossover_strategy": ("random",),
    "enable_adaptation": (False,),
    # Fixed parameters
    "selection_acceptance_rate_warmup": 0.3,
    "num_elites": 2,
    "learning_rate_time": 0.5,
    "learning_rate_space": 0.5,
    "executor_type": "sequential",
    "num_workers": 1,
}

# Production hyperparameters for ablation studies
PRODUCTION_HYPER = {
    "population_size": (128, 256),
    "generations": (1, 2, 4),
    "mutation_iterations": (1, 3),
    "gd_iterations": (1, 2),
    "crossover_rounds": (0, 1, 2),
    "crossover_strategy": ("minimal_cost", "random"),
    "selection_quantile": (0.1, 0.25),
    "selection_acceptance_rate": (0.0, 0.25),
    "hazard_penalty_multiplier": (0.0, 100.0),
    "mutation_width_fraction_warmup": (0.99,),
    "mutation_displacement_fraction_warmup": (0.1, 0.25),
    "enable_adaptation": (True, False),
    "adaptation_scale_W": (0.5, 0.8),
    "adaptation_scale_D": (0.707, 0.894),
    # Fixed parameters
    "selection_acceptance_rate_warmup": 0.3,
    "mutation_width_fraction": 0.9,
    "mutation_displacement_fraction": 0.1,
    "num_elites": 2,
    "learning_rate_time": 0.5,
    "learning_rate_space": 0.5,
    "time_increment": 1200.0,
    "distance_increment": 10000.0,
    "executor_type": "sequential",
    "num_workers": 1,
}

# Experiment configurations
EXPERIMENTS = {
    "quick": {
        "n_samples": 5,
        "param_space": {
            "journey": {
                "route": ROUTES,
                "time_start": ("2021-01-01T00:00:00",),
                "speed_knots": (10.0,),
                "time_resolution_hours": 6.0,
            },
            "forcing": FORCING_PATHS,
            "hyper": QUICK_HYPER,
        },
        "output_prefix": "results_quick",
    },
    "ablation_baseline": {
        "n_samples": 2_000,
        "param_space": {
            "journey": {
                "route": ROUTES,
                "time_start": tuple(
                    f"2021-{mm:02d}-01T00:00:00" for mm in range(1, 12 + 1)
                ),
                "speed_knots": (8.0, 10.0, 12.0),
                "time_resolution_hours": 6.0,
            },
            "forcing": {**FORCING_SCENARIOS["baseline"], "scenario_name": "baseline"},
            "hyper": PRODUCTION_HYPER,
        },
        "output_prefix": "results_ablation_baseline",
    },
    "ablation_no_currents": {
        "n_samples": 2_000,
        "param_space": {
            "journey": {
                "route": ROUTES,
                "time_start": tuple(
                    f"2021-{mm:02d}-01T00:00:00" for mm in range(1, 12 + 1)
                ),
                "speed_knots": (8.0, 10.0, 12.0),
                "time_resolution_hours": 6.0,
            },
            "forcing": {
                **FORCING_SCENARIOS["no_currents"],
                "scenario_name": "no_currents",
            },
            "hyper": PRODUCTION_HYPER,
        },
        "output_prefix": "results_ablation_no_currents",
    },
    "ablation_no_winds": {
        "n_samples": 2_000,
        "param_space": {
            "journey": {
                "route": ROUTES,
                "time_start": tuple(
                    f"2021-{mm:02d}-01T00:00:00" for mm in range(1, 12 + 1)
                ),
                "speed_knots": (8.0, 10.0, 12.0),
                "time_resolution_hours": 6.0,
            },
            "forcing": {**FORCING_SCENARIOS["no_winds"], "scenario_name": "no_winds"},
            "hyper": PRODUCTION_HYPER,
        },
        "output_prefix": "results_ablation_no_winds",
    },
    "ablation_no_waves": {
        "n_samples": 2_000,
        "param_space": {
            "journey": {
                "route": ROUTES,
                "time_start": tuple(
                    f"2021-{mm:02d}-01T00:00:00" for mm in range(1, 12 + 1)
                ),
                "speed_knots": (8.0, 10.0, 12.0),
                "time_resolution_hours": 6.0,
            },
            "forcing": {**FORCING_SCENARIOS["no_waves"], "scenario_name": "no_waves"},
            "hyper": PRODUCTION_HYPER,
        },
        "output_prefix": "results_ablation_no_waves",
    },
}
