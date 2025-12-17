# Hyperparameter Tuning Experiment (tuning_003)

Parsl-based hyperparameter tuning for ship routing optimization.

## Key Difference from tuning_002

In tuning_003, `offspring_size` is computed as a ratio of `population_size`:
- `offspring_ratio` is sampled from {0.25, 0.5, 1.0}
- `offspring_size = int(population_size * offspring_ratio)`

This tests the impact of varying offspring pool sizes relative to population size, addressing the bug in tuning_002 where offspring_size was fixed at 4.

**Example configurations:**
- If `population_size=64`, `offspring_size` ∈ {16, 32, 64}
- If `population_size=128`, `offspring_size` ∈ {32, 64, 128}
- If `population_size=256`, `offspring_size` ∈ {64, 128, 256}

## Quick Start

### Local Testing
```bash
python run_tuning.py --experiment quick --execution local-small
```

### Production on SLURM
```bash
# Submit individual ablation experiments
sbatch run_baseline.job
sbatch run_no_currents.job
sbatch run_no_waves.job
sbatch run_no_winds.job
```

## Available Experiments

- `quick`: 5 configs (minimal smoke test)
- `ablation_baseline`: 4000 configs, full forcing
- `ablation_no_currents`: 4000 configs, currents disabled
- `ablation_no_waves`: 4000 configs, waves disabled
- `ablation_no_winds`: 4000 configs, winds disabled

## Available Execution Configs

- `local-small`: 2 workers, 5min timeout (quick testing)
- `local-large`: 8 workers, 10min timeout (local development)
- `nesh-test`: 4 workers, 2 nodes, 1h (HPC testing)
- `nesh-prod-50`: 50 workers, 50 nodes, 4h (production runs)

## Monitoring

### Check SLURM jobs
```bash
squeue -u $USER  # See orchestrator + Parsl worker jobs
```

### Watch orchestrator progress
```bash
tail -f slurm_logs/baseline_*.out
tail -f slurm_logs/no_currents_*.out
tail -f slurm_logs/no_waves_*.out
tail -f slurm_logs/no_winds_*.out
```

### View results
```bash
ls -lh results/results_*.msgpack
```

## Analysis

Use notebooks/ for post-experiment analysis and `scripts/load_tuning_results.py` for loading results.

## Architecture

- **Orchestrator**: Job scripts run `run_tuning.py` on a compute node
- **Workers**: Parsl's HighThroughputExecutor spawns SLURM worker jobs
- **Results**: Collected directly by orchestrator, serialized to msgpack
- **Scaling**: Auto-scaling based on task queue (htex_auto_scale)

## Implementation Notes

The custom sampling logic in `run_tuning.py` extracts `offspring_ratio` from the parameter space and computes `offspring_size` before creating the `HyperParams` dataclass. This allows the ratio-based relationship to be maintained while keeping the core library (`config_factory.py`) unchanged.
