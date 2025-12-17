# Hyperparameter Tuning Experiment (tuning_002)

Parsl-based hyperparameter tuning for ship routing optimization.

## Quick Start

### Local Testing
```bash
python run_tuning.py --experiment test --execution local-small
```

### Production on SLURM
```bash
sbatch submit_tuning.job
# Edit submit_tuning.job line 28 to change experiment/execution/seed
```

## Available Experiments

- `test`: 40 configs (2 directions × 20 runs), small parameter space
- `production`: 2000 configs (2 directions × 1000 runs), full parameter space
- `quick`: 5 configs (minimal smoke test)

## Available Execution Configs

- `local-small`: 2 workers, 5min timeout (quick testing)
- `local-large`: 8 workers, 10min timeout (local development)
- `nesh-test`: 4 workers, 2 nodes, 1h (HPC testing)
- `nesh-prod`: 8 workers, 10 nodes, 4h (production runs)

## Monitoring

### Check SLURM jobs
```bash
squeue -u $USER  # See orchestrator + Parsl worker jobs
```

### Watch orchestrator progress
```bash
tail -f slurm_logs/orchestrator_*.out
```

### View results
```bash
ls -lh results/results_*.msgpack
```

## Analysis

Use notebooks/ for post-experiment analysis and `scripts/load_tuning_results.py` for loading results.

## Architecture

- **Orchestrator**: `submit_tuning.job` runs `run_tuning.py` on a compute node
- **Workers**: Parsl's HighThroughputExecutor spawns SLURM worker jobs
- **Results**: Collected directly by orchestrator, serialized to msgpack
- **Scaling**: Auto-scaling based on task queue (htex_auto_scale)
