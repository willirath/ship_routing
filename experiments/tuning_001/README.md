# Hyperparameter Tuning Experiment

## Structure

- `job_scripts/` - SLURM job scripts
- `notebooks/` - Analysis notebooks
- `scripts/` - Python utilities
- `results/` - Output files (.msgpack)
- `figures/` - Generated plots
- `data_large/` - Symlink to forcing data

## Usage

Run from this directory:

```bash
# Single job
sbatch job_scripts/run_experiment.job

# Full experiment (multiple jobs for, e.g., months, speeds, etc.)
./job_scripts/start_exp.sh
```

## Analysis

Open notebooks from `notebooks/` directory. They import from `scripts/` via symlink.
