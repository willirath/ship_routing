# Ship Routing

## Development environment

This repository is managed with [Pixi](https://pixi.sh), which takes care of the
Conda-based environment, editable install, and repeatable tasks.

```bash
pixi install        # create the environment declared in pixi.toml
pixi run tests      # run pytest
pixi run lint       # check formatting with black
pixi shell          # drop into an activated shell
pixi run jupyterlab # (optional) start the notebook environment
```

If you need additional tools, add them to [pixi.toml](pixi.toml) so every contributor
inherits the same setup.

## HPC Setup

On HPC systems with limited HOME quota, use the provided script to redirect Pixi's cache directories to scratch storage:

```bash
source setup_pixi_hpc.sh <scratch_directory>  # e.g., $SCRATCH, $WORK, (defaults to $PWD)
```

This configures `PIXI_CACHE_DIR`, `PIXI_HOME`, `RATTLER_CACHE_DIR`, and `CONDA_PKGS_DIRS` to use the specified directory. If no directory is provided, the current working directory is used by default.

See [setup_pixi_hpc.sh](setup_pixi_hpc.sh) for details.

## Data Setup

- [data/test/](data/test/): Included, ~10 MB
- [data/large/](data/large/):  Download separately, ~23 GB, use `pixi run download-data`

See [data/README.md](data/README.md) for more details.

## Testing

See [tests/](tests/).

```bash
pixi run tests
pixi run coverage
```

## TODO: 
- document benchmarking
- document example run
- document hyperparameter tuning