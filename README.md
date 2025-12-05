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

The default environment targets Python 3.11 and uses the Conda-Forge channel.
If you need additional tools, add them to `pixi.toml` so every contributor
inherits the same setup.

## Data Setup

- **Test data** (`data/test/`): Included, ~10 MB
- **Large data** (`data/large/`): Download separately, ~23 GB

### Quick Start
Tests only (data included):
```bash
pixi run tests
```

Examples (requires large data):
```bash
pixi run download-data  # One-time setup
python doc/examples/example_routing.py
```

See `data/README.md` for details.

## Example routing workflow

The repository contains a minimal script at `doc/examples/example_routing.py`
that demonstrates how to configure a journey, point the routing app to forcing
datasets, and run the optimisation pipeline. To run it:

1. Download the forcing datasets (currents, winds, waves) by running
   `pixi run download-data`. This will clone the data from
   `https://git.geomar.de/willi-rath/ship_routing_data` into `data/large/`.
2. Activate the Pixi environment (`pixi shell`) and execute
   `python doc/examples/example_routing.py`.
