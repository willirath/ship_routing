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

## Example routing workflow

The repository contains a minimal script at `doc/examples/example_routing.py`
that demonstrates how to configure a journey, point the routing app to forcing
datasets, and run the optimisation pipeline. To run it:

1. Clone or otherwise obtain the forcing datasets (currents, winds, waves) and
   place them under `doc/examples/data_large/`. For example, you can mirror
   `https://git.geomar.de/willi-rath/ship_routing_data` and symlink or copy the
   `*.zarr` directories so they match the filenames referenced in the script.
2. Activate the Pixi environment (`pixi shell`) and execute
   `python doc/examples/example_routing.py`.
