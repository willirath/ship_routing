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
