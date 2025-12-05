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

If you need additional tools, add them to `pixi.toml` so every contributor
inherits the same setup.

## Data Setup

- `data/test/`: Included, ~10 MB
- `data/large/`:  Download separately, ~23 GB, use `pixi run download-data`

See `data/README.md` for more details.

## Testing

See `tests/`.

```bash
pixi run tests
pixi run coverage
```

## TODO: 
- document benchmarking
- document example run
- document hyperparameter tuning