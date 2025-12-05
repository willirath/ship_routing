# Ship Routing Data

See subdirectory READMEs for details.

## Directory Structure

- `test/` - Small test datasets (~10 MB, in version control)
  - Used by test suite and small examples
  - Sourced from CMEMS with spatial/temporal downsampling

- `large/` - Full-resolution forcing data (~23 GB, NOT in version control)
  - Used by larger examples, hyperparameter tuning, benchmarking, and in scientific analysis
  - Sourced from https://git.geomar.de/willi-rath/ship_routing_data.git

## Setup

Download large data using:
```bash
pixi run download-data
pixi run check-data
```