# Ship Routing Data

## Directory Structure

- `test/` - Small test datasets (~10 MB, in version control)
  - Used by pytest test suite
  - Sourced from CMEMS with spatial/temporal downsampling

- `large/` - Full-resolution forcing data (~23 GB, NOT in version control)
  - Used by examples and production runs
  - Sourced from https://git.geomar.de/willi-rath/ship_routing_data.git

## Setup

**For tests only:** Test data is included in repo
```bash
pixi run tests
```

**For examples:** Download large data
```bash
pixi run download-data
pixi run check-data
```

## Data Sources

All data from Copernicus Marine Service (CMEMS):
- Currents: `cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m`
- Waves: `cmems_mod_glo_wav_my_0.2deg_PT3H-i`
- Winds: `cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H`

See subdirectory READMEs for details.
