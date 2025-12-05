# Test Data

Small datasets for pytest suite.

## Contents
- `currents/` - Ocean currents (u, v)
- `waves/` - Significant wave height
- `winds/` - Wind velocity components
- `segmentation/` - CSV files for route tests

## Provenance
Derived from CMEMS with downsampling to ~1-4 MB per file.
See subdirectory READMEs for processing scripts.

## Usage in Code

Tests discover data via shared conftest.py:
```python
# In tests/conftest.py:
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_DIR = PROJECT_ROOT / "data" / "test"
```

Examples discover data locally:
```python
# In doc/examples/*.py:
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LARGE_DATA_DIR = PROJECT_ROOT / "data" / "large"
```
