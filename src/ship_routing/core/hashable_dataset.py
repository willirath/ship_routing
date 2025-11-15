"""Hashable xarray Dataset wrapper used for caching."""

from __future__ import annotations

import xarray as xr


class HashableDataset(xr.Dataset):
    __slots__ = []

    def __hash__(self) -> int:
        # Note: this assumes the dataset identity is sufficient for caching purposes.
        return hash(id(self))


def make_hashable(ds: xr.Dataset) -> HashableDataset:
    """Return the dataset wrapped as a HashableDataset."""
    return HashableDataset(ds)
