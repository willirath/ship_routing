"""Population data structures for genetic algorithm optimization."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .routes import Route


@dataclass(frozen=True)
class PopulationMember:
    """Bundle of a route and its associated cost value.

    Attributes
    ----------
    route : Route
        The route.
    cost : float
        The cost associated with this route.
    """

    route: Route
    cost: float = np.nan


@dataclass
class Population:
    """Collection of population members for genetic algorithm.

    Attributes
    ----------
    members : list[PopulationMember]
        The population members.
    """

    members: list[PopulationMember]

    @property
    def size(self) -> int:
        """Return the number of members in the population."""
        return len(self.members)

    @classmethod
    def from_seed_member(
        cls,
        seed_member: PopulationMember,
        size: int,
    ) -> Population:
        """Create a population from a seed route.

        Creates a population of the specified size where all members
        are deep copies of the seed route with cost initialized to 0.0.

        Parameters
        ----------
        seed_member : PopulationMember
            The initial member to use as template.
        size : int
            Number of members in the population.

        Returns
        -------
        Population
            A new population with `size` members.
        """
        members = [seed_member for _ in range(size)]
        return cls(members=members)

    def sort(self) -> Population:
        """Return a new population sorted by cost (ascending).

        Returns
        -------
        Population
            A new population with members sorted by cost.
        """
        sorted_members = sorted(self.members, key=lambda m: m.cost)
        return Population(members=sorted_members)
