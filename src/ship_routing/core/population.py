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

    @classmethod
    def from_dict(cls, data: dict) -> PopulationMember:
        """Construct PopulationMember from dict.

        Parameters
        ----------
        data : dict
            Dictionary with 'route' and 'cost' keys.

        Returns
        -------
        PopulationMember
            A new PopulationMember instance.
        """
        return cls(
            route=Route.from_dict(data["route"]),
            cost=float(data["cost"]),
        )

    def to_dict(self) -> dict:
        """Return a JSON-friendly representation.

        Returns
        -------
        dict
            Dictionary with 'route' and 'cost' keys.
        """
        return {
            "route": self.route.to_dict(),
            "cost": float(self.cost),
        }

    @property
    def cost_valid(self):
        return not (
            np.isnan(self.cost)
            or np.isinf(self.cost)
            or np.isneginf(self.cost)
            or (self.cost <= 0)
        )


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
    def from_members(cls, members) -> Population:
        """Create population from sequence of members.

        Parameters
        ----------
        members : Sequence[PopulationMember]
            Members to include in population

        Returns
        -------
        Population
            New population containing the provided members
        """
        return cls(members=list(members))

    def add_member(self, member: PopulationMember) -> Population:
        """Return new population with additional member appended.

        Implements P ∪ {r} operator from pseudocode.

        Parameters
        ----------
        member : PopulationMember
            Member to add

        Returns
        -------
        Population
            New population with member appended
        """
        return Population(members=self.members + [member])

    def sort(self) -> Population:
        """Return a new population sorted by cost (ascending).

        Returns
        -------
        Population
            A new population with members sorted by cost.
        """
        sorted_members = sorted(self.members, key=lambda m: m.cost)
        return Population(members=sorted_members)

    @classmethod
    def from_dict(cls, data: dict) -> Population:
        """Construct Population from dict.

        Parameters
        ----------
        data : dict
            Dictionary with 'members' key containing list of member dicts.

        Returns
        -------
        Population
            A new Population instance.
        """
        members = [
            PopulationMember.from_dict(member_data) for member_data in data["members"]
        ]
        return cls(members=members)

    def to_dict(self) -> dict:
        """Return a JSON-friendly representation.

        Returns
        -------
        dict
            Dictionary with 'members' key containing list of member dicts.
        """
        return {
            "members": [member.to_dict() for member in self.members],
        }

    def remove_invalid(self):
        self.members = [m for m in self.members if m.cost_valid]
