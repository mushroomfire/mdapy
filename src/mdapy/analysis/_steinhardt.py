# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Steinhardt bond-orientational order parameters."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy._core import analysis as _native
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built


@dataclass(frozen=True)
class SteinhardtResult(Result):
    """Order parameters of each atom.

    Attributes
    ----------
    degrees : tuple of int
        The degrees l.
    q : numpy.ndarray
        (N, len(degrees)) the parameter q_l of each atom and degree.
    average : bool
        Whether the parameters are averaged over the neighbors.
    """

    degrees: tuple[int, ...]
    q: np.ndarray
    average: bool

    def __getitem__(self, degree: int) -> np.ndarray:
        """(N,) the parameter of one degree."""
        try:
            return self.q[:, self.degrees.index(degree)]
        except ValueError:
            raise InvalidParameter(f"degree {degree} was not computed") from None

    def columns(self) -> dict[str, np.ndarray]:
        suffix = "_average" if self.average else ""
        return {f"q{d}{suffix}": self[d] for d in self.degrees}


class SteinhardtParameters:
    """Steinhardt bond-orientational order parameters q_l.

    Parameters
    ----------
    degrees : sequence of int, optional
        The degrees l, from 0 to 20. Default (4, 6).
    num_neighbors : int, optional
        The neighbors of an atom are its num_neighbors nearest; default 12
        unless ``cutoff`` is given.
    cutoff : float, optional
        The neighbors of an atom are those closer than cutoff, in Å.
    average : bool, optional
        Average the harmonics of each atom with those of its neighbors first
        (Lechner and Dellago). Default False.
    """

    def __init__(
        self,
        degrees: Sequence[int] = (4, 6),
        *,
        num_neighbors: int | None = None,
        cutoff: float | None = None,
        average: bool = False,
    ):
        self.degrees = tuple(int(d) for d in degrees)
        if not self.degrees or any(not 0 <= d <= 20 for d in self.degrees):
            raise InvalidParameter(f"degrees must be from 0 to 20, not {degrees!r}")
        if num_neighbors is not None and cutoff is not None:
            raise InvalidParameter("give num_neighbors or cutoff, not both")
        if cutoff is not None and not cutoff > 0:
            raise InvalidParameter(f"cutoff must be positive, not {cutoff!r}")
        if cutoff is None:
            num_neighbors = 12 if num_neighbors is None else int(num_neighbors)
            if num_neighbors < 1:
                raise InvalidParameter("num_neighbors must be at least 1")
        self.num_neighbors = num_neighbors
        self.cutoff = None if cutoff is None else float(cutoff)
        self.average = bool(average)

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> SteinhardtResult:
        """The parameters of each atom.

        Parameters
        ----------
        system : System
            The atoms.
        neighbors : NeighborList, optional
            Neighbors of this system: within at least ``cutoff``, or at least
            ``num_neighbors`` per atom.

        Returns
        -------
        SteinhardtResult
            q_l of each atom; atoms without neighbors have 0.
        """
        if self.cutoff is not None:
            nl = _given_or_built(system, self.cutoff, neighbors).within(self.cutoff)
        elif neighbors is None:
            nl = NeighborList.nearest(system, self.num_neighbors)
        else:
            nl = _given_or_built(system, 0.0, neighbors)
            if len(system) and nl.counts.min() < self.num_neighbors:
                raise InvalidParameter(
                    f"an atom has {nl.counts.min()} neighbors in the list, "
                    f"fewer than {self.num_neighbors}"
                )
            nl = nl.sorted()._first(self.num_neighbors)
        q = _native.steinhardt(
            nl.offsets,
            nl.j,
            np.ascontiguousarray(nl.vectors),
            list(self.degrees),
            self.average,
            progress,
            _native_token(cancel),
        )
        return SteinhardtResult(self.degrees, q, self.average)

    def __repr__(self) -> str:
        reach = (
            f"cutoff={self.cutoff!r}"
            if self.cutoff is not None
            else f"num_neighbors={self.num_neighbors}"
        )
        return f"SteinhardtParameters(degrees={self.degrees}, {reach}, average={self.average})"
