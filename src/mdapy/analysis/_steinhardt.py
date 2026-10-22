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
from mdapy.neighbor._neighbor_list import (
    _given_or_built,
    _nearest_given_or_built,
    _voronoi_given_or_built,
)


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
        unless ``cutoff`` or ``voronoi`` is given.
    cutoff : float, optional
        The neighbors of an atom are those closer than cutoff, in Å.
    voronoi : bool, optional
        The neighbors of an atom are those that share a face of its Voronoi
        cell. Default False.
    weighted : bool, optional
        With voronoi, weight each neighbor by the area of its face. Default
        False.
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
        voronoi: bool = False,
        weighted: bool = False,
        average: bool = False,
    ):
        self.degrees = tuple(int(d) for d in degrees)
        if not self.degrees or any(not 0 <= d <= 20 for d in self.degrees):
            raise InvalidParameter(f"degrees must be from 0 to 20, not {degrees!r}")
        if (num_neighbors is not None) + (cutoff is not None) + bool(voronoi) > 1:
            raise InvalidParameter("give one of num_neighbors, cutoff and voronoi")
        if weighted and not voronoi:
            raise InvalidParameter("weighted needs voronoi")
        if cutoff is not None and not cutoff > 0:
            raise InvalidParameter(f"cutoff must be positive, not {cutoff!r}")
        if cutoff is None and not voronoi:
            num_neighbors = 12 if num_neighbors is None else int(num_neighbors)
            if num_neighbors < 1:
                raise InvalidParameter("num_neighbors must be at least 1")
        self.num_neighbors = num_neighbors
        self.cutoff = None if cutoff is None else float(cutoff)
        self.voronoi = bool(voronoi)
        self.weighted = bool(weighted)
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
            Neighbors of this system: within at least ``cutoff``, at least
            ``num_neighbors`` per atom, or, with ``voronoi``, Voronoi
            neighbors.

        Returns
        -------
        SteinhardtResult
            q_l of each atom; atoms without neighbors have 0.
        """
        if self.voronoi:
            nl = _voronoi_given_or_built(system, neighbors)
        elif self.cutoff is not None:
            nl = _given_or_built(system, self.cutoff, neighbors).within(self.cutoff)
        else:
            nl = _nearest_given_or_built(system, self.num_neighbors, neighbors)
        q = _native.steinhardt(
            nl.offsets,
            nl.j,
            np.ascontiguousarray(nl.vectors),
            nl.areas if self.weighted else None,
            list(self.degrees),
            self.average,
            progress,
            _native_token(cancel),
        )
        return SteinhardtResult(self.degrees, q, self.average)

    def __repr__(self) -> str:
        if self.voronoi:
            reach = f"voronoi=True, weighted={self.weighted}"
        elif self.cutoff is not None:
            reach = f"cutoff={self.cutoff!r}"
        else:
            reach = f"num_neighbors={self.num_neighbors}"
        return f"SteinhardtParameters(degrees={self.degrees}, {reach}, average={self.average})"
