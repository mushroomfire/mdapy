# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

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
    _run,
    _view,
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
    solid_bonds : numpy.ndarray or None
        (N,) int32 the number of solid-like bonds of each atom, with
        ``solid``; else None.
    solid : numpy.ndarray or None
        (N,) bool whether each atom is solid-like, with ``solid``; else None.
    """

    degrees: tuple[int, ...]
    q: np.ndarray
    average: bool
    solid_bonds: np.ndarray | None = None
    solid: np.ndarray | None = None

    def __getitem__(self, degree: int) -> np.ndarray:
        """(N,) the parameter of one degree."""
        try:
            return self.q[:, self.degrees.index(degree)]
        except ValueError:
            raise InvalidParameter(f"degree {degree} was not computed") from None

    def columns(self) -> dict[str, np.ndarray]:
        suffix = "_average" if self.average else ""
        columns = {f"q{d}{suffix}": self[d] for d in self.degrees}
        if self.solid is not None:
            columns["solid"] = self.solid
            columns["solid_bonds"] = self.solid_bonds
        return columns


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
    solid : bool, optional
        Also tell solid-like atoms from liquid-like ones by their bonds (ten
        Wolde, Ruiz-Montero and Frenkel). Default False.
    solid_degree : int, optional
        With solid, the degree l of the harmonics of the bonds. Default 6.
    bond_threshold : float, optional
        With solid, a bond is solid-like when the normalized product of the
        harmonics of its two atoms exceeds it. Default 0.7.
    min_solid_bonds : int, optional
        With solid, an atom is solid-like with at least so many solid-like
        bonds. Default 7.
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
        solid: bool = False,
        solid_degree: int = 6,
        bond_threshold: float = 0.7,
        min_solid_bonds: int = 7,
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
        if not 0 <= int(solid_degree) <= 20 or int(solid_degree) != solid_degree:
            raise InvalidParameter(
                f"solid_degree must be from 0 to 20, not {solid_degree!r}"
            )
        if not -1 <= bond_threshold <= 1:
            raise InvalidParameter(
                f"bond_threshold must be from -1 to 1, not {bond_threshold!r}"
            )
        if int(min_solid_bonds) != min_solid_bonds or min_solid_bonds < 0:
            raise InvalidParameter(
                f"min_solid_bonds must be an integer of at least 0, not {min_solid_bonds!r}"
            )
        self.solid = bool(solid)
        self.solid_degree = int(solid_degree)
        self.bond_threshold = float(bond_threshold)
        self.min_solid_bonds = int(min_solid_bonds)

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
            q_l of each atom; atoms without neighbors have 0. With solid,
            also the solid-like bonds and atoms.
        """
        k = self.num_neighbors
        nearest = (
            not self.voronoi
            and self.cutoff is None
            and not self.average
            and not self.solid
        )
        if nearest and neighbors is None and (system.box.pbc.any() or len(system) > k):
            # the nearest neighbors found as it goes, without a list
            box = system.box
            q = _run(
                _native.steinhardt_nearest,
                np.ascontiguousarray(system["positions"]),
                np.ascontiguousarray(box.matrix),
                np.ascontiguousarray(box.origin),
                np.ascontiguousarray(box.pbc),
                k,
                list(self.degrees),
                progress,
                _native_token(cancel),
            )
            return SteinhardtResult(self.degrees, q, self.average)
        if self.voronoi:
            nl = _voronoi_given_or_built(system, neighbors)
            count, cutoff = 0, 0.0
        elif self.cutoff is not None:
            nl = _given_or_built(system, self.cutoff, neighbors)
            count, cutoff = 0, self.cutoff
        else:
            nl = _nearest_given_or_built(system, self.num_neighbors, neighbors)
            count, cutoff = self.num_neighbors, 0.0
        weights = nl.areas if self.weighted else None
        with nl._reading():
            q = _native.steinhardt(
                *_view(system, nl),
                count,
                cutoff,
                weights,
                list(self.degrees),
                self.average,
                progress,
                _native_token(cancel),
            )
            if not self.solid:
                return SteinhardtResult(self.degrees, q, self.average)
            bonds = _native.solid_bonds(
                *_view(system, nl),
                count,
                cutoff,
                weights,
                self.solid_degree,
                self.bond_threshold,
                progress,
                _native_token(cancel),
            )
        return SteinhardtResult(
            self.degrees, q, self.average, bonds, bonds >= self.min_solid_bonds
        )

    def __repr__(self) -> str:
        if self.voronoi:
            reach = f"voronoi=True, weighted={self.weighted}"
        elif self.cutoff is not None:
            reach = f"cutoff={self.cutoff!r}"
        else:
            reach = f"num_neighbors={self.num_neighbors}"
        solid = (
            f", solid=True, solid_degree={self.solid_degree}, "
            f"bond_threshold={self.bond_threshold!r}, min_solid_bonds={self.min_solid_bonds}"
            if self.solid
            else ""
        )
        return (
            f"SteinhardtParameters(degrees={self.degrees}, {reach}, "
            f"average={self.average}{solid})"
        )
