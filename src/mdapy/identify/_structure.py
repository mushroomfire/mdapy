# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Common neighbor analysis and the centro-symmetry parameter."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy._core import identify as _native
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import (
    _given_or_built,
    _nearest_given_or_built,
    _run,
)


def _nearest(
    kernel: Any, system: System, k: int, progress: Any, cancel: Any
) -> np.ndarray:
    """A kernel of the k nearest neighbors of every atom, found as it goes,
    without a list."""
    box = system.box
    return _run(
        kernel,
        np.ascontiguousarray(system.positions),
        np.ascontiguousarray(box.matrix),
        np.ascontiguousarray(box.origin),
        np.ascontiguousarray(box.pbc),
        k,
        progress,
        _native_token(cancel),
    )


@dataclass(frozen=True)
class StructureResult(Result):
    """Structure type of each atom.

    Attributes
    ----------
    structure : numpy.ndarray
        (N,) int8 type of each atom, an index into ``names``.
    names : tuple of str
        Name of each type.
    colors : tuple of tuple of float
        Default RGB color of each type, from 0 to 1.
    """

    structure: np.ndarray
    names: tuple[str, ...]
    colors: tuple[tuple[float, float, float], ...]

    def counts(self) -> dict[str, int]:
        """Number of atoms of each type, by name."""
        found = np.bincount(self.structure, minlength=len(self.names))
        return {name: int(c) for name, c in zip(self.names, found)}

    def columns(self) -> dict[str, np.ndarray]:
        return {"structure": self.structure}


class CommonNeighborAnalysis:
    """Common neighbor analysis: fcc, hcp, bcc and icosahedral atoms.

    Parameters
    ----------
    cutoff : float, optional
        Cutoff of the neighbors and of the bonds between them, in Å. Without
        it, each atom has its own cutoff from its nearest neighbors
        (adaptive common neighbor analysis).
    """

    #: Names of the structure types.
    NAMES = ("other", "fcc", "hcp", "bcc", "ico")
    #: Their default colors.
    COLORS = (
        (0.95, 0.95, 0.95),
        (0.4, 1.0, 0.4),
        (1.0, 0.4, 0.4),
        (0.4, 0.4, 1.0),
        (0.95, 0.8, 0.2),
    )

    def __init__(self, cutoff: float | None = None):
        if cutoff is not None and not cutoff > 0:
            raise InvalidParameter(f"cutoff must be positive, not {cutoff!r}")
        self.cutoff = None if cutoff is None else float(cutoff)

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> StructureResult:
        """The structure type of each atom.

        Parameters
        ----------
        system : System
            The atoms.
        neighbors : NeighborList, optional
            Neighbors of this system: with a cutoff, within at least the
            cutoff; without, at least the 14 nearest of each atom.

        Returns
        -------
        StructureResult
            Types ``other``, ``fcc``, ``hcp``, ``bcc`` and ``ico``.
        """
        # without periodic images, small systems have fewer neighbors
        k = 14 if system.box.pbc.any() else min(14, len(system) - 1)
        if self.cutoff is not None and neighbors is None:
            # the neighbors found on the grid, without a list
            box = system.box
            structure = _native.common_neighbor_analysis_search(
                np.ascontiguousarray(system.positions),
                np.ascontiguousarray(box.matrix),
                np.ascontiguousarray(box.origin),
                np.ascontiguousarray(box.pbc),
                self.cutoff,
                progress,
                _native_token(cancel),
            )
            return StructureResult(structure, self.NAMES, self.COLORS)
        if self.cutoff is None and neighbors is None and k > 0:
            # the nearest neighbors found as it goes, without a list
            structure = _nearest(
                _native.common_neighbor_analysis_nearest, system, k, progress, cancel
            )
            return StructureResult(structure, self.NAMES, self.COLORS)
        if self.cutoff is not None:
            nl = _given_or_built(system, self.cutoff, neighbors).within(self.cutoff)
        elif k > 0:
            nl = _nearest_given_or_built(system, k, neighbors)
        else:
            nl = None
        if nl is None:
            structure = np.zeros(len(system), dtype=np.int8)
        else:
            structure = _native.common_neighbor_analysis(
                nl.offsets,
                np.ascontiguousarray(nl.vectors),
                self.cutoff or 0.0,
                progress,
                _native_token(cancel),
            )
        return StructureResult(structure, self.NAMES, self.COLORS)

    def __repr__(self) -> str:
        return f"CommonNeighborAnalysis(cutoff={self.cutoff!r})"


class AcklandJonesAnalysis:
    """Ackland-Jones bond-angle analysis: fcc, hcp, bcc and icosahedral atoms."""

    #: Names of the structure types.
    NAMES = CommonNeighborAnalysis.NAMES
    #: Their default colors.
    COLORS = CommonNeighborAnalysis.COLORS

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> StructureResult:
        """The structure type of each atom.

        Parameters
        ----------
        system : System
            The atoms.
        neighbors : NeighborList, optional
            Neighbors of this system with at least the 14 nearest of each
            atom.

        Returns
        -------
        StructureResult
            Types ``other``, ``fcc``, ``hcp``, ``bcc`` and ``ico``.

        Raises
        ------
        InvalidParameter
            If an atom has fewer than 14 neighbors in the given list.
        """
        # without periodic images, small systems have fewer neighbors
        k = 14 if system.box.pbc.any() else min(14, len(system) - 1)
        if k < 6:
            return StructureResult(
                np.zeros(len(system), dtype=np.int8), self.NAMES, self.COLORS
            )
        if neighbors is None:
            # the nearest neighbors found as it goes, without a list
            structure = _nearest(
                _native.ackland_jones_nearest, system, k, progress, cancel
            )
            return StructureResult(structure, self.NAMES, self.COLORS)
        nl = _nearest_given_or_built(system, k, neighbors)
        structure = _native.ackland_jones(
            nl.offsets,
            np.ascontiguousarray(nl.vectors),
            progress,
            _native_token(cancel),
        )
        return StructureResult(structure, self.NAMES, self.COLORS)

    def __repr__(self) -> str:
        return "AcklandJonesAnalysis()"


class DiamondStructure:
    """Cubic and hexagonal diamond structures, from the arrangement of the
    second neighbors of each atom."""

    #: Names of the structure types.
    NAMES = ("other", "cubic_diamond", "hexagonal_diamond")
    #: Their default colors.
    COLORS = ((0.95, 0.95, 0.95), (0.075, 0.24, 0.94), (1.0, 0.6, 0.2))

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> StructureResult:
        """The structure type of each atom.

        Parameters
        ----------
        system : System
            The atoms.
        neighbors : NeighborList, optional
            Neighbors of this system with at least the 4 nearest of each atom.

        Returns
        -------
        StructureResult
            Types ``other``, ``cubic_diamond`` and ``hexagonal_diamond``.

        Raises
        ------
        InvalidParameter
            If an atom has fewer than 4 neighbors.
        """
        nl = _nearest_given_or_built(system, 4, neighbors)
        n = len(system)
        structure = _native.identify_diamond(
            np.ascontiguousarray(nl.j.reshape(n, 4)),
            np.ascontiguousarray(nl.vectors).reshape(n, 4, 3),
            progress,
            _native_token(cancel),
        )
        return StructureResult(structure, self.NAMES, self.COLORS)

    def __repr__(self) -> str:
        return "DiamondStructure()"


class ChillPlus:
    """CHILL+: ice, clathrate hydrate and liquid water, from the correlation
    of the bond order of neighboring molecules.

    Parameters
    ----------
    cutoff : float, optional
        Neighbors are the molecules closer than cutoff, in Å. Default 3.5,
        for the oxygen atoms of water.
    """

    #: Names of the structure types.
    NAMES = (
        "other",
        "hexagonal_ice",
        "cubic_ice",
        "interfacial_ice",
        "hydrate",
        "interfacial_hydrate",
    )
    #: Their default colors.
    COLORS = (
        (0.95, 0.95, 0.95),
        (0.0, 0.8, 1.0),
        (1.0, 0.6, 0.0),
        (0.6, 0.2, 0.6),
        (0.2, 0.6, 0.2),
        (0.6, 0.8, 0.2),
    )

    def __init__(self, cutoff: float = 3.5):
        if not cutoff > 0:
            raise InvalidParameter(f"cutoff must be positive, not {cutoff!r}")
        self.cutoff = float(cutoff)

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> StructureResult:
        """The type of each molecule.

        Parameters
        ----------
        system : System
            One atom per molecule, for water its oxygen atoms.
        neighbors : NeighborList, optional
            A neighbor list of this system reaching at least ``cutoff``.

        Returns
        -------
        StructureResult
            Types ``other``, ``hexagonal_ice``, ``cubic_ice``,
            ``interfacial_ice``, ``hydrate`` and ``interfacial_hydrate``.
        """
        nl = _given_or_built(system, self.cutoff, neighbors).within(self.cutoff)
        structure = _native.chill_plus(
            nl.offsets,
            nl.j,
            np.ascontiguousarray(nl.vectors),
            progress,
            _native_token(cancel),
        )
        return StructureResult(structure, self.NAMES, self.COLORS)

    def __repr__(self) -> str:
        return f"ChillPlus(cutoff={self.cutoff!r})"


@dataclass(frozen=True)
class CentroSymmetryResult(Result):
    """The centro-symmetry parameter of each atom.

    Attributes
    ----------
    csp : numpy.ndarray
        (N,) float64 parameter in Å².
    """

    csp: np.ndarray

    def columns(self) -> dict[str, np.ndarray]:
        return {"csp": self.csp}


class CentroSymmetryParameter:
    """The centro-symmetry parameter: zero for an atom whose neighbors come
    in opposite pairs.

    Parameters
    ----------
    num_neighbors : int, optional
        Number of nearest neighbors, even: 12 for fcc, 8 for bcc. Default 12.
    """

    def __init__(self, num_neighbors: int = 12):
        if num_neighbors < 2 or num_neighbors % 2:
            raise InvalidParameter(
                f"num_neighbors must be even and at least 2, not {num_neighbors!r}"
            )
        self.num_neighbors = int(num_neighbors)

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> CentroSymmetryResult:
        """The parameter of each atom.

        Parameters
        ----------
        system : System
            The atoms.
        neighbors : NeighborList, optional
            Neighbors of this system with at least ``num_neighbors`` per atom.

        Returns
        -------
        CentroSymmetryResult
            The parameter of each atom.

        Raises
        ------
        InvalidParameter
            If an atom has fewer neighbors.
        """
        k = self.num_neighbors
        if neighbors is None and k <= 14 and (system.box.pbc.any() or len(system) > k):
            # the nearest neighbors found as it goes, without a list
            csp = _nearest(_native.centro_symmetry_nearest, system, k, progress, cancel)
            return CentroSymmetryResult(csp)
        nl = _nearest_given_or_built(system, k, neighbors)
        vectors = np.ascontiguousarray(nl.vectors).reshape(len(system), k, 3)
        csp = _native.centro_symmetry(vectors, progress, _native_token(cancel))
        return CentroSymmetryResult(csp)

    def __repr__(self) -> str:
        return f"CentroSymmetryParameter(num_neighbors={self.num_neighbors})"


@dataclass(frozen=True)
class CommonNeighborParameterResult(Result):
    """The common neighbor parameter of each atom.

    Attributes
    ----------
    cnp : numpy.ndarray
        (N,) float64 parameter in Å²; NaN for an atom without neighbors.
    """

    cnp: np.ndarray

    def columns(self) -> dict[str, np.ndarray]:
        return {"cnp": self.cnp}


class CommonNeighborParameter:
    """The common neighbor parameter: zero for an atom whose neighbors and
    their common neighbors are arranged symmetrically, as in fcc and bcc.

    Parameters
    ----------
    cutoff : float
        Neighbors and common neighbors are closer than cutoff, in Å.
    """

    def __init__(self, cutoff: float):
        if not cutoff > 0:
            raise InvalidParameter(f"cutoff must be positive, not {cutoff!r}")
        self.cutoff = float(cutoff)

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> CommonNeighborParameterResult:
        """The parameter of each atom.

        Parameters
        ----------
        system : System
            The atoms.
        neighbors : NeighborList, optional
            A neighbor list of this system with at least the cutoff.

        Returns
        -------
        CommonNeighborParameterResult
            The parameter of each atom.
        """
        if neighbors is None:
            # the neighbors found on the grid, without a list
            box = system.box
            cnp = _native.common_neighbor_parameter_search(
                np.ascontiguousarray(system.positions),
                np.ascontiguousarray(box.matrix),
                np.ascontiguousarray(box.origin),
                np.ascontiguousarray(box.pbc),
                self.cutoff,
                progress,
                _native_token(cancel),
            )
        else:
            nl = _given_or_built(system, self.cutoff, neighbors).within(self.cutoff)
            cnp = _native.common_neighbor_parameter(
                nl.offsets,
                np.ascontiguousarray(nl.vectors),
                self.cutoff,
                progress,
                _native_token(cancel),
            )
        return CommonNeighborParameterResult(cnp)

    def __repr__(self) -> str:
        return f"CommonNeighborParameter(cutoff={self.cutoff!r})"
