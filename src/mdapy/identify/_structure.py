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
from mdapy.neighbor._neighbor_list import _given_or_built


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
        if self.cutoff is not None:
            nl = _given_or_built(system, self.cutoff, neighbors).within(self.cutoff)
        elif neighbors is not None:
            nl = _given_or_built(system, 0.0, neighbors).sorted()._first(14)
        else:
            # without periodic images, small systems have fewer neighbors
            k = 14 if system.box.pbc.any() else min(14, len(system) - 1)
            nl = NeighborList.nearest(system, k) if k > 0 else None
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


class IdentifyDiamond:
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
        if neighbors is None:
            nl = NeighborList.nearest(system, 4)
        else:
            nl = _given_or_built(system, 0.0, neighbors)
            if len(system) and nl.counts.min() < 4:
                raise InvalidParameter(
                    f"an atom has {nl.counts.min()} neighbors in the list, fewer than 4"
                )
            nl = nl.sorted()._first(4)
        n = len(system)
        structure = _native.identify_diamond(
            np.ascontiguousarray(nl.j.reshape(n, 4)),
            np.ascontiguousarray(nl.vectors).reshape(n, 4, 3),
            progress,
            _native_token(cancel),
        )
        return StructureResult(structure, self.NAMES, self.COLORS)

    def __repr__(self) -> str:
        return "IdentifyDiamond()"


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
        if neighbors is None:
            nl = NeighborList.nearest(system, k)
        else:
            nl = _given_or_built(system, 0.0, neighbors)
            if len(system) and nl.counts.min() < k:
                raise InvalidParameter(
                    f"an atom has {nl.counts.min()} neighbors in the list, fewer than {k}"
                )
            nl = nl.sorted()._first(k)
        vectors = np.ascontiguousarray(nl.vectors).reshape(len(system), k, 3)
        csp = _native.centro_symmetry(vectors, progress, _native_token(cancel))
        return CentroSymmetryResult(csp)

    def __repr__(self) -> str:
        return f"CentroSymmetryParameter(num_neighbors={self.num_neighbors})"
