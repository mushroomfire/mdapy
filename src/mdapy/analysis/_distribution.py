# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Radial distribution functions and coordination numbers."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy._core import analysis as _native
from mdapy.analysis._bonds import _Bonds, _elements
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _given_or_built


@dataclass(frozen=True)
class RadialDistributionResult(Result):
    """A radial distribution function and its partial functions.

    Attributes
    ----------
    r : numpy.ndarray
        (bins,) centers of the bins in Å.
    edges : numpy.ndarray
        (bins + 1,) edges of the bins in Å.
    g : numpy.ndarray
        (bins,) the radial distribution function.
    coordination : numpy.ndarray
        (bins,) mean number of neighbors closer than the upper edge of each bin.
    partial : dict
        (bins,) partial function of each pair of elements ``(a, b)``, ``a``
        not after ``b`` by atomic number; empty without an element column.
    """

    r: np.ndarray
    edges: np.ndarray
    g: np.ndarray
    coordination: np.ndarray
    partial: dict[tuple[str, str], np.ndarray]

    def pair(self, a: str, b: str) -> np.ndarray:
        """The partial function of elements a and b, in either order."""
        if (a, b) in self.partial:
            return self.partial[a, b]
        if (b, a) in self.partial:
            return self.partial[b, a]
        raise InvalidParameter(f"no partial function for {(a, b)}")


class RadialDistributionFunction:
    """The radial distribution function g(r) and its partial functions.

    Parameters
    ----------
    r_max : float
        Largest distance in Å.
    bins : int, optional
        Number of bins of equal width. Default 200.
    """

    def __init__(self, r_max: float, bins: int = 200):
        if not r_max > 0:
            raise InvalidParameter(f"r_max must be positive, not {r_max!r}")
        if bins < 1:
            raise InvalidParameter(f"bins must be at least 1, not {bins!r}")
        self.r_max = float(r_max)
        self.bins = int(bins)

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> RadialDistributionResult:
        """The functions of a system.

        Parameters
        ----------
        system : System
            The atoms; partial functions need an ``element`` column.
        neighbors : NeighborList, optional
            A neighbor list of this system with a cutoff of at least ``r_max``.

        Returns
        -------
        RadialDistributionResult
            The functions, normalized by the number density of the box.
        """
        n, bins = len(system), self.bins
        found = _elements(system)
        names, index = found if found is not None else ((), np.zeros(n, dtype=np.int64))
        types = max(len(names), 1)
        if neighbors is None:
            # counted during the search, without storing the pairs
            box = system.box
            table = _native.pair_histogram(
                np.ascontiguousarray(system.positions),
                np.ascontiguousarray(box.matrix),
                np.ascontiguousarray(box.origin),
                np.ascontiguousarray(box.pbc),
                self.r_max,
                bins,
                index.astype(np.int32),
                types,
                progress,
                _native_token(cancel),
            )
        else:
            nl = _given_or_built(system, self.r_max, neighbors)
            with nl._reading():
                table = _native.pair_histogram_list(
                    nl.offsets,
                    nl.j,
                    nl.distances,
                    self.r_max,
                    bins,
                    index.astype(np.int32),
                    types,
                    progress,
                    _native_token(cancel),
                )
        edges = np.linspace(0.0, self.r_max, bins + 1)
        shells = 4.0 / 3.0 * np.pi * np.diff(edges**3)
        volume = system.box.volume
        counts = table.sum(axis=(0, 1)).astype(np.float64)
        g = counts * volume / (n * n * shells) if n else np.zeros(bins)
        coordination = np.cumsum(counts) / n if n else np.zeros(bins)
        partial = {}
        if found is not None and n:
            sizes = np.bincount(index, minlength=types)
            for a in range(types):
                for b in range(a, types):
                    partial[names[a], names[b]] = (
                        table[a, b] * volume / (sizes[a] * sizes[b] * shells)
                    )
        return RadialDistributionResult(
            r=0.5 * (edges[1:] + edges[:-1]),
            edges=edges,
            g=g,
            coordination=coordination,
            partial=partial,
        )

    def __repr__(self) -> str:
        return f"RadialDistributionFunction(r_max={self.r_max!r}, bins={self.bins})"


@dataclass(frozen=True)
class CoordinationResult(Result):
    """Number of neighbors of each atom.

    Attributes
    ----------
    coordination : numpy.ndarray
        (N,) int64 number of neighbors.
    partial : dict
        (N,) int64 number of neighbors of each element, by element; empty
        without an element column.
    """

    coordination: np.ndarray
    partial: dict[str, np.ndarray]

    def columns(self) -> dict[str, np.ndarray]:
        columns = {"coordination": self.coordination}
        for element, counts in self.partial.items():
            columns[f"coordination_{element}"] = counts
        return columns


class CoordinationNumber:
    """The number of neighbors of each atom within a cutoff.

    Parameters
    ----------
    cutoff : float, optional
        Cutoff in Å for every pair.
    cutoffs : dict, optional
        Cutoff in Å for each pair of elements ``(a, b)``, in either order;
        pairs not given are never neighbors. Give ``cutoff`` or ``cutoffs``.
    """

    def __init__(
        self,
        cutoff: float | None = None,
        *,
        cutoffs: Mapping[tuple[str, str], float] | None = None,
    ):
        self.__bonds = _Bonds(cutoff, cutoffs)
        self.cutoff, self.cutoffs = self.__bonds.cutoff, self.__bonds.cutoffs

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> CoordinationResult:
        """The counts of a system.

        Parameters
        ----------
        system : System
            The atoms; per-element cutoffs and counts need an ``element`` column.
        neighbors : NeighborList, optional
            A neighbor list of this system with at least the largest cutoff.

        Returns
        -------
        CoordinationResult
            The counts of each atom.
        """
        table, found = self.__bonds.counts(system, neighbors, progress, cancel)
        partial = {}
        if found is not None:
            partial = {name: table[:, t].copy() for t, name in enumerate(found[0])}
        return CoordinationResult(table.sum(axis=1), partial)

    def __repr__(self) -> str:
        return f"CoordinationNumber({self.__bonds})"


@dataclass(frozen=True)
class StructureFactorResult(Result):
    """A static structure factor and its partial factors.

    Attributes
    ----------
    q : numpy.ndarray
        (bins,) centers of the bins of |q| in 1/Å.
    edges : numpy.ndarray
        (bins + 1,) edges of the bins in 1/Å.
    s : numpy.ndarray
        (bins,) the structure factor; with the direct method, averaged over
        the vectors of each bin and NaN in bins without vectors.
    vectors : numpy.ndarray or None
        (bins,) int64 number of vectors in each bin, one of each pair q, -q,
        with the direct method; None with the Debye method.
    partial : dict
        (bins,) partial factor S_ab of each pair of elements ``(a, b)``,
        ``a`` not after ``b`` by atomic number, normalized so that ``s`` is
        the sum of S_aa and twice S_ab; empty without an element column.
    faber_ziman : dict
        (bins,) Faber-Ziman partial factor A_ab of each pair, keyed as
        ``partial``.
    method : str
        ``"direct"`` or ``"debye"``.
    """

    q: np.ndarray
    edges: np.ndarray
    s: np.ndarray
    vectors: np.ndarray | None
    partial: dict[tuple[str, str], np.ndarray]
    faber_ziman: dict[tuple[str, str], np.ndarray]
    method: str

    def pair(self, a: str, b: str) -> np.ndarray:
        """The partial factor S_ab of elements a and b, in either order."""
        if (a, b) in self.partial:
            return self.partial[a, b]
        if (b, a) in self.partial:
            return self.partial[b, a]
        raise InvalidParameter(f"no partial factor for {(a, b)}")


#: Ways to compute the structure factor: the sum over the wave vectors of
#: the box, or the Fourier transform of the radial distribution function.
METHODS = ("direct", "debye")

# Width of the bins of the radial distribution function of the Debye method,
# in Å.
_DEBYE_DR = 0.01


class StructureFactor:
    """The static structure factor S(q) of a periodic system and its partial
    factors.

    Parameters
    ----------
    q_max : float
        Largest |q| in 1/Å.
    bins : int, optional
        Number of bins of |q| of equal width. Default 200.
    method : {"direct", "debye"}, optional
        Sum over the wave vectors that fit the box (default), or transform
        of the radial distribution function up to ``r_max``.
    r_max : float, optional
        Largest distance of the Debye method in Å; default half the length
        of the longest lattice vector.
    """

    def __init__(
        self,
        q_max: float,
        bins: int = 200,
        *,
        method: str = "direct",
        r_max: float | None = None,
    ):
        if not q_max > 0:
            raise InvalidParameter(f"q_max must be positive, not {q_max!r}")
        if bins < 1:
            raise InvalidParameter(f"bins must be at least 1, not {bins!r}")
        if method not in METHODS:
            raise InvalidParameter(f"method must be one of {METHODS}, not {method!r}")
        if r_max is not None and (method != "debye" or not r_max > 0):
            raise InvalidParameter(
                f"r_max must be positive and is used by the Debye method only, not {r_max!r}"
            )
        self.q_max = float(q_max)
        self.bins = int(bins)
        self.method = method
        self.r_max = None if r_max is None else float(r_max)

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> StructureFactorResult:
        """The structure factor of a system.

        Parameters
        ----------
        system : System
            The atoms in a box periodic in every direction; partial factors
            need an ``element`` column.
        neighbors : NeighborList, optional
            With the Debye method, a neighbor list of this system reaching
            ``r_max``.

        Returns
        -------
        StructureFactorResult
            The factor averaged over shells of |q|.
        """
        if not system.box.pbc.all():
            raise InvalidParameter(
                "the structure factor needs a box periodic in every direction"
            )
        if neighbors is not None and self.method != "debye":
            raise InvalidParameter("the direct method takes no neighbor list")
        n, bins = len(system), self.bins
        found = _elements(system)
        names, index = found if found is not None else ((), np.zeros(n, dtype=np.int64))
        types = max(len(names), 1)
        sizes = np.bincount(index, minlength=types)
        fractions = sizes / n if n else np.zeros(types)
        edges = np.linspace(0.0, self.q_max, bins + 1)
        q = 0.5 * (edges[1:] + edges[:-1])
        if self.method == "direct":
            sums, vectors = _native.structure_factor(
                np.ascontiguousarray(system.positions),
                np.ascontiguousarray(system.box.matrix),
                index.astype(np.int32),
                types,
                self.q_max,
                bins,
                0,
                progress,
                _native_token(cancel),
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                s = sums.sum(axis=(0, 1)) / n / vectors if n else np.full(bins, np.nan)
                pairs = {
                    (a, b): sums[a, b] / n / vectors
                    for a in range(types)
                    for b in range(a, types)
                }
        else:
            vectors = None
            r_max = self.r_max or 0.5 * float(
                np.linalg.norm(system.box.matrix, axis=1).max()
            )
            rdf = RadialDistributionFunction(
                r_max, max(1, int(np.ceil(r_max / _DEBYE_DR)))
            ).compute(system, neighbors=neighbors, progress=progress, cancel=cancel)
            r, dr = rdf.r, rdf.edges[1] - rdf.edges[0]
            # 4 pi rho r sin(qr) / q w(r) dr, w a window that goes to zero at
            # r_max against the ripples of the cutoff
            density = n / system.box.volume
            kernel = (
                4.0 * np.pi * density * dr
                * np.sin(np.outer(q, r)) * (r * np.sinc(r / r_max)) / q[:, None]
            )  # fmt: skip
            s = 1.0 + kernel @ (rdf.g - 1.0) if n else np.full(bins, np.nan)
            pairs = {}
            if found is not None and n:
                for a in range(types):
                    for b in range(a, types):
                        fz = 1.0 + kernel @ (rdf.pair(names[a], names[b]) - 1.0)
                        pairs[a, b] = fractions[a] * fractions[b] * (fz - 1.0) + (
                            fractions[a] if a == b else 0.0
                        )
        partial, faber_ziman = {}, {}
        if found is not None and n:
            for (a, b), value in pairs.items():
                partial[names[a], names[b]] = value
                faber_ziman[names[a], names[b]] = (
                    value - (fractions[a] if a == b else 0.0)
                ) / (fractions[a] * fractions[b]) + 1.0
        return StructureFactorResult(
            q=q,
            edges=edges,
            s=s,
            vectors=vectors,
            partial=partial,
            faber_ziman=faber_ziman,
            method=self.method,
        )

    def __repr__(self) -> str:
        extra = f", r_max={self.r_max!r}" if self.r_max is not None else ""
        return (
            f"StructureFactor(q_max={self.q_max!r}, bins={self.bins}, "
            f"method={self.method!r}{extra})"
        )
