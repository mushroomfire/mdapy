# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Molecules and their chemical formulas."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Any

import numpy as np
import polars as pl

from mdapy.analysis._bonds import ClusterAnalysis
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.elements import SYMBOLS, covalent_radii, vdw_radii
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.neighbor import NeighborList


def _hill(kinds: list[str]) -> list[str]:
    """Elements in the order of Hill formulas: carbon, hydrogen, then the
    others alphabetically; without carbon, all alphabetically."""
    if "C" in kinds:
        return ["C"] + (["H"] if "H" in kinds else []) + sorted(set(kinds) - {"C", "H"})
    return sorted(kinds)


def _formula(kinds: list[str], counts: np.ndarray) -> str:
    return "".join(
        k if c == 1 else f"{k}{c}" for k, c in zip(kinds, counts.tolist()) if c > 0
    )


@dataclass(frozen=True)
class SpeciesResult(Result):
    """Molecules of bonded atoms and their chemical formulas.

    Attributes
    ----------
    molecule : numpy.ndarray
        (N,) int64 molecule of each atom, numbered from 1 by decreasing size,
        ties by their first atom.
    formulas : numpy.ndarray
        (M,) chemical formula of each molecule, in Hill order.
    sizes : numpy.ndarray
        (M,) int64 number of atoms of each molecule.
    """

    molecule: np.ndarray
    formulas: np.ndarray
    sizes: np.ndarray

    def counts(self) -> pl.DataFrame:
        """The number of molecules of each formula, the most common first."""
        names, counts = np.unique(self.formulas, return_counts=True)
        table = pl.DataFrame(
            {"formula": names.astype(str), "count": counts.astype(np.int64)},
            schema={"formula": pl.String, "count": pl.Int64},
        )
        return table.sort(["count", "formula"], descending=[True, False])

    def columns(self) -> dict[str, np.ndarray]:
        return {"molecule": self.molecule}


class ChemicalSpecies:
    """Molecules: atoms joined by bonds shorter than their radii allow, and
    their chemical formulas.

    Two atoms of radii r_i and r_j are bonded when closer than
    ``scale * (r_i + r_j) + tolerance``, or than the cutoff given for their
    pair of elements.

    Parameters
    ----------
    radii : {"covalent", "vdw"} or dict, optional
        The radius of each element: the covalent radii (Cordero et al.
        2008), the van der Waals radii (Alvarez 2013), or radii in Å by
        element. Default "covalent".
    scale : float, optional
        Factor on the sum of the radii. Default 1.
    tolerance : float, optional
        Length added to it, in Å. Default 0.45, so that with covalent radii
        the bonds are those of Jmol and of mdapy's pictures; with van der
        Waals radii, a scale of 0.6 and no tolerance give the bonds of VMD.
    cutoffs : dict, optional
        Bond length in Å for pairs of elements ``(a, b)``, in either order,
        instead of the radii; 0 for pairs never bonded.
    """

    def __init__(
        self,
        radii: str | Mapping[str, float] = "covalent",
        *,
        scale: float = 1.0,
        tolerance: float = 0.45,
        cutoffs: Mapping[tuple[str, str], float] | None = None,
    ):
        if isinstance(radii, str):
            if radii not in ("covalent", "vdw"):
                raise InvalidParameter(
                    f'radii must be "covalent", "vdw" or radii by element, not {radii!r}'
                )
        else:
            radii = dict(radii)
            for element, value in radii.items():
                if element not in SYMBOLS[1:]:
                    raise InvalidParameter(f"unknown element {element!r} in radii")
                if not value > 0:
                    raise InvalidParameter(f"the radius of {element} must be positive")
        self.radii = radii
        if not scale > 0:
            raise InvalidParameter(f"scale must be positive, not {scale!r}")
        if not tolerance >= 0:
            raise InvalidParameter(f"tolerance must not be negative, not {tolerance!r}")
        self.scale, self.tolerance = float(scale), float(tolerance)
        self.cutoffs = {}
        for (a, b), value in (cutoffs or {}).items():
            for element in (a, b):
                if element not in SYMBOLS[1:]:
                    raise InvalidParameter(f"unknown element {element!r} in cutoffs")
            if not value >= 0:
                raise InvalidParameter(f"the cutoff of {(a, b)} must not be negative")
            self.cutoffs[a, b] = self.cutoffs[b, a] = float(value)

    def __radii(self, kinds: list[str]) -> np.ndarray:
        if self.radii == "covalent":
            return covalent_radii(kinds)
        if self.radii == "vdw":
            return vdw_radii(kinds)
        missing = [k for k in kinds if k not in self.radii]
        if missing:
            raise InvalidParameter(f"no radius given for {missing}")
        return np.array([self.radii[k] for k in kinds])

    def bond_lengths(self, elements: Iterable[str]) -> dict[tuple[str, str], float]:
        """The longest bond between each pair of the elements.

        Parameters
        ----------
        elements : iterable of str
            Chemical symbols.

        Returns
        -------
        dict
            The bond length in Å of each pair ``(a, b)``, a before b in the
            order given; 0 for pairs never bonded.

        Raises
        ------
        InvalidParameter
            If an element has no radius and its pairs no cutoffs.
        """
        kinds = list(dict.fromkeys(elements))
        lengths = {}
        needed = [k for k in kinds if any((k, o) not in self.cutoffs for o in kinds)]
        radius = dict(zip(needed, self.__radii(needed))) if needed else {}
        for a, b in combinations_with_replacement(kinds, 2):
            if (a, b) in self.cutoffs:
                lengths[a, b] = self.cutoffs[a, b]
            else:
                lengths[a, b] = self.scale * (radius[a] + radius[b]) + self.tolerance
        return lengths

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> SpeciesResult:
        """The molecules of a system.

        Parameters
        ----------
        system : System
            The atoms, with an ``element`` column.
        neighbors : NeighborList, optional
            A neighbor list of this system with at least the longest bond.

        Returns
        -------
        SpeciesResult
            The molecule of each atom and the formula of each molecule.

        Raises
        ------
        MissingProperty
            If the system has no ``element`` column.
        InvalidParameter
            If an element has no radius.
        """
        if "element" not in system.table.columns:
            raise MissingProperty("chemical species need the element column")
        names = system.table["element"].cast(pl.String)
        kinds = _hill(names.unique().to_list())
        lengths = {
            pair: value for pair, value in self.bond_lengths(kinds).items() if value > 0
        }
        n = len(system)
        if lengths:
            found = ClusterAnalysis(cutoffs=lengths).compute(
                system, neighbors=neighbors, progress=progress, cancel=cancel
            )
            molecule, sizes = found.cluster, found.sizes
        else:  # no pair ever bonded: each atom alone
            molecule, sizes = np.arange(n, dtype=np.int64), np.ones(n, dtype=np.int64)
        code = names.replace_strict(
            kinds, list(range(len(kinds))), return_dtype=pl.Int64
        )
        count = len(sizes)
        composition = np.bincount(
            molecule * len(kinds) + code.to_numpy(), minlength=count * len(kinds)
        ).reshape(count, len(kinds))
        unique, inverse = np.unique(composition, axis=0, return_inverse=True)
        formulas = np.array([_formula(kinds, row) for row in unique], dtype=object)
        return SpeciesResult(
            molecule=molecule + 1,
            formulas=formulas[inverse.reshape(-1)],
            sizes=sizes,
        )

    def count(self, frames: Iterable[System]) -> pl.DataFrame:
        """The number of molecules of each formula in each frame.

        Parameters
        ----------
        frames : iterable of System
            The frames, such as a trajectory.

        Returns
        -------
        polars.DataFrame
            A ``frame`` column, from 0, and a column of counts for each
            formula, the most common over all frames first.
        """
        rows = []
        for frame in frames:
            table = self.compute(frame).counts()
            rows.append(dict(zip(table["formula"], table["count"])))
        totals: dict[str, int] = {}
        for row in rows:
            for formula, c in row.items():
                totals[formula] = totals.get(formula, 0) + c
        order = sorted(totals, key=lambda f: (-totals[f], f))
        return pl.DataFrame(
            {"frame": np.arange(len(rows), dtype=np.int64)}
            | {
                f: np.array([row.get(f, 0) for row in rows], dtype=np.int64)
                for f in order
            },
        )

    def __repr__(self) -> str:
        radii = self.radii if isinstance(self.radii, str) else "by element"
        return (
            f"ChemicalSpecies(radii={radii!r}, scale={self.scale!r}, "
            f"tolerance={self.tolerance!r})"
        )
