# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Polyhedral template matching (Larsen, Schmidt and Schiøtz 2016)."""

from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy._core import identify as _native
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.elements import SYMBOLS
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.identify._structure import StructureResult
from mdapy.neighbor import NeighborList
from mdapy.neighbor._neighbor_list import _nearest_given_or_built

# the nearest neighbors that are ordered and matched
_NEIGHBORS = 18


@dataclass(frozen=True)
class PTMResult(StructureResult):
    """Polyhedral template matching of each atom.

    Attributes
    ----------
    structure : numpy.ndarray
        (N,) int8 type of each atom, an index into ``names``.
    names : tuple of str
        Name of each type.
    colors : tuple of tuple of float
        Default RGB color of each type, from 0 to 1.
    rmsd : numpy.ndarray
        (N,) root-mean-square deviation from the best template, 0 for an atom
        that matches none.
    interatomic_distance : numpy.ndarray
        (N,) distance in Å between nearest neighbors of the matched lattice,
        0 for none.
    orientation : numpy.ndarray
        (N, 4) the rotation of the matched lattice from its standard
        orientation as a quaternion (x, y, z, w), zeros for none.
    correspondences : numpy.ndarray
        (N,) uint64 the order of the neighbors matching the template, as
        OVITO encodes it.
    template_index : numpy.ndarray
        (N,) int8 which of the equivalent templates of a structure matched.
    deformation_gradient : numpy.ndarray or None
        (N, 3, 3) the elastic deformation gradient from the template to the
        neighbors, when computed.
    ordering : numpy.ndarray or None
        (N,) int8 chemical ordering, an index into ``ordering_names``, when
        identified.
    ordering_names : tuple of str
        Name of each ordering type.
    neighbors : numpy.ndarray or None
        (N, 16) the neighbors of each atom in the order of the points of its
        template, -1 after its count, when asked for.
    """

    rmsd: np.ndarray
    interatomic_distance: np.ndarray
    orientation: np.ndarray
    correspondences: np.ndarray
    template_index: np.ndarray
    deformation_gradient: np.ndarray | None = None
    ordering: np.ndarray | None = None
    ordering_names: tuple[str, ...] = ()
    neighbors: np.ndarray | None = None

    def columns(self) -> dict[str, np.ndarray]:
        found = {
            "structure": self.structure,
            "rmsd": self.rmsd,
            "interatomic_distance": self.interatomic_distance,
            "orientation": self.orientation,
        }
        if self.deformation_gradient is not None:
            found["deformation_gradient"] = self.deformation_gradient.reshape(-1, 9)
        if self.ordering is not None:
            found["ordering"] = self.ordering
        return found


class PolyhedralTemplateMatching:
    """Polyhedral template matching: the structure, orientation and
    deformation of the neighborhood of each atom, and its chemical ordering.

    Parameters
    ----------
    structures : iterable of str, optional
        The structures to look for, of ``NAMES`` but other. Default fcc, hcp
        and bcc.
    rmsd_cutoff : float, optional
        Atoms whose root-mean-square deviation from the best template is
        larger are other; 0 for no cutoff. Default 0.1.
    deformation_gradient : bool, optional
        Compute the elastic deformation gradient of each atom. Default False.
    ordering : bool, optional
        Identify the chemical ordering of each atom from the elements.
        Default False.
    template_neighbors : bool, optional
        Give the neighbors of each atom in the order of its template.
        Default False.
    """

    #: Names of the structure types.
    NAMES = (
        "other",
        "fcc",
        "hcp",
        "bcc",
        "ico",
        "sc",
        "cubic_diamond",
        "hexagonal_diamond",
        "graphene",
    )
    #: Their default colors, OVITO's.
    COLORS = (
        (0.95, 0.95, 0.95),
        (0.4, 1.0, 0.4),
        (1.0, 0.4, 0.4),
        (0.4, 0.4, 1.0),
        (0.95, 0.8, 0.2),
        (0.6275, 0.0784, 0.9961),
        (0.0745, 0.6275, 0.9961),
        (0.9961, 0.5373, 0.0),
        (0.6275, 0.4706, 0.9961),
    )
    #: Names of the chemical ordering types.
    ORDERING_NAMES = (
        "other",
        "pure",
        "L10",
        "L12_A",
        "L12_B",
        "B2",
        "zincblende_wurtzite",
        "boron_nitride",
    )
    #: Their default colors, OVITO's.
    ORDERING_COLORS = (
        (0.95, 0.95, 0.95),
        (0.40, 0.62, 1.00),
        (1.00, 0.60, 0.20),
        (0.35, 0.85, 0.35),
        (1.00, 0.35, 0.35),
        (0.75, 0.40, 1.00),
        (0.20, 0.88, 0.90),
        (1.00, 0.90, 0.20),
    )

    def __init__(
        self,
        structures: Iterable[str] = ("fcc", "hcp", "bcc"),
        *,
        rmsd_cutoff: float = 0.1,
        deformation_gradient: bool = False,
        ordering: bool = False,
        template_neighbors: bool = False,
    ):
        if isinstance(structures, str):
            structures = (structures,)
        self.structures = tuple(structures)
        if not self.structures:
            raise InvalidParameter("give at least one structure to look for")
        for name in self.structures:
            if name not in self.NAMES[1:]:
                raise InvalidParameter(
                    f"unknown structure {name!r}; the structures are {self.NAMES[1:]}"
                )
        if not rmsd_cutoff >= 0:
            raise InvalidParameter(
                f"rmsd_cutoff must not be negative, not {rmsd_cutoff!r}"
            )
        self.rmsd_cutoff = float(rmsd_cutoff)
        self.deformation_gradient = bool(deformation_gradient)
        self.ordering = bool(ordering)
        self.template_neighbors = bool(template_neighbors)

    def compute(
        self,
        system: System,
        *,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> PTMResult:
        """The structure of each atom.

        Parameters
        ----------
        system : System
            The atoms, with an ``element`` column for the ordering.
        neighbors : NeighborList, optional
            Neighbors of this system with at least the 18 nearest of each
            atom.

        Returns
        -------
        PTMResult
            The structure types, orientations and the other results.

        Raises
        ------
        InvalidParameter
            If an atom has fewer than 18 neighbors in the given list.
        MissingProperty
            If the ordering is asked for without an element column.
        """
        n = len(system)
        types = None
        if self.ordering:
            if "element" not in system.table.columns:
                raise MissingProperty("the chemical ordering needs the element column")
            names = system.table["element"].cast(str).to_numpy()
            numbers = {symbol: z for z, symbol in enumerate(SYMBOLS)}
            types = np.array([numbers[e] for e in names], dtype=np.int32)
        # without periodic images, small systems have fewer neighbors
        k = _NEIGHBORS if system.box.pbc.any() else min(_NEIGHBORS, max(n - 1, 0))
        if k > 0:
            nl = _nearest_given_or_built(system, k, neighbors)
            offsets, j, shifts, reading = nl.offsets, nl.j, nl.shifts, nl._reading()
        else:
            offsets = np.zeros(n + 1, dtype=np.int64)
            j = np.zeros(0, dtype=np.int32)
            shifts = np.zeros((0, 3), dtype=np.int8)
            reading = nullcontext()
        bits = sum(1 << (self.NAMES.index(name) - 1) for name in self.structures)
        with reading:
            found = _native.polyhedral_template_matching(
                np.ascontiguousarray(system["positions"]),
                np.ascontiguousarray(system.box.matrix),
                offsets,
                j,
                shifts,
                types,
                bits,
                self.deformation_gradient,
                self.template_neighbors,
                progress,
                _native_token(cancel),
            )
        structure = found["structure"]
        if self.rmsd_cutoff > 0:
            # as OVITO: the other results stay those of the best template
            structure[found["rmsd"] > self.rmsd_cutoff] = 0
        return PTMResult(
            structure,
            self.NAMES,
            self.COLORS,
            found["rmsd"],
            found["interatomic_distance"],
            found["orientation"],
            found["correspondences"],
            found["template_index"],
            found.get("deformation_gradient"),
            found.get("ordering"),
            self.ORDERING_NAMES if self.ordering else (),
            found.get("neighbors"),
        )

    def __repr__(self) -> str:
        return (
            f"PolyhedralTemplateMatching(structures={self.structures!r}, "
            f"rmsd_cutoff={self.rmsd_cutoff!r})"
        )
