# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The elastic strain of the atoms of a crystal against the ideal crystal."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy._core import analysis as _native
from mdapy.analysis._dislocations import _CRYSTALS, DislocationAnalysis
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter

_HEXAGONAL = ("hcp", "hexagonal_diamond")


@dataclass(frozen=True)
class ElasticStrainResult(Result):
    """The elastic strain of each atom.

    Attributes
    ----------
    structure : numpy.ndarray
        (N,) int8 structure of each atom, an index into ``names``.
    names : tuple of str
        Names of the structures: other, fcc, hcp, bcc, cubic_diamond,
        hexagonal_diamond.
    colors : tuple of tuple of float
        RGB color of each structure.
    strain_tensor : numpy.ndarray
        (N, 3, 3) elastic strain of each atom of the crystal in the spatial
        frame; zero for the other atoms.
    volumetric : numpy.ndarray
        (N,) a third of the trace of the strain; zero for the atoms not of
        the crystal.
    """

    structure: np.ndarray
    names: tuple[str, ...]
    colors: tuple[tuple[float, float, float], ...]
    strain_tensor: np.ndarray
    volumetric: np.ndarray

    def columns(self) -> dict[str, np.ndarray]:
        return {
            "structure": self.structure,
            "elastic_strain": self.strain_tensor,
            "elastic_volumetric_strain": self.volumetric,
        }


class ElasticStrain:
    """The elastic strain of each atom of a crystal: the deformation that
    maps the neighbors of the ideal crystal onto its own, as OVITO's elastic
    strain calculation.

    Parameters
    ----------
    crystal : str
        The crystal: ``"fcc"``, ``"hcp"``, ``"bcc"``, ``"cubic_diamond"`` or
        ``"hexagonal_diamond"``. Its atoms are found as by the dislocation
        analysis, with the stacking faults of fcc and hcp, and of the
        diamonds, in the other structure.
    a : float
        Lattice constant of the unstrained crystal in Å: of the cubic cell,
        or a of the hexagonal cell.
    c : float, optional
        Lattice constant c of the unstrained hexagonal cell in Å; default
        that of the ideal ratio, a √(8/3). Only for hcp and hexagonal
        diamond.
    """

    NAMES = DislocationAnalysis.NAMES
    COLORS = DislocationAnalysis.COLORS

    def __init__(self, crystal: str, a: float, c: float | None = None):
        if crystal not in _CRYSTALS:
            raise InvalidParameter(
                f"crystal must be one of {', '.join(_CRYSTALS)}, not {crystal!r}"
            )
        if not a > 0:
            raise InvalidParameter(f"a must be positive, not {a!r}")
        if c is not None:
            if crystal not in _HEXAGONAL:
                raise InvalidParameter(
                    f"c is only for hexagonal crystals, not {crystal}"
                )
            if not c > 0:
                raise InvalidParameter(f"c must be positive, not {c!r}")
        self.crystal = crystal
        self.a = float(a)
        self.c = None if c is None else float(c)

    def compute(
        self, system: System, *, progress: Any = None, cancel: Any = None
    ) -> ElasticStrainResult:
        """The elastic strain of the atoms of a system.

        Parameters
        ----------
        system : System
            The atoms, no two at the same position.

        Returns
        -------
        ElasticStrainResult
            The structure and the elastic strain of each atom.

        Raises
        ------
        InvalidParameter
            For a box so short along a periodic direction that an atom meets
            its own periodic image: replicate the system first.
        """
        box = system.box
        positions = system["positions"]
        fractional = (positions - box.origin) @ box.inverse
        moved = np.where(box.pbc, np.floor(fractional), 0.0) @ box.matrix
        if self.crystal in _HEXAGONAL:
            # the ideal vectors of the hexagonal lattices are in units of the
            # cubic constant with the same nearest neighbors, c/a ideal
            scale = self.a * np.sqrt(2.0)
            axial = 1.0 if self.c is None else self.c / self.a / np.sqrt(8.0 / 3.0)
        else:
            scale, axial = self.a, 1.0
        try:
            structure, strain = _native.elastic_strain(
                np.ascontiguousarray(positions - moved),
                np.ascontiguousarray(box.matrix),
                np.ascontiguousarray(box.inverse),
                np.ascontiguousarray(box.origin),
                np.ascontiguousarray(box.pbc),
                _CRYSTALS[self.crystal],
                scale,
                axial,
                progress,
                _native_token(cancel),
            )
        except (ValueError, RuntimeError) as error:  # periodic images too close
            raise InvalidParameter(
                f"{error}: replicate the system along its short periodic directions"
            ) from None
        index = np.zeros(8, dtype=np.int8)
        for k, name in enumerate(self.NAMES[1:], start=1):
            index[_CRYSTALS[name]] = k
        strain = strain.reshape(-1, 3, 3)
        return ElasticStrainResult(
            structure=index[structure],
            names=self.NAMES,
            colors=self.COLORS,
            strain_tensor=strain,
            volumetric=np.trace(strain, axis1=1, axis2=2) / 3,
        )

    def __repr__(self) -> str:
        c = "" if self.c is None else f", c={self.c!r}"
        return f"ElasticStrain({self.crystal!r}, {self.a!r}{c})"
