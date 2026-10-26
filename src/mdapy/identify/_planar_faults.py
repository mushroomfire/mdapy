# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Planar faults of fcc crystals from polyhedral template matching."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy._core import identify as _native
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter
from mdapy.identify._ptm import PolyhedralTemplateMatching, PTMResult
from mdapy.neighbor import NeighborList


@dataclass(frozen=True)
class PlanarFaultResult(Result):
    """The planar fault type of each atom.

    Attributes
    ----------
    fault : numpy.ndarray
        (N,) int8 type of each atom, an index into ``names``.
    names : tuple of str
        Name of each type.
    colors : tuple of tuple of float
        Default RGB color of each type, from 0 to 1.
    interatomic_distance : numpy.ndarray
        (N,) distance in Å between nearest neighbors of the lattice each
        atom matched, from polyhedral template matching.
    """

    fault: np.ndarray
    names: tuple[str, ...]
    colors: tuple[tuple[float, float, float], ...]
    interatomic_distance: np.ndarray

    def counts(self) -> dict[str, int]:
        """Number of atoms of each type, by name."""
        found = np.bincount(self.fault, minlength=len(self.names))
        return {name: int(c) for name, c in zip(self.names, found)}

    def areas(self) -> dict[str, float]:
        """Area in Å² of the intrinsic stacking faults, each two layers of
        atoms, of the coherent twin boundaries, each one, and of the
        extrinsic stacking faults, each two."""
        d2 = self.interatomic_distance**2
        return {
            "intrinsic stacking fault": float(
                np.sqrt(3) / 4 * d2[self.fault == 2].sum()
            ),
            "coherent twin boundary": float(np.sqrt(3) / 2 * d2[self.fault == 3].sum()),
            "extrinsic stacking fault": float(
                np.sqrt(3) / 4 * d2[self.fault == 5].sum()
            ),
        }

    def columns(self) -> dict[str, np.ndarray]:
        return {"planar_fault": self.fault}


class PlanarFaults:
    """Planar faults of fcc crystals: the hcp-like atoms of stacking faults
    and twin boundaries, found by polyhedral template matching.

    Parameters
    ----------
    structures : iterable of str, optional
        The structures polyhedral template matching looks for; fcc and hcp
        among them. Default fcc, hcp and bcc.
    rmsd_cutoff : float, optional
        Atoms whose root-mean-square deviation from the best template is
        larger are other; 0 for no cutoff. Default 0.1.
    extrinsic : bool, optional
        Also find extrinsic stacking faults, two planes of hcp-like atoms
        with one plane of fcc atoms between them, which are otherwise two
        coherent twin boundaries. Default False.
    """

    NAMES = (
        "non-hcp",
        "other hcp",
        "intrinsic stacking fault",
        "coherent twin boundary",
        "multilayer stacking fault",
        "extrinsic stacking fault",
    )
    # the colors of OVITO
    COLORS = (
        (1.0, 1.0, 1.0),
        (1.0, 0.4, 0.4),
        (0.9, 0.7, 0.2),
        (0.4, 0.4, 1.0),
        (0.0, 1.0, 0.9),
        (0.7, 0.3, 0.9),
    )

    def __init__(
        self,
        structures: tuple[str, ...] = ("fcc", "hcp", "bcc"),
        *,
        rmsd_cutoff: float = 0.1,
        extrinsic: bool = False,
    ):
        if isinstance(structures, str):
            structures = (structures,)
        self.structures = tuple(structures)
        if not {"fcc", "hcp"} <= set(self.structures):
            raise InvalidParameter("planar faults need the structures fcc and hcp")
        self.__ptm = PolyhedralTemplateMatching(
            self.structures, rmsd_cutoff=rmsd_cutoff, template_neighbors=True
        )
        self.rmsd_cutoff = self.__ptm.rmsd_cutoff
        self.extrinsic = bool(extrinsic)

    def compute(
        self,
        system: System,
        *,
        ptm: PTMResult | None = None,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> PlanarFaultResult:
        """The planar fault type of each atom.

        Parameters
        ----------
        system : System
            The atoms.
        ptm : PTMResult, optional
            Polyhedral template matching of system already done, with the
            neighbors in the order of the templates; done here when not
            given.
        neighbors : NeighborList, optional
            Neighbors of system with at least the 18 nearest of each atom,
            for the template matching.

        Returns
        -------
        PlanarFaultResult
            The fault types.

        Raises
        ------
        InvalidParameter
            If ptm is not of system or has no neighbors in the order of the
            templates.
        """
        if ptm is None:
            ptm = self.__ptm.compute(
                system, neighbors=neighbors, progress=progress, cancel=cancel
            )
        elif ptm.neighbors is None or len(ptm.structure) != len(system):
            raise InvalidParameter(
                "ptm must be of this system, from PolyhedralTemplateMatching("
                "template_neighbors=True)"
            )
        try:
            fault = _native.planar_faults(
                np.ascontiguousarray(ptm.structure),
                np.ascontiguousarray(ptm.neighbors),
                self.extrinsic,
                progress,
                _native_token(cancel),
            )
        except ValueError as error:
            raise InvalidParameter(str(error)) from None
        return PlanarFaultResult(
            fault, self.NAMES, self.COLORS, ptm.interatomic_distance
        )

    def __repr__(self) -> str:
        extra = ", extrinsic=True" if self.extrinsic else ""
        return (
            f"PlanarFaults(structures={self.structures!r}, "
            f"rmsd_cutoff={self.rmsd_cutoff!r}{extra})"
        )
