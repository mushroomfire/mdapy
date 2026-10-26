# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Point defects by Wigner-Seitz cells of a reference lattice."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy._core import neighbor as _search
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter


@dataclass(frozen=True)
class WignerSeitzResult(Result):
    """Occupancy of the reference sites.

    Attributes
    ----------
    site : numpy.ndarray
        (N,) reference site of each atom, the nearest one (the first of
        equally near sites), as ``mdapy.index_dtype``.
    occupancy : numpy.ndarray
        (M,) int64 number of atoms on each reference site.
    """

    site: np.ndarray
    occupancy: np.ndarray

    @property
    def vacancies(self) -> np.ndarray:
        """Reference sites without atoms."""
        return np.flatnonzero(self.occupancy == 0)

    @property
    def interstitials(self) -> np.ndarray:
        """Reference sites with more than one atom."""
        return np.flatnonzero(self.occupancy > 1)

    @property
    def vacancy_count(self) -> int:
        """Number of empty sites."""
        return int(np.sum(self.occupancy == 0))

    @property
    def interstitial_count(self) -> int:
        """Number of atoms beyond one on each site."""
        return int(np.sum(np.maximum(self.occupancy - 1, 0)))

    def columns(self) -> dict[str, np.ndarray]:
        return {"site": self.site, "occupancy": self.occupancy[self.site]}


class WignerSeitz:
    """Wigner-Seitz analysis: each atom belongs to the cell of its nearest
    reference site; empty sites are vacancies and crowded ones hold
    interstitials.

    Parameters
    ----------
    reference : System
        The reference sites, usually the perfect crystal.
    affine : bool, optional
        Map the atoms into the reference box by their fractional coordinates
        before assigning them, which removes a homogeneous deformation of the
        box. Default False.
    """

    def __init__(self, reference: System, *, affine: bool = False):
        if len(reference) == 0:
            raise InvalidParameter("the reference has no sites")
        self.reference = reference
        self.affine = bool(affine)
        box = reference.box
        self.__tree = _search.KdTree(
            np.ascontiguousarray(reference.positions),
            np.ascontiguousarray(box.matrix),
            np.ascontiguousarray(box.origin),
            np.ascontiguousarray(box.pbc),
        )

    def compute(
        self, system: System, *, progress: Any = None, cancel: Any = None
    ) -> WignerSeitzResult:
        """The site of each atom and the occupancy of each site.

        Parameters
        ----------
        system : System
            The atoms, any number of them, in a box with the periodicity of
            the reference.

        Returns
        -------
        WignerSeitzResult
            The site of each atom and the number of atoms on each site.
        """
        box = self.reference.box
        if not np.array_equal(system.box.pbc, box.pbc):
            raise InvalidParameter("the system and the reference differ in periodicity")
        positions = system.positions
        if self.affine:
            positions = box.to_cartesian(system.box.to_fractional(positions))
        nearest, _, _ = self.__tree.nearest(
            np.ascontiguousarray(positions), 1, progress, _native_token(cancel)
        )
        site = nearest[:, 0]
        occupancy = np.bincount(site, minlength=len(self.reference))
        return WignerSeitzResult(site, occupancy)

    def __repr__(self) -> str:
        return f"WignerSeitz(sites={len(self.reference)}, affine={self.affine})"
