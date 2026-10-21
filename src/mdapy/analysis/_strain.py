# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Atomic strain against a reference configuration."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy._core import analysis as _native
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList


@dataclass(frozen=True)
class StrainResult(Result):
    """The strain of each atom against the reference.

    Attributes
    ----------
    strain : numpy.ndarray
        (N, 3, 3) Green-Lagrange strain; NaN where the reference neighbors
        do not span three dimensions.
    shear : numpy.ndarray
        (N,) von Mises shear strain.
    volumetric : numpy.ndarray
        (N,) volumetric strain, a third of the trace.
    d2min : numpy.ndarray
        (N,) non-affine squared displacement in Å², the part of the neighbor
        vectors that the deformation gradient does not explain.
    """

    strain: np.ndarray
    shear: np.ndarray
    volumetric: np.ndarray
    d2min: np.ndarray

    def columns(self) -> dict[str, np.ndarray]:
        return {
            "shear_strain": self.shear,
            "volumetric_strain": self.volumetric,
            "d2min": self.d2min,
        }


class AtomicStrain:
    """Atomic strain: the deformation gradient that best maps each atom's
    neighbor vectors in a reference configuration onto the current ones.

    Parameters
    ----------
    reference : System
        The reference configuration, with the same atoms in the same order as
        the systems to compare.
    cutoff : float
        Neighbors in the reference closer than cutoff, in Å, define the
        neighbor vectors.
    affine : bool, optional
        Measure the current vectors in the reference box, which removes the
        homogeneous deformation of the box. Default False.
    """

    def __init__(self, reference: System, cutoff: float, *, affine: bool = False):
        if not cutoff > 0:
            raise InvalidParameter(f"cutoff must be positive, not {cutoff!r}")
        self.reference = reference
        self.cutoff = float(cutoff)
        self.affine = bool(affine)
        self._pairs = NeighborList.build(reference, self.cutoff)
        self._fractional = reference.box.to_fractional(reference.positions)

    def compute(
        self, system: System, *, progress: Any = None, cancel: Any = None
    ) -> StrainResult:
        """The strain of each atom of a configuration.

        Parameters
        ----------
        system : System
            The current configuration: the atoms of the reference, in the
            same order, each moved by less than half the box.

        Returns
        -------
        StrainResult
            The strain of each atom.
        """
        reference = self.reference
        if len(system) != len(reference):
            raise InvalidParameter(
                f"the system has {len(system)} atoms, the reference {len(reference)}"
            )
        if not np.array_equal(system.box.pbc, reference.box.pbc):
            raise InvalidParameter("the system and the reference differ in periodicity")
        if "id" in system.table.columns and "id" in reference.table.columns:
            if not np.array_equal(
                system.table["id"].to_numpy(), reference.table["id"].to_numpy()
            ):
                raise InvalidParameter(
                    "the atoms are not in the order of the reference"
                )
        # displacements in fractional coordinates, each by the nearest image
        displacement = system.box.to_fractional(system.positions) - self._fractional
        periodic = system.box.pbc
        displacement[:, periodic] -= np.round(displacement[:, periodic])
        matrix = reference.box.matrix if self.affine else system.box.matrix
        strain, d2min = _native.atomic_strain(
            self._pairs.offsets,
            self._pairs.j,
            np.ascontiguousarray(self._pairs.shifts),
            np.ascontiguousarray(self._fractional),
            np.ascontiguousarray(displacement),
            np.ascontiguousarray(reference.box.matrix),
            np.ascontiguousarray(matrix),
            progress,
            _native_token(cancel),
        )
        e = strain
        shear = np.sqrt(
            e[:, 0, 1] ** 2
            + e[:, 0, 2] ** 2
            + e[:, 1, 2] ** 2
            + (
                (e[:, 0, 0] - e[:, 1, 1]) ** 2
                + (e[:, 1, 1] - e[:, 2, 2]) ** 2
                + (e[:, 0, 0] - e[:, 2, 2]) ** 2
            )
            / 6.0
        )
        volumetric = np.trace(e, axis1=1, axis2=2) / 3.0
        return StrainResult(
            strain=strain, shear=shear, volumetric=volumetric, d2min=d2min
        )

    def __repr__(self) -> str:
        return f"AtomicStrain(cutoff={self.cutoff!r}, affine={self.affine})"
