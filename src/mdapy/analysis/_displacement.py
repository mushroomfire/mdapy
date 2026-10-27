# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The displacement of each atom from a reference configuration."""

from dataclasses import dataclass

import numpy as np

from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.errors import InvalidParameter


@dataclass(frozen=True)
class AtomicDisplacementResult(Result):
    """The displacements of the atoms.

    Attributes
    ----------
    displacement : numpy.ndarray
        (N, 3) the displacement of each atom from its reference position in
        Å; its column ``displacement``.
    magnitude : numpy.ndarray
        (N,) its length in Å; its column ``displacement_magnitude``.
    """

    displacement: np.ndarray
    magnitude: np.ndarray

    def columns(self) -> dict[str, np.ndarray]:
        return {
            "displacement": self.displacement,
            "displacement_magnitude": self.magnitude,
        }


class AtomicDisplacement:
    """The displacement of each atom from its position in a reference
    configuration.

    Parameters
    ----------
    reference : System
        The reference configuration: the atoms of the systems to compare,
        matched by the column ``id`` when both have it, otherwise in the
        same order.
    minimum_image : bool, optional
        Take along each periodic direction the periodic image of the
        reference position nearest the atom. Default True.
    affine : bool, optional
        Measure the displacements in the reference box: the current
        positions mapped into it first, which removes the homogeneous
        deformation of the box. Default False.
    """

    def __init__(
        self, reference: System, *, minimum_image: bool = True, affine: bool = False
    ):
        self.reference = reference
        self.minimum_image = bool(minimum_image)
        self.affine = bool(affine)

    def compute(self, system: System) -> AtomicDisplacementResult:
        """The displacements of the atoms of a configuration.

        Parameters
        ----------
        system : System
            The current configuration.

        Returns
        -------
        AtomicDisplacementResult
            The displacement of each atom, in its order.

        Raises
        ------
        InvalidParameter
            For atoms not in the reference, or a reference of another
            number of atoms without ids.
        """
        reference = self.reference
        start = reference["positions"]
        if "id" in system.table.columns and "id" in reference.table.columns:
            ids, known = system["id"], reference["id"]
            order = np.argsort(known, kind="stable")
            rows = order[np.clip(np.searchsorted(known[order], ids), 0, len(known) - 1)]
            if len(known) == 0 or not np.array_equal(known[rows], ids):
                raise InvalidParameter("an atom of the system is not in the reference")
            start = start[rows]
        elif len(system) != len(reference):
            raise InvalidParameter(
                f"the system has {len(system)} atoms, the reference {len(reference)}, "
                "and no ids to match them"
            )
        box = reference.box if self.affine else system.box
        current = system["positions"]
        if self.affine:
            fractional = system.box.to_fractional(current)
            current = fractional @ reference.box.matrix + reference.box.origin
        moved = current - start
        if self.minimum_image:
            moved = box.minimum_image(moved)
        return AtomicDisplacementResult(
            displacement=moved, magnitude=np.linalg.norm(moved, axis=1)
        )

    def __repr__(self) -> str:
        return (
            f"AtomicDisplacement(minimum_image={self.minimum_image!r}, "
            f"affine={self.affine!r})"
        )
