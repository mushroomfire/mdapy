# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The atoms on one side of a plane, or in a slab about it."""

import numpy as np
from numpy.typing import ArrayLike

from mdapy.core.system import System
from mdapy.errors import InvalidParameter

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def slice_mask(
    system: System,
    normal: ArrayLike | str,
    distance: float = 0.0,
    *,
    width: float = 0.0,
    invert: bool = False,
    miller: bool = False,
    through: ArrayLike | None = None,
    atoms: ArrayLike | None = None,
) -> np.ndarray:
    """Which atoms :func:`slice` keeps.

    Parameters
    ----------
    system : System
        The atoms.
    normal : array_like or str
        (3,) the normal of the plane in Cartesian coordinates, of any
        length, or an axis "x", "y" or "z"; with miller, the Miller indices
        hkl of the plane in the vectors of the box.
    distance : float, optional
        The plane holds the points x with n . x = distance in Å, n the unit
        normal; with miller, it lies distance interplanar spacings d_hkl from
        the origin of the box. Default 0.
    width : float, optional
        Keep the atoms closer to the plane than width / 2 in Å instead.
        Default 0: keep those on the side the normal points away from.
    invert : bool, optional
        Keep the other atoms. Default False.
    miller : bool, optional
        The normal holds Miller indices. Default False.
    through : array_like, optional
        (3,) a point in Å the plane passes through, instead of distance.
    atoms : array_like, optional
        A boolean mask or indices of the atoms the plane acts on; the others
        are kept. Default all.

    Returns
    -------
    numpy.ndarray
        (N,) bool whether each atom is kept.

    Raises
    ------
    InvalidParameter
        For a normal that is zero or not three numbers or an axis, Miller
        indices given as an axis, a negative width, a point not of three
        numbers, both a distance and a point, and atoms that are not a mask
        or indices of the system.
    """
    if isinstance(normal, str):
        if normal not in _AXES or miller:
            raise InvalidParameter(
                f"normal must be x, y, z or three numbers, not {normal!r}"
                if not miller
                else "Miller indices must be three numbers"
            )
        normal = _AXES[normal]
    n = np.asarray(normal, dtype=float)
    if n.shape != (3,) or not np.isfinite(n).all() or not np.linalg.norm(n) > 0:
        raise InvalidParameter("normal must be three numbers, not all zero")
    width = float(width)
    if not width >= 0:
        raise InvalidParameter(f"width must be at least 0, not {width!r}")
    box = system.box
    positions = system["positions"]
    # the Cartesian normal, of length 1 or, of Miller indices, 1 / d_hkl
    g = box.inverse @ n if miller else n / np.linalg.norm(n)
    spacing = 1.0 / np.linalg.norm(g)
    if through is not None:
        if distance != 0:
            raise InvalidParameter("give distance or through, not both")
        point = np.asarray(through, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise InvalidParameter("through must be a point of three numbers")
        side = (positions - point) @ g * (spacing if miller else 1.0)
    elif miller:
        side = ((positions - box.origin) @ g - float(distance)) * spacing
    else:
        side = positions @ g - float(distance)
    keep = (np.abs(side) <= width / 2 if width > 0 else side <= 0) != bool(invert)
    if atoms is not None:
        acting = np.asarray(atoms)
        if acting.dtype == bool:
            if acting.shape != (len(system),):
                raise InvalidParameter(
                    f"a mask of atoms must have {len(system)} entries"
                )
        else:
            index = acting.reshape(-1)
            if index.size and (
                index.dtype.kind not in "iu"
                or index.min() < 0
                or index.max() >= len(system)
            ):
                raise InvalidParameter("atoms must be a mask or indices of the system")
            acting = np.zeros(len(system), dtype=bool)
            acting[index.astype(np.int64)] = True
        keep |= ~acting
    return keep


# named after the Slice modifier of OVITO
def slice(
    system: System,
    normal: ArrayLike | str,
    distance: float = 0.0,
    *,
    width: float = 0.0,
    invert: bool = False,
    miller: bool = False,
    through: ArrayLike | None = None,
    atoms: ArrayLike | None = None,
) -> System:
    """The atoms on one side of a plane, or in a slab about it.

    Parameters
    ----------
    system : System
        The atoms.
    normal : array_like or str
        (3,) the normal of the plane in Cartesian coordinates, of any
        length, or an axis "x", "y" or "z"; with miller, the Miller indices
        hkl of the plane in the vectors of the box.
    distance : float, optional
        The plane holds the points x with n . x = distance in Å, n the unit
        normal; with miller, it lies distance interplanar spacings d_hkl from
        the origin of the box. Default 0.
    width : float, optional
        Keep the atoms closer to the plane than width / 2 in Å instead.
        Default 0: keep those on the side the normal points away from.
    invert : bool, optional
        Keep the other atoms. Default False.
    miller : bool, optional
        The normal holds Miller indices. Default False.
    through : array_like, optional
        (3,) a point in Å the plane passes through, instead of distance.
    atoms : array_like, optional
        A boolean mask or indices of the atoms the plane acts on; the others
        are kept. Default all.

    Returns
    -------
    System
        The atoms kept, in their order, in the same box.

    Raises
    ------
    InvalidParameter
        As :func:`slice_mask`.
    """
    keep = slice_mask(
        system,
        normal,
        distance,
        width=width,
        invert=invert,
        miller=miller,
        through=through,
        atoms=atoms,
    )
    return system.gather(np.flatnonzero(keep))
