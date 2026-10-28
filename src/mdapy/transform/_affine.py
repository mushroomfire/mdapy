# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""An affine transformation of a system: its atoms and its box moved,
rotated, sheared or scaled."""

from collections.abc import Sequence

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from mdapy.core._atoms import _mask
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import InvalidParameter

# The columns of vectors that turn with the atoms, as OVITO's vector
# properties do; no translation applies to them.
VECTORS = ("velocities", "forces", "displacement")


def affine(
    system: System,
    matrix: ArrayLike,
    translation: ArrayLike = (0.0, 0.0, 0.0),
    *,
    box: bool = True,
    vectors: Sequence[str] = VECTORS,
    atoms: ArrayLike | None = None,
) -> System:
    """The system with its atoms, and its box, mapped by an affine
    transformation.

    Parameters
    ----------
    system : System
        The system to transform.
    matrix : array_like
        (3, 3) linear part M of the transformation, acting on Cartesian
        column vectors: a position x goes to M x + t.
    translation : array_like, optional
        (3,) translation t in Å. Default none.
    box : bool, optional
        Whether the box transforms too: its vectors by M, its origin by M
        and t. Default True.
    vectors : sequence of str, optional
        The columns of vectors, of three components, that turn with M
        without the translation, where the system has them. Default
        ``velocities``, ``forces`` and ``displacement``.
    atoms : array_like, optional
        A boolean mask or indices of the atoms that move, with their
        vectors; the others stay as they are. Default all of them.

    Returns
    -------
    System
        The system with the positions, the vectors and, with ``box``, the
        box transformed; its other columns and its information as they are.

    Raises
    ------
    InvalidParameter
        If the matrix is not (3, 3) of finite numbers, or singular while the
        box transforms; if the translation is not 3 finite numbers; if
        atoms is not a mask or indices of the system; or if a column of
        ``vectors`` does not hold three numbers per atom.
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        raise InvalidParameter(f"matrix must be (3, 3) finite numbers, not {matrix!r}")
    t = np.asarray(translation, dtype=float)
    if t.shape != (3,) or not np.all(np.isfinite(t)):
        raise InvalidParameter(
            f"translation must be 3 finite numbers, not {translation!r}"
        )
    if box and abs(np.linalg.det(m)) < 1e-12:
        raise InvalidParameter("a singular matrix would flatten the box")
    n = len(system)
    moved = np.ones(n, dtype=bool) if atoms is None else _mask(atoms, n)
    columns = {}
    positions = np.array(system["positions"], dtype=float)
    positions[moved] = positions[moved] @ m.T + t
    columns["positions"] = positions
    table = system.table
    for name in vectors:
        if name not in table.columns:
            continue
        dtype = table.schema[name]
        if not (isinstance(dtype, pl.Array) and dtype.shape == (3,)):
            raise InvalidParameter(
                f"column {name!r} is not a vector of three numbers but {dtype}"
            )
        values = np.array(system[name], dtype=float)
        values[moved] = values[moved] @ m.T
        columns[name] = values
    result = system.with_columns(**columns)
    if box:
        old = system.box
        result = result.with_box(
            Box(old.matrix @ m.T, origin=m @ old.origin + t, pbc=old.pbc)
        )
    return result
