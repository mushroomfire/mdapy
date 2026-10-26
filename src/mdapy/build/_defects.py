# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Point defects, stacking faults and surfaces."""

import numpy as np
from numpy.typing import ArrayLike

from mdapy._core import neighbor as _search
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.elements import SYMBOLS
from mdapy.errors import InvalidParameter

_AXES = {"x": 0, "y": 1, "z": 2}


def _chosen(
    system: System,
    count: int | None,
    atoms: ArrayLike | None,
    seed: int | np.random.Generator | None,
) -> np.ndarray:
    """The indices of count atoms at random, or of the given atoms."""
    n = len(system)
    if (count is None) == (atoms is None):
        raise InvalidParameter("give either count or atoms")
    if atoms is not None:
        chosen = np.asarray(atoms)
        if chosen.dtype == bool:
            if chosen.shape != (n,):
                raise InvalidParameter(f"a mask of atoms must have {n} entries")
            return np.flatnonzero(chosen)
        chosen = chosen.astype(np.int64).reshape(-1)
        if ((chosen < 0) | (chosen >= n)).any() or len(np.unique(chosen)) != len(
            chosen
        ):
            raise InvalidParameter("atoms must be distinct indices of the system")
        return chosen
    if (
        isinstance(count, bool)
        or not isinstance(count, int | np.integer)
        or not 0 <= count <= n
    ):
        raise InvalidParameter(f"count must be an integer from 0 to {n}, not {count!r}")
    return np.random.default_rng(seed).choice(n, size=int(count), replace=False)


def add_vacancies(
    system: System,
    count: int | None = None,
    *,
    atoms: ArrayLike | None = None,
    seed: int | np.random.Generator | None = None,
) -> System:
    """The system without some of its atoms.

    Parameters
    ----------
    system : System
        The atoms.
    count : int, optional
        Number of atoms removed at random.
    atoms : array_like, optional
        Instead of count, the indices of the atoms removed, or a boolean mask.
    seed : int or numpy.random.Generator, optional
        Seed of the random choice.

    Returns
    -------
    System
        The other atoms, in their order.
    """
    keep = np.ones(len(system), dtype=bool)
    keep[_chosen(system, count, atoms, seed)] = False
    return system.filter(keep)


def substitute(
    system: System,
    element: str,
    count: int | None = None,
    *,
    atoms: ArrayLike | None = None,
    seed: int | np.random.Generator | None = None,
) -> System:
    """The system with some of its atoms of another element.

    Parameters
    ----------
    system : System
        The atoms.
    element : str
        The new element.
    count : int, optional
        Number of atoms changed at random.
    atoms : array_like, optional
        Instead of count, the indices of the atoms changed, or a boolean mask.
    seed : int or numpy.random.Generator, optional
        Seed of the random choice.

    Returns
    -------
    System
        The same atoms, those chosen of the element.
    """
    if element not in SYMBOLS[1:]:
        raise InvalidParameter(f"unknown element {element!r}")
    chosen = _chosen(system, count, atoms, seed)
    if "element" in system.table.columns:
        elements = np.array(system.table["element"].to_list(), dtype=object)
    else:
        elements = np.full(len(system), "X", dtype=object)
    elements[chosen] = element
    return system.with_columns(element=elements.tolist())


def add_interstitials(
    system: System,
    element: str,
    count: int | None = None,
    *,
    positions: ArrayLike | None = None,
    min_distance: float = 1.0,
    seed: int | np.random.Generator | None = None,
    attempts: int = 1000,
) -> System:
    """The system with atoms added at given or random positions.

    Parameters
    ----------
    system : System
        The atoms.
    element : str
        The element of the added atoms.
    count : int, optional
        Number of atoms added at random positions in the box, each at least
        min_distance from every other atom.
    positions : array_like, optional
        Instead of count, the (M, 3) positions of the added atoms in Å.
    min_distance : float, optional
        With count, the smallest distance in Å to the other atoms. Default 1.
    seed : int or numpy.random.Generator, optional
        Seed of the random positions.
    attempts : int, optional
        With count, the number of random positions tried for each atom
        before giving up. Default 1000.

    Returns
    -------
    System
        The atoms of the system followed by the added ones; columns other
        than the positions and the elements are those of the system, the
        added atoms taking the values of the first atom.
    """
    if element not in SYMBOLS[1:]:
        raise InvalidParameter(f"unknown element {element!r}")
    box = system.box
    if (count is None) == (positions is None):
        raise InvalidParameter("give either count or positions")
    if positions is not None:
        added = np.asarray(positions, dtype=float).reshape(-1, 3)
    else:
        if (
            isinstance(count, bool)
            or not isinstance(count, int | np.integer)
            or count < 0
        ):
            raise InvalidParameter(
                f"count must be a non-negative integer, not {count!r}"
            )
        if not min_distance >= 0:
            raise InvalidParameter(
                f"min_distance must not be negative, not {min_distance!r}"
            )
        rng = np.random.default_rng(seed)
        frame = (
            np.ascontiguousarray(box.matrix),
            np.ascontiguousarray(box.origin),
            np.ascontiguousarray(box.pbc),
        )
        tree = (
            _search.KdTree(np.ascontiguousarray(system.positions), *frame)
            if len(system)
            else None
        )
        found: list[np.ndarray] = []
        for _ in range(int(count)):
            for _ in range(attempts):
                candidate = box.to_cartesian(rng.random((1, 3)))
                if tree is not None:
                    _, _, d = tree.nearest(np.ascontiguousarray(candidate), 1)
                    if d[0, 0] < min_distance:
                        continue
                if found:
                    others = np.array(found)
                    step = box.to_fractional(others) - box.to_fractional(candidate)
                    step[:, box.pbc] -= np.round(step[:, box.pbc])
                    if np.linalg.norm(step @ box.matrix, axis=1).min() < min_distance:
                        continue
                found.append(candidate[0])
                break
            else:
                raise InvalidParameter(
                    f"found no place {min_distance} Å from every atom after {attempts} tries"
                )
        added = np.array(found).reshape(-1, 3)
    if not len(system):
        return System(added, box, elements=[element] * len(added))
    table = system.table
    if "element" not in table.columns:
        system = system.with_columns(element=["X"] * len(system))
    # rows like the first atom's, with the new positions, element and ids
    rows = System(table=system.table[[0] * len(added)], box=box)
    extra = rows.with_positions(added).with_columns(element=[element] * len(added))
    if "id" in table.columns:
        first = int(table["id"].max()) + 1
        extra = extra.with_columns(id=np.arange(first, first + len(added)))
    return System(table=system.table.vstack(extra.table), box=box, info=system.info)


def _axis(axis: int | str) -> int:
    if isinstance(axis, str):
        if axis not in _AXES:
            raise InvalidParameter(f"axis must be x, y, z or 0, 1, 2, not {axis!r}")
        return _AXES[axis]
    if axis not in (0, 1, 2):
        raise InvalidParameter(f"axis must be x, y, z or 0, 1, 2, not {axis!r}")
    return int(axis)


def slip(
    system: System,
    displacement: ArrayLike,
    *,
    height: float | None = None,
    axis: int | str = 2,
) -> System:
    """The system with the atoms above a plane moved, such as to make a
    stacking fault.

    Parameters
    ----------
    system : System
        The atoms.
    displacement : array_like
        (3,) displacement in Å of the atoms above the plane, along it.
    height : float, optional
        The plane, at this fractional coordinate along the box vector of
        axis. Default 0.5, the middle of the box.
    axis : int or str, optional
        The box vector normal to the plane, 0, 1, 2 or x, y, z. Default z.

    Returns
    -------
    System
        The atoms, those above the plane moved. Along a periodic axis the box
        vector moves with them, so that the box holds the one plane of slip.
    """
    k = _axis(axis)
    box = system.box
    shift = np.asarray(displacement, dtype=float)
    if shift.shape != (3,):
        raise InvalidParameter("displacement must have three components")
    others = [i for i in range(3) if i != k]
    normal = np.cross(box.matrix[others[0]], box.matrix[others[1]])
    if abs(shift @ normal) > 1e-9 * np.linalg.norm(shift) * np.linalg.norm(normal):
        raise InvalidParameter("displacement must lie in the plane")
    height = 0.5 if height is None else float(height)
    fractional = box.to_fractional(system.positions)[:, k]
    above = (
        fractional - np.floor(fractional) >= height
        if box.pbc[k]
        else fractional >= height
    )
    positions = system.positions.copy()
    positions[above] += shift
    matrix = box.matrix.copy()
    if box.pbc[k]:
        matrix[k] += shift
    moved = system.with_box(Box(matrix, origin=box.origin, pbc=box.pbc))
    return moved.with_positions(positions)


def add_vacuum(system: System, thickness: float, *, axis: int | str = 2) -> System:
    """The system in a box longer along one of its vectors, such as for a
    slab with two surfaces.

    Parameters
    ----------
    system : System
        The atoms.
    thickness : float
        Length in Å added to the box along the vector of axis.
    axis : int or str, optional
        The box vector made longer, 0, 1, 2 or x, y, z. Default z.

    Returns
    -------
    System
        The same atoms, unwrapped across the boundary of that axis so that
        none is cut off, in the longer box; the periodicity is kept.
    """
    k = _axis(axis)
    if not thickness >= 0:
        raise InvalidParameter(f"thickness must not be negative, not {thickness!r}")
    box = system.box
    vector = box.matrix[k]
    matrix = box.matrix.copy()
    matrix[k] = vector * (1 + thickness / np.linalg.norm(vector))
    # every atom with its fractional coordinate along axis in [0, 1)
    fractional = box.to_fractional(system.positions)
    positions = system.positions - np.floor(fractional[:, k])[:, None] * vector
    longer = system.with_box(Box(matrix, origin=box.origin, pbc=box.pbc))
    return longer.with_positions(positions)
