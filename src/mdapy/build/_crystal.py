# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Perfect crystals, along any crystal directions."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import gcd, sqrt

import numpy as np
import polars as pl

from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.elements import ELEMENT_DTYPE
from mdapy.errors import InvalidParameter


@dataclass(frozen=True)
class _Structure:
    """A crystal structure: its cell, the fractional positions of its atoms,
    the sublattice of each, and the numbers of elements it takes."""

    hexagonal: bool
    basis: tuple[tuple[float, float, float], ...]
    sublattice: tuple[int, ...]
    elements: tuple[int, ...]
    c_ratio: float | None = None  # default c / a of a hexagonal structure


_T = 1 / 3
_FCC = ((0, 0, 0), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5))
_TETRAHEDRAL = ((0.25, 0.25, 0.25), (0.75, 0.75, 0.25), (0.75, 0.25, 0.75), (0.25, 0.75, 0.75))  # fmt: skip
_IDEAL = sqrt(8 / 3)

STRUCTURES = {
    "sc": _Structure(False, ((0, 0, 0),), (0,), (1,)),
    "bcc": _Structure(False, ((0, 0, 0), (0.5, 0.5, 0.5)), (0, 1), (1, 2)),
    "fcc": _Structure(False, _FCC, (0, 0, 1, 1), (1, 2)),
    "diamond": _Structure(False, _FCC + _TETRAHEDRAL, (0,) * 4 + (1,) * 4, (1, 2)),
    "cscl": _Structure(False, ((0, 0, 0), (0.5, 0.5, 0.5)), (0, 1), (2,)),
    "rocksalt": _Structure(
        False,
        _FCC + ((0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5), (0.5, 0.5, 0.5)),
        (0,) * 4 + (1,) * 4,
        (2,),
    ),
    "zincblende": _Structure(False, _FCC + _TETRAHEDRAL, (0,) * 4 + (1,) * 4, (2,)),
    "fluorite": _Structure(
        False,
        _FCC
        + tuple(
            (x, y, z) for z in (0.25, 0.75) for y in (0.25, 0.75) for x in (0.25, 0.75)
        ),
        (0,) * 4 + (1,) * 8,
        (2,),
    ),  # fmt: skip
    "l1_2": _Structure(
        False,
        ((0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0, 0, 0)),
        (0, 0, 0, 1),
        (2,),
    ),  # fmt: skip
    "perovskite": _Structure(
        False,
        ((0.5, 0.5, 0.5), (0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)),
        (0, 1, 2, 2, 2),
        (3,),
    ),
    "hcp": _Structure(True, ((0, 0, 0), (_T, 2 * _T, 0.5)), (0, 1), (1, 2), _IDEAL),
    "wurtzite": _Structure(
        True,
        ((_T, 2 * _T, 0), (2 * _T, _T, 0.5), (_T, 2 * _T, 3 / 8), (2 * _T, _T, 7 / 8)),
        (0, 0, 1, 1),
        (1, 2),
        _IDEAL,
    ),
    "lonsdaleite": _Structure(
        True,
        ((_T, 2 * _T, 0), (2 * _T, _T, 0.5), (_T, 2 * _T, 3 / 8), (2 * _T, _T, 7 / 8)),
        (0, 0, 0, 0),
        (1,),
        _IDEAL,
    ),
    "graphite": _Structure(
        True,
        ((0, 0, 0), (0, 0, 0.5), (_T, 2 * _T, 0), (2 * _T, _T, 0.5)),
        (0, 1, 0, 1),
        (1, 2),
    ),  # fmt: skip
    "graphene": _Structure(True, ((0, 0, 0), (_T, 2 * _T, 0)), (0, 1), (1, 2)),
}


def _cell(structure: _Structure, a: float, c: float) -> np.ndarray:
    """The lattice vectors of the conventional cell, as rows."""
    if structure.hexagonal:
        return np.array([[a, 0, 0], [-0.5 * a, 0.5 * sqrt(3) * a, 0], [0, 0, c]])
    return a * np.eye(3)


def _directions(orientation: Sequence[Sequence[int]], hexagonal: bool) -> np.ndarray:
    """(3, 3) integer directions in units of the cell vectors, each reduced
    to coprime integers; four-index hexagonal directions [hkil] become
    [2h + k, h + 2k, l]."""
    try:
        rows = [tuple(direction) for direction in orientation]
    except TypeError:
        rows = []
    if len(rows) != 3:
        raise InvalidParameter("orientation must be three directions")
    directions = []
    for row in rows:
        try:
            values = np.array(row, dtype=float)
        except (TypeError, ValueError):
            raise InvalidParameter(f"{row!r} is not a direction of integers") from None
        if values.ndim != 1 or not np.array_equal(values, np.round(values)):
            raise InvalidParameter(f"{row!r} is not a direction of integers")
        v = [int(x) for x in values]
        if hexagonal and len(v) == 4:
            h, k, i, w = v
            if h + k + i != 0:
                raise InvalidParameter(f"the direction {row!r} needs h + k + i = 0")
            v = [2 * h + k, h + 2 * k, w]
        elif len(v) != 3:
            kind = "three or four" if hexagonal else "three"
            raise InvalidParameter(f"a direction has {kind} indices, not {row!r}")
        divisor = gcd(*map(abs, v))
        if divisor == 0:
            raise InvalidParameter("a direction must not be zero")
        directions.append([x // divisor for x in v])
    return np.array(directions, dtype=float)


def _is_translation(
    shift: np.ndarray, basis: np.ndarray, sublattice: np.ndarray
) -> bool:
    """Whether a shift, in fractional coordinates of the cell, maps every
    atom onto an atom of the same sublattice."""
    moved = basis + shift
    difference = moved[:, None, :] - basis[None, :, :]
    same = (np.abs(difference - np.round(difference)) < 1e-8).all(axis=2)
    return bool((same & (sublattice[:, None] == sublattice[None, :])).any(axis=1).all())


def _oriented(
    cell: np.ndarray,
    basis: np.ndarray,
    sublattice: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cell spanned by the shortest translations along the directions:
    its vectors as rows, and the fractional positions and sublattices of the
    atoms inside it."""
    vectors = directions @ cell
    if abs(np.linalg.det(vectors)) < 1e-9 * np.linalg.norm(vectors) ** 3:
        raise InvalidParameter("the directions must be linearly independent")
    for k in range(3):
        # the shortest translation along the direction
        for parts in range(12, 1, -1):
            if _is_translation(directions[k] / parts, basis, sublattice):
                vectors[k] /= parts
                break
    # the lattice points of the cells around the new one, kept inside it
    inverse = np.linalg.inv(vectors)
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    span = corners @ vectors @ np.linalg.inv(cell)
    low = np.floor(span.min(axis=0)) - 1
    high = np.ceil(span.max(axis=0)) + 1
    cells = np.stack(
        np.meshgrid(
            *[np.arange(lo, hi + 1) for lo, hi in zip(low, high)], indexing="ij"
        ),
        -1,
    ).reshape(-1, 1, 3)
    fractional = ((cells + basis) @ cell @ inverse).reshape(-1, 3)
    kinds = np.broadcast_to(sublattice, cells.shape[:1] + sublattice.shape).reshape(-1)
    inside = ((fractional > -1e-9) & (fractional < 1 - 1e-9)).all(axis=1)
    fractional = np.clip(fractional[inside], 0.0, None)
    kinds = kinds[inside]
    expected = round(abs(np.linalg.det(vectors) / np.linalg.det(cell)) * len(basis))
    if len(fractional) != expected:  # pragma: no cover - a guard on the geometry
        raise RuntimeError(f"found {len(fractional)} atoms instead of {expected}")
    return vectors, fractional, kinds


def _upright(vectors: np.ndarray) -> np.ndarray:
    """Lattice vectors of the same lengths and angles with the first along x
    and the second in the xy plane, right-handed."""
    a, b, c = np.linalg.norm(vectors, axis=1)
    cos_gamma = vectors[0] @ vectors[1] / (a * b)
    cos_beta = vectors[0] @ vectors[2] / (a * c)
    cos_alpha = vectors[1] @ vectors[2] / (b * c)
    sin_gamma = sqrt(1 - cos_gamma**2)
    upright = np.zeros((3, 3))
    upright[0, 0] = a
    upright[1, :2] = b * cos_gamma, b * sin_gamma
    upright[2, 0] = c * cos_beta
    upright[2, 1] = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    upright[2, 2] = sqrt(max(c * c - upright[2, 0] ** 2 - upright[2, 1] ** 2, 0.0))
    upright[np.abs(upright) < 1e-12 * max(a, b, c)] = 0.0
    return upright


def build_crystal(
    structure: str,
    a: float,
    *,
    c: float | None = None,
    elements: str | Sequence[str] = "X",
    orientation: Sequence[Sequence[int]] | None = None,
    repeat: Sequence[int] = (1, 1, 1),
) -> System:
    """A perfect crystal in a periodic box, optionally along chosen crystal
    directions.

    Parameters
    ----------
    structure : str
        One of ``mdapy.build.STRUCTURES``: "sc", "bcc", "fcc", "diamond",
        "cscl", "rocksalt", "zincblende", "fluorite", "l1_2", "perovskite",
        "hcp", "wurtzite", "lonsdaleite", "graphite" or "graphene".
    a : float
        Lattice constant in Å.
    c : float, optional
        Lattice constant c of a hexagonal structure in Å; by default the
        ideal sqrt(8/3) a, and needed for graphite and graphene.
    elements : str or sequence of str, optional
        The element of every atom, or of each sublattice of the structure.
        Default "X".
    orientation : sequence of three directions, optional
        The crystal directions along the three box vectors: [hkl] in the
        cubic cell, orthogonal; or [uvw] or [hkil] in the hexagonal cell,
        linearly independent. Default the cell itself.
    repeat : sequence of int, optional
        Number of periods along each box vector. Default (1, 1, 1).

    Returns
    -------
    System
        The crystal, ordered by z, then y, then x.
    """
    if structure not in STRUCTURES:
        raise InvalidParameter(
            f"structure must be one of {tuple(STRUCTURES)}, not {structure!r}"
        )
    kind = STRUCTURES[structure]
    if not a > 0:
        raise InvalidParameter(f"a must be positive, not {a!r}")
    if kind.hexagonal:
        if c is None and kind.c_ratio is None:
            raise InvalidParameter(f"{structure} needs c")
        c = kind.c_ratio * a if c is None else c
        if not c > 0:
            raise InvalidParameter(f"c must be positive, not {c!r}")
    elif c is not None:
        raise InvalidParameter("c is for hexagonal structures only")
    names = (elements,) if isinstance(elements, str) else tuple(elements)
    allowed = sorted({1, *kind.elements})
    if len(names) not in allowed:
        raise InvalidParameter(
            f"{structure} takes {' or '.join(map(str, allowed))} elements, "
            f"not {len(names)}"
        )
    counts = np.array(repeat)
    if (
        counts.shape != (3,)
        or not np.issubdtype(counts.dtype, np.integer)
        or (counts < 1).any()
    ):
        raise InvalidParameter(
            f"repeat must be three positive integers, not {repeat!r}"
        )

    cell = _cell(kind, a, c)
    basis = np.array(kind.basis, dtype=float)
    sublattice = (
        np.array(kind.sublattice) if len(names) > 1 else np.zeros(len(basis), int)
    )
    if orientation is None:
        vectors, fractional, kinds = cell, basis, sublattice
    else:
        directions = _directions(orientation, kind.hexagonal)
        if not kind.hexagonal:
            products = directions @ directions.T
            if np.abs(products - np.diag(np.diag(products))).max() > 0:
                raise InvalidParameter(
                    f"the directions of a cubic crystal must be orthogonal, not "
                    f"{directions.astype(int).tolist()}"
                )
        vectors, fractional, kinds = _oriented(cell, basis, sublattice, directions)
        vectors = _upright(vectors)
    try:
        symbols = pl.Series(names, dtype=pl.String).cast(ELEMENT_DTYPE)
    except pl.exceptions.InvalidOperationError:
        raise InvalidParameter(f"unknown element in {names}") from None
    cells, atoms = _in_order(fractional, counts)
    positions = (cells + fractional[atoms]) @ vectors
    table = pl.DataFrame(
        {
            "positions": pl.Series(positions, dtype=pl.Array(pl.Float64, 3)),
            "element": symbols.gather(kinds[atoms]),
        }
    )
    return System(table=table, box=Box(vectors * counts[:, None]))


def _in_order(
    fractional: np.ndarray, counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The atoms of the crystal of counts cells: the (N, 3) cell of each and
    its atom in the cell, ordered by plane along the third vector, then the
    second, then the first; with the first vector along x and the second in
    the xy plane, by z, then y, then x."""
    planes = np.round(fractional, 9)  # the same plane within rounding
    layers = []  # per plane along the third vector: its rows along the first
    for c in np.unique(planes[:, 2]):
        on = planes[:, 2] == c
        rows = []
        for b in np.unique(planes[on, 1]):
            row = np.flatnonzero(on & (planes[:, 1] == b))
            rows.append(row[np.argsort(planes[row, 0], kind="stable")])
        layers.append(rows)
    n1, n2, n3 = (int(n) for n in counts)
    first, second, atom = [], [], []
    for rows in layers:
        # a plane of one cell's height: for each cell along the second
        # vector, its rows, each through the cells along the first
        line_first = np.concatenate([np.repeat(np.arange(n1), len(r)) for r in rows])
        line_atom = np.concatenate([np.tile(r, n1) for r in rows])
        first.append(np.tile(line_first, n2))
        second.append(np.repeat(np.arange(n2), len(line_first)))
        atom.append(np.tile(line_atom, n2))
    layer = len(fractional) * n1 * n2
    cells = np.empty((layer * n3, 3))
    cells[:, 0] = np.tile(np.concatenate(first), n3)
    cells[:, 1] = np.tile(np.concatenate(second), n3)
    cells[:, 2] = np.repeat(np.arange(n3), layer)
    return cells, np.tile(np.concatenate(atom), n3)
