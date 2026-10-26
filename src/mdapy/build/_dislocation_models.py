# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Models of straight dislocations in cubic crystals from anisotropic
elasticity: a cylinder with a fixed shell, and a periodic quadrupole."""

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mdapy.build._anisotropic import StraightDislocation, rotated, stiffness
from mdapy.build._crystal import STRUCTURES, build_crystal
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.errors import InvalidParameter

# periodic images of the dipole summed along each box vector, both ways
_IMAGES = 10


@dataclass(frozen=True)
class Dislocation:
    """A straight dislocation in a cubic crystal.

    Attributes
    ----------
    structure : str
        A cubic structure of :func:`build_crystal`, such as "bcc", "fcc" or
        "diamond".
    orientation : tuple of three directions
        The crystal directions along x, y and z, orthogonal and
        right-handed: x in the glide plane, y normal to it, z along the
        line.
    burgers : tuple of float
        The Burgers vector in the axes of the cubic cell, in units of the
        lattice constant, in the glide plane.
    partial : bool
        A single Shockley partial, which bounds a stacking fault on its -x
        side. Default False.
    core : tuple of float
        Where the line lies: (s, t), s rows of atoms along x from a row of
        the plane of atoms below it and t of the way to the plane above.
        Default (0.5, 0.5).
    """

    structure: str
    orientation: tuple
    burgers: tuple
    partial: bool = False
    core: tuple = (0.5, 0.5)

    def __post_init__(self):
        if self.structure not in STRUCTURES or STRUCTURES[self.structure].hexagonal:
            raise InvalidParameter(
                f"structure must be a cubic structure, not {self.structure!r}"
            )
        axes = np.asarray(self.orientation, dtype=float)
        if axes.shape != (3, 3):
            raise InvalidParameter("orientation must be three directions")
        unit = axes / np.linalg.norm(axes, axis=1)[:, None]
        if (
            not np.allclose(unit @ unit.T, np.eye(3), atol=1e-9)
            or np.linalg.det(unit) < 0
        ):
            raise InvalidParameter("orientation must be orthogonal and right-handed")
        b = np.asarray(self.burgers, dtype=float)
        if b.shape != (3,) or not np.linalg.norm(b) > 0:
            raise InvalidParameter("burgers must be a nonzero vector of three numbers")
        if abs(b @ unit[1]) > 1e-9:
            raise InvalidParameter("burgers must lie in the glide plane, normal to y")


def _dislocations() -> dict[str, Dislocation]:
    bcc_plane = (-1, 1, 0)
    fcc_screw = ((1, 1, -2), (1, 1, 1), (1, -1, 0))
    return {
        "bcc screw": Dislocation(
            "bcc",
            ((1, 1, -2), bcc_plane, (1, 1, 1)),
            (0.5, 0.5, 0.5),
            core=(0.5, 1 / 3),
        ),
        "bcc edge": Dislocation(
            "bcc", ((1, 1, 1), bcc_plane, (-1, -1, 2)), (0.5, 0.5, 0.5)
        ),
        "bcc mixed": Dislocation(
            "bcc", ((-1, -1, -2), bcc_plane, (1, 1, -1)), (0.5, 0.5, 0.5)
        ),
        "bcc <100> edge": Dislocation(
            "bcc", ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (1.0, 0.0, 0.0)
        ),
        "bcc <100>{110} edge": Dislocation(
            "bcc", ((1, 0, 0), (0, 1, 1), (0, -1, 1)), (1.0, 0.0, 0.0)
        ),
        "fcc screw": Dislocation("fcc", fcc_screw, (0.5, -0.5, 0.0)),
        "fcc edge": Dislocation(
            "fcc", ((1, -1, 0), (1, 1, 1), (-1, -1, 2)), (0.5, -0.5, 0.0)
        ),
        "diamond screw": Dislocation("diamond", fcc_screw, (0.5, -0.5, 0.0)),
        "diamond 60": Dislocation("diamond", fcc_screw, (0.5, 0.0, -0.5)),
        "diamond 30 partial": Dislocation(
            "diamond", fcc_screw, (1 / 6, -1 / 3, 1 / 6), partial=True
        ),
        "diamond 90 partial": Dislocation(
            "diamond", fcc_screw, (1 / 6, 1 / 6, -1 / 3), partial=True
        ),
    }


DISLOCATIONS: dict[str, Dislocation] = _dislocations()


def _lookup(dislocation: Any) -> Dislocation:
    if isinstance(dislocation, Dislocation):
        return dislocation
    if dislocation not in DISLOCATIONS:
        raise InvalidParameter(
            f"dislocation must be a Dislocation or one of {tuple(DISLOCATIONS)}, "
            f"not {dislocation!r}"
        )
    return DISLOCATIONS[dislocation]


def _fault_vectors(structure: str, normal: np.ndarray) -> np.ndarray:
    """The shifts of the crystal above a {111} plane that make an intrinsic
    stacking fault, in units of a: the nearest neighbors above the plane
    projected on it."""
    if structure not in ("fcc", "diamond"):
        raise InvalidParameter("only fcc and diamond dislocations split into partials")
    n = normal / np.linalg.norm(normal)
    neighbors = np.array(
        [v for v in product((-0.5, 0.0, 0.5), repeat=3) if np.count_nonzero(v) == 2]
    )
    up = neighbors[neighbors @ n > 1e-9]
    if len(up) != 3:
        raise InvalidParameter("partials need a {111} glide plane")
    return up - np.outer(up @ n, n)


def _partials(spec: Dislocation, sign: int) -> list[np.ndarray]:
    """The Burgers vectors, in units of a and the cubic axes, of the
    Shockley partials of sign times the dislocation, from -x to +x."""
    b = sign * np.asarray(spec.burgers, dtype=float)
    normal = np.asarray(spec.orientation[1], dtype=float)
    faults = _fault_vectors(spec.structure, normal)
    size = np.linalg.norm(faults[0])
    for f in faults:
        rest = b - f
        if abs(np.linalg.norm(rest) - size) < 1e-9:
            return [rest, f]
    raise InvalidParameter(
        "the Burgers vector does not split into two Shockley partials"
    )


def _levels(values: np.ndarray, period: float) -> np.ndarray:
    """The distinct values modulo period, in order, the same within 1e-6."""
    v = np.round(np.mod(values, period), 6)
    return np.unique(np.where(v >= np.round(period, 6), 0.0, v))


class _Setup:
    """A dislocation in its oriented cell: the frame, the elastic field, the
    sites of its lines and the spacing of the rows along x."""

    def __init__(self, spec, a, elastic, elements):
        self.spec = spec
        self.a = float(a)
        if not self.a > 0:
            raise InvalidParameter(f"a must be positive, not {a!r}")
        axes = np.asarray(spec.orientation, dtype=float)
        self.frame = axes / np.linalg.norm(axes, axis=1)[:, None]
        self.field = StraightDislocation(rotated(stiffness(elastic), self.frame))
        self.elements = elements
        self.cell = build_crystal(
            spec.structure, self.a, elements=elements, orientation=spec.orientation
        )
        self.size = self.cell.box.matrix.diagonal()
        x, y = (self.cell["positions"] % self.size)[:, :2].T
        planes = _levels(y, self.size[1])
        gaps = np.diff(np.append(planes, planes[0] + self.size[1]))
        # the glide set of diamond between the planes closest together
        k = int(np.argmin(gaps)) if spec.structure == "diamond" else 0
        self.low, self.gap = planes[k], gaps[k]
        on = np.isclose(y, self.low, atol=1e-6) | np.isclose(
            y, self.low + self.size[1], atol=1e-6
        )
        rows = _levels(x[on], self.size[0])
        self.row, self.spacing = rows[0], self.size[0] / len(rows)

    def crystal(self, repeat) -> System:
        return build_crystal(
            self.spec.structure,
            self.a,
            elements=self.elements,
            orientation=self.spec.orientation,
            repeat=repeat,
        )

    def site(self, sign: int) -> np.ndarray:
        """(x, y) of the line of sign times the dislocation in the first
        cell: the opposite one in the mirrored site."""
        s, t = self.spec.core
        if sign < 0:
            s, t = s + 0.5, 1 - t
        return np.array([self.row + s * self.spacing, self.low + t * self.gap])

    def lines(self, sign: int, at: np.ndarray, rows: int) -> list:
        """The lines of sign times the dislocation from the site at: (x, y,
        Burgers vector in Å in the frame), from -x to +x."""
        if self.spec.partial:
            vectors = [sign * np.asarray(self.spec.burgers, dtype=float)]
        elif rows > 0:
            vectors = _partials(self.spec, sign)
        else:
            vectors = [sign * np.asarray(self.spec.burgers, dtype=float)]
        return [
            (at + [k * rows * self.spacing, 0.0], self.a * self.frame @ v)
            for k, v in enumerate(vectors)
        ]

    def rows(self, dissociation: float) -> int:
        d = float(dissociation)
        if not d >= 0:
            raise InvalidParameter(
                f"dissociation must be at least 0, not {dissociation!r}"
            )
        if d > 0 and self.spec.partial:
            raise InvalidParameter("a partial dislocation does not split")
        return int(round(d / self.spacing))


def dislocation_cylinder(
    dislocation: str | Dislocation,
    a: float,
    elastic: ArrayLike,
    radius: float,
    *,
    elements: str | tuple = "X",
    length: int = 1,
    fixed: float = 6.0,
    dissociation: float = 0.0,
) -> System:
    """A cylinder of crystal around a straight dislocation along its axis,
    displaced as anisotropic elasticity predicts, the atoms of its outer
    shell fixed.

    Parameters
    ----------
    dislocation : str or Dislocation
        The dislocation: a name of ``mdapy.build.DISLOCATIONS`` or one of
        your own.
    a : float
        Lattice constant in Å.
    elastic : array_like
        The elastic constants in the axes of the cubic cell: (C11, C12, C44)
        or the (6, 6) Voigt matrix, in any unit.
    radius : float
        Radius of the cylinder in Å.
    elements : str or sequence of str, optional
        As for :func:`build_crystal`. Default "X".
    length : int, optional
        Number of periods of the crystal along the line. Default 1.
    fixed : float, optional
        Thickness in Å of the outer shell whose atoms are fixed. Default 6.
    dissociation : float, optional
        Distance in Å between the two Shockley partials of an fcc or diamond
        dislocation, rounded to whole rows of atoms; 0 for a perfect one.
        Default 0.

    Returns
    -------
    System
        The cylinder along z in a box periodic along z only, the line at
        the middle of x and y; ``movable`` is False for the atoms of the
        fixed shell.
    """
    setup = _Setup(_lookup(dislocation), a, elastic, elements)
    radius = float(radius)
    if not radius > 0:
        raise InvalidParameter(f"radius must be positive, not {radius!r}")
    if not float(fixed) >= 0:
        raise InvalidParameter(f"fixed must be at least 0, not {fixed!r}")
    if isinstance(length, bool) or int(length) != length or length < 1:
        raise InvalidParameter(f"length must be a positive integer, not {length!r}")
    rows = setup.rows(dissociation)
    width = rows * setup.spacing
    count = np.ceil((2 * radius + width) / setup.size[:2]).astype(int) + 2
    crystal = setup.crystal((count[0], count[1], int(length)))
    at = (
        setup.site(1)
        + (count[:2] // 2 - [width / 2 // setup.size[0], 0]) * setup.size[:2]
    )
    lines = setup.lines(1, at, rows)
    middle = at + [width / 2, 0.0]
    distance = np.linalg.norm(crystal["positions"][:, :2] - middle, axis=1)
    keep = distance <= radius
    crystal = crystal.filter(keep)
    reference = crystal["positions"]
    positions = reference.copy()
    for point, burgers in lines:
        x = reference[:, :2] - point
        positions += setup.field(x[:, 0], x[:, 1], burgers)
    movable = np.repeat((distance[keep] <= radius - float(fixed))[:, None], 3, axis=1)
    side = 2 * radius + width + 10.0
    lz = crystal.box.matrix[2, 2]
    corner = np.array([middle[0] - side / 2, middle[1] - side / 2, 0.0])
    box = Box(np.diag([side, side, lz]), origin=corner, pbc=(False, False, True))
    return crystal.with_box(box).with_columns(positions=positions, movable=movable)


def _segments(lines: list) -> list:
    """The cuts between consecutive lines of zero total Burgers vector: from
    point to point, with the Burgers vector of all lines before."""
    segments, total = [], np.zeros(3)
    for (start, burgers), (end, _) in zip(lines[:-1], lines[1:]):
        total = total + burgers
        segments.append((start, end, total.copy()))
    return segments


def _summed(field, points, segments, c1, c2):
    """The displacements at points (N, 2) of the segments and their images
    within _IMAGES box vectors c1, c2 (2,) both ways."""
    u = np.zeros((len(points), 3))
    for start, end, burgers in segments:
        d = end - start
        cut = np.arctan2(d[1], d[0])
        for m in range(-_IMAGES, _IMAGES + 1):
            for n in range(-_IMAGES, _IMAGES + 1):
                x = points - m * c1 - n * c2
                p, q = x - start, x - end
                u += field(p[:, 0], p[:, 1], burgers, cut)
                u -= field(q[:, 0], q[:, 1], burgers, cut)
    return u


def dislocation_quadrupole(
    dislocation: str | Dislocation,
    a: float,
    elastic: ArrayLike,
    repeat: tuple[int, int, int],
    *,
    elements: str | tuple = "X",
    dissociation: float = 0.0,
) -> System:
    """A periodic crystal with a dipole of straight dislocations of opposite
    Burgers vectors, whose periodic images make a quadrupolar lattice,
    displaced as anisotropic elasticity predicts.

    Parameters
    ----------
    dislocation : str or Dislocation
        The dislocation: a name of ``mdapy.build.DISLOCATIONS`` or one of
        your own, not a partial.
    a : float
        Lattice constant in Å.
    elastic : array_like
        The elastic constants in the axes of the cubic cell: (C11, C12, C44)
        or the (6, 6) Voigt matrix, in any unit.
    repeat : tuple of int
        Periods of the oriented crystal along x, y and z, at least 2 along x
        and y.
    elements : str or sequence of str, optional
        As for :func:`build_crystal`. Default "X".
    dissociation : float, optional
        Distance in Å between the two Shockley partials of each fcc or
        diamond dislocation, rounded to whole rows of atoms; 0 for perfect
        ones. Default 0.

    Returns
    -------
    System
        The crystal in a periodic box tilted by the plastic strain of the
        dipole: the dislocation at a quarter of x and y, the opposite one
        half a box away along both. With an edge part, the slab of atoms
        between their glide planes that the cut sweeps is taken away.
    """
    spec = _lookup(dislocation)
    if spec.partial:
        raise InvalidParameter("a quadrupole needs a perfect dislocation")
    setup = _Setup(spec, a, elastic, elements)
    counts = np.asarray(repeat)
    if (
        counts.shape != (3,)
        or not np.issubdtype(counts.dtype, np.integer)
        or (counts < 1).any()
        or (counts[:2] < 2).any()
    ):
        raise InvalidParameter(
            f"repeat must be three positive integers, at least 2 along x and y, not {repeat!r}"
        )
    rows = setup.rows(dissociation)
    crystal = setup.crystal(tuple(int(n) for n in counts))
    cell = setup.size[:2]
    c1 = np.array([crystal.box.matrix[0, 0], 0.0])
    c2 = np.array([0.0, crystal.box.matrix[1, 1]])
    burgers = setup.a * setup.frame @ np.asarray(spec.burgers, dtype=float)
    # the opposite dislocation half a box away along x, and along y the
    # way that makes the climb of the cut take atoms away, not add them
    half = (counts[:2] // 2) * cell * [1, -1 if burgers[0] > 0 else 1]
    first = setup.site(1) + (counts[:2] // 4) * cell
    second = setup.site(-1) + (counts[:2] // 4) * cell + half
    plus, minus = setup.lines(1, first, rows), setup.lines(-1, second, rows)
    # the cuts: the faults of the partials, the glide plane of the first,
    # across the planes between atoms, then the glide plane of the second
    x = _levels(crystal["positions"][:, 0], cell[0])
    middles = x + np.diff(np.append(x, x[0] + cell[0])) / 2
    target = (plus[-1][0][0] + minus[0][0][0]) / 2
    choices = np.floor(target / cell[0]) * cell[0] + np.concatenate(
        [middles - cell[0], middles, middles + cell[0]]
    )
    across = choices[np.argmin(np.abs(choices - target))]
    bends = [np.array([across, plus[-1][0][1]]), np.array([across, minus[0][0][1]])]
    segments = _segments(plus + [(bend, np.zeros(3)) for bend in bends] + minus)
    if abs(burgers[0]) > 1e-9 * np.linalg.norm(burgers):
        # the slab the cut across the planes sweeps
        reference = crystal["positions"][:, :2]
        low, high = sorted([bends[0][1], bends[1][1]])
        inside = ((reference[:, 0] - across) % c1[0] < abs(burgers[0])) & (
            (reference[:, 1] - low) % c2[1] < high - low
        )
        crystal = crystal.filter(~inside)
    area = c1[0] * c2[1]
    # the box takes the plastic strain of the cuts, crossed counterclockwise
    tilts = []
    for c in (c1, c2):
        tilt = np.zeros(3)
        for start, end, vector in segments:
            d = end - start
            tilt -= vector * (-d[1] * c[0] + d[0] * c[1]) / area
        tilts.append(tilt)
    # the sum of images is periodic only up to a linear term, set by
    # matching the tilts at points of the box
    rng = np.random.default_rng(20261027)
    probes = rng.uniform(0, 1, (8, 2)) @ np.array([c1, c2])
    base = _summed(setup.field, probes, segments, c1, c2)
    steps = [
        (_summed(setup.field, probes + c, segments, c1, c2) - base).mean(axis=0) - tilt
        for c, tilt in zip((c1, c2), tilts)
    ]
    gradient = np.column_stack(steps) @ np.linalg.inv(np.column_stack([c1, c2]))
    positions = crystal["positions"].copy()
    xy = positions[:, :2]
    positions += _summed(setup.field, xy, segments, c1, c2) - xy @ gradient.T
    matrix = crystal.box.matrix.copy()
    matrix[0] += tilts[0]
    matrix[1] += tilts[1]
    return crystal.with_box(Box(matrix, origin=crystal.box.origin)).with_columns(
        positions=positions
    )
