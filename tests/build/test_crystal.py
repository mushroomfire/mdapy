# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.build import STRUCTURES, build_crystal
from mdapy.errors import InvalidParameter

from atomsk_reference import ATOMSK, atomsk_crystal
from lattices import same_crystal


def neighbors(system, cutoff):
    """For each atom, the distances and elements of its neighbors closer
    than cutoff, nearest first, by brute force over the 27 nearest images
    (the box must be thicker than the cutoff)."""
    shifts = np.array(np.meshgrid(*[[-1, 0, 1]] * 3, indexing="ij")).reshape(3, -1).T
    x = system["positions"]
    d = x[None, :, None] - x[:, None, None] + (shifts @ system.box.matrix)[None, None]
    r = np.linalg.norm(d, axis=-1).reshape(len(system), -1)
    kinds = np.repeat(np.array(system.table["element"].to_list()), len(shifts))
    rows = []
    for row in r:
        keep = (row > 1e-9) & (row < cutoff)
        order = np.argsort(row[keep], kind="stable")
        rows.append((row[keep][order], kinds[keep][order]))
    return rows


# coordination and nearest distance / a of the cubic structures of one element
SHELLS = {"sc": (6, 1.0), "bcc": (8, np.sqrt(3) / 2), "fcc": (12, np.sqrt(0.5)), "diamond": (4, np.sqrt(3) / 4)}  # fmt: skip


@pytest.mark.parametrize("structure", SHELLS)
def test_cubic_crystals(structure):
    a = 3.0
    coordination, nearest = SHELLS[structure]
    system = build_crystal(structure, a, elements="Cu", repeat=(2, 3, 2))
    per_cell = len(STRUCTURES[structure].basis)
    assert len(system) == 12 * per_cell
    np.testing.assert_allclose(system.box.matrix, np.diag([2 * a, 3 * a, 2 * a]))
    for r, _ in neighbors(system, nearest * a * 1.05):
        assert len(r) == coordination
        np.testing.assert_allclose(r, nearest * a, rtol=1e-12)
    assert system.table["element"].to_list() == ["Cu"] * len(system)
    x, y, z = system["positions"].T
    assert (np.lexsort((x, y, z)) == np.arange(len(system))).all()


@pytest.mark.parametrize(
    "structure, elements, first, counts",
    [
        # each element: its nearest neighbors (number, distance / a, element)
        (
            "cscl",
            ("Cs", "Cl"),
            {"Cs": (8, np.sqrt(3) / 2, "Cl"), "Cl": (8, np.sqrt(3) / 2, "Cs")},
            (1, 1),
        ),  # fmt: skip
        (
            "rocksalt",
            ("Na", "Cl"),
            {"Na": (6, 0.5, "Cl"), "Cl": (6, 0.5, "Na")},
            (1, 1),
        ),
        (
            "zincblende",
            ("Ga", "As"),
            {"Ga": (4, np.sqrt(3) / 4, "As"), "As": (4, np.sqrt(3) / 4, "Ga")},
            (1, 1),
        ),  # fmt: skip
        (
            "fluorite",
            ("Ca", "F"),
            {"Ca": (8, np.sqrt(3) / 4, "F"), "F": (4, np.sqrt(3) / 4, "Ca")},
            (1, 2),
        ),  # fmt: skip
        ("l1_2", ("Ni", "Al"), {"Al": (12, np.sqrt(0.5), "Ni")}, (3, 1)),
        (
            "perovskite",
            ("Sr", "Ti", "O"),
            {"Ti": (6, 0.5, "O"), "Sr": (12, np.sqrt(0.5), "O")},
            (1, 1, 3),
        ),  # fmt: skip
        ("bcc", ("Fe", "Al"), {"Fe": (8, np.sqrt(3) / 2, "Al")}, (1, 1)),
        ("diamond", ("Si", "C"), {"Si": (4, np.sqrt(3) / 4, "C")}, (1, 1)),
    ],
)
def test_ordered_crystals(structure, elements, first, counts):
    a = 4.0
    system = build_crystal(structure, a, elements=elements, repeat=(2, 2, 2))
    kinds = system.table["element"].to_list()
    np.testing.assert_array_equal(
        [kinds.count(e) for e in elements],
        np.array(counts) * len(system) // sum(counts),
    )
    for (r, around), kind in zip(neighbors(system, 0.9 * a), kinds):
        if kind in first:
            number, distance, other = first[kind]
            assert (np.abs(r - distance * a) < 1e-9).sum() == number
            assert set(around[np.abs(r - distance * a) < 1e-9]) == {other}
            assert r.min() == pytest.approx(distance * a, rel=1e-12)


def test_fcc_of_two_elements_in_layers():
    system = build_crystal("fcc", 3.6, elements=("Cu", "Au"), repeat=(2, 2, 2))
    kinds = np.array(system.table["element"].to_list())
    # Cu on the planes z = 0, a, ...; Au on z = a / 2, ...
    z = system["positions"][:, 2] / 1.8
    np.testing.assert_array_equal(kinds == "Cu", np.round(z) % 2 == 0)


def test_hexagonal_crystals():
    a = 2.5
    hcp = build_crystal("hcp", a, repeat=(4, 4, 3))
    np.testing.assert_allclose(
        hcp.box.matrix, [[4 * a, 0, 0], [-2 * a, 2 * np.sqrt(3) * a, 0], [0, 0, 3 * np.sqrt(8 / 3) * a]]
    )  # fmt: skip
    for r, _ in neighbors(hcp, 1.05 * a):
        assert len(r) == 12
        np.testing.assert_allclose(r, a, rtol=1e-12)
    stretched = build_crystal("hcp", a, c=2 * a, repeat=(4, 4, 2))
    for r, _ in neighbors(stretched, 1.05 * a):
        assert len(r) == 6  # the neighbors above and below move away
    # a tetrahedron of the other element around each atom
    wurtzite = build_crystal("wurtzite", 3.19, c=3.19 * np.sqrt(8 / 3), elements=("Ga", "N"), repeat=(4, 4, 2))  # fmt: skip
    for (r, around), kind in zip(neighbors(wurtzite, 2.2), wurtzite.table["element"]):
        assert len(r) == 4 and kind not in set(around)
        np.testing.assert_allclose(r, 3.19 * np.sqrt(3 / 8), rtol=1e-12)
    lonsdaleite = build_crystal("lonsdaleite", 2.51, repeat=(4, 4, 2))
    assert all(len(r) == 4 for r, _ in neighbors(lonsdaleite, 1.7))
    for name in ("graphene", "graphite"):
        sheet = build_crystal(name, 2.46, c=6.7, repeat=(4, 4, 1))
        for r, _ in neighbors(sheet, 1.5):
            assert len(r) == 3
            np.testing.assert_allclose(r, 2.46 / np.sqrt(3), rtol=1e-12)
    assert len(build_crystal("graphite", 2.46, c=6.7)) == 2 * len(build_crystal("graphene", 2.46, c=6.7))  # fmt: skip


@pytest.mark.parametrize(
    "structure, orientation, lengths",
    [
        (
            "fcc",
            [(1, 1, 0), (-1, 1, 2), (1, -1, 1)],
            [np.sqrt(2) / 2, np.sqrt(6) / 2, np.sqrt(3)],
        ),  # fmt: skip
        (
            "bcc",
            [(1, 1, 1), (-1, 1, 0), (-1, -1, 2)],
            [np.sqrt(3) / 2, np.sqrt(2), np.sqrt(6)],
        ),  # fmt: skip
        (
            "diamond",
            [(1, 1, 0), (-1, 1, 0), (0, 0, 1)],
            [np.sqrt(2) / 2, np.sqrt(2) / 2, 1.0],
        ),  # fmt: skip
        ("sc", [(2, 0, 0), (0, 3, 3), (0, -1, 1)], [1.0, np.sqrt(2), np.sqrt(2)]),
        # a mirror image, as the frame is left-handed
        (
            "fcc",
            [(1, 1, 0), (1, -1, 0), (0, 0, 1)],
            [np.sqrt(2) / 2, np.sqrt(2) / 2, 1.0],
        ),  # fmt: skip
    ],
)
def test_oriented_crystals(structure, orientation, lengths):
    a = 3.0
    repeat = np.ceil(4.5 / np.array(lengths)).astype(int)  # beyond the cutoff
    system = build_crystal(structure, a, orientation=orientation, repeat=repeat)
    np.testing.assert_allclose(
        system.box.matrix, np.diag(np.array(lengths) * a * repeat)
    )
    per_cell = len(STRUCTURES[structure].basis)
    assert len(system) == pytest.approx(system.box.volume / a**3 * per_cell)
    # the same crystal: the same shells of neighbors around every atom
    shells = neighbors(build_crystal(structure, a, repeat=(3, 3, 3)), 1.5 * a)[0][0]
    for r, _ in neighbors(system, 1.5 * a):
        np.testing.assert_allclose(r, shells, rtol=1e-9)
    assert by_planes(system)


def by_planes(system):
    """Whether the atoms are ordered by z, then y, then x, with values the
    same within rounding taken as equal."""
    x, y, z = np.round(system["positions"], 7).T
    return (np.lexsort((x, y, z)) == np.arange(len(system))).all()


@pytest.mark.parametrize(
    "structure, orientation",
    [
        ("fcc", [(1, 1, -2), (1, -1, 0), (1, 1, 1)]),
        ("diamond", [(1, 2, 3), (-1, -1, 1), (5, -4, 1)]),
        ("hcp", [(2, -1, -1, 0), (0, 1, -1, 1), (0, 0, 0, 1)]),
    ],
)
def test_ordered_by_planes(structure, orientation):
    # atoms of one plane whose z differ by rounding are ordered by y and x
    system = build_crystal(structure, 3.0, orientation=orientation, repeat=(2, 3, 2))
    assert by_planes(system)
    first = system.box.to_fractional(system["positions"][:1])
    np.testing.assert_allclose(first, 0.0, atol=1e-12)


def test_oriented_hexagonal_crystals():
    a, c = 3.21, 5.21
    # orthogonal: x [10-10], y [-12-10], z [0001], as [uvw] or [hkil]
    for orientation in ([(1, 0, -1, 0), (-1, 2, -1, 0), (0, 0, 0, 1)], [(2, 1, 0), (0, 1, 0), (0, 0, 1)]):  # fmt: skip
        system = build_crystal("hcp", a, c=c, orientation=orientation)
        np.testing.assert_allclose(
            system.box.matrix, np.diag([np.sqrt(3) * a, a, c]), atol=1e-12
        )
        assert len(system) == 4
    # not orthogonal: the first along x, the second in the xy plane
    tilted = build_crystal("hcp", a, c=c, orientation=[(0, 0, 0, 1), (2, -3, 1, 0), (-1, -2, 3, 0)])  # fmt: skip
    h = tilted.box.matrix
    assert h[0, 1] == h[0, 2] == h[1, 2] == 0 and np.linalg.det(h) > 0
    # [2-310] is [1-40]: a1 - 4 a2
    np.testing.assert_allclose(np.linalg.norm(h, axis=1), [c, np.sqrt(21) * a, np.sqrt(21) * a])  # fmt: skip
    assert len(tilted) == pytest.approx(
        tilted.box.volume / (np.sqrt(3) / 2 * a * a * c) * 2
    )
    shells = neighbors(build_crystal("hcp", a, c=c, repeat=(4, 4, 3)), 1.5 * a)[0][0]
    for r, _ in neighbors(build_crystal("hcp", a, c=c, orientation=[(1, 0, -1, 0), (-1, 2, -1, 0), (0, 0, 0, 1)], repeat=(2, 4, 2)), 1.5 * a):  # fmt: skip
        np.testing.assert_allclose(r, shells, rtol=1e-9)


def test_shortest_periods():
    # fluorite repeats every half [110], although atomsk takes the whole
    system = build_crystal("fluorite", 5.46, elements=("Ca", "F"), orientation=[(1, 1, 0), (-1, 1, 0), (0, 0, 1)])  # fmt: skip
    np.testing.assert_allclose(system.box.matrix.diagonal(), [5.46 / np.sqrt(2)] * 2 + [5.46])  # fmt: skip
    # with two elements bcc is CsCl, which does not repeat every half [111]
    one = build_crystal("bcc", 2.87, orientation=[(1, 1, 1), (-1, 1, 0), (-1, -1, 2)])
    two = build_crystal("bcc", 2.87, elements=("Fe", "Al"), orientation=[(1, 1, 1), (-1, 1, 0), (-1, -1, 2)])  # fmt: skip
    assert two.box.matrix[0, 0] == pytest.approx(2 * one.box.matrix[0, 0], rel=1e-12)


CASES = [
    ("fcc", 3.615, "Cu", None, None),
    ("fcc", 3.615, "Cu", None, [(1, 1, 0), (-1, 1, 2), (1, -1, 1)]),
    ("fcc", 3.6, ("Cu", "Au"), None, [(1, 1, 0), (-1, 1, 0), (0, 0, 1)]),
    ("fcc", 3.6, "Cu", None, [(1, 1, 0), (1, -1, 0), (0, 0, 1)]),
    ("bcc", 2.87, "Fe", None, [(1, 2, 1), (-1, 0, 1), (1, -1, 1)]),
    ("sc", 3.0, "Po", None, [(1, 2, 0), (-2, 1, 0), (0, 0, 1)]),
    ("diamond", 5.43, "Si", None, [(1, 1, 0), (-1, 1, 0), (0, 0, 1)]),
    ("zincblende", 5.65, ("Ga", "As"), None, [(1, 1, 2), (-1, 1, 0), (-1, -1, 1)]),
    ("rocksalt", 5.64, ("Na", "Cl"), None, [(1, 1, 0), (-1, 1, 0), (0, 0, 1)]),
    ("fluorite", 5.46, ("Ca", "F"), None, [(1, 1, 0), (-1, 1, 0), (0, 0, 1)]),
    ("l1_2", 3.57, ("Ni", "Al"), None, [(1, 1, 0), (-1, 1, 0), (0, 0, 1)]),
    ("perovskite", 3.905, ("Sr", "Ti", "O"), None, [(0, 0, 1), (1, 1, 0), (1, -1, 0)]),
    ("cscl", 4.12, ("Cs", "Cl"), None, [(1, 1, 1), (-1, 1, 0), (-1, -1, 2)]),
    ("hcp", 3.21, "Mg", 5.21, None),
    ("hcp", 3.21, "Mg", 5.21, [(1, 0, -1, 0), (-1, 2, -1, 0), (0, 0, 0, 1)]),
    (
        "hcp",
        3.21,
        "Mg",
        5.213,
        [(0, 0, 0, 1), (2, -3, 1, 0), (-1, -2, 3, 0)],
    ),  # atomsk does not duplicate it
    ("wurtzite", 3.19, ("Ga", "N"), 5.18, None),
    (
        "wurtzite",
        3.19,
        ("Ga", "N"),
        5.18,
        [(1, 0, -1, 0), (-1, 2, -1, 0), (0, 0, 0, 1)],
    ),  # fmt: skip
    ("graphite", 2.46, "C", 6.7, None),
]


@pytest.mark.skipif(ATOMSK is None, reason="atomsk is not installed")
@pytest.mark.parametrize("structure, a, elements, c, orientation", CASES)
def test_as_atomsk(tmp_path, structure, a, elements, c, orientation):
    """The same atoms as atomsk --create, up to a translation; where atomsk
    takes a longer period, in a box of as many of ours. (atomsk halves
    [hkl] of bcc with h, k and l odd also for two elements, which is CsCl:
    compared as cscl.)"""
    tilted = (
        orientation is not None and len(orientation[0]) == 4 and orientation[0][3] == 1
    )
    repeat = (1, 1, 1) if tilted else (2, 1, 2)
    theirs = atomsk_crystal(tmp_path, structure, a, elements, c=c, orientation=orientation, repeat=repeat)  # fmt: skip
    ours = build_crystal(structure, a, c=c, elements=elements, orientation=orientation)
    ratio = np.linalg.norm(theirs.box.matrix, axis=1) / np.linalg.norm(
        ours.box.matrix, axis=1
    )
    repeat = np.round(ratio).astype(int)
    np.testing.assert_allclose(ratio, repeat, rtol=1e-9)
    ours = build_crystal(structure, a, c=c, elements=elements, orientation=orientation, repeat=repeat)  # fmt: skip
    assert same_crystal(ours, theirs, tolerance=1e-7)


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"structure": "bct"}, "structure must be one of"),
        ({"a": 0.0}, "a must be positive"),
        ({"repeat": (1, 0, 1)}, "repeat must be three positive integers"),
        ({"repeat": (1.5, 1, 1)}, "repeat must be three positive integers"),
        ({"orientation": [(1, 0, 0), (0, 1, 0)]}, "three directions"),
        ({"orientation": 3}, "three directions"),
        (
            {"orientation": [(1, 0, 0), (0, 1.5, 0), (0, 0, 1)]},
            "not a direction of integers",
        ),
        (
            {"orientation": [(1, 0, 0), ("y",), (0, 0, 1)]},
            "not a direction of integers",
        ),
        ({"orientation": [(1, 0, 0), (0, 0, 0), (0, 0, 1)]}, "must not be zero"),
        ({"orientation": [(1, 1, 0), (1, 0, 0), (0, 0, 1)]}, "must be orthogonal"),
        ({"orientation": [(1, 0, 0, 0), (0, 1, 0), (0, 0, 1)]}, "three indices"),
        ({"c": 4.0}, "c is for hexagonal structures only"),
        ({"elements": ("Cu", "Au", "Ag")}, "fcc takes 1 or 2 elements, not 3"),
        ({"elements": "Cuu"}, "unknown element"),
        ({"structure": "hcp", "c": -1.0}, "c must be positive"),
        ({"structure": "graphene"}, "graphene needs c"),
        (
            {"structure": "hcp", "orientation": [(1, 0, 0), (2, 0, 0), (0, 0, 1)]},
            "linearly independent",
        ),  # fmt: skip
        (
            {"structure": "hcp", "orientation": [(1, 0, 0, 0), (0, 1, 0), (0, 0, 1)]},
            "h \\+ k \\+ i = 0",
        ),  # fmt: skip
        (
            {"structure": "hcp", "orientation": [(1, 0), (0, 1, 0), (0, 0, 1)]},
            "three or four indices",
        ),  # fmt: skip
        ({"structure": "rocksalt", "elements": "Na"}, None),
    ],
)
def test_invalid_arguments(arguments, message):
    options = {"structure": "fcc", "a": 3.6} | arguments
    structure, a = options.pop("structure"), options.pop("a")
    if message is None:  # one element on every sublattice is fcc
        assert len(build_crystal(structure, a, **options)) == 8
        return
    with pytest.raises(InvalidParameter, match=message):
        build_crystal(structure, a, **options)


def test_a_system():
    system = build_crystal("fcc", 3.6, elements="Ni", repeat=(3, 3, 3))
    assert isinstance(system, mdapy.System) and system.box.pbc.all()
