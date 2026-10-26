# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest

import mdapy
from mdapy.build import (
    add_interstitials,
    add_vacancies,
    add_vacuum,
    build_crystal,
    slip,
    substitute,
)
from mdapy.errors import InvalidParameter
from mdapy.identify import CommonNeighborAnalysis

FCC111 = [(1, 1, -2), (-1, 1, 0), (1, 1, 1)]  # z normal to a {111} plane


def test_vacancies():
    crystal = build_crystal("fcc", 3.6, elements="Cu", repeat=(3, 3, 3))
    holes = add_vacancies(crystal, 5, seed=1)
    assert len(holes) == len(crystal) - 5
    np.testing.assert_array_equal(
        holes["positions"], add_vacancies(crystal, 5, seed=1)["positions"]
    )
    kept = np.ones(len(crystal), bool)
    kept[[0, 7]] = False
    np.testing.assert_array_equal(
        add_vacancies(crystal, atoms=[7, 0])["positions"], crystal["positions"][kept]
    )
    np.testing.assert_array_equal(
        add_vacancies(crystal, atoms=~kept)["positions"], crystal["positions"][kept]
    )
    assert len(add_vacancies(crystal, 0)) == len(crystal)
    for arguments, message in [
        ({}, "either count or atoms"),
        ({"count": 1, "atoms": [0]}, "either count or atoms"),
        ({"count": 1000}, "count must be an integer"),
        ({"count": 1.5}, "count must be an integer"),
        ({"atoms": [0, 0]}, "distinct indices"),
        ({"atoms": [-1]}, "distinct indices"),
        ({"atoms": np.ones(3, bool)}, "mask of atoms"),
    ]:
        with pytest.raises(InvalidParameter, match=message):
            add_vacancies(crystal, **arguments)


def test_substitutions():
    crystal = build_crystal("fcc", 3.6, elements="Cu", repeat=(3, 3, 3))
    alloy = substitute(crystal, "Ni", 10, seed=2)
    kinds = alloy.table["element"].to_list()
    assert kinds.count("Ni") == 10 and kinds.count("Cu") == len(crystal) - 10
    np.testing.assert_array_equal(alloy["positions"], crystal["positions"])
    assert substitute(crystal, "Au", atoms=[3]).table["element"][3] == "Au"
    bare = mdapy.System(crystal["positions"], crystal.box)
    assert substitute(bare, "Ag", atoms=[0]).table["element"].to_list()[:2] == [
        "Ag",
        "X",
    ]
    with pytest.raises(InvalidParameter, match="unknown element"):
        substitute(crystal, "Qq", 1)


def test_interstitials():
    crystal = build_crystal("fcc", 3.6, elements="Cu", repeat=(3, 3, 3)).with_columns(
        id=np.arange(1, 109)
    )
    octahedral = [[1.8, 1.8, 1.8]]  # the center of a cubic cell
    one = add_interstitials(crystal, "Cu", positions=octahedral)
    assert len(one) == 109 and one.table["id"][-1] == 109
    np.testing.assert_array_equal(one["positions"][-1], octahedral[0])
    nl = mdapy.NeighborList.build(one, 1.85)
    assert nl.counts[-1] == 6  # an octahedron of nearest atoms
    many = add_interstitials(crystal, "H", 20, min_distance=1.0, seed=3)
    assert many.table["element"].to_list()[-20:] == ["H"] * 20
    distances = mdapy.NeighborList.build(many, 1.0).distances
    assert len(distances) == 0
    np.testing.assert_array_equal(
        many["positions"],
        add_interstitials(crystal, "H", 20, min_distance=1.0, seed=3)["positions"],
    )
    empty = mdapy.System(np.zeros((0, 3)), mdapy.Box(10.0), elements=[])
    assert len(add_interstitials(empty, "Ar", 3, min_distance=2.0, seed=4)) == 3
    for arguments, message in [
        ({"element": "Qq", "count": 1}, "unknown element"),
        ({"element": "H"}, "either count or positions"),
        ({"element": "H", "count": -1}, "non-negative integer"),
        ({"element": "H", "count": 1, "min_distance": -1.0}, "must not be negative"),
        ({"element": "H", "count": 1, "min_distance": 5.0, "attempts": 10}, "no place"),
    ]:
        with pytest.raises(InvalidParameter, match=message):
            add_interstitials(crystal, **arguments)


@pytest.mark.parametrize("periodic", [True, False])
def test_stacking_fault(periodic):
    """An intrinsic stacking fault: two layers of hcp atoms in fcc."""
    a = 3.615
    crystal = build_crystal(
        "fcc", a, elements="Cu", orientation=FCC111, repeat=(2, 3, 4)
    )
    if not periodic:
        crystal = crystal.with_box(
            mdapy.Box(crystal.box.matrix, pbc=(True, True, False))
        )
    partial = np.array([a / np.sqrt(6), 0, 0])  # 1/6 [11-2]
    faulted = slip(crystal, partial)
    structures = CommonNeighborAnalysis().compute(faulted)
    counts = structures.counts()
    layer = len(crystal) // 12
    if periodic:
        assert counts["hcp"] == 2 * layer and counts["fcc"] == len(crystal) - 2 * layer
        np.testing.assert_allclose(
            faulted.box.matrix[2], crystal.box.matrix[2] + partial
        )
    else:
        np.testing.assert_array_equal(faulted.box.matrix, crystal.box.matrix)
        assert counts["hcp"] == 2 * layer
    # a lattice translation, 1/2 [-110] along y, restores the crystal
    whole = slip(crystal, [0, a / np.sqrt(2), 0], height=0.3, axis="z")
    np.testing.assert_array_equal(
        CommonNeighborAnalysis().compute(whole).structure,
        CommonNeighborAnalysis().compute(crystal).structure,
    )
    with pytest.raises(InvalidParameter, match="lie in the plane"):
        slip(crystal, [0, 0, 1.0])
    with pytest.raises(InvalidParameter, match="three components"):
        slip(crystal, [1.0, 0])
    with pytest.raises(InvalidParameter, match="axis must be"):
        slip(crystal, partial, axis="w")
    with pytest.raises(InvalidParameter, match="axis must be"):
        slip(crystal, partial, axis=3)


def test_vacuum():
    crystal = build_crystal("fcc", 3.6, elements="Cu", repeat=(3, 3, 4))
    shifted = crystal.with_columns(
        positions=crystal["positions"] + [0, 0, -0.5]
    )  # some atoms below 0
    slab = add_vacuum(shifted, 10.0)
    assert slab.box.matrix[2, 2] == pytest.approx(14.4 + 10.0)
    fractional = slab.box.to_fractional(slab["positions"])[:, 2]
    assert fractional.min() >= 0 and fractional.max() < 14.4 / 24.4
    # the atoms of the two surfaces lose their fcc neighborhood
    counts = CommonNeighborAnalysis().compute(slab).counts()
    assert counts["fcc"] == len(crystal) - 2 * 18
    side = add_vacuum(crystal, 5.0, axis="x")
    assert side.box.matrix[0, 0] == pytest.approx(10.8 + 5.0)
    with pytest.raises(InvalidParameter, match="must not be negative"):
        add_vacuum(crystal, -1.0)
