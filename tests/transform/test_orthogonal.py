# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

from mdapy import Box, System
from mdapy.build import build_crystal
from mdapy.core.topology import Topology
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics import primitive_cell
from mdapy.transform import orthogonal_cell, replicate

from atomsk_reference import ATOMSK, atomsk_apply

# boxes in the form of LAMMPS, a along x and b in the x-y plane, as atomsk reads
# them: hexagonal, centred rectangular, tilted in x-z, and both
BOXES = {
    "hexagonal": Box([[3.21, 0, 0], [-1.605, 3.21 * np.sqrt(3) / 2, 0], [0, 0, 5.21]]),
    "centred rectangular": Box([[3.0, 0, 0], [-1.5, 2.2, 0], [0, 0, 4.0]]),
    "tilted in x-z": Box([[3.0, 0, 0], [0, 2.5, 0], [1.5, 0, 4.0]]),
    "both": Box([[3.0, 0, 0], [-1.0, 2.5, 0], [1.5, -1.25, 4.0]]),
}


def crystal_in(box, seed=0):
    """Three atoms of three elements at random in a box."""
    fractional = np.random.default_rng(seed).random((3, 3))
    return System(fractional @ box.matrix, box, elements=["Cu", "Ni", "Al"])


def same_crystal(a, b, tolerance=1e-6):
    """Whether two systems in equal boxes hold the same atoms, in any order and
    any periodic image."""
    if len(a) != len(b) or not np.allclose(a.box.matrix, b.box.matrix, atol=1e-9):
        return False
    inverse = np.linalg.inv(a.box.matrix)
    kinds_a, kinds_b = np.asarray(a["element"]), np.asarray(b["element"])
    for k in range(len(a)):
        d = (b["positions"] - a["positions"][k]) @ inverse
        d -= np.round(d)
        hit = np.flatnonzero(np.linalg.norm(d @ a.box.matrix, axis=1) < tolerance)
        if len(hit) != 1 or kinds_b[hit[0]] != kinds_a[k]:
            return False
    return True


@pytest.mark.skipif(ATOMSK is None, reason="atomsk is not installed")
@pytest.mark.parametrize("name", BOXES)
def test_as_atomsk(tmp_path, name):
    system = crystal_in(BOXES[name])
    ours = orthogonal_cell(system, max_search=10)
    theirs = atomsk_apply(tmp_path, system, "-orthogonal-cell")
    assert ours.box.is_orthogonal
    assert same_crystal(ours, theirs)
    # a box of several periods as well
    many = replicate(system, (2, 3, 1))
    assert same_crystal(
        orthogonal_cell(many, max_search=10),
        atomsk_apply(tmp_path, many, "-orthogonal-cell"),
    )


@pytest.mark.parametrize("name", BOXES)
def test_the_same_crystal(name):
    system = crystal_in(BOXES[name], seed=1)
    cell = orthogonal_cell(system, max_search=10)
    assert np.allclose(cell.box.matrix, np.diag(np.diagonal(cell.box.matrix)))
    assert np.allclose(cell.box.origin, system.box.origin)
    # as many atoms per volume, and every one at a site of the crystal
    assert cell.box.volume / len(cell) == pytest.approx(
        system.box.volume / len(system), rel=1e-12
    )
    copies = len(cell) // len(system)
    kinds = np.asarray(cell["element"])
    inverse = np.linalg.inv(system.box.matrix)
    for k, element in enumerate(system["element"]):
        mine = np.flatnonzero(kinds == element)
        assert len(mine) == copies
        offsets = (cell["positions"][mine] - system["positions"][k]) @ inverse
        np.testing.assert_allclose(offsets, np.round(offsets), atol=1e-9)
        # each copy at another point of the lattice of the cell
        whole = np.round(offsets).astype(int) @ np.linalg.inv(
            np.round(cell.box.matrix @ inverse)
        )
        whole -= np.floor(whole + 1e-9)
        assert len(np.unique(np.round(whole, 9), axis=0)) == copies


def test_hexagonal_crystals():
    hcp = build_crystal("hcp", 3.21, c=5.21, elements="Mg")
    cell = orthogonal_cell(hcp)
    assert len(cell) == 4
    np.testing.assert_allclose(
        np.diagonal(cell.box.matrix), [3.21, 3.21 * np.sqrt(3), 5.21], rtol=1e-14
    )
    wurtzite = orthogonal_cell(
        build_crystal("wurtzite", 3.19, c=5.18, elements=("Ga", "N"))
    )
    assert len(wurtzite) == 8
    assert sorted(wurtzite["element"].tolist()) == ["Ga"] * 4 + ["N"] * 4


def test_the_smallest_cell():
    hcp = build_crystal("hcp", 3.21, c=5.21, elements="Mg")
    many = replicate(hcp, (2, 3, 2))
    # as atomsk, from the box: along y 6 a sqrt(3), 24 times the atoms
    assert len(orthogonal_cell(many)) == 96
    small = orthogonal_cell(many, minimal=True)
    assert same_crystal(small, orthogonal_cell(hcp))
    # from primitive cells to the cubic cells of fcc and rock salt
    copper = build_crystal("fcc", 3.615, elements="Cu")
    cubic = orthogonal_cell(primitive_cell(copper))
    assert same_crystal(cubic, copper)
    salt = build_crystal("rocksalt", 5.64, elements=("Na", "Cl"))
    assert same_crystal(orthogonal_cell(primitive_cell(salt)), salt)
    assert same_crystal(orthogonal_cell(replicate(salt, 2), minimal=True), salt)


def test_an_orthogonal_box_is_kept():
    copper = build_crystal("fcc", 3.615, elements="Cu", repeat=(2, 1, 1))
    assert orthogonal_cell(copper) is copper
    assert len(orthogonal_cell(copper, minimal=True)) == 4


def test_columns_ids_and_images():
    box = BOXES["hexagonal"]
    system = (
        crystal_in(box)
        .with_columns(
            velocities=np.arange(9.0).reshape(3, 3),
            id=[4, 8, 6],
            molecule=[1, 0, 2],
            image=np.array([[0, 0, 0], [1, -1, 0], [0, 2, -1]], dtype=np.int32),
        )
        .with_info(energy=-3.0, note="kept")
    )
    cell = orthogonal_cell(system)
    copies = len(cell) // 3
    np.testing.assert_array_equal(
        cell["velocities"], np.tile(system["velocities"], (copies, 1))
    )
    np.testing.assert_array_equal(
        cell["id"], np.concatenate([np.array([4, 8, 6]) + 8 * c for c in range(copies)])
    )
    np.testing.assert_array_equal(
        cell["molecule"],
        np.concatenate([[1 + 2 * c, 0, 2 + 2 * c] for c in range(copies)]),
    )
    assert "energy" not in cell.info and cell.info["note"] == "kept"
    # the unwrapped positions of the first copy are those of the system
    unwrapped = cell["positions"] + cell["image"] @ cell.box.matrix
    original = system["positions"] + system["image"] @ box.matrix
    np.testing.assert_allclose(unwrapped[:3], original, atol=1e-12)


@pytest.mark.parametrize("imaged", [False, True])
def test_bonds_across_the_boundary(imaged):
    """Molecules of two atoms lying across the boundary of a hexagonal cell:
    every bond of the orthogonal cell has the vector of its molecule."""
    box = BOXES["hexagonal"]
    first = np.array([[0.1, 0.1, 0.5], [0.6, 0.55, 0.2]]) @ box.matrix
    bond = np.array([[-0.9, -0.3, 0.0], [0.4, -1.1, 0.3]])
    positions = np.concatenate([first, first + bond])
    columns = {}
    if imaged:
        columns["image"] = np.zeros((4, 3), dtype=np.int32)
    wrapped, moved = box.wrap(positions)
    if imaged:
        columns["image"] = moved.astype(np.int32)
    system = System(
        wrapped,
        box,
        elements=["C", "C", "O", "O"],
        topology=Topology(bonds=[[0, 2], [1, 3]], angles=[[2, 0, 1]]),
    ).with_columns(**columns)
    cell = orthogonal_cell(system)
    bonds = cell.topology.bonds["atoms"].to_numpy()
    x = cell["positions"]
    vectors = cell.box.minimum_image(x[bonds[:, 1]] - x[bonds[:, 0]])
    copies = len(cell) // 4
    np.testing.assert_allclose(vectors, np.tile(bond, (copies, 1)), atol=1e-12)
    # each atom in one bond: the copies do not share atoms
    assert len(bonds) == 2 * copies and len(np.unique(bonds)) == 2 * len(bonds)
    # the angle joins the two molecules: each atom where its image flags put
    # it, or else at the image nearest the first atom of the term, as
    # replicate takes it
    angles = cell.topology.angles["atoms"].to_numpy()
    assert len(angles) == copies
    expected = np.array([first[0] - positions[2], first[1] - positions[2]])
    if imaged:
        unwrapped = x + cell["image"] @ cell.box.matrix
    else:
        expected = box.minimum_image(expected)
    for a, b, c in angles:
        if imaged:
            found = np.array([unwrapped[b] - unwrapped[a], unwrapped[c] - unwrapped[a]])
        else:
            found = cell.box.minimum_image(np.array([x[b] - x[a], x[c] - x[a]]))
        np.testing.assert_allclose(found, expected, atol=1e-12)


def test_errors():
    slab = System(np.zeros((1, 3)), Box(5.0, pbc=(True, True, False)), elements="Cu")
    with pytest.raises(InvalidParameter, match="periodic in every direction"):
        orthogonal_cell(slab)
    system = crystal_in(BOXES["hexagonal"])
    for value in (0, 1.5, True):
        with pytest.raises(InvalidParameter, match="max_search must be"):
            orthogonal_cell(system, max_search=value)
    with pytest.raises(InvalidParameter, match="tolerance must be positive"):
        orthogonal_cell(system, tolerance=0.0)
    turned = crystal_in(Box([[3.0, 0, 0], [1.0, 2.5, 0], [0.37, 0.71, 4.0]]))
    with pytest.raises(InvalidParameter, match="along z"):
        orthogonal_cell(turned, max_search=5)
