# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import sys

import numpy as np
import pytest

import mdapy
from mdapy.analysis import SpaceGroup
from mdapy.build import build_crystal
from mdapy.errors import InvalidParameter

pytest.importorskip("spglib")

CRYSTALS = {
    "fcc": ("fcc", 3.6, "Cu", 225, "Fm-3m"),
    "bcc": ("bcc", 2.87, "Fe", 229, "Im-3m"),
    "hcp": ("hcp", 3.2, "Mg", 194, "P6_3/mmc"),
    "diamond": ("diamond", 5.43, "Si", 227, "Fd-3m"),
    "L1_2": ("l1_2", 3.75, ("Cu", "Au"), 221, "Pm-3m"),
    "rocksalt": ("rocksalt", 5.64, ("Na", "Cl"), 225, "Fm-3m"),
    "wurtzite": ("wurtzite", 3.25, ("Zn", "O"), 186, "P6_3mc"),
}


def rotated(system, seed=2):
    """The crystal turned with its box and moved off the origin."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q[:, 0] *= np.sign(np.linalg.det(q))
    return system.with_box(
        mdapy.Box(system.box.matrix @ q.T, origin=[1.0, -2.0, 0.5])
    ).with_columns(positions=system["positions"] @ q.T + [1.0, -2.0, 0.5])


def maps_onto_itself(system, rotation, translation):
    """Whether the operation takes each atom onto an atom of its element."""
    box = system.box
    moved = box.to_fractional(system["positions"] @ rotation.T + translation)
    apart = moved[:, None] - box.to_fractional(system["positions"])[None]
    close = np.abs(apart - np.round(apart)).max(axis=2) < 1e-8
    elements = system.table["element"].to_numpy()
    j = np.argmax(close, axis=1)
    return bool(close.any(axis=1).all() and (elements[j] == elements).all())


@pytest.mark.parametrize("name", CRYSTALS)
@pytest.mark.parametrize("placed", [False, True])
def test_space_groups_of_crystals(name, placed):
    structure, a, elements, number, symbol = CRYSTALS[name]
    system = build_crystal(structure, a, elements=elements)
    if placed:
        system = rotated(system)
    result = SpaceGroup().compute(system)
    assert (result.number, result.symbol) == (number, symbol)
    k = len(result.rotations)
    assert result.rotations.shape == (k, 3, 3) and result.translations.shape == (k, 3)
    # orthogonal operations that map the crystal onto itself
    np.testing.assert_allclose(
        result.rotations @ result.rotations.transpose(0, 2, 1),
        np.broadcast_to(np.eye(3), (k, 3, 3)),
        atol=1e-12,
    )
    for rotation, translation in zip(result.rotations, result.translations):
        assert maps_onto_itself(system, rotation, translation)
    assert not result.rotations.flags.writeable and result.columns() == {}


def test_equivalent_atoms():
    cell = build_crystal("l1_2", 3.75, elements=("Cu", "Au"))
    result = SpaceGroup().compute(cell)
    elements = cell.table["element"].to_list()
    # the Au corner is alone, the three Cu face centers are equivalent
    au = elements.index("Au")
    assert result.equivalent[au] == au
    cu = [i for i, e in enumerate(elements) if e == "Cu"]
    assert len(set(result.equivalent[cu].tolist())) == 1
    assert result.equivalent.dtype == np.int64


def test_kinds_of_atoms_and_tolerance():
    cell = build_crystal("cscl", 3.0, elements=("Cs", "Cl"))
    assert SpaceGroup().compute(cell).symbol == "Pm-3m"
    # without elements types tell atoms apart, and without either all alike
    types = mdapy.System(cell["positions"], cell.box, types=[1, 2])
    assert SpaceGroup().compute(types).symbol == "Pm-3m"
    alike = mdapy.System(cell["positions"], cell.box)
    assert SpaceGroup().compute(alike).symbol == "Im-3m"
    # an atom moved by more than the tolerance breaks the symmetry
    moved = cell["positions"].copy()
    moved[1, 0] += 1e-3
    shifted = cell.with_columns(positions=moved)
    assert SpaceGroup().compute(shifted).number < 221
    assert SpaceGroup(tolerance=1e-2).compute(shifted).symbol == "Pm-3m"


def test_errors(monkeypatch):
    cell = build_crystal("fcc", 3.6, elements="Cu")
    with pytest.raises(InvalidParameter, match="tolerance must be positive"):
        SpaceGroup(tolerance=0)
    with pytest.raises(InvalidParameter, match="periodic"):
        SpaceGroup().compute(cell.with_box(mdapy.Box(3.6, pbc=False)))
    monkeypatch.setitem(sys.modules, "spglib", None)
    with pytest.raises(InvalidParameter, match="pip install spglib"):
        SpaceGroup().compute(cell)
    assert repr(SpaceGroup()) == "SpaceGroup(tolerance=1e-05)"
