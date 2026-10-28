# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.analysis import ElasticStrain
from mdapy.build import build_crystal, dislocation_quadrupole
from mdapy.errors import Cancelled, InvalidParameter

FE = (243.3, 145.0, 116.1)
CRYSTALS = [
    ("fcc", "fcc", 3.615, None),
    ("hcp", "hcp", 3.21, 5.21),
    ("bcc", "bcc", 2.865, None),
    ("cubic_diamond", "diamond", 5.431, None),
    ("hexagonal_diamond", "lonsdaleite", 2.51, 4.12),
]


def deformed(system, gradient):
    """The system and its box deformed by a gradient."""
    box = mdapy.Box(system.box.matrix @ gradient.T, pbc=system.box.pbc)
    return system.with_box(box, scale_positions=True)


def spatial(gradient):
    inverse = np.linalg.inv(gradient)
    return (np.eye(3) - inverse.T @ inverse) / 2


@pytest.mark.parametrize("crystal, structure, a, c", CRYSTALS)
def test_a_homogeneous_deformation(crystal, structure, a, c):
    """Every atom of a crystal and its box deformed by one gradient, which
    turns it too, has the strain of that gradient in the spatial frame."""
    cell = build_crystal(structure, a, c=c, repeat=(5, 5, 5))
    rng = np.random.default_rng(1)
    gradient = np.eye(3) + 0.02 * rng.standard_normal((3, 3))
    result = ElasticStrain(crystal, a, c).compute(deformed(cell, gradient))
    assert (np.asarray(result.names)[result.structure] == crystal).all()
    expected = spatial(gradient)
    np.testing.assert_allclose(
        result.strain_tensor, np.broadcast_to(expected, (len(cell), 3, 3)), atol=1e-12
    )
    np.testing.assert_allclose(result.volumetric, np.trace(expected) / 3, atol=1e-12)


@pytest.mark.parametrize("crystal, structure, a, c", CRYSTALS)
def test_the_lattice_constants(crystal, structure, a, c):
    """A crystal analyzed with other lattice constants is strained by their
    ratios: the same along a and b, that of c along the c axis of the
    hexagonal crystals, which builds along z."""
    cell = build_crystal(structure, a, c=c, repeat=(5, 5, 5))
    if c is None:
        result = ElasticStrain(crystal, 1.01 * a).compute(cell)
        gradient = np.eye(3) / 1.01
    else:
        result = ElasticStrain(crystal, 1.01 * a, 0.98 * c).compute(cell)
        gradient = np.diag([1 / 1.01, 1 / 1.01, 1 / 0.98])
    np.testing.assert_allclose(result.strain_tensor[0], spatial(gradient), atol=1e-12)
    np.testing.assert_allclose(
        result.strain_tensor, result.strain_tensor[:1].repeat(len(cell), 0), atol=1e-12
    )


def test_the_ideal_ratio_of_hexagonal_crystals():
    """Without c, the ideal ratio of a √(8/3)."""
    for crystal, structure, a in (
        ("hcp", "hcp", 3.2),
        ("hexagonal_diamond", "lonsdaleite", 2.5),
    ):
        cell = build_crystal(structure, a, repeat=(5, 5, 5))
        default = ElasticStrain(crystal, a).compute(cell)
        given = ElasticStrain(crystal, a, a * np.sqrt(8 / 3)).compute(cell)
        np.testing.assert_allclose(
            default.strain_tensor, given.strain_tensor, atol=1e-14
        )
        np.testing.assert_allclose(default.strain_tensor, 0, atol=1e-12)


def test_invariance():
    """The strain of a dislocation turns with the system, does not change
    with a move, and follows the atoms in any order."""
    system = dislocation_quadrupole("bcc screw", 2.865, FE, (6, 10, 5), elements="Fe")
    result = ElasticStrain("bcc", 2.865).compute(system)
    assert (result.structure == 3).mean() > 0.9 and np.abs(
        result.volumetric
    ).max() > 1e-3
    t = np.radians(30)
    turn = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    fractional = system["positions"] @ system.box.inverse
    turned = system.with_box(
        mdapy.Box(system.box.matrix @ turn.T, origin=[3.0, -1.0, 2.0])
    ).with_columns(positions=fractional @ system.box.matrix @ turn.T + [3.0, -1.0, 2.0])
    again = ElasticStrain("bcc", 2.865).compute(turned)
    np.testing.assert_array_equal(again.structure, result.structure)
    np.testing.assert_allclose(
        again.strain_tensor, turn @ result.strain_tensor @ turn.T, atol=1e-12
    )
    order = np.random.default_rng(0).permutation(len(system))
    shuffled = ElasticStrain("bcc", 2.865).compute(
        system.with_table(system.table[order])
    )
    np.testing.assert_allclose(
        shuffled.strain_tensor, result.strain_tensor[order], atol=1e-14
    )


def test_other_atoms_and_results():
    """Atoms not of the crystal have no strain; the columns of the result."""
    cell = build_crystal("fcc", 3.615, elements="Cu", repeat=(5, 5, 5))
    rng = np.random.default_rng(2)
    liquid = cell.with_columns(
        positions=cell["positions"] + rng.normal(0, 0.6, (len(cell), 3))
    )
    result = ElasticStrain("fcc", 3.615).compute(liquid)
    other = result.structure != 1
    assert other.any()
    assert (result.strain_tensor[other] == 0).all() and (
        result.volumetric[other] == 0
    ).all()
    assert set(result.columns()) == {
        "structure",
        "elastic_strain",
        "elastic_volumetric_strain",
    }
    with_columns = cell.with_columns(ElasticStrain("fcc", 3.615).compute(cell))
    assert with_columns["elastic_strain"].shape == (len(cell), 3, 3)
    assert repr(ElasticStrain("hcp", 3.2, 5.2)) == "ElasticStrain('hcp', 3.2, c=5.2)"
    assert repr(ElasticStrain("fcc", 3.615)) == "ElasticStrain('fcc', 3.615)"


def test_the_same_for_any_number_of_threads():
    system = dislocation_quadrupole("bcc screw", 2.865, FE, (6, 10, 5), elements="Fe")
    found = []
    for threads in (1, 4):
        with mdapy.num_threads(threads):
            found.append(ElasticStrain("bcc", 2.865).compute(system).strain_tensor)
    np.testing.assert_array_equal(found[0], found[1])


def test_progress_cancel_and_errors():
    system = dislocation_quadrupole("bcc screw", 2.865, FE, (6, 10, 5), elements="Fe")
    calls = []
    ElasticStrain("bcc", 2.865).compute(
        system, progress=lambda done, total: calls.append((done, total))
    )
    assert calls and calls[-1][0] == calls[-1][1]
    token = mdapy.CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        ElasticStrain("bcc", 2.865).compute(system, cancel=token)
    with pytest.raises(InvalidParameter, match="crystal"):
        ElasticStrain("sc", 3.0)
    for a in (0, -1, np.nan):
        with pytest.raises(InvalidParameter, match="a must be positive"):
            ElasticStrain("fcc", a)
    with pytest.raises(InvalidParameter, match="only for hexagonal"):
        ElasticStrain("fcc", 3.615, 5.0)
    with pytest.raises(InvalidParameter, match="c must be positive"):
        ElasticStrain("hcp", 3.2, 0)
    # one period along the lines, or two, whose neighbors along them are
    # one atom through two images
    for periods in (1, 2):
        thin = dislocation_quadrupole(
            "bcc screw", 2.865, FE, (6, 10, periods), elements="Fe"
        )
        with pytest.raises(InvalidParameter, match="own periodic image"):
            ElasticStrain("bcc", 2.865).compute(thin)
