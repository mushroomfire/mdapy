# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The affine transformation: as OVITO's, and its invariances."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.errors import InvalidParameter
from mdapy.transform import affine, unwrap

MATRIX = [[1.1, 0.2, 0.0], [0.05, 0.95, 0.1], [0.0, 0.3, 1.2]]
SHIFT = [1.0, -2.0, 0.5]


@pytest.fixture(scope="module")
def crystal():
    """A triclinic cell of copper, off the origin, its atoms shaken and
    moving, with forces and displacements."""
    cell = build_crystal("fcc", 3.615, elements="Cu", repeat=(3, 3, 3))
    box = mdapy.Box(
        cell.box.matrix + [[0, 0, 0], [1.5, 0, 0], [0.7, -0.4, 0]], origin=(2, -1, 3)
    )
    rng = np.random.default_rng(0)
    n = len(cell)
    return cell.with_box(box, scale_positions=True).with_columns(
        positions=cell.with_box(box, scale_positions=True)["positions"]
        + rng.normal(0, 0.1, (n, 3)),
        velocities=rng.normal(0, 5, (n, 3)),
        forces=rng.normal(0, 1, (n, 3)),
        displacement=rng.normal(0, 0.5, (n, 3)),
    )


@pytest.mark.parametrize("only_selected", [False, True])
def test_as_ovito(crystal, only_selected):
    """Positions, vectors and cell as OVITO's modifier moves them, all atoms
    or the selected ones."""
    pytest.importorskip("ovito")
    from ovito.modifiers import AffineTransformationModifier
    from ovito.pipeline import Pipeline

    from ovito_reference import ovito_source

    selection = np.arange(len(crystal)) % 3 == 0
    ours = affine(crystal, MATRIX, SHIFT, atoms=selection if only_selected else None)
    source = ovito_source(crystal)
    particles = source.data.particles_
    for column, name in (
        ("velocities", "Velocity"),
        ("forces", "Force"),
        ("displacement", "Displacement"),
    ):
        particles.create_property(name, data=crystal[column])
    particles.create_property("Selection", data=selection.astype(np.int32))
    pipeline = Pipeline(source=source)
    pipeline.modifiers.append(
        AffineTransformationModifier(
            transformation=np.c_[MATRIX, SHIFT], only_selected=only_selected
        )
    )
    data = pipeline.compute()
    np.testing.assert_allclose(ours["positions"], data.particles.positions, atol=1e-10)
    for column, name in (
        ("velocities", "Velocity"),
        ("forces", "Force"),
        ("displacement", "Displacement"),
    ):
        np.testing.assert_allclose(ours[column], data.particles[name], atol=1e-10)
    np.testing.assert_allclose(
        ours.box.matrix, np.asarray(data.cell)[:, :3].T, atol=1e-10
    )
    np.testing.assert_allclose(ours.box.origin, np.asarray(data.cell)[:, 3], atol=1e-10)


def test_rotation_keeps_distances(crystal):
    """A rotation turns every distance and vector length into itself; the
    other columns and the information stay."""
    angle = np.radians(30)
    rotation = [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]
    turned = affine(crystal, rotation, SHIFT)
    before = crystal["positions"][1:] - crystal["positions"][0]
    after = turned["positions"][1:] - turned["positions"][0]
    np.testing.assert_allclose(
        np.linalg.norm(after, axis=1), np.linalg.norm(before, axis=1)
    )
    np.testing.assert_allclose(
        np.linalg.norm(turned["velocities"], axis=1),
        np.linalg.norm(crystal["velocities"], axis=1),
    )
    np.testing.assert_allclose(turned.box.volume, crystal.box.volume)
    assert (turned["element"] == crystal["element"]).all()
    # the fractional coordinates stay with the box
    np.testing.assert_allclose(
        turned.box.to_fractional(turned["positions"]),
        crystal.box.to_fractional(crystal["positions"]),
        atol=1e-12,
    )


def test_images_stay_valid(crystal):
    """With the box, image flags still count the same lattice vectors: the
    unwrapped positions transform as the positions."""
    rng = np.random.default_rng(1)
    flagged = crystal.with_columns(image=rng.integers(-2, 3, (len(crystal), 3)))
    np.testing.assert_allclose(
        unwrap(affine(flagged, MATRIX, SHIFT))["positions"],
        affine(unwrap(flagged), MATRIX, SHIFT)["positions"],
        atol=1e-10,
    )


def test_atoms_alone(crystal):
    """Without the box the atoms move in the old box; without vectors these
    stay; a singular matrix may then flatten the atoms."""
    moved = affine(crystal, MATRIX, SHIFT, box=False, vectors=())
    assert moved.box == crystal.box
    np.testing.assert_array_equal(moved["velocities"], crystal["velocities"])
    flat = affine(crystal, np.diag([1.0, 1.0, 0.0]), box=False)
    assert (flat["positions"][:, 2] == 0).all()


@pytest.mark.parametrize(
    "options, message",
    [
        ({"matrix": np.eye(2)}, "matrix must be"),
        ({"matrix": [[np.nan] * 3] * 3}, "matrix must be"),
        ({"matrix": np.diag([1.0, 1.0, 0.0])}, "singular"),
        ({"translation": (1, 2)}, "translation must be"),
        ({"atoms": [True]}, "a mask of atoms must have"),
        ({"atoms": [108]}, "atoms must be a mask or indices"),
        ({"vectors": ("element",)}, "not a vector"),
    ],
)
def test_errors(crystal, options, message):
    options = {"matrix": MATRIX, **options}
    with pytest.raises(InvalidParameter, match=message):
        affine(crystal, **options)
