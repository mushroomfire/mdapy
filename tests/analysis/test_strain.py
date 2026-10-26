# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest

import mdapy
from mdapy.analysis import AtomicStrain
from mdapy.errors import InvalidParameter

from lattices import crystal

F0 = np.array([[1.05, 0.02, 0.0], [0.0, 0.97, 0.01], [0.03, 0.0, 1.02]])


def deformed(system, gradient, wrap=True):
    box = mdapy.Box(system.box.matrix @ gradient.T, pbc=system.box.pbc)
    moved = system.with_box(box).with_columns(
        positions=system["positions"] @ gradient.T
    )
    return (
        moved.with_columns(positions=box.wrap(moved["positions"])[0]) if wrap else moved
    )


def reference_strain(reference, current, cutoff):
    """Least squares per atom with NumPy, displacements unwrapped by hand."""
    nl = mdapy.NeighborList.build(reference, cutoff)
    s0 = reference.box.to_fractional(reference["positions"])
    u = current.box.to_fractional(current["positions"]) - s0
    u -= np.round(u)
    strains, d2 = [], []
    for i in range(len(reference)):
        js = nl.j[nl.offsets[i] : nl.offsets[i + 1]]
        n = nl.shifts[nl.offsets[i] : nl.offsets[i + 1]]
        f0 = s0[js] + n - s0[i]
        d0 = f0 @ reference.box.matrix
        d = (f0 + u[js] - u[i]) @ current.box.matrix
        gradient = np.linalg.lstsq(d0, d, rcond=None)[0].T  # d = F d0
        strains.append(0.5 * (gradient.T @ gradient - np.eye(3)))
        d2.append(((d - d0 @ gradient.T) ** 2).sum())
    return np.array(strains), np.array(d2)


def test_a_homogeneous_deformation():
    reference = crystal("fcc", 3.6, (4, 4, 4), jitter=0.05, seed=1)
    current = deformed(reference, F0)  # atoms wrapped back into the new box
    result = AtomicStrain(reference, 3.0, strain_tensor=True, d2min=True).compute(
        current
    )
    expected = 0.5 * (F0.T @ F0 - np.eye(3))
    np.testing.assert_allclose(
        result.strain_tensor,
        np.broadcast_to(expected, (len(reference), 3, 3)),
        atol=1e-13,
    )
    np.testing.assert_allclose(result.d2min, 0.0, atol=1e-20)
    e = expected
    shear = np.sqrt(
        e[0, 1] ** 2 + e[0, 2] ** 2 + e[1, 2] ** 2
        + ((e[0, 0] - e[1, 1]) ** 2 + (e[1, 1] - e[2, 2]) ** 2 + (e[0, 0] - e[2, 2]) ** 2) / 6
    )  # fmt: skip
    np.testing.assert_allclose(result.shear, shear, rtol=1e-12)
    np.testing.assert_allclose(result.volumetric, np.trace(e) / 3, rtol=1e-12)
    removed = AtomicStrain(reference, 3.0, affine=True, strain_tensor=True)
    np.testing.assert_allclose(removed.compute(current).strain_tensor, 0.0, atol=1e-13)


def test_a_rotation_is_no_strain():
    reference = crystal("bcc", 2.87, (4, 4, 4), jitter=0.05, seed=2)
    angle = 0.3
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    strain = AtomicStrain(reference, 3.0, strain_tensor=True)
    result = strain.compute(deformed(reference, rotation))
    np.testing.assert_allclose(result.strain_tensor, 0.0, atol=1e-13)


def test_matches_least_squares():
    reference = crystal("fcc", 3.6, (3, 3, 3), jitter=0.05, seed=3)
    rng = np.random.default_rng(4)
    current = deformed(reference, F0, wrap=False)
    current = current.with_columns(
        positions=current["positions"] + rng.normal(scale=0.1, size=(len(current), 3))
    )
    everything = AtomicStrain(reference, 3.2, strain_tensor=True, d2min=True)
    result = everything.compute(current)
    strain, d2 = reference_strain(reference, current, 3.2)
    np.testing.assert_allclose(result.strain_tensor, strain, atol=1e-12)
    np.testing.assert_allclose(result.d2min, d2, rtol=1e-10, atol=1e-14)
    assert result.d2min.min() > 0
    table = current.with_columns(result).table
    assert {"shear_strain", "volumetric_strain", "d2min"} <= set(table.columns)
    with mdapy.num_threads(3):
        np.testing.assert_array_equal(
            everything.compute(current).strain_tensor, result.strain_tensor
        )
    bare = AtomicStrain(reference, 3.2).compute(current)
    assert bare.strain_tensor is None and bare.d2min is None
    np.testing.assert_array_equal(bare.shear, result.shear)
    np.testing.assert_array_equal(bare.volumetric, result.volumetric)
    assert set(bare.columns()) == {"shear_strain", "volumetric_strain"}
    assert repr(everything) == "AtomicStrain(cutoff=3.2, affine=False)"


def test_flat_neighborhoods_are_nan():
    layer = mdapy.System(
        [[x * 2.5, y * 2.5, 5.0] for x in range(4) for y in range(4)],
        mdapy.Box([10.0, 10.0, 10.0], pbc=(True, True, False)),
        elements="C",
    )
    result = AtomicStrain(layer, 2.6, strain_tensor=True, d2min=True).compute(layer)
    assert np.isnan(result.strain_tensor).all() and np.isnan(result.d2min).all()
    assert np.isnan(result.shear).all() and np.isnan(result.volumetric).all()
    bare = AtomicStrain(layer, 2.6).compute(layer)
    assert np.isnan(bare.shear).all() and np.isnan(bare.volumetric).all()


def test_errors():
    reference = crystal("fcc", 3.6, (2, 2, 2))
    with pytest.raises(InvalidParameter, match="cutoff must be positive"):
        AtomicStrain(reference, 0.0)
    strain = AtomicStrain(reference, 3.0)
    with pytest.raises(InvalidParameter, match="the reference 32"):
        strain.compute(crystal("fcc", 3.6, (2, 2, 1)))
    with pytest.raises(InvalidParameter, match="periodicity"):
        strain.compute(reference.with_box(mdapy.Box(reference.box.matrix, pbc=False)))
    numbered = reference.with_columns(id=np.arange(1, 33))
    shuffled = numbered.with_columns(id=np.arange(32, 0, -1))
    with pytest.raises(InvalidParameter, match="not in the order of the reference"):
        AtomicStrain(numbered, 3.0).compute(shuffled)


@pytest.mark.parametrize("affine", [False, True])
def test_a_given_list_gives_what_a_search_gives(affine):
    rng = np.random.default_rng(4)
    reference = crystal("fcc", 3.6, (4, 4, 4), jitter=0.1, seed=4)
    current = deformed(reference, F0).with_columns(
        positions=deformed(reference, F0)["positions"] + rng.normal(0, 0.05, (256, 3))
    )
    options = {"affine": affine, "strain_tensor": True, "d2min": True}
    searched = AtomicStrain(reference, 3.0, **options).compute(current)
    for cutoff in (3.0, 4.5):  # the list's own cutoff, or a longer one
        nl = mdapy.NeighborList.build(reference, cutoff)
        listed = AtomicStrain(reference, 3.0, neighbors=nl, **options)
        result = listed.compute(current)
        for name in ("strain_tensor", "shear", "volumetric", "d2min"):
            np.testing.assert_allclose(
                getattr(result, name), getattr(searched, name), rtol=1e-12, atol=1e-14
            )
    strain, d2 = reference_strain(reference, current, 3.0)
    if not affine:
        np.testing.assert_allclose(searched.strain_tensor, strain, atol=1e-10)
        np.testing.assert_allclose(searched.d2min, d2, atol=1e-10)
    with pytest.raises(InvalidParameter, match="below the required"):
        AtomicStrain(reference, 3.0, neighbors=mdapy.NeighborList.build(reference, 2.0))
    with pytest.raises(InvalidParameter, match="other positions or another box"):
        AtomicStrain(reference, 3.0, neighbors=mdapy.NeighborList.build(current, 3.0))
