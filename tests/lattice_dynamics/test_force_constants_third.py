# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Third order force constants fitted with the second: a Morse potential
between nearest neighbors, whose derivatives are known in closed form, gives
them exactly from the forces of its Taylor expansion, and to the square of
the displacement from its own forces. symfc on the same data is in
test_force_constants_symfc.py."""

import dataclasses

import numpy as np
import pytest

from mdapy.build import build_crystal
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics import (
    ForceConstants,
    _force_constants,
    displaced_supercells,
    force_constants,
    primitive_cell,
)
from mdapy.lattice_dynamics._phonons import _supercell, _supercell_matrix

# the fits use the symmetry of the crystals, found by spglib, which the
# development install has
pytest.importorskip("spglib")

D, ALPHA, R0, REACH = 0.4, 1.6, 2.45, 3.0  # eV, 1/Å, Å, Å


def derivatives(r):
    """The Morse energy of a pair and its first three derivatives."""
    e = np.exp(-ALPHA * (r - R0))
    return (
        D * (e * e - 2 * e),
        D * ALPHA * (-2 * e * e + 2 * e),
        D * ALPHA**2 * (4 * e * e - 2 * e),
        D * ALPHA**3 * (-8 * e * e + 2 * e),
    )


def tensors(vectors):
    """The gradient, Hessian and third derivatives of the energy of pairs of
    these vectors, with respect to the vector."""
    r = np.linalg.norm(vectors, axis=1)
    e = vectors / r[:, None]
    _, first, second, third = derivatives(r)
    eye = np.eye(3)
    ee = np.einsum("ma,mb->mab", e, e)
    eee = np.einsum("ma,mb,mc->mabc", e, e, e)
    gradient = first[:, None] * e
    hessian = second[:, None, None] * ee + (first / r)[:, None, None] * (eye - ee)
    mixed = (
        np.einsum("ab,mc->mabc", eye, e)
        + np.einsum("ac,mb->mabc", eye, e)
        + np.einsum("bc,ma->mabc", eye, e)
    )
    cubic = third[:, None, None, None] * eee + (second / r - first / r**2)[
        :, None, None, None
    ] * (mixed - 3 * eee)
    return gradient, hessian, cubic


def bonds(big):
    positions = big["positions"]
    vectors = big.box.minimum_image(
        (positions[None] - positions[:, None]).reshape(-1, 3)
    ).reshape(len(big), len(big), 3)
    lengths = np.linalg.norm(vectors, axis=2)
    i, j = np.nonzero((lengths > 1e-9) & (lengths < REACH))
    return i, j, vectors[i, j]


def taylor(big, u):
    """The forces of the expansion of the energy to third order."""
    i, j, vectors = bonds(big)
    gradient, hessian, cubic = tensors(vectors)
    s = u[j] - u[i]
    pull = (
        gradient
        + np.einsum("mab,mb->ma", hessian, s)
        + 0.5 * np.einsum("mabc,mb,mc->ma", cubic, s, s)
    )
    forces = np.zeros_like(u)
    np.add.at(forces, i, pull)
    return forces


def exact(big, u):
    """The forces of the potential itself."""
    positions = big["positions"] + u
    i, j, _ = bonds(big)
    gradient, _, _ = tensors(big.box.minimum_image(positions[j] - positions[i]))
    forces = np.zeros_like(u)
    np.add.at(forces, i, gradient)
    return forces


def closed(cell, big):
    """The second and third order force constants of the potential."""
    n, count = len(cell), len(big)
    i, j, vectors = bonds(big)
    _, hessian, cubic = tensors(vectors)
    second = np.zeros((n, count, 3, 3))
    third = np.zeros((n, count, count, 3, 3, 3))
    for a, b, h, t in zip(i, j, hessian, cubic):
        if a >= n:
            continue
        second[a, b] -= h
        second[a, a] += h
        third[a, b, b] -= t
        third[a, a, b] += t
        third[a, b, a] += t
        third[a, a, a] -= t
    return second, third


CASES = {
    "fcc": ("fcc", 3.6, 4),
    "hcp": ("hcp", 2.55, (4, 4, 3)),
    "diamond": ("diamond", 6.0, 4),
    # a cutoff longer than a third of the supercell
    "fcc 3x3x3": ("fcc", 3.6, 3),
}


def crystal(name):
    structure, a, supercell = CASES[name]
    cell = build_crystal(structure, a, elements="Cu")
    if structure != "hcp":
        cell = primitive_cell(cell)
    matrix = _supercell_matrix(supercell)
    return cell, supercell, _supercell(cell, matrix)


def frames(cell, supercell, big, forces, displacement, count=2):
    moved = displaced_supercells(
        cell,
        supercell=supercell,
        method="random",
        displacement=displacement,
        count=count,
        seed=1,
    )
    return [
        f.with_columns(forces=forces(big, f["positions"] - big["positions"]))
        for f in moved
    ]


@pytest.mark.parametrize("name", CASES)
def test_the_taylor_expansion_comes_back_exactly(name):
    cell, supercell, big = crystal(name)
    second, third = closed(cell, big)
    assert np.abs(exact(big, np.zeros((len(big), 3)))).max() < 1e-12  # at rest
    fit = force_constants(
        cell,
        frames(cell, supercell, big, taylor, 0.05),
        cutoff=REACH,
        cutoff_third=REACH,
    )
    assert fit.cutoff_third == REACH and fit.parameters_third > 0
    assert fit.error < 1e-12
    np.testing.assert_allclose(fit.second, second, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fit.third, third, rtol=0, atol=1e-12)
    # the order of the two atoms of the supercell, and the sum rule
    np.testing.assert_allclose(
        fit.third, fit.third.transpose(0, 2, 1, 3, 5, 4), rtol=0, atol=1e-13
    )
    np.testing.assert_allclose(fit.third.sum(axis=2), 0, atol=1e-12)


def test_the_forces_of_the_potential_to_the_square_of_the_displacement():
    """With displacements by plus and minus, the errors of both orders fall
    as the square of the displacement."""
    cell, supercell, big = crystal("fcc")
    second, third = closed(cell, big)
    errors = []
    for h in (0.02, 0.01):
        fit = force_constants(
            cell,
            frames(cell, supercell, big, exact, h),
            cutoff=REACH,
            cutoff_third=REACH,
        )
        errors.append(
            (
                np.linalg.norm(fit.second - second) / np.linalg.norm(second),
                np.linalg.norm(fit.third - third) / np.linalg.norm(third),
            )
        )
    for coarse, fine in zip(*errors):
        assert 3.5 < coarse / fine < 4.5
    assert errors[1][1] < 1e-3


def test_the_second_order_is_that_of_a_fit_of_it_alone():
    """The forces of plus and minus a pattern: the third order is even in
    the displacements and the second odd, so they part."""
    cell, supercell, big = crystal("hcp")
    moved = frames(cell, supercell, big, exact, 0.03)
    alone = force_constants(cell, moved, cutoff=REACH)
    both = force_constants(cell, moved, cutoff=REACH, cutoff_third=REACH)
    np.testing.assert_allclose(both.second, alone.second, rtol=0, atol=1e-12)
    assert both.error < alone.error
    assert alone.third is None and alone.cutoff_third is None
    assert alone.parameters_third is None


def test_third_order_needs_random_displacements():
    cell, supercell, big = crystal("fcc")
    moved = displaced_supercells(cell, supercell=supercell)
    moved = [
        f.with_columns(forces=exact(big, f["positions"] - big["positions"]))
        for f in moved
    ]
    with pytest.raises(InvalidParameter, match="third order needs random"):
        force_constants(cell, moved, cutoff_third=REACH)


def test_force_constants_with_third_order():
    cell, supercell, big = crystal("fcc")
    second, third = closed(cell, big)
    constants = ForceConstants(cell, supercell, second, third)
    np.testing.assert_array_equal(constants.third, third)
    with pytest.raises(ValueError):
        constants.third[0, 0, 0, 0, 0, 0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        constants.third = third
    assert ForceConstants(cell, supercell, second).third is None
    with pytest.raises(InvalidParameter, match="third must be of shape"):
        ForceConstants(cell, supercell, second, third[:, :1])
    with pytest.raises(InvalidParameter, match="third must be finite"):
        ForceConstants(cell, supercell, second, np.full_like(third, np.nan))


def test_errors(monkeypatch):
    cell, supercell, big = crystal("fcc")
    moved = frames(cell, supercell, big, exact, 0.01, count=1)
    with pytest.raises(InvalidParameter, match="cutoff_third must be positive"):
        force_constants(cell, moved, cutoff_third=5.2)
    monkeypatch.setattr(_force_constants, "_MEMORY", 1e5)
    with pytest.raises(InvalidParameter, match="GB: too many"):
        force_constants(cell, moved, cutoff=REACH, cutoff_third=REACH)
    second, third = closed(cell, big)
    with pytest.raises(InvalidParameter, match="GB: too many"):
        ForceConstants(cell, supercell, second, third)
