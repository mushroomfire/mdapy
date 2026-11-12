# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Mode Grüneisen parameters: springs of a simple cubic lattice at rest,
whose frequencies change with the volume in closed form. phonopy on the same
force constants is in test_force_constants_phonopy.py."""

import numpy as np
import pytest

from mdapy.build import build_crystal
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics import phonons

from calculators.test_testing import Harmonic

A = 2.5


def scaled(cell, volume):
    s = volume ** (1 / 3)
    return cell.with_box(cell.box.matrix * s).with_columns(
        positions=cell["positions"] * s
    )


@pytest.fixture(scope="module")
def three():
    """Springs at rest between nearest neighbors, at three volumes."""
    cell = build_crystal("sc", A, elements="Ar")
    springs = Harmonic(k=1.3, r0=A)
    return [
        phonons(
            scaled(cell, v), springs, supercell=3, displacement=1e-5, symmetry=False
        )
        for v in (1 - 1e-4, 1.0, 1 + 1e-4)
    ]


def test_springs_of_a_simple_cubic_lattice(three):
    """Along x the mode feels k (1 - cos q_x a) along the bonds and, in
    tension, k (1 - r0/a) (2 - cos q_y a - cos q_z a) across them: at rest
    gamma = -(2 - cos q_y a - cos q_z a) / (6 (1 - cos q_x a))."""
    smaller, middle, larger = three
    q = np.random.default_rng(0).random((10, 3)) - 0.5
    found = middle.gruneisen(q, smaller=smaller, larger=larger)
    c = np.cos(2 * np.pi * q)
    each = [
        -(2 - c[:, (x + 1) % 3] - c[:, (x + 2) % 3]) / (6 * (1 - c[:, x]))
        for x in range(3)
    ]
    each = np.stack(each, axis=1)
    order = np.argsort(1 - c, axis=1)  # the modes by frequency
    expected = np.take_along_axis(each, order, axis=1)
    np.testing.assert_allclose(found, expected, rtol=1e-5)
    single = middle.gruneisen(q[0], smaller=smaller, larger=larger)
    np.testing.assert_allclose(single, found[0])


def test_the_translations_have_none(three):
    smaller, middle, larger = three
    found = middle.gruneisen([0, 0, 0], smaller=smaller, larger=larger)
    assert np.isnan(found).all()


def test_errors(three):
    smaller, middle, larger = three
    with pytest.raises(InvalidParameter, match="smaller and larger"):
        middle.gruneisen([0.1, 0.2, 0.3], smaller=larger, larger=smaller)
    with pytest.raises(InvalidParameter, match="wave vector"):
        middle.gruneisen([0.1, 0.2], smaller=smaller, larger=larger)
    cell = build_crystal("fcc", A * np.sqrt(2), elements="Ar")
    other = phonons(cell, Harmonic(k=1.3, r0=A), supercell=2, symmetry=False)
    with pytest.raises(InvalidParameter, match="same atoms"):
        middle.gruneisen([0.1, 0.2, 0.3], smaller=smaller, larger=other)
