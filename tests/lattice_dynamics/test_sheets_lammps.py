# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Graphene, with the Tersoff potential in LAMMPS. Skipped where LAMMPS is
not installed."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import LAMMPS
from mdapy.lattice_dynamics import elastic_constants, phonons
from mdapy.simulation import minimize

pytest.importorskip("lammps")

# the parameters of carbon by Lindsay and Broido (2010), as in the file
# BNC.tersoff of LAMMPS
TERSOFF = (
    "C C C 3.0 1.0 0.0 3.8049e4 4.3484 -0.93000 0.72751 1.5724e-7 "
    "2.2119 430.00 1.95 0.15 3.4879 1393.6\n"
)


@pytest.fixture(scope="module")
def potential(tmp_path_factory):
    path = tmp_path_factory.mktemp("tersoff") / "C.tersoff"
    path.write_text(TERSOFF)
    return LAMMPS(f"pair_style tersoff\npair_coeff * * {path} C", elements=["C"])


@pytest.fixture(scope="module")
def graphene(potential):
    crystal = build_crystal("graphene", 2.46, c=20.0, elements="C")
    crystal = crystal.with_columns(positions=crystal["positions"] + [0, 0, 10.0])
    sheet = crystal.with_box(mdapy.Box(crystal.box.matrix, pbc=(True, True, False)))
    result = minimize(
        sheet, potential, cell="full", max_force=1e-6, max_stress=1e-4, max_steps=5000
    )
    assert result.converged
    return result.system


def test_the_constants_are_those_of_an_isotropic_sheet(potential, graphene):
    c = elastic_constants(graphene, potential, symmetry=False).constants
    assert c[0, 0] == pytest.approx(c[1, 1], rel=1e-4)
    assert c[2, 2] == pytest.approx((c[0, 0] - c[0, 1]) / 2, rel=1e-3)
    assert abs(c[0, 2]) < 1e-3 * c[0, 0]
    assert 200 < c[0, 0] < 500  # N/m, the range of measurements and potentials


def test_the_vibrations_across_the_sheet_are_quadratic(potential, graphene):
    result = phonons(graphene, potential, supercell=(8, 8, 1), symmetry=False)
    np.testing.assert_allclose(result.frequencies([0, 0, 0])[:3], 0.0, atol=1e-3)
    low = result.frequencies([[x, 0, 0] for x in (0.02, 0.04)])[:, 0]
    assert low[1] / low[0] == pytest.approx(4.0, rel=0.05)
    linear = result.frequencies([[x, 0, 0] for x in (0.02, 0.04)])[:, 1]
    assert linear[1] / linear[0] == pytest.approx(2.0, rel=0.05)
    dos = result.dos(20)
    assert np.trapezoid(dos.dos, dos.frequencies) == pytest.approx(3.0, rel=0.02)


def test_as_in_a_periodic_box_with_vacuum(potential, graphene):
    periodic = graphene.with_box(mdapy.Box(graphene.box.matrix))
    q = [[0.1, 0.2, 0], [1 / 3, 1 / 3, 0], [0.5, 0, 0]]
    sheet = phonons(graphene, potential, supercell=(6, 6, 1), symmetry=False)
    other = phonons(periodic, potential, supercell=(6, 6, 1), symmetry=False)
    np.testing.assert_allclose(sheet.frequencies(q), other.frequencies(q), atol=1e-6)
