# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Phonons and elastic constants with the symmetry of the crystal: the same
results from fewer computations."""

import sys

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics import (
    elastic_constants,
    phonons,
    primitive_cell,
    quasi_harmonic,
    thermal_elastic_constants,
)
from mdapy.simulation import minimize

from calculators.eam_files import write_setfl

pytest.importorskip("spglib")


class Counting:
    """A calculator that counts its computations."""

    def __init__(self, inner):
        self.inner, self.calls = inner, 0
        self.cutoff, self.implemented = inner.cutoff, inner.implemented

    def compute(self, system, properties):
        self.calls += 1
        return self.inner.compute(system, properties)


@pytest.fixture(scope="module")
def potential(tmp_path_factory):
    return EAM(write_setfl(tmp_path_factory.mktemp("eam") / "CuNi.eam.alloy"))


def relaxed(potential, structure, a, elements="Cu"):
    return minimize(
        build_crystal(structure, a, elements=elements),
        potential,
        cell="full",
        max_force=1e-10,
        max_stress=1e-8,
        max_steps=20000,
    ).system


@pytest.fixture(scope="module")
def crystals(potential):
    fcc = relaxed(potential, "fcc", 3.29)
    # a monoclinic cell: the cubic cell sheared, its atoms relaxed
    box = mdapy.Box(fcc.box.matrix + [[0, 0, 0], [0, 0, 0], [0.3, 0, 0]])
    sheared = fcc.with_box(box).with_columns(
        positions=box.to_cartesian(fcc.box.to_fractional(fcc["positions"]))
    )
    return {
        "fcc primitive": (primitive_cell(fcc), 1, 6),
        "fcc cubic": (fcc, 1, 24),
        "bcc primitive": (primitive_cell(relaxed(potential, "bcc", 2.6)), 1, 6),
        "L1_2": (relaxed(potential, "l1_2", 3.29, ("Cu", "Ni")), 3, 24),
        "monoclinic": (
            minimize(sheared, potential, max_force=1e-10, max_steps=20000).system,
            3,
            24,
        ),
        "hcp": (relaxed(potential, "hcp", 2.33), 2, 12),
    }


@pytest.mark.parametrize(
    "name",
    ["fcc primitive", "fcc cubic", "bcc primitive", "L1_2", "monoclinic", "hcp"],
)
def test_phonons_from_fewer_displacements(potential, crystals, name):
    cell, fewer, all_of_them = crystals[name]
    counting = Counting(potential)
    plain = phonons(cell, counting, supercell=3, symmetry=False)
    assert counting.calls == all_of_them
    counting.calls = 0
    symmetric = phonons(cell, counting, supercell=3, symmetry=True)
    assert counting.calls == fewer
    # rotations by 90° take displacements along axes onto axes: the same
    # force constants; those of hcp turn them by 60°, which changes the
    # error of the differences, of the order of the square of the
    # displacement
    tolerance = 1e-3 if name == "hcp" else 1e-9
    np.testing.assert_allclose(
        symmetric.force_constants, plain.force_constants, rtol=0, atol=tolerance
    )


def test_hcp_differences_shrink_with_the_displacement(potential, crystals):
    cell = crystals["hcp"][0]
    differences = []
    for h in (0.01, 0.002):
        plain = phonons(cell, potential, supercell=3, displacement=h, symmetry=False)
        symmetric = phonons(cell, potential, supercell=3, displacement=h, symmetry=True)
        differences.append(
            np.abs(plain.force_constants - symmetric.force_constants).max()
        )
    assert differences[1] < differences[0] / 15


def test_by_default_whenever_spglib_is_installed(potential, crystals, monkeypatch):
    cell = crystals["fcc primitive"][0]
    counting = Counting(potential)
    phonons(cell, counting, supercell=3)
    assert counting.calls == 1
    monkeypatch.setitem(sys.modules, "spglib", None)
    counting.calls = 0
    phonons(cell, counting, supercell=3)
    assert counting.calls == 6
    with pytest.raises(InvalidParameter, match="pip install spglib"):
        phonons(cell, potential, supercell=3, symmetry=True)


def test_elastic_constants_from_fewer_strains(potential, crystals):
    cell = crystals["fcc cubic"][0]
    counting = Counting(potential)
    plain = elastic_constants(cell, counting, relax=False, symmetry=False)
    assert counting.calls == 12
    counting.calls = 0
    symmetric = elastic_constants(cell, counting, relax=False, symmetry=True)
    # stretches along x both ways, a shear one way: the others are images
    assert counting.calls == 3
    np.testing.assert_allclose(symmetric.constants, plain.constants, rtol=0, atol=1e-6)
    relaxed_ = elastic_constants(cell, potential, max_force=1e-8, symmetry=True)
    np.testing.assert_allclose(
        relaxed_.constants,
        elastic_constants(cell, potential, max_force=1e-8, symmetry=False).constants,
        rtol=0,
        atol=1e-6,
    )


@pytest.fixture(scope="module")
def hexagonal(potential, crystals):
    cell = crystals["hcp"][0]
    qha = quasi_harmonic(cell, potential, [0.0, 300.0], supercell=4, mesh=8)
    return qha, thermal_elastic_constants(qha, potential, supercell=4, mesh=8)


def test_thermal_elastic_constants_of_hcp(potential, hexagonal):
    qha, result = hexagonal
    for c in result.isothermal:
        # hexagonal: C44 = C55 and C66 = (C11 - C12) / 2
        assert c[3, 3] == pytest.approx(c[4, 4], rel=1e-4)
        assert c[5, 5] == pytest.approx((c[0, 0] - c[0, 1]) / 2, rel=1e-2)
        # equal to second order; the differences of finite strains differ a
        # little in the plane, where only the symmetry of the crystal holds
        assert c[0, 0] == pytest.approx(c[1, 1], rel=1e-3)


def test_thin_supercells_are_close_to_thick_ones(potential, crystals):
    # a supercell of 4 primitive cells is thinner than the force constants
    # reach; the strained cells keep the periodic images of the cell, so
    # that it is close to one of 6
    cell = crystals["fcc primitive"][0]
    constants = []
    for size in (4, 6):
        qha = quasi_harmonic(cell, potential, [300.0], supercell=size, mesh=8)
        result = thermal_elastic_constants(qha, potential, supercell=size, mesh=8)
        constants.append(result.isothermal[0])
    np.testing.assert_allclose(constants[0], constants[1], rtol=0, atol=0.5)


def test_thermal_elastic_constants_without_symmetry(potential, hexagonal):
    qha, result = hexagonal
    plain = thermal_elastic_constants(
        qha, potential, supercell=4, mesh=8, symmetry=False
    )
    np.testing.assert_allclose(
        plain.isothermal, result.isothermal, rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(plain.adiabatic, result.adiabatic, rtol=1e-4, atol=1e-6)
