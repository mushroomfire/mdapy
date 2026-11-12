# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Thermal conductivity in the relaxation-time approximation: the
irreducible wave vectors give what the whole mesh gives, the widths go as
the square of the third order and as the temperature where it is high, the
symmetry of the crystal holds. phono3py on the same force constants is in
test_conductivity_phono3py.py."""

import dataclasses

import numpy as np
import pytest

import mdapy
from mdapy import units
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics import (
    ForceConstants,
    displaced_supercells,
    force_constants,
    primitive_cell,
    thermal_conductivity,
)
from mdapy.simulation import minimize

from calculators.eam_files import write_setfl

pytest.importorskip("spglib")


@pytest.fixture(scope="module")
def copper(tmp_path_factory):
    """The force constants of fcc copper of the test EAM, to third order of
    the nearest neighbors."""
    potential = EAM(write_setfl(tmp_path_factory.mktemp("eam") / "CuNi.eam.alloy"))
    cell = minimize(
        build_crystal("fcc", 3.29, elements="Cu"),
        potential,
        cell="full",
        max_force=1e-10,
        max_stress=1e-8,
        max_steps=10000,
    ).system
    cell = primitive_cell(cell)
    frames = displaced_supercells(
        cell, supercell=4, method="random", displacement=0.01, count=3, seed=3
    )
    frames = [
        f.with_columns(forces=potential.compute(f, ("forces",)).forces) for f in frames
    ]
    return force_constants(cell, frames, cutoff_third=2.8)


def test_the_irreducible_wave_vectors_give_the_whole_mesh(copper):
    """The widths of the modes of the irreducible wave vectors are those of
    the whole mesh; the conductivity of the cubic crystal is one number."""
    reduced = thermal_conductivity(copper, [300.0], mesh=6, sigma=0.2)
    whole = thermal_conductivity(copper, [300.0], mesh=6, sigma=0.2, symmetry=False)
    # time reversal alone: q and -q
    assert len(whole.q) == 112 and whole.weights.sum() == 216
    assert len(reduced.q) == 16 and reduced.weights.sum() == 216
    at = [int(np.flatnonzero((whole.q == q).all(axis=1))[0]) for q in reduced.q]
    np.testing.assert_allclose(
        reduced.gamma, whole.gamma[:, at], rtol=1e-10, atol=1e-14
    )
    np.testing.assert_allclose(reduced.frequencies, whole.frequencies[at])
    kappa = reduced.kappa[0]
    np.testing.assert_allclose(kappa, kappa[0, 0] * np.eye(3), atol=1e-10 * kappa[0, 0])
    # a mesh of less symmetry than the crystal keeps the rotations that map
    # it onto itself
    reduced = thermal_conductivity(copper, [300.0], mesh=(4, 4, 6), sigma=0.2)
    whole = thermal_conductivity(
        copper, [300.0], mesh=(4, 4, 6), sigma=0.2, symmetry=False
    )
    assert len(whole.q) > len(reduced.q) and reduced.weights.sum() == 96
    at = [int(np.flatnonzero((whole.q == q).all(axis=1))[0]) for q in reduced.q]
    np.testing.assert_allclose(
        reduced.gamma, whole.gamma[:, at], rtol=1e-10, atol=1e-14
    )
    assert repr(reduced) == (
        "ThermalConductivity(1 temperatures, 34 irreducible wave vectors)"
    )


def test_the_same_for_any_number_of_threads(copper):
    before = mdapy.get_num_threads()
    try:
        mdapy.set_num_threads(1)
        one = thermal_conductivity(copper, [300.0], mesh=5, sigma=0.2)
        mdapy.set_num_threads(4)
        four = thermal_conductivity(copper, [300.0], mesh=5, sigma=0.2)
    finally:
        mdapy.set_num_threads(before)
    np.testing.assert_array_equal(one.gamma, four.gamma)
    np.testing.assert_array_equal(one.kappa, four.kappa)


def test_widths_go_as_the_square_of_the_third_order(copper):
    doubled = ForceConstants(
        copper.cell, copper.supercell, copper.second, 2 * copper.third
    )
    one = thermal_conductivity(copper, [300.0], mesh=5, sigma=0.2)
    two = thermal_conductivity(doubled, [300.0], mesh=5, sigma=0.2)
    np.testing.assert_allclose(two.gamma, 4 * one.gamma, rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(two.kappa, one.kappa / 4, rtol=1e-12)
    none = ForceConstants(
        copper.cell, copper.supercell, copper.second, 0 * copper.third
    )
    still = thermal_conductivity(none, [300.0], mesh=5, sigma=0.2)
    assert (still.gamma == 0).all() and np.isinf(still.lifetimes).all()
    assert (still.kappa == 0).all()


def test_hot_crystals(copper):
    """Far above the Debye temperature the occupations go as T: the widths
    double from 3000 to 6000 K, the conductivity halves, and each mode holds
    k_B of heat."""
    hot = thermal_conductivity(copper, [3000.0, 6000.0], mesh=5, sigma=0.2)
    moving = hot.frequencies > 0.1
    ratio = hot.gamma[1][moving] / hot.gamma[0][moving]
    np.testing.assert_allclose(ratio, 2.0, rtol=0.01)
    assert hot.kappa[1, 0, 0] == pytest.approx(hot.kappa[0, 0, 0] / 2, rel=0.01)
    np.testing.assert_allclose(
        hot.heat_capacities[1][moving], units.BOLTZMANN, rtol=1e-3
    )
    np.testing.assert_allclose(
        hot.lifetimes[0][moving], 1 / (4 * np.pi * hot.gamma[0][moving])
    )


def test_the_results_cannot_change(copper):
    result = thermal_conductivity(copper, 300.0, mesh=4, sigma=0.3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.kappa = None
    for name in ("kappa", "gamma", "q", "weights", "group_velocities"):
        with pytest.raises(ValueError):
            getattr(result, name)[0] = 0


def test_errors(copper):
    second = ForceConstants(copper.cell, copper.supercell, copper.second)
    with pytest.raises(InvalidParameter, match="third order"):
        thermal_conductivity(second, [300.0], mesh=4, sigma=0.2)
    for temperatures in ([0.0], [-10.0], [[300.0]]):
        with pytest.raises(InvalidParameter, match="temperatures"):
            thermal_conductivity(copper, temperatures, mesh=4, sigma=0.2)
    with pytest.raises(InvalidParameter, match="sigma"):
        thermal_conductivity(copper, [300.0], mesh=4, sigma=0.0)
    with pytest.raises(InvalidParameter, match="mesh"):
        thermal_conductivity(copper, [300.0], mesh=0, sigma=0.2)
    side = 2.5
    plane = np.array(
        [[side, 0, 0], [-side / 2, side * np.sqrt(3) / 2, 0], [0, 0, 10.0]]
    )
    sheet = mdapy.System(
        np.array([[0, 0, 5.0]]),
        mdapy.Box(plane, pbc=(True, True, False)),
        elements=["C"],
    )
    flat = ForceConstants(
        sheet, 2, np.zeros((1, 4, 3, 3)), np.zeros((1, 4, 4, 3, 3, 3))
    )
    with pytest.raises(InvalidParameter, match="periodic in every direction"):
        thermal_conductivity(flat, [300.0], mesh=4, sigma=0.2)


def single(values, levels):
    """(M, L) the tetrahedron weights of M points, of (M, G, 4) values at the
    corners of their tetrahedra, at the levels: those of three phonons whose
    second frequencies are zero, from the kernel."""
    from mdapy._core import analysis

    count, around, _ = values.shape
    found = []
    for level in levels:
        joined, _ = analysis.tetrahedron_triplet_weights(
            np.ascontiguousarray(values[..., None], dtype=float),
            np.zeros((count, around, 4, 1)),
            np.array([level]),
            1e-10,
        )
        found.append(joined)
    return np.stack(found, axis=1)


@pytest.mark.parametrize(
    "corners",
    [
        (0.0, 1.0, 2.0, 3.5),
        (2.0, 0.0, 3.0, 1.0),
        (1.0, 1.0, 2.0, 3.0),
        (0.0, 2.0, 2.0, 2.0),
    ],
)
def test_the_weights_of_a_tetrahedron_integrate_to_one(corners):
    """The weight of a corner of one tetrahedron, over all levels, is a
    quarter of it, which the function returns as one; the point itself may
    be any corner, and corners may be equal."""
    corners = np.array(corners)
    levels = np.linspace(-0.5, 4.0, 4501)
    for first in range(4):
        order = [first] + [k for k in range(4) if k != first]
        weights = single(corners[order][None, None], levels)[0]
        assert np.trapezoid(weights, levels) == pytest.approx(1.0, abs=2e-3)


def test_the_weights_sum_to_the_density_of_states(copper):
    """Over all points of the mesh, the weights of the frequencies of the
    modes are the density of states of the tetrahedron method of Phonons."""
    from mdapy.lattice_dynamics._conductivity import _tetrahedra_around

    phonons = copper.phonons()
    counts = np.array([6, 6, 6])
    axes = [np.arange(c) for c in counts]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    frequencies = phonons.frequencies(grid / counts)
    reciprocal = 2 * np.pi * np.linalg.inv(copper.cell.box.matrix).T
    around = _tetrahedra_around(counts, reciprocal)
    corners = (grid[:, None, None] + around) % counts
    flat = (corners[..., 0] * counts[1] + corners[..., 1]) * counts[2] + corners[..., 2]
    levels = np.linspace(0.5, 8.0, 40)
    total = sum(
        single(frequencies[flat, band], levels).sum(axis=0) for band in range(3)
    ) / len(grid)
    expected = phonons.dos(6, frequencies=levels).dos  # per atom, one atom
    np.testing.assert_allclose(total, expected, rtol=1e-10, atol=1e-12)


def test_the_tetrahedra_keep_the_symmetry(copper):
    result = thermal_conductivity(copper, [300.0], mesh=5)
    kappa = result.kappa[0]
    np.testing.assert_allclose(kappa, kappa[0, 0] * np.eye(3), atol=1e-10 * kappa[0, 0])
