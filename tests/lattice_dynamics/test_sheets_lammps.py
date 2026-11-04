# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Graphene, with the Tersoff potential in LAMMPS. Skipped where LAMMPS is
not installed."""

import numpy as np
import pytest

import mdapy
from mdapy import units
from mdapy.build import build_crystal
from mdapy.calculators import LAMMPS
from mdapy.elements import atomic_masses
from mdapy.lattice_dynamics import elastic_constants, phonons
from mdapy.simulation import MolecularDynamics, minimize
from mdapy.transform import replicate

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


def test_the_constants_as_second_derivatives_of_the_energy(potential, graphene):
    """The relaxed energy under small strains, against the stress of the constants."""
    area = np.linalg.norm(np.cross(graphene.box.matrix[0], graphene.box.matrix[1]))
    rest = potential.compute(graphene).energy

    def energy(xx, yy, xy):
        strain = np.array([[1 + xx, xy / 2, 0], [xy / 2, 1 + yy, 0], [0, 0, 1]])
        box = mdapy.Box(graphene.box.matrix @ strain, pbc=graphene.box.pbc)
        stretched = graphene.with_box(box, scale_positions=True)
        relaxed = minimize(stretched, potential, max_force=1e-7, max_steps=3000)
        return potential.compute(relaxed.system).energy

    step = 0.002

    def curvature(*mode):
        up = energy(*(step * m for m in mode))
        down = energy(*(-step * m for m in mode))
        return (up + down - 2 * rest) / step**2 / area * units.EV_PER_A2_TO_J_PER_M2

    c = elastic_constants(graphene, potential, symmetry=False).constants
    c11, c22 = curvature(1, 0, 0), curvature(0, 1, 0)
    assert c[0, 0] == pytest.approx(c11, rel=1e-3)
    assert c[1, 1] == pytest.approx(c22, rel=1e-3)
    assert c[0, 1] == pytest.approx((curvature(1, 1, 0) - c11 - c22) / 2, abs=0.5)
    assert c[2, 2] == pytest.approx(curvature(0, 0, 1), rel=1e-3)


def test_against_phonopy(potential, graphene):
    """With the same forces, in a periodic box with vacuum for phonopy."""
    phonopy = pytest.importorskip("phonopy")
    from phonopy.structure.atoms import PhonopyAtoms

    counts, distance = (6, 6, 1), 0.01
    ours = phonons(
        graphene, potential, supercell=counts, displacement=distance, symmetry=False
    )
    names = graphene.table["element"].cast(str).to_list()
    unit = PhonopyAtoms(
        symbols=names,
        cell=graphene.box.matrix,
        scaled_positions=graphene.box.to_fractional(graphene["positions"]),
        masses=atomic_masses(names),
    )
    theirs = phonopy.Phonopy(
        unit, supercell_matrix=np.diag(counts), primitive_matrix="P"
    )
    theirs.generate_displacements(distance=distance, is_plusminus=True)
    theirs.forces = np.array(
        [
            potential.compute(
                mdapy.System(s.positions, mdapy.Box(s.cell), elements=s.symbols),
                ("forces",),
            ).forces
            for s in theirs.supercells_with_displacements
        ]
    )
    theirs.produce_force_constants()
    q = (np.random.default_rng(0).random((30, 3)) - 0.5) * [1, 1, 0]
    theirs.run_qpoints(q)
    np.testing.assert_allclose(
        ours.frequencies(q), theirs.qpoints.frequencies, atol=0.02
    )


def test_a_flat_sheet_runs(potential, graphene):
    with MolecularDynamics(graphene, potential, timestep=0.0005) as md:
        md.velocities(20.0, seed=1)
        md.run(100, ensemble="nvt", temperature=10.0)
        assert md.system.box.matrix[2, 2] == pytest.approx(graphene.box.matrix[2, 2])


def test_a_tension_stretches_the_sheet_as_its_constants_say(potential, graphene):
    sheet = replicate(graphene, (4, 4, 1))
    tension = 2.0  # N/m
    pressure = -10 * tension / sheet.box.matrix[2, 2]
    c = elastic_constants(graphene, potential, symmetry=False).constants

    def mean_area(target):
        with MolecularDynamics(sheet, potential, timestep=0.0005) as md:
            md.velocities(10.0, seed=2)
            md.run(
                10000,
                ensemble="npt",
                temperature=5.0,
                pressure=target,
                barostat="xy",
                barostat_damping=0.2,
                thermo=20,
            )
            assert md.system.box.matrix[2, 2] == pytest.approx(sheet.box.matrix[2, 2])
            volume = md.thermo["volume"].to_numpy()
        return volume[len(volume) // 2 :].mean()

    strain = mean_area(pressure) / mean_area(0.0) - 1
    assert strain == pytest.approx(2 * tension / (c[0, 0] + c[0, 1]), rel=0.1)


def test_as_in_a_periodic_box_with_vacuum(potential, graphene):
    periodic = graphene.with_box(mdapy.Box(graphene.box.matrix))
    q = [[0.1, 0.2, 0], [1 / 3, 1 / 3, 0], [0.5, 0, 0]]
    sheet = phonons(graphene, potential, supercell=(6, 6, 1), symmetry=False)
    other = phonons(periodic, potential, supercell=(6, 6, 1), symmetry=False)
    np.testing.assert_allclose(sheet.frequencies(q), other.frequencies(q), atol=1e-6)
