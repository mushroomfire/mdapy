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
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics import (
    SheetElasticResult,
    SheetQuasiHarmonicResult,
    elastic_constants,
    md_elastic_constants,
    phonons,
    quasi_harmonic,
)
from mdapy.lattice_dynamics._quasi_harmonic import _Fit
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


def mean_box(system, potential, barostat):
    """The mean box of the second half of a run near 0 K under no stress."""
    with MolecularDynamics(system, potential, timestep=0.0005) as md:
        md.velocities(10.0, seed=2)
        md.run(
            4000,
            ensemble="npt",
            temperature=5.0,
            pressure=0.0,
            barostat=barostat,
            barostat_damping=0.2,
            save_every=20,
        )
        boxes = np.array([f.box.matrix for f in md.frames])
    return boxes[len(boxes) // 2 :].mean(axis=0)


def strained(system, tensor):
    deformation = np.eye(3) + np.asarray(tensor)
    box = system.box
    return system.with_box(
        mdapy.Box(box.matrix @ deformation.T, origin=box.origin, pbc=box.pbc)
    ).with_columns(
        positions=box.origin + (system["positions"] - box.origin) @ deformation.T
    )


@pytest.mark.parametrize("barostat", ["iso", "aniso", "tri"])
def test_the_barostats_of_a_sheet(potential, graphene, barostat):
    """Stretched along x and sheared in its plane, a sheet under no stress:
    "iso" scales its vectors alike, keeping their shape; "aniso" relaxes
    each length, keeping the shear; "tri" relaxes the shear too. The height
    stays."""
    sheet = replicate(graphene, (4, 4, 1))
    start = strained(sheet, [[0.02, 0.01, 0], [0, 0, 0], [0, 0, 0]])
    box = mean_box(start, potential, barostat)
    rest, begin = sheet.box.matrix, start.box.matrix
    assert box[2, 2] == pytest.approx(rest[2, 2], rel=1e-9)
    if barostat == "iso":
        # the shape of the start, scaled
        scale = np.linalg.norm(box[0]) / np.linalg.norm(begin[0])
        np.testing.assert_allclose(box[:2], scale * begin[:2], atol=1e-6)
        assert scale < 1
    else:
        # the lengths relax: x near its rest, as the sheet is stiff
        assert box[0, 0] == pytest.approx(rest[0, 0], rel=2e-3)
        # the tilt, x of the second vector, stays or relaxes
        tilt = box[1, 0]
        expected = begin[1, 0] if barostat == "aniso" else rest[1, 0]
        assert tilt == pytest.approx(expected, abs=0.01 * np.linalg.norm(rest[1]))


def test_the_constants_from_dynamics_near_0_k(potential, graphene):
    """At 1 K the sheet hardly moves: the static constants in N/m, but for
    the strain of 1 %, with their errors."""
    sheet = replicate(graphene, (4, 4, 1))
    static = elastic_constants(graphene, potential, symmetry=False).constants
    result = md_elastic_constants(
        sheet,
        potential,
        1.0,
        timestep=0.0005,
        equilibration=500,
        production=1000,
        every=10,
    )
    assert isinstance(result, SheetElasticResult)
    np.testing.assert_allclose(result.constants, static, atol=1e-2 * static.max())
    assert result.errors.shape == (3, 3) and result.errors.max() < 1.0
    assert result.area_modulus() == pytest.approx(
        (static[0, 0] + static[0, 1]) / 2, rel=1e-2
    )


QHA = {"supercell": (6, 6, 1), "mesh": 20, "symmetry": False}


@pytest.fixture(scope="module")
def heating(potential, graphene):
    temperatures = np.array([0.0, 99.0, 100.0, 101.0, 299.0, 300.0, 301.0, 600.0])
    return quasi_harmonic(graphene, potential, temperatures, **QHA)


def test_the_quasi_harmonic_sheet(graphene, heating):
    """Graphene of this potential: its zero-point vibrations stretch it, and
    then it shrinks as it warms, as its vibrations across the plane soften
    when it is stretched."""
    assert isinstance(heating, SheetQuasiHarmonicResult)
    area = np.linalg.norm(np.cross(*graphene.box.matrix[:2])) / len(graphene)
    np.testing.assert_allclose(heating.areas, area * np.linspace(1, 1.04, 9))
    assert heating.areas.shape == (9,) and heating.free_energies.shape == (9, 8)
    assert heating.area[0] > area
    assert np.all(np.diff(heating.area) < 0)
    assert heating.thermal_expansion[0] == 0 and np.all(
        heating.thermal_expansion[1:] < 0
    )
    assert np.all(np.diff(heating.gibbs_energy) < 0)
    assert np.all(np.diff(heating.area_modulus) < 0)
    for cell in heating.cells:  # the height stays
        assert cell.box.matrix[2, 2] == graphene.box.matrix[2, 2]


@pytest.mark.parametrize("t", [100.0, 300.0])
def test_quasi_harmonic_identities_of_a_sheet(heating, t):
    k = list(heating.temperatures).index(t)
    g, a = heating.gibbs_energy, heating.area
    # S = -dG/dT, alpha = dA/dT / A, C_P = -T d2G/dT2, by central differences
    assert -(g[k + 1] - g[k - 1]) / 2 == pytest.approx(heating.entropy[k], rel=1e-4)
    assert (a[k + 1] - a[k - 1]) / 2 / a[k] == pytest.approx(
        heating.thermal_expansion[k], rel=2e-2
    )
    assert -t * (g[k + 1] - 2 * g[k] + g[k - 1]) == pytest.approx(
        heating.heat_capacity[k], rel=2e-2
    )


def test_the_static_area_modulus(potential, graphene, heating):
    """The curvature of the static energy in the area, at its minimum, the
    first area, is the area modulus of the elastic constants, in N/m."""
    c = elastic_constants(graphene, potential, symmetry=False).constants
    fit = _Fit(heating.areas, heating.energies, -1.0)
    a0, h = heating.areas[0], 1e-4 * heating.areas[0]
    second = (fit(a0 + h) - 2 * fit(a0) + fit(a0 - h)) / h**2
    modulus = a0 * second * units.EV_PER_A2_TO_J_PER_M2
    assert modulus == pytest.approx((c[0, 0] + c[0, 1]) / 2, rel=5e-3)
    assert heating.area_modulus[0] < modulus


def test_a_sheet_under_tension(potential, graphene, heating):
    """dG/dp = A, the pressure in the plane in N/m; and the cells under
    tensions, from pressures in N/m."""
    h = 0.5
    pulled = [
        quasi_harmonic(graphene, potential, [300.0], pressure=p, **QHA) for p in (-h, h)
    ]
    assert pulled[0].pressure == -h and pulled[0].area[0] > pulled[1].area[0]
    slope = (pulled[1].gibbs_energy[0] - pulled[0].gibbs_energy[0]) / (
        2 * h / units.EV_PER_A2_TO_J_PER_M2
    )
    k = list(heating.temperatures).index(300.0)
    assert slope == pytest.approx(heating.area[k], rel=1e-3)
    tensions = quasi_harmonic(
        graphene,
        potential,
        [300.0],
        pressures=[-3.0, -2.0, -1.0, -0.5, -0.25],
        **QHA,
    )
    c = elastic_constants(graphene, potential, symmetry=False).constants
    # the cells stretched by their tension, the least by it over the area
    # modulus; the equilibrium at 300 K among them
    a0 = np.linalg.norm(np.cross(*graphene.box.matrix[:2])) / len(graphene)
    assert np.all(np.diff(tensions.areas) > 0)
    assert tensions.areas[0] / a0 - 1 == pytest.approx(
        0.25 / ((c[0, 0] + c[0, 1]) / 2), rel=0.01
    )
    assert tensions.area[0] == pytest.approx(heating.area[k], rel=1e-4)


def test_a_compressed_sheet_bends(potential, graphene):
    with pytest.raises(InvalidParameter, match="at area ratio 0.99.*bends"):
        quasi_harmonic(
            graphene, potential, [300.0], volumes=np.linspace(0.99, 1.03, 5), **QHA
        )
