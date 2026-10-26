# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Phonons against phonopy, with the same forces. Skipped where phonopy is
not installed."""

import numpy as np
import pytest

import mdapy
from mdapy import units
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.elements import atomic_masses
from mdapy.lattice_dynamics import phonons, primitive_cell
from mdapy.lattice_dynamics._phonons import _supercell
from mdapy.simulation import minimize

from calculators.eam_files import write_setfl

phonopy = pytest.importorskip("phonopy")

# phonopy derives the force constants of other directions from one by
# symmetry; with central differences both are right to the square of the
# displacement, small enough here
DISPLACEMENT = 0.002


@pytest.fixture(scope="module")
def potential(tmp_path_factory):
    return EAM(write_setfl(tmp_path_factory.mktemp("eam") / "CuNi.eam.alloy"))


def relaxed(potential, structure, elements):
    start = build_crystal(structure, 3.29, elements=elements)
    return minimize(
        start, potential, cell="full", max_force=1e-10, max_stress=1e-8, max_steps=10000
    ).system


def both(cell, potential, counts):
    """The phonons of mdapy, and phonopy's with the cell as its primitive
    cell, from the forces of the same potential."""
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    ours = phonons(cell, potential, supercell=counts, displacement=DISPLACEMENT)
    names = cell.table["element"].cast(str).to_list()
    unit = PhonopyAtoms(
        symbols=names,
        cell=cell.box.matrix,
        scaled_positions=cell.box.to_fractional(cell["positions"]),
        masses=atomic_masses(names),
    )
    theirs = Phonopy(unit, supercell_matrix=np.diag(counts), primitive_matrix="P")
    theirs.generate_displacements(distance=DISPLACEMENT, is_plusminus=True)
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
    return ours, theirs


CELLS = {
    "fcc cubic cell": ("fcc", "Cu", False, (3, 3, 3)),
    "fcc primitive cell": ("fcc", "Cu", True, (4, 4, 4)),
    "L1_2 Cu3Ni": ("l1_2", ("Cu", "Ni"), False, (3, 3, 3)),
}


@pytest.mark.parametrize("name", CELLS)
def test_against_phonopy(potential, name):
    structure, elements, primitive, counts = CELLS[name]
    cell = relaxed(potential, structure, elements)
    if primitive:
        cell = primitive_cell(cell)
    n = len(cell)
    ours, theirs = both(cell, potential, counts)
    # the force constants, phonopy's supercell atoms matched to mdapy's
    big = _supercell(cell, np.diag(counts))
    apart = (
        big.box.to_fractional(big["positions"])[:, None]
        - theirs.supercell.scaled_positions
    )
    apart -= np.round(apart)
    ours_of = np.argmin(np.abs(apart).sum(axis=2), axis=0)
    constants = np.zeros_like(ours.force_constants)
    for a, s in enumerate(theirs.primitive.p2s_map):
        constants[a, ours_of] = theirs.force_constants[s]
    np.testing.assert_allclose(ours.force_constants, constants, atol=1e-4)
    # frequencies at any wave vector
    q = np.random.default_rng(0).random((30, 3)) - 0.5
    theirs.run_qpoints(q)
    np.testing.assert_allclose(
        ours.frequencies(q), theirs.qpoints.frequencies, atol=1e-4
    )
    # thermal properties per atom
    temperatures = [0.0, 100.0, 300.0, 900.0]
    theirs.run_mesh([10, 10, 10], is_gamma_center=True)
    theirs.run_thermal_properties(temperatures=temperatures, cutoff_frequency=1e-3)
    reference = theirs.thermal_properties
    thermal = ours.thermal(temperatures, mesh=10)
    per_atom = units.KJ_PER_MOL_TO_EV / n
    np.testing.assert_allclose(
        thermal.free_energy, reference.free_energy * per_atom, rtol=1e-5, atol=1e-8
    )
    np.testing.assert_allclose(
        thermal.entropy, reference.entropy * per_atom / 1000, rtol=1e-4, atol=1e-10
    )
    np.testing.assert_allclose(
        thermal.heat_capacity,
        reference.heat_capacity * per_atom / 1000,
        rtol=1e-4,
        atol=1e-10,
    )
    # densities of states: the tetrahedra of the whole mesh, and Gaussians
    theirs.run_mesh(
        [10, 10, 10],
        is_gamma_center=True,
        is_mesh_symmetry=False,
        with_eigenvectors=True,
    )
    theirs.run_total_dos(
        use_tetrahedron_method=True, freq_min=0, freq_max=9, freq_pitch=0.02
    )
    grid = theirs.total_dos.frequency_points
    dos = ours.dos(10, frequencies=grid).dos
    np.testing.assert_allclose(dos, theirs.total_dos.dos / n, atol=2e-3 * dos.max())
    theirs.run_projected_dos(sigma=0.1, freq_min=0, freq_max=9, freq_pitch=0.02)
    projected = ours.dos(10, frequencies=grid, sigma=0.1, projected=True).projected
    np.testing.assert_allclose(
        projected, theirs.projected_dos.projected_dos.T, atol=2e-3 * projected.max()
    )


def test_band_structure_against_phonopy(potential):
    cell = primitive_cell(relaxed(potential, "fcc", "Cu"))
    ours, theirs = both(cell, potential, (4, 4, 4))
    bands = ours.band_structure("fcc", points=21)
    theirs.run_qpoints(bands.q)
    np.testing.assert_allclose(bands.frequencies, theirs.qpoints.frequencies, atol=1e-4)
