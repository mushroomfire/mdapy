# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Force constants and frequencies against the dynamical matrix of LAMMPS
(its PHONON package). Skipped where LAMMPS or the package is missing."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.elements import atomic_masses
from mdapy.lattice_dynamics import phonons
from mdapy.lattice_dynamics._phonons import _THZ, _displaced_forces, _supercell
from mdapy.simulation import minimize

from calculators.eam_files import write_setfl

lammps = pytest.importorskip("lammps")


def lammps_dynamical_matrix(tmp_path, system, potential_file, displacement):
    """The (3N, 3N) dynamical matrix of the system by LAMMPS, in eV/(Å² amu)."""
    data = tmp_path / "supercell.data"
    mdapy.write(data, system, atom_style="atomic")
    output = tmp_path / "dynmat.dat"
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        if "PHONON" not in lmp.installed_packages:
            pytest.skip("LAMMPS has no PHONON package")
        lmp.commands_string(
            f"""
            units metal
            atom_style atomic
            atom_modify map array
            boundary p p p
            read_data {data}
            mass 1 {float(atomic_masses(["Cu"])[0])!r}
            pair_style eam/alloy
            pair_coeff * * {potential_file} Cu
            dynamical_matrix all regular {displacement!r} file {output} binary yes
            """
        )
    finally:
        lmp.close()
    n = 3 * len(system)
    return np.fromfile(output, dtype=np.float64).reshape(n, n)


def test_against_lammps(tmp_path):
    potential_file = write_setfl(tmp_path / "CuNi.eam.alloy")
    potential = EAM(potential_file)
    relaxed = minimize(
        build_crystal("fcc", 3.29, elements="Cu"),
        potential,
        cell="full",
        max_force=1e-10,
        max_stress=1e-8,
        max_steps=10000,
    ).system
    # an orthogonal box, which LAMMPS takes as it is
    cell = relaxed.with_box(mdapy.Box(np.diag(np.diag(relaxed.box.matrix))))
    counts = 3 * np.eye(3, dtype=int)
    displacement = 0.01
    reference = lammps_dynamical_matrix(
        tmp_path, _supercell(cell, counts), potential_file, displacement
    )
    # the rows of the atoms of the first cell, before any symmetrizing
    mass = atomic_masses(["Cu"])[0]
    raw = _displaced_forces(cell, potential, counts, displacement) / mass
    rows = raw.transpose(0, 2, 1, 3).reshape(3 * len(cell), -1)
    np.testing.assert_allclose(rows, reference[: 3 * len(cell)], atol=1e-9)
    # the frequencies at the wave vectors of the supercell are the
    # frequencies of the supercell at Γ
    result = phonons(cell, potential, supercell=counts, displacement=displacement)
    grid = np.stack(np.meshgrid(*[np.arange(3) / 3] * 3, indexing="ij"), -1)
    ours = np.sort(result.frequencies(grid.reshape(-1, 3)).reshape(-1))
    eigenvalues = np.linalg.eigvalsh(0.5 * (reference + reference.T))
    theirs = np.sort(np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * _THZ)
    np.testing.assert_allclose(ours, theirs, atol=1e-4)
