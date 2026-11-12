# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Fitted force constants against symfc, which phonopy installs, on the same
displacements and forces of an EAM potential: random displacements, whose
forces are not harmonic, so that the two fits must agree as fits. Skipped
where symfc is not installed."""

import numpy as np
import pytest

from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.lattice_dynamics import (
    displaced_supercells,
    force_constants,
    primitive_cell,
)
from mdapy.lattice_dynamics._phonons import _supercell
from mdapy.simulation import minimize

from calculators.eam_files import write_setfl

symfc = pytest.importorskip("symfc")

CRYSTALS = {
    "fcc primitive cell": ("fcc", 3.29, "Cu", 4, True),
    "fcc cubic cell": ("fcc", 3.29, "Cu", 2, False),
    "bcc": ("bcc", 2.7, "Cu", 3, False),
    "hcp": ("hcp", 2.33, "Cu", (3, 3, 2), False),
    "L1_2 Cu3Ni": ("l1_2", 3.29, ("Cu", "Ni"), 2, False),
}


@pytest.fixture(scope="module")
def potential(tmp_path_factory):
    return EAM(write_setfl(tmp_path_factory.mktemp("eam") / "CuNi.eam.alloy"))


@pytest.mark.parametrize("name", CRYSTALS)
def test_against_symfc(potential, name):
    from symfc import Symfc
    from symfc.utils.utils import SymfcAtoms

    structure, a, elements, supercell, primitive = CRYSTALS[name]
    cell = minimize(
        build_crystal(structure, a, elements=elements),
        potential,
        cell="full",
        max_force=1e-10,
        max_stress=1e-8,
        max_steps=10000,
    ).system
    if primitive:
        cell = primitive_cell(cell)
    frames = displaced_supercells(
        cell, supercell=supercell, method="random", displacement=0.03, count=2, seed=3
    )
    frames = [
        f.with_columns(forces=potential.compute(f, ("forces",)).forces) for f in frames
    ]
    ours = force_constants(cell, frames)
    big = _supercell(cell, ours.supercell)
    atoms = SymfcAtoms(
        cell=big.box.matrix,
        scaled_positions=big.box.to_fractional(big["positions"]),
        numbers=big.table["element"].to_physical().to_numpy() + 1,
    )
    theirs = Symfc(atoms)
    theirs.displacements = np.array([f["positions"] - big["positions"] for f in frames])
    theirs.forces = np.array([f["forces"] for f in frames])
    theirs.run(orders=[2])
    # symfc's compact force constants are of the atoms of its primitive cell
    rows = theirs.p2s_map
    np.testing.assert_allclose(
        ours.second[rows], theirs.force_constants[2], rtol=0, atol=1e-12
    )


# supercells that keep every operation of the crystal, and cutoffs between
# shells, which symfc leaves out when at a shell
CUTOFFS = {
    "fcc primitive cell": ("fcc", 3.29, "Cu", 4, True, 4.5),
    "fcc cubic cell": ("fcc", 3.29, "Cu", 3, False, 4.8),
    "bcc": ("bcc", 2.7, "Cu", 4, False, 5.0),
    "hcp": ("hcp", 2.33, "Cu", (4, 4, 3), False, 4.4),
    "L1_2 Cu3Ni": ("l1_2", 3.29, ("Cu", "Ni"), 3, False, 4.8),
}


@pytest.mark.parametrize("name", CUTOFFS)
def test_against_symfc_within_a_cutoff(potential, name):
    from symfc import Symfc
    from symfc.utils.utils import SymfcAtoms

    structure, a, elements, supercell, primitive, cutoff = CUTOFFS[name]
    cell = minimize(
        build_crystal(structure, a, elements=elements),
        potential,
        cell="full",
        max_force=1e-10,
        max_stress=1e-8,
        max_steps=10000,
    ).system
    if primitive:
        cell = primitive_cell(cell)
    frames = displaced_supercells(
        cell, supercell=supercell, method="random", displacement=0.01, count=2, seed=3
    )
    frames = [
        f.with_columns(forces=potential.compute(f, ("forces",)).forces) for f in frames
    ]
    ours = force_constants(cell, frames, cutoff=cutoff)
    big = _supercell(cell, ours.supercell)
    atoms = SymfcAtoms(
        cell=big.box.matrix,
        scaled_positions=big.box.to_fractional(big["positions"]),
        numbers=big.table["element"].to_physical().to_numpy() + 1,
    )
    theirs = Symfc(atoms, cutoff={2: cutoff})
    theirs.displacements = np.array([f["positions"] - big["positions"] for f in frames])
    theirs.forces = np.array([f["forces"] for f in frames])
    theirs.run(orders=[2])
    np.testing.assert_allclose(
        ours.second[theirs.p2s_map], theirs.force_constants[2], rtol=0, atol=1e-12
    )
