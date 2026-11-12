# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Thermal conductivity against phono3py, with the same second and third
order force constants, mesh and Gaussian: the conductivity and the width of
every mode. Skipped where phono3py is not installed."""

import warnings

import numpy as np
import pytest

from mdapy.analysis._symmetry import _rotations
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.elements import atomic_masses
from mdapy.lattice_dynamics import (
    displaced_supercells,
    force_constants,
    primitive_cell,
    thermal_conductivity,
)
from mdapy.lattice_dynamics._conductivity import _irreducible
from mdapy.lattice_dynamics._force_constants import _phonopy_atoms
from mdapy.simulation import minimize

from calculators.eam_files import write_setfl

phono3py = pytest.importorskip("phono3py")
pytest.importorskip("spglib")

CRYSTALS = {
    "fcc primitive cell": ("fcc", 3.29, "Cu", 4, True, (9, 9, 9)),
    "hcp": ("hcp", 2.33, "Cu", (4, 4, 3), False, (6, 6, 4)),
    "L1_2 Cu3Ni": ("l1_2", 3.29, ("Cu", "Ni"), 3, False, (5, 5, 5)),
}
TEMPERATURES = [100.0, 300.0, 600.0]
SIGMA = 0.2


@pytest.fixture(scope="module")
def potential(tmp_path_factory):
    return EAM(write_setfl(tmp_path_factory.mktemp("eam") / "CuNi.eam.alloy"))


@pytest.mark.parametrize("name", CRYSTALS)
def test_against_phono3py(potential, name):
    from phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms

    structure, a, elements, supercell, primitive, mesh = CRYSTALS[name]
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
        cell, supercell=supercell, method="random", displacement=0.01, count=3, seed=3
    )
    frames = [
        f.with_columns(forces=potential.compute(f, ("forces",)).forces) for f in frames
    ]
    constants = force_constants(cell, frames, cutoff_third=2.8)
    ours = thermal_conductivity(constants, TEMPERATURES, mesh=mesh, sigma=SIGMA)

    order = _phonopy_atoms(constants.supercell, len(cell))
    names = cell.table["element"].cast(str).to_list()
    unit = PhonopyAtoms(
        symbols=names,
        cell=cell.box.matrix,
        scaled_positions=cell.box.to_fractional(cell["positions"]),
        masses=atomic_masses(names),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theirs = Phono3py(
            unit,
            supercell_matrix=constants.supercell.T,
            primitive_matrix="P",
            log_level=0,
        )
    theirs.fc2 = np.ascontiguousarray(constants.second[:, order])
    theirs.fc3 = np.ascontiguousarray(constants.third[:, order][:, :, order])
    theirs.mesh_numbers = list(mesh)
    theirs.sigmas = [SIGMA]
    theirs.init_phph_interaction()
    theirs.run_thermal_conductivity(temperatures=TEMPERATURES)
    result = theirs.thermal_conductivity
    # xx, yy, zz, yz, xz, xy; phono3py's physical constants are older, by a
    # few parts in 10^7
    kappa = ours.kappa
    found = np.stack(
        [
            kappa[:, 0, 0],
            kappa[:, 1, 1],
            kappa[:, 2, 2],
            kappa[:, 1, 2],
            kappa[:, 0, 2],
            kappa[:, 0, 1],
        ],
        axis=1,
    )
    expected = result.kappa[0]
    np.testing.assert_allclose(
        found, expected, rtol=0, atol=1e-5 * np.abs(expected).max()
    )
    assert len(ours.q) == len(result.qpoints)
    # the width of each mode, at the irreducible wave vector of ours that
    # stands for each of phono3py's
    counts = np.array(mesh)
    reciprocal = 2 * np.pi * np.linalg.inv(cell.box.matrix).T
    _, first, _ = _irreducible(counts, reciprocal, _rotations(cell, True))
    grid = np.round(result.qpoints * counts).astype(np.int64) % counts
    index = (grid[:, 0] * counts[1] + grid[:, 1]) * counts[2] + grid[:, 2]
    slots = np.searchsorted(np.unique(first), first[index])
    gamma = result.gamma[0]  # (T, Q, 3n)
    np.testing.assert_allclose(
        ours.gamma[:, slots], gamma, rtol=0, atol=1e-5 * gamma.max()
    )
    np.testing.assert_allclose(ours.frequencies[slots], result.frequencies, atol=1e-5)
