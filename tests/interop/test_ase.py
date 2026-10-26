# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.errors import InvalidParameter
from mdapy.interop import from_ase, to_ase

ase = pytest.importorskip("ase")
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from ase.constraints import FixAtoms, FixBondLength, FixCartesian  # noqa: E402

# ASE itself sets array shapes, which NumPy 2.5 deprecates.


def sample():
    rng = np.random.default_rng(0)
    box = mdapy.Box(
        [[8.0, 0, 0], [1.0, 7.0, 0], [0.5, -0.5, 9.0]],
        origin=[1.0, 2.0, -3.0],
        pbc=(True, False, True),
    )
    n = 6
    return (
        mdapy.System(
            rng.uniform(0, 5, size=(n, 3)),
            box,
            elements=["Cu", "Ni", "O", "H", "Cu", "Ni"],
        )
        .with_columns(
            mass=rng.uniform(1, 60, n),
            velocities=rng.normal(size=(n, 3)),
            charge=rng.normal(size=n),
            movable=np.array([[True] * 3, [False] * 3, [True, False, True]] * 2),
            forces=rng.normal(size=(n, 3)),
            weight=rng.uniform(size=n),
            flags=rng.integers(0, 5, size=(n, 3)).astype(np.int32),
            type_label=["a", "b", "c", "d", "e", "f"],
        )
        .with_info(energy=-12.5, stress=np.diag([1.0, 2.0, 3.0]) + 0.5, label="sample")
    )


def test_round_trip():
    system = sample()
    atoms = to_ase(system)
    back = from_ase(atoms)
    for name in ("positions", "mass", "velocities", "charge", "forces", "weight"):
        np.testing.assert_allclose(
            back.table[name].to_numpy(), system.table[name].to_numpy(), rtol=1e-14
        )
    np.testing.assert_array_equal(
        back.table["movable"].to_numpy(), system.table["movable"].to_numpy()
    )
    np.testing.assert_array_equal(
        back.table["flags"].to_numpy(), system.table["flags"].to_numpy()
    )
    assert back.table["element"].to_list() == system.table["element"].to_list()
    np.testing.assert_array_equal(back.box.matrix, system.box.matrix)
    np.testing.assert_array_equal(back.box.origin, system.box.origin)
    np.testing.assert_array_equal(back.box.pbc, system.box.pbc)
    assert back.info["energy"] == -12.5 and back.info["label"] == "sample"
    np.testing.assert_allclose(back.info["stress"], system.info["stress"], rtol=1e-14)
    assert "type_label" not in back.table.columns  # strings stay behind


def test_what_ase_sees():
    system = sample()
    atoms = to_ase(system)
    assert atoms.get_chemical_symbols() == ["Cu", "Ni", "O", "H", "Cu", "Ni"]
    # ASE velocities are in Å per its unit of time, sqrt(amu Å^2 / eV)
    unit = np.sqrt(mdapy.units.AMU_A2_PER_PS2_TO_EV)
    np.testing.assert_allclose(
        atoms.get_velocities(), system.table["velocities"].to_numpy() * unit
    )
    kinetic = (
        0.5
        * np.sum(
            system.table["mass"].to_numpy()[:, None]
            * system.table["velocities"].to_numpy() ** 2
        )
        * mdapy.units.AMU_A2_PER_PS2_TO_EV
    )
    assert atoms.get_kinetic_energy() == pytest.approx(kinetic, rel=1e-13)
    assert atoms.get_potential_energy() == -12.5
    # ASE stress in eV/Å^3, Voigt order xx yy zz yz xz xy
    np.testing.assert_allclose(
        atoms.get_stress() * mdapy.units.EV_PER_A3_TO_GPA,
        [1.5, 2.5, 3.5, 0.5, 0.5, 0.5],
    )
    fixed = atoms.constraints[0]
    assert isinstance(fixed, FixAtoms) and list(fixed.index) == [1, 4]
    np.testing.assert_allclose(atoms.get_celldisp().ravel(), [1.0, 2.0, -3.0])
    assert atoms.info["label"] == "sample" and "energy" not in atoms.info


def test_virial_becomes_stress():
    system = sample().with_info(stress=None, virial=np.eye(3) * 2.0)
    atoms = to_ase(system)
    np.testing.assert_allclose(
        atoms.get_stress()[:3], -2.0 / system.box.volume, rtol=1e-14
    )


def test_atoms_made_by_ase():
    atoms = ase.Atoms(
        "Cu2", positions=[[0, 0, 0], [1.8, 1.8, 0]], cell=[3.6] * 3, pbc=True
    )
    system = from_ase(atoms)
    assert system.table.columns == ["positions", "element"]
    assert system.info == {}
    atoms.set_constraint([FixCartesian(0, mask=[True, False, True]), FixAtoms([1])])
    atoms.calc = SinglePointCalculator(atoms, energy=1.0, stress=np.eye(3) * 0.01)
    system = from_ase(atoms)
    np.testing.assert_array_equal(
        system.table["movable"].to_numpy(), [[False, True, False], [False] * 3]
    )
    np.testing.assert_allclose(
        system.info["stress"], np.eye(3) * 0.01 * mdapy.units.EV_PER_A3_TO_GPA
    )
    plain = mdapy.System([[0.0, 0, 0]], mdapy.Box(3.0))
    assert to_ase(plain).get_chemical_symbols() == ["X"]
    atoms.set_constraint(FixBondLength(0, 1))
    with pytest.raises(InvalidParameter, match="FixBondLengths has no equivalent"):
        from_ase(atoms)
