# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Files exchanged with ASE. Skipped where ASE is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy import units

ase = pytest.importorskip("ase")
ase_io = pytest.importorskip("ase.io")
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402

# ASE itself sets array shapes, which NumPy 2.5 deprecates.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array:DeprecationWarning"
)

# ASE uses CODATA 2014 constants; mdapy uses the exact SI values and
# CODATA 2022, so unit conversions agree to about 1e-8.
RTOL = 1e-7


def ase_atoms():
    rng = np.random.default_rng(0)
    atoms = ase.Atoms(
        "Cu3Zr",
        positions=rng.random((4, 3)) * 4,
        cell=[[4.0, 0.0, 0.0], [0.5, 4.0, 0.0], [0.2, 0.3, 4.0]],
        pbc=[True, True, False],
    )
    atoms.set_momenta(rng.standard_normal((4, 3)))
    atoms.set_initial_charges([0.5, -0.5, 0.25, -0.25])
    atoms.info["config_type"] = "test"
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-12.5,
        forces=rng.standard_normal((4, 3)),
        stress=rng.standard_normal(6) * 0.01,
    )
    return atoms


def test_mdapy_reads_ase_files(tmp_path):
    path = tmp_path / "ase.xyz"
    ase_io.write(path, ase_atoms(), format="extxyz")
    # ASE writes 8 decimals; compare with what ASE reads from the same file.
    atoms = ase_io.read(path, format="extxyz")
    system = mdapy.read(path)
    assert system.table["element"].to_list() == atoms.get_chemical_symbols()
    np.testing.assert_array_equal(system["positions"], atoms.positions)
    np.testing.assert_array_equal(system.box.matrix, atoms.cell[:])
    np.testing.assert_array_equal(system.box.pbc, atoms.pbc)
    np.testing.assert_allclose(
        system.table["velocities"].to_numpy(),
        atoms.get_velocities() * 1000 * ase.units.fs,  # ASE units -> Å/ps
        rtol=RTOL,
    )
    np.testing.assert_array_equal(system.table["forces"].to_numpy(), atoms.get_forces())
    assert system.info["energy"] == atoms.get_potential_energy()
    stress = atoms.get_stress(voigt=False) / ase.units.GPa
    np.testing.assert_allclose(system.info["stress"], stress, rtol=RTOL)
    assert system.info["config_type"] == "test"
    np.testing.assert_array_equal(system.table["charge"], atoms.get_initial_charges())


def test_ase_reads_mdapy_files(tmp_path):
    rng = np.random.default_rng(1)
    system = mdapy.System(
        rng.random((3, 3)) * 5,
        mdapy.Box([[5.0, 0, 0], [0, 5.0, 0], [1.0, 0, 5.0]], pbc=[True, False, True]),
        elements=["O", "H", "H"],
        info={
            "energy": -14.2,
            "stress": np.diag([1.0, 2.0, 3.0]),
            "label": "a b",
            "n": 3,
        },
    ).with_columns(forces=rng.standard_normal((3, 3)), charge=[-0.8, 0.4, 0.4])
    path = tmp_path / "mdapy.xyz"
    mdapy.write(path, system)
    atoms = ase_io.read(path, format="extxyz")
    assert atoms.get_chemical_symbols() == ["O", "H", "H"]
    np.testing.assert_array_equal(atoms.positions, system["positions"])
    np.testing.assert_array_equal(atoms.cell[:], system.box.matrix)
    np.testing.assert_array_equal(atoms.pbc, system.box.pbc)
    assert atoms.get_potential_energy() == -14.2
    np.testing.assert_array_equal(atoms.get_forces(), system.table["forces"].to_numpy())
    np.testing.assert_allclose(
        atoms.get_stress(voigt=False),
        np.diag([1.0, 2.0, 3.0]) * ase.units.GPa,
        rtol=RTOL,
    )
    assert atoms.info["label"] == "a b" and atoms.info["n"] == 3
    # ASE reads a charge property as charges computed by a calculator.
    np.testing.assert_array_equal(atoms.get_charges(), [-0.8, 0.4, 0.4])
    assert units.EV_PER_A3_TO_GPA == pytest.approx(1 / ase.units.GPa, rel=RTOL)
