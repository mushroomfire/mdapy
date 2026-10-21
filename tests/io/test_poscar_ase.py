# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""POSCAR files exchanged with ASE. Skipped where ASE is not installed."""

import numpy as np
import pytest

import mdapy

ase = pytest.importorskip("ase")
ase_io = pytest.importorskip("ase.io")
from ase.constraints import FixAtoms  # noqa: E402

pytestmark = pytest.mark.filterwarnings(
    "ignore:Setting the shape on a NumPy array:DeprecationWarning"
)


def test_mdapy_reads_ase_files(tmp_path):
    rng = np.random.default_rng(0)
    atoms = ase.Atoms(
        "Cu2Zr2",
        scaled_positions=rng.random((4, 3)),
        cell=[[4.0, 0, 0], [0.5, 4.0, 0], [0.2, 0.3, 4.0]],
        pbc=True,
    )
    atoms.set_velocities(rng.standard_normal((4, 3)) * 0.01)
    atoms.set_constraint(FixAtoms([1]))
    path = tmp_path / "POSCAR"
    ase_io.write(path, atoms, format="vasp", direct=True)
    expected = ase_io.read(path, format="vasp")  # what the file holds
    system = mdapy.read(path)
    assert system.table["element"].to_list() == expected.get_chemical_symbols()
    np.testing.assert_array_equal(system.box.matrix, expected.cell[:])
    np.testing.assert_allclose(system.positions, expected.positions, atol=1e-14)
    movable = system.table["movable"].to_numpy()
    assert movable.tolist() == [[True] * 3, [False] * 3, [True] * 3, [True] * 3]
    # ASE zeroes the velocity of the fixed atom when reading; the file has it
    free = [0, 2, 3]
    np.testing.assert_allclose(
        system.table["velocities"].to_numpy()[free] / 1000,  # Å/ps to Å/fs
        expected.get_velocities()[free] / (ase.units.Ang / ase.units.fs),
        rtol=1e-12,
    )


def test_ase_reads_mdapy_files(tmp_path):
    rng = np.random.default_rng(1)
    box = mdapy.Box([[4.0, 0, 0], [0.5, 4.0, 0], [0.2, 0.3, 4.0]])
    system = mdapy.System(
        box.to_cartesian(rng.random((4, 3))), box, elements=["O", "Ti", "O", "Ti"]
    ).with_columns(velocities=rng.standard_normal((4, 3)))
    path = tmp_path / "POSCAR"
    mdapy.write(path, system)
    atoms = ase_io.read(path, format="vasp")
    order = [0, 2, 1, 3]  # grouped by element
    assert atoms.get_chemical_symbols() == ["O", "O", "Ti", "Ti"]
    np.testing.assert_array_equal(atoms.cell[:], box.matrix)
    np.testing.assert_allclose(atoms.positions, system.positions[order], atol=1e-14)
    np.testing.assert_allclose(
        atoms.get_velocities() / (ase.units.Ang / ase.units.fs) * 1000,
        system.table["velocities"].to_numpy()[order],
        rtol=1e-12,
    )
    # ASE turns atoms whose three flags are F into FixAtoms
    fixed = system.with_columns(movable=[[True] * 3, [False] * 3] * 2)
    mdapy.write(path, fixed)
    constraints = ase_io.read(path, format="vasp").constraints
    assert [c.get_indices().tolist() for c in constraints] == [[2, 3]]
