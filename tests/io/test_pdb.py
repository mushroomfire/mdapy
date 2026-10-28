# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""PDB files: the atoms of each model and the bonds of CONECT, against
OVITO and ASE reading the same files."""

import numpy as np
import pytest

import mdapy
from mdapy.errors import FileFormatError

PEPTIDE = """\
HEADER    a small peptide written by hand
CRYST1   20.000   21.000   22.000  90.00  90.00  90.00 P 1           1
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.400   2.100   3.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   3.500   3.100  0.50 12.30           C
ATOM      4  O   ALA A   1       2.300   4.500   3.200  1.00  0.00           O
ATOM      5 1HB  ALA A   1       2.900   1.200   4.000  1.00  0.00
HETATM    6 CA    CA B   2      10.000  10.000  10.000  1.00  0.00          CA2+
HETATM    7 CL1  LIG B   3      11.000  10.500   9.000  1.00  0.00
TER
CONECT    1    2
CONECT    2    1    3    5
CONECT    3    2    4    4
END
"""


# with every element in columns 77-78, which ASE needs
LABELED = PEPTIDE.replace(
    "4.000  1.00  0.00\n", "4.000  1.00  0.00           H\n"
).replace("9.000  1.00  0.00\n", "9.000  1.00  0.00          CL\n")


def test_atoms_names_residues_and_bonds(tmp_path):
    path = tmp_path / "peptide.pdb"
    path.write_text(PEPTIDE)
    assert mdapy.detect_format(path) == "pdb"
    system = mdapy.read(path)
    assert system["id"].tolist() == [1, 2, 3, 4, 5, 6, 7]
    # elements from columns 77-78, or else from the atom name
    assert system["element"].tolist() == ["N", "C", "C", "O", "H", "Ca", "Cl"]
    assert system["atom_name"].tolist() == ["N", "CA", "C", "O", "1HB", "CA", "CL1"]
    assert system["residue"].tolist() == ["ALA"] * 5 + ["CA", "LIG"]
    assert system["residue_number"].tolist() == [1] * 5 + [2, 3]
    np.testing.assert_allclose(system["positions"][2], [3.0, 3.5, 3.1])
    np.testing.assert_allclose(system.box.matrix, np.diag([20.0, 21.0, 22.0]))
    bonds = system.topology.bonds["atoms"].to_list()
    assert sorted(map(sorted, bonds)) == [[0, 1], [1, 2], [1, 4], [2, 3]]


def test_as_ovito_and_ase(tmp_path):
    path = tmp_path / "peptide.pdb"
    path.write_text(LABELED)
    ours = mdapy.read(path)
    ase_io = pytest.importorskip("ase.io")
    theirs = ase_io.read(path)
    np.testing.assert_allclose(ours["positions"], theirs.positions)
    assert ours["element"].tolist() == theirs.get_chemical_symbols()
    assert ours["atom_name"].tolist() == theirs.arrays["atomtypes"].tolist()
    assert ours["residue"].tolist() == theirs.arrays["residuenames"].tolist()
    assert ours["residue_number"].tolist() == theirs.arrays["residuenumbers"].tolist()
    np.testing.assert_allclose(ours.box.matrix, theirs.cell.array)

    pytest.importorskip("ovito")
    from ovito.io import import_file

    data = import_file(str(path)).compute()
    particles = data.particles
    types = particles.particle_types
    assert ours["element"].tolist() == [types.type_by_id(t).name for t in types]
    np.testing.assert_array_equal(ours["id"], particles.identifiers)
    np.testing.assert_allclose(ours["positions"], particles.positions)
    np.testing.assert_allclose(ours.box.matrix, np.asarray(data.cell)[:, :3].T)
    # OVITO adds bonds between atoms close enough, here Ca and Cl 1.5 Å
    # apart: covalent bonds
    from mdapy.visualization import covalent_bonds

    theirs_bonds = {
        tuple(sorted(b)) for b in np.asarray(particles.bonds.topology).tolist()
    }
    ours_bonds = {tuple(sorted(b)) for b in ours.topology.bonds["atoms"].to_list()}
    close = covalent_bonds(ours)
    close = {(int(i), int(j)) for i, j in zip(close.i, close.j) if i < j}
    assert ours_bonds < theirs_bonds and theirs_bonds - ours_bonds == {(5, 6)} <= close


def test_models_as_frames(tmp_path):
    """Models are frames; the cell before the first applies to all, unless a
    model gives its own."""
    ase_build = pytest.importorskip("ase.build")
    import ase.io

    molecule = ase_build.molecule("CH3CH2OH")
    molecule.set_cell([12.0, 13.0, 14.0])
    molecule.pbc = True  # so that ASE writes the cell
    frames = []
    for k in range(3):
        moved = molecule.copy()
        moved.positions += 0.1 * k
        frames.append(moved)
    ase.io.write(tmp_path / "models.pdb", frames)
    trajectory = mdapy.read_trajectory(tmp_path / "models.pdb")
    assert len(trajectory) == 3
    for ours, theirs in zip(trajectory, ase.io.read(tmp_path / "models.pdb", ":")):
        np.testing.assert_allclose(ours["positions"], theirs.positions, atol=1e-3)
        assert ours["element"].tolist() == theirs.get_chemical_symbols()
        np.testing.assert_allclose(ours.box.matrix, theirs.cell.array, atol=1e-3)
    text = (
        "CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1\n"
        "MODEL        1\n"
        "ATOM      1  O   HOH     1       1.000   1.000   1.000\n"
        "MODEL        2\n"
        "CRYST1   11.000   11.000   11.000  90.00  90.00 120.00 P 1\n"
        "ATOM      1  O   HOH     1       2.000   1.000   1.000\n"
    )
    (tmp_path / "two.pdb").write_text(text)
    first, second = mdapy.read_trajectory(tmp_path / "two.pdb")
    np.testing.assert_allclose(first.box.matrix, 10 * np.eye(3))
    np.testing.assert_allclose(np.linalg.norm(second.box.matrix, axis=1), 11.0)
    np.testing.assert_allclose(second["positions"], [[2.0, 1.0, 1.0]])


def test_molecules_without_a_cell(tmp_path):
    """No CRYST1, or one of 1 Å: a box about the atoms, not periodic."""
    for header in ("", "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1\n"):
        (tmp_path / "free.pdb").write_text(
            header + "HETATM    1  C   MOL     1       1.000   2.000   3.000\n"
            "HETATM    2  O   MOL     1       2.200   2.000   3.000\nEND\n"
        )
        system = mdapy.read(tmp_path / "free.pdb")
        assert not system.box.pbc.any() and system.topology is None
        assert system["element"].tolist() == ["C", "O"]


@pytest.mark.parametrize(
    "text, message",
    [
        ("ATOM      1  O   HOH     1       1.000   x       1.000\n", "a coordinate"),
        ("ATOM      1  O   HOH     1       1.000\n", "ends before its coordinates"),
        ("ATOM      1 QQ   HOH     1       1.000   1.000   1.000\n", "no element"),
        ("CRYST1   10.000   1x.000\n", "a cell parameter"),
        (
            "ATOM      1  O   HOH     1       1.000   1.000   1.000\nCONECT    1    x\n",
            "not a serial",
        ),
        (
            "ATOM      1  O   HOH     1       1.000   1.000   1.000\nCONECT    1    9\n",
            "bonds atom 9, which the model lacks",
        ),
    ],
)
def test_errors(tmp_path, text, message):
    (tmp_path / "bad.pdb").write_text(text)
    with pytest.raises(FileFormatError, match=message):
        mdapy.read(tmp_path / "bad.pdb")
