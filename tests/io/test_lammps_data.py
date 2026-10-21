# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import polars as pl
import pytest

import mdapy
from mdapy.errors import (
    FileFormatError,
    InvalidParameter,
    MissingProperty,
    UnsupportedFormat,
)

WATER = """\
LAMMPS data file via write_data, version 30 Sep 2026, timestep = 500, units = real

6 atoms
2 atom types
4 bonds
1 bond types
2 angles
1 angle types

0.0 20.0 xlo xhi
0.0 20.0 ylo yhi
0.0 20.0 zlo zhi

Atom Type Labels

1 OW
2 HW

Masses

1 15.9994 # O
2 1.008 # H

Pair Coeffs # lj/cut/coul/long

1 0.1553 3.166
2 0 0

Bond Coeffs # harmonic

1 554.1349 1.0

Atoms # full

1 1 OW -0.8476 1.0 1.0 1.0 0 0 0
2 1 HW 0.4238 1.8 1.5 1.0 0 0 0
3 1 2 0.4238 0.4 1.6 1.0 0 0 0
4 2 1 -0.8476 5.0 5.0 5.0 1 0 -1  # comment
5 2 2 0.4238 5.8 5.5 5.0 1 0 -1
6 2 2 0.4238 4.4 5.6 5.0 1 0 -1

Velocities

1 0.001 0.002 0.003
2 0 0 0
3 0 0 0
4 0 0 0
5 0 0 0
6 -0.004 0 0

Bonds

1 1 1 2
2 1 1 3
3 1 4 5
4 1 4 6

Angles

1 1 2 1 3
2 1 5 4 6
"""


def read_text(tmp_path, text, name="t.data", **options):
    path = tmp_path / name
    path.write_text(text)
    return mdapy.read(path, **options)


def round_trip(tmp_path, system, **options):
    path = tmp_path / "rt.data"
    mdapy.write(path, system, **options)
    return mdapy.read(path)


def atoms_file(header, atoms, hint=" # atomic", sections=""):
    return f"title\n\n{header}\n\nAtoms{hint}\n\n{atoms}\n{sections}"


# ---------------------------------------------------------------- reading


def test_read_full_style_file(tmp_path):
    system = read_text(tmp_path, WATER)
    table = system.table
    assert table["id"].to_list() == [1, 2, 3, 4, 5, 6]
    assert table["molecule"].to_list() == [1, 1, 1, 2, 2, 2]
    assert table["type"].to_list() == [1, 2, 2, 1, 2, 2]
    assert table["type_label"].to_list() == ["OW", "HW", "HW", "OW", "HW", "HW"]
    assert table["element"].to_list() == ["O", "H", "H", "O", "H", "H"]
    assert table["mass"].to_list() == [15.9994, 1.008, 1.008, 15.9994, 1.008, 1.008]
    assert table["charge"].to_list() == [-0.8476, 0.4238, 0.4238] * 2
    np.testing.assert_array_equal(system.positions[3], [5.0, 5.0, 5.0])
    assert table["image"].to_list()[4] == [1, 0, -1]
    # real units: Å/fs in the file, Å/ps in mdapy, scaled exactly in decimal
    assert table["velocities"].to_list()[0] == [1.0, 2.0, 3.0]
    assert table["velocities"].to_list()[5] == [-4.0, 0.0, 0.0]
    assert system.box == mdapy.Box(20.0)
    assert system.info["timestep"] == 500
    lammps = system.info["lammps"]
    assert lammps["atom_style"] == "full" and lammps["units"] == "real"
    assert (lammps["atom_types"], lammps["bond_types"], lammps["angle_types"]) == (
        2,
        1,
        1,
    )
    assert lammps["masses"] == (15.9994, 1.008)
    assert lammps["type_labels"]["atom"] == ("OW", "HW")
    assert lammps["coeffs"]["Pair Coeffs"] == {
        "style": "lj/cut/coul/long",
        "lines": ("1 0.1553 3.166", "2 0 0"),
    }
    assert lammps["coeffs"]["Bond Coeffs"]["lines"] == ("1 554.1349 1.0",)
    topology = system.topology
    assert topology.bonds["atoms"].to_list() == [[0, 1], [0, 2], [3, 4], [3, 5]]
    assert topology.angles["atoms"].to_list() == [[1, 0, 2], [4, 3, 5]]
    assert topology.bonds["type"].to_list() == [1, 1, 1, 1]


@pytest.mark.parametrize(
    "style, line, expected",
    [
        ("atomic", "7 2 1.5 2.5 3.5", {"type": 2}),
        ("charge", "7 2 -0.5 1.5 2.5 3.5", {"type": 2, "charge": -0.5}),
        ("bond", "7 3 2 1.5 2.5 3.5", {"molecule": 3, "type": 2}),
        ("angle", "7 3 2 1.5 2.5 3.5", {"molecule": 3, "type": 2}),
        ("molecular", "7 3 2 1.5 2.5 3.5", {"molecule": 3, "type": 2}),
        ("full", "7 3 2 -0.5 1.5 2.5 3.5", {"molecule": 3, "type": 2, "charge": -0.5}),
    ],
)
def test_each_atom_style(tmp_path, style, line, expected):
    text = atoms_file("1 atoms\n2 atom types\n0 10 xlo xhi", line, f" # {style}")
    system = read_text(tmp_path, text)
    row = system.table.row(0, named=True)
    assert row["id"] == 7 and row["positions"] == [1.5, 2.5, 3.5]
    for name, value in expected.items():
        assert row[name] == value
    assert system.info["lammps"]["atom_style"] == style
    assert "image" not in system.table.columns
    with_images = read_text(tmp_path, text.replace(line, line + " 1 -2 0"))
    assert with_images.table["image"].to_list() == [[1, -2, 0]]


def test_header_defaults_comments_and_extra(tmp_path):
    text = (
        "title without units\n\n"
        "  # a comment line\n"
        "2 atoms # trailing comment\n\n"
        "1 atom types\n"
        "2 extra bond per atom\n"
        "4 extra special per atom\n"
        "0.0 3.0 xlo xhi\n"
        "0.5 1.0 1.5 xy xz yz\n\n"
        "Atoms\n\n"
        "1 1 0 0 0\n2 1 1 1 1\n"
    )
    system = read_text(tmp_path, text)
    box = system.box
    np.testing.assert_array_equal(
        box.matrix, [[3.0, 0.0, 0.0], [0.5, 1.0, 0.0], [1.0, 1.5, 1.0]]
    )
    np.testing.assert_array_equal(box.origin, [0.0, -0.5, -0.5])  # default bounds
    assert box.pbc.all()
    assert system.info["lammps"]["extra"] == {"bond": 2, "special": 4}
    assert system.info["lammps"]["units"] == "metal"
    assert "timestep" not in system.info


def test_general_triclinic_box(tmp_path):
    text = (
        "t\n\n1 atoms\n1 atom types\n"
        "4 0.5 0 avec\n0 3 1 bvec\n0.2 0 5 cvec\n1 2 3 abc origin\n\n"
        "Atoms # atomic\n\n1 1 1 2 3\n"
    )
    box = read_text(tmp_path, text).box
    np.testing.assert_array_equal(box.matrix, [[4, 0.5, 0], [0, 3, 1], [0.2, 0, 5]])
    np.testing.assert_array_equal(box.origin, [1, 2, 3])


def test_atom_style_is_inferred_when_unambiguous(tmp_path):
    system = read_text(tmp_path, atoms_file("1 atoms\n1 atom types", "1 1 0 0 0", ""))
    assert system.info["lammps"]["atom_style"] == "atomic"
    system = read_text(
        tmp_path, atoms_file("1 atoms\n1 atom types", "1 1 1 0.5 0 0 0", "")
    )
    assert system.info["lammps"]["atom_style"] == "full"
    header = "1 atoms\n1 atom types\n0 bonds\n1 bond types"
    system = read_text(tmp_path, atoms_file(header, "1 4 1 0 0 0", ""))
    assert system.info["lammps"]["atom_style"] == "molecular"
    assert system.table["molecule"].to_list() == [4]
    with pytest.raises(FileFormatError, match="pass atom_style='charge'"):
        read_text(tmp_path, atoms_file("1 atoms\n1 atom types", "1 1 0.5 0 0 0", ""))
    system = read_text(
        tmp_path,
        atoms_file("1 atoms\n1 atom types", "1 1 0.5 0 0 0", ""),
        atom_style="charge",
    )
    assert system.table["charge"].to_list() == [0.5]


def test_atom_style_option_overrides_the_comment(tmp_path):
    text = atoms_file("1 atoms\n1 atom types", "1 1 0.5 0 0 0", " # molecular")
    assert read_text(tmp_path, text, atom_style="charge").table["charge"].to_list() == [
        0.5
    ]


def test_type_labels_everywhere(tmp_path):
    text = """title

3 atoms
2 atom types
2 bonds
2 bond types

Atom Type Labels

1 C
2 H

Bond Type Labels

1 C-H
2 C-C

Masses

C 12.011
H 1.008

Atoms # molecular

1 1 C 0 0 0
2 1 H 1 0 0
3 1 1 2 0 0

Bonds

1 C-H 1 2
2 1 1 3
"""
    system = read_text(tmp_path, text)
    assert system.table["type"].to_list() == [1, 2, 1]
    assert system.table["element"].to_list() == ["C", "H", "C"]  # from the labels
    assert system.topology.bonds["type"].to_list() == [1, 1]
    assert system.topology.bonds["type_label"].to_list() == ["C-H", "C-H"]
    assert system.info["lammps"]["type_labels"]["bond"] == ("C-H", "C-C")


def test_every_coefficient_section(tmp_path):
    sections = ""
    sizes = {
        "Pair Coeffs": 2, "PairIJ Coeffs": 3, "Bond Coeffs": 1, "Angle Coeffs": 1,
        "Dihedral Coeffs": 1, "Improper Coeffs": 1, "BondBond Coeffs": 1,
        "BondAngle Coeffs": 1, "UreyBradley Coeffs": 1, "MiddleBondTorsion Coeffs": 1,
        "EndBondTorsion Coeffs": 1, "AngleTorsion Coeffs": 1,
        "AngleAngleTorsion Coeffs": 1, "BondBond13 Coeffs": 1, "AngleAngle Coeffs": 1,
    }  # fmt: skip
    for name, n in sizes.items():
        lines = "\n".join(f"{i + 1}  0.5   1.5 # note" for i in range(n))
        sections += f"\n{name}\n\n{lines}\n"
    header = (
        "1 atoms\n2 atom types\n1 bond types\n1 angle types\n"
        "1 dihedral types\n1 improper types"
    )
    text = f"title\n\n{header}\n{sections}\nAtoms # full\n\n1 1 1 0 0 0 0\n"
    coeffs = read_text(tmp_path, text).info["lammps"]["coeffs"]
    assert list(coeffs) == list(sizes)
    assert coeffs["PairIJ Coeffs"]["lines"] == ("1 0.5 1.5", "2 0.5 1.5", "3 0.5 1.5")
    assert coeffs["AngleAngle Coeffs"] == {"style": None, "lines": ("1 0.5 1.5",)}


# ---------------------------------------------------------------- elements


def elements_file(masses="", labels=""):
    return (
        "t\n\n2 atoms\n2 atom types\n\n"
        + (f"Atom Type Labels\n\n{labels}\n\n" if labels else "")
        + (f"Masses\n\n{masses}\n\n" if masses else "")
        + "Atoms # atomic\n\n1 1 0 0 0\n2 2 1 1 1\n"
    )


@pytest.mark.parametrize(
    "masses, labels, elements",
    [
        ("1 63.546 # Cu\n2 91.224 # Zr", "", ["Cu", "Zr"]),  # comments
        ("1 63.546 # Ag\n2 91.224 # Zr", "", ["Ag", "Zr"]),  # comments beat masses
        ("1 63.546\n2 91.224", "1 Al\n2 Ni", ["Al", "Ni"]),  # labels beat masses
        ("1 63.546\n2 91.224", "1 ca\n2 Zr", ["Cu", "Zr"]),  # labels must be exact
        ("1 63.546\n2 91.224", "", ["Cu", "Zr"]),  # masses
        ("1 63.546 # Cu\n2 14.027 # CH2", "", None),  # CH2 is not an element
        ("1 1.0\n2 2.0", "", None),
        ("", "", None),
    ],
)
def test_elements_are_guessed_only_when_every_type_is_known(
    tmp_path, masses, labels, elements
):
    table = read_text(tmp_path, elements_file(masses, labels)).table
    if elements is None:
        assert "element" not in table.columns
    else:
        assert table["element"].to_list() == elements


def test_elements_option(tmp_path):
    table = read_text(tmp_path, elements_file(), elements=["Fe", "C"]).table
    assert table["element"].to_list() == ["Fe", "C"]
    with pytest.raises(InvalidParameter, match="names 1 elements for 2 atom types"):
        read_text(tmp_path, elements_file(), elements=["Fe"])
    with pytest.raises(InvalidParameter, match="unknown symbol"):
        read_text(tmp_path, elements_file(), elements=["Fe", "Qq"])


# ------------------------------------------------------------------ units


def test_units_from_the_title_and_the_option(tmp_path):
    text = WATER.replace("units = real", "units = lj")
    with pytest.raises(UnsupportedFormat, match="in lj units"):
        read_text(tmp_path, text)
    system = read_text(tmp_path, text, units="metal")
    assert system.table["velocities"].to_list()[0] == [0.001, 0.002, 0.003]
    assert system.info["lammps"]["units"] == "metal"
    with pytest.raises(InvalidParameter, match="units must be 'metal' or 'real'"):
        read_text(tmp_path, WATER, units="si")


def test_unsupported_atom_styles(tmp_path):
    text = atoms_file("1 atoms\n1 atom types", "1 1 1 1 0 0 0", " # sphere")
    with pytest.raises(UnsupportedFormat, match="atom style 'sphere' is not supported"):
        read_text(tmp_path, text)
    with pytest.raises(InvalidParameter, match="atom_style must be one of"):
        read_text(tmp_path, text, atom_style="sphere")
    with pytest.raises(UnsupportedFormat, match="ellipsoids need atom styles"):
        read_text(tmp_path, "t\n\n1 atoms\n1 ellipsoids\n")
    text = atoms_file(
        "1 atoms\n1 atom types", "1 1 0 0 0", sections="\nLines\n\n1 0 0 1 1\n"
    )
    with pytest.raises(UnsupportedFormat, match="Lines section needs"):
        read_text(tmp_path, text)


def test_file_without_atoms(tmp_path):
    system = read_text(tmp_path, "t\n\n0 atoms\n1 atom types\n0 2 xlo xhi\n")
    assert len(system) == 0 and system.info["lammps"]["atom_style"] == "atomic"
    system = read_text(tmp_path, "t\n\n0 atoms\n1 atom types\n\nAtoms # full\n")
    assert len(system) == 0 and system.info["lammps"]["atom_style"] == "full"
    assert len(round_trip(tmp_path, system)) == 0


def test_bonds_without_atoms(tmp_path):
    text = (
        "t\n\n0 atoms\n1 atom types\n1 bonds\n1 bond types\n\n"
        "Atoms # molecular\n\nBonds\n\n1 1 1 2\n"
    )
    with pytest.raises(FileFormatError, match="no atom has ID 1") as info:
        read_text(tmp_path, text)
    assert info.value.line == 12


def test_values_the_system_rejects(tmp_path):
    text = atoms_file("1 atoms\n1 atom types", "1 1 0 nan 0")
    with pytest.raises(FileFormatError, match="finite"):
        read_text(tmp_path, text)


# ----------------------------------------------------------------- errors


@pytest.mark.parametrize(
    "text, line, message",
    [
        (atoms_file("1 atoms\n1 atom types", "1 1 0 0"), 8, "found 4"),
        (atoms_file("1 atoms\n1 atom types", "1 1 0 0 0 1"), 8, "found 6"),
        (
            atoms_file("1 atoms\n1 atom types", "1 1 0 0", ""),
            8,
            "4 values, which fits none",
        ),
        (atoms_file("1 atoms\n1 atom types", "1 1 0 0 0 1", ""), 8, "fit both charge"),
        (atoms_file("1 atoms\n1 atom types", "1 2 0 0 0"), 8, "atom type 2 is not"),
        (
            atoms_file("1 atoms\n1 atom types", "1 Cu 0 0 0"),
            8,
            "unknown atom type label",
        ),
        (
            atoms_file("2 atoms\n1 atom types", "1 1 0 0 0\n1 1 0 0 0"),
            9,
            "IDs are repeated",
        ),
        (atoms_file("1 atoms\n1 atom types", "1 1 x 0 0"), 8, "cannot read 'x'"),
        (atoms_file("2 atoms\n1 atom types", "1 1 0 0 0"), 9, "expected 2 lines"),
        (
            atoms_file("1 atoms\n1 atom types", "1 1 0 0 0", sections="\nFoo\n\n1\n"),
            10,
            "unknown section 'Foo'",
        ),
        (
            atoms_file(
                "1 atoms\n1 atom types", "1 1 0 0 0", sections="\nAtoms\n\n1 1 0 0 0\n"
            ),
            10,
            "section 'Atoms' appears twice",
        ),
        (
            atoms_file(
                "1 atoms\n1 atom types",
                "1 1 0 0 0",
                sections="\nVelocities\n\n2 0 0 0\n",
            ),
            12,
            "no atom has ID 2",
        ),
        ("t\n\nx atoms\n", 3, "expected a count, found 'x'"),
        ("t\n\n1 atoms\n0 x xlo xhi\n", 4, "expected numbers"),
        (
            "t\n\n1 atoms\n1 atom types\n\nMasses\n\n1\n",
            8,
            "expected a type and a mass",
        ),
        ("t\n\n1 atoms\n1 atom types\n\nMasses\n\n1 x\n", 8, "expected numbers"),
        (
            "t\n\n1 atom types\n\nAtom Type Labels\n\nC 1\n",
            7,
            "expected a type and a label",
        ),
        (
            "t\n\n1 atom types\n\nAtom Type Labels\n\n2 C\n",
            7,
            "atom type 2 is not between",
        ),
        (
            "t\n\n2 atom types\n\nAtom Type Labels\n\n1 C\n1 H\n",
            8,
            "repeated or invalid",
        ),
        (
            "t\n\n2 atom types\n\nAtom Type Labels\n\n1 C\n2 C\n",
            8,
            "repeated or invalid",
        ),
        ("t\n\n1 atom types\n\nAtom Type Labels\n\n1 2C\n", 7, "repeated or invalid"),
        ("t\n\n1 atom types\n\nPair Coeffs\n\n# nothing\n", 7, "expected coefficients"),
    ],
)
def test_errors_name_the_line(tmp_path, text, line, message):
    with pytest.raises(FileFormatError, match=message) as info:
        read_text(tmp_path, text)
    assert info.value.line == line


BONDED = (
    "t\n\n2 atoms\n1 atom types\n1 bonds\n1 bond types\n\n"
    "Atoms # molecular\n\n1 1 1 0 0 0\n2 1 1 1 0 0\n\nBonds\n\n{bond}\n"
)


@pytest.mark.parametrize(
    "bond, line, message",
    [
        ("1 1 1 3", 15, "no atom has ID 3"),
        ("1 2 1 2", 15, "bond type 2 is not between"),
        ("1 C-C 1 2", 15, "unknown bond type label 'C-C'"),
    ],
)
def test_bond_errors(tmp_path, bond, line, message):
    with pytest.raises(FileFormatError, match=message) as info:
        read_text(tmp_path, BONDED.format(bond=bond))
    assert info.value.line == line


@pytest.mark.parametrize(
    "text, message",
    [
        (BONDED.format(bond="1 1 1 1"), "repeats an atom"),
        (
            "t\n\n2 atoms\n1 atom types\n1 bonds\n1 bond types\n\n"
            "Atoms # atomic\n\n1 1 0 0 0\n2 1 1 0 0\n\nBonds\n\n1 1 1 2\n",
            "atom style atomic cannot have bonds",
        ),
        ("t\n\n2 atoms\n", "2 atoms but there is no Atoms section"),
        (
            "t\n\n1 atoms\n1 atom types\n1 bonds\n1 bond types\n\nAtoms # molecular\n\n1 1 1 0 0 0\n",
            "1 bonds but there is no Bonds section",
        ),
        (
            "t\n\n1 atoms\n1 atom types\n1 bond types\n\nAtoms # atomic\n\n1 1 0 0 0\n",
            "atom style atomic cannot have bonds",
        ),
        ("t\n\n0 1 xlo xhi\n1 0 0 avec\n", "both by bounds"),
        ("t\n\n1 0 0 avec\n0 1 0 bvec\n", "no cvec"),
        ("t\n\n1 0 0 avec\n2 0 0 bvec\n0 0 1 cvec\n", "linearly dependent"),
    ],
)
def test_file_errors(tmp_path, text, message):
    with pytest.raises(FileFormatError, match=message):
        read_text(tmp_path, text)


# ---------------------------------------------------------------- writing


def test_round_trip_of_a_full_file(tmp_path):
    system = read_text(tmp_path, WATER)
    back = round_trip(tmp_path, system, units="real")
    assert back.table.equals(system.table)
    assert back.box == system.box
    assert back.info == system.info
    assert back.topology == system.topology


def test_written_text(tmp_path):
    path = tmp_path / "w.data"
    mdapy.write(path, read_text(tmp_path, WATER), units="real")
    text = path.read_text()
    assert text.startswith(
        "LAMMPS data file via mdapy "
        + mdapy.__version__
        + ", timestep = 500, units = real\n"
    )
    assert "\nMasses\n\n1 15.9994 # O\n2 1.008 # H\n" in text
    assert "\nPair Coeffs # lj/cut/coul/long\n\n1 0.1553 3.166\n2 0 0\n" in text
    assert "\nAtoms # full\n\n1 1 1 -0.8476 1 1 1 0 0 0\n" in text
    assert "\nVelocities\n\n1 0.001 0.002 0.003\n" in text
    assert "\nAngles\n\n1 1 2 1 3\n2 1 5 4 6\n" in text


def test_coefficients_keep_their_units(tmp_path):
    system = read_text(tmp_path, WATER)
    with pytest.raises(InvalidParameter, match="in real units and cannot be written"):
        mdapy.write(tmp_path / "m.data", system)
    lammps = dict(system.info["lammps"])
    del lammps["coeffs"]
    back = round_trip(tmp_path, system.with_info(lammps=lammps))  # metal units
    np.testing.assert_array_equal(
        back.table["velocities"].to_numpy(), system.table["velocities"].to_numpy()
    )
    assert back.info["lammps"]["units"] == "metal"


def test_types_from_elements(tmp_path):
    system = mdapy.System(
        np.arange(12.0).reshape(4, 3),
        mdapy.Box(20.0),
        elements=["Zr", "Cu", "Zr", "Al"],
    )
    back = round_trip(tmp_path, system)
    assert back.table["type"].to_list() == [1, 2, 1, 3]  # order of first appearance
    assert back.table["element"].to_list() == ["Zr", "Cu", "Zr", "Al"]
    assert back.info["lammps"]["masses"] == (91.224, 63.546, 26.9815385)
    back = round_trip(tmp_path, system, elements=["Al", "Cu", "Zr", "Ni"])
    assert back.table["type"].to_list() == [3, 2, 3, 1]
    assert back.info["lammps"]["atom_types"] == 4  # Ni has no atoms
    assert back.info["lammps"]["masses"][3] == 58.6934
    with pytest.raises(InvalidParameter, match="lacks the elements"):
        mdapy.write(tmp_path / "e.data", system, elements=["Cu", "Zr"])
    typed = system.with_columns(type=[1, 1, 1, 2])
    with pytest.raises(InvalidParameter, match="has a type column"):
        mdapy.write(tmp_path / "e.data", typed, elements=["Cu", "Zr"])


def test_masses_comments_only_for_types_of_one_element(tmp_path):
    system = mdapy.System(
        np.zeros((3, 3)), mdapy.Box(5.0), elements=["Cu", "Zr", "Ni"], types=[1, 1, 2]
    )
    path = tmp_path / "m.data"
    with pytest.raises(
        InvalidParameter, match=r"no mass is known for atom types \[1\]"
    ):
        mdapy.write(path, system)
    mdapy.write(path, system.with_columns(mass=[60.0, 60.0, 58.6934]))
    text = path.read_text()
    assert "\n1 60.0\n2 58.6934 # Ni\n" in text


def test_info_masses_fill_types_without_atoms(tmp_path):
    info = {"lammps": {"atom_types": 3, "masses": [1.0, float("nan"), 3.0, 4.0]}}
    system = mdapy.System(np.zeros((1, 3)), 5.0, types=[2], info=info)
    with pytest.raises(
        InvalidParameter, match=r"no mass is known for atom types \[2\]"
    ):
        mdapy.write(tmp_path / "m.data", system)
    back = round_trip(tmp_path, system.with_columns(mass=[2.0]))
    assert back.info["lammps"]["masses"] == (1.0, 2.0, 3.0)


def test_topology_labels_from_the_tables(tmp_path):
    bonds = pl.DataFrame({"atoms": [[0, 1]], "type": [1], "type_label": ["H-H"]})
    system = mdapy.System(
        [[0.0, 0, 0], [0.7, 0, 0]],
        5.0,
        elements="H",
        topology=mdapy.Topology(bonds=bonds),
    )
    back = round_trip(tmp_path, system)
    assert back.topology.bonds["type_label"].to_list() == ["H-H"]
    assert back.info["lammps"]["type_labels"]["bond"] == ("H-H",)


def test_unsupported_style_in_info(tmp_path):
    system = mdapy.System(
        np.zeros((1, 3)), 5.0, info={"lammps": {"atom_style": "sphere"}}
    )
    with pytest.raises(UnsupportedFormat, match="'sphere' is not supported"):
        mdapy.write(tmp_path / "s.data", system)


def test_systems_without_types_or_elements(tmp_path):
    system = mdapy.System(np.zeros((2, 3)), mdapy.Box(5.0))
    back = round_trip(tmp_path, system)
    assert back.table["type"].to_list() == [1, 1]
    assert "Masses" not in (tmp_path / "rt.data").read_text()
    with pytest.raises(InvalidParameter, match="needs an element column"):
        mdapy.write(tmp_path / "e.data", system, elements=["Cu"])


def test_default_atom_styles(tmp_path):
    base = mdapy.System([[0.0, 0, 0], [1.0, 0, 0]], mdapy.Box(5.0), elements="H")
    styles = {
        "atomic": base,
        "charge": base.with_columns(charge=[0.5, -0.5]),
        "molecular": base.with_topology(mdapy.Topology(bonds=[[0, 1]])),
        "full": base.with_columns(charge=[0.5, -0.5], molecule=[1, 1]),
    }
    for style, system in styles.items():
        back = round_trip(tmp_path, system)
        assert back.info["lammps"]["atom_style"] == style
    back = round_trip(tmp_path, styles["molecular"])
    assert back.table["molecule"].to_list() == [0, 0]  # no molecule column: ID 0
    assert back.topology.bonds["atoms"].to_list() == [[0, 1]]


def test_unused_types_masses_and_labels_are_kept(tmp_path):
    text = """t

1 atoms
3 atom types
2 bond types

Atom Type Labels

1 A
2 B
3 C

Bond Type Labels

1 A-B
2 B-C

Masses

1 1.5
2 2.5
3 3.5

Atoms # molecular

5 0 2 0 0 0
"""
    system = read_text(tmp_path, text)
    back = round_trip(tmp_path, system)
    assert back.info["lammps"]["masses"] == (1.5, 2.5, 3.5)
    assert back.info["lammps"]["type_labels"] == {
        "atom": ("A", "B", "C"),
        "bond": ("A-B", "B-C"),
    }
    assert back.table["id"].to_list() == [5] and back.table["type_label"].to_list() == [
        "B"
    ]


def test_extra_per_atom_and_images_round_trip(tmp_path):
    text = "t\n\n1 atoms\n1 atom types\n3 extra bond per atom\n\nAtoms # atomic\n\n1 1 0 0 0 1 2 3\n"
    back = round_trip(tmp_path, read_text(tmp_path, text))
    assert back.info["lammps"]["extra"] == {"bond": 3}
    assert back.table["image"].to_list() == [[1, 2, 3]]


def test_restricted_boxes_are_written_as_bounds(tmp_path):
    box = mdapy.Box([[4.0, 0, 0], [1.0, 5.0, 0], [0.5, -1.0, 6.0]])
    system = mdapy.System([[1.0, 2.0, 3.0]], box, elements="Cu")
    back = round_trip(tmp_path, system)
    assert back.box == box
    assert "1.0 0.5 -1.0 xy xz yz" in (tmp_path / "rt.data").read_text()
    shifted = mdapy.Box(box.matrix, origin=[0.1, -0.2, 0.3])
    back = round_trip(tmp_path, mdapy.System([[1.0, 2.0, 3.0]], shifted, elements="Cu"))
    # hi = lo + length is rounded, so the lengths can differ in the last bit
    np.testing.assert_allclose(back.box.matrix, shifted.matrix, rtol=1e-15)
    np.testing.assert_array_equal(back.box.origin, shifted.origin)


def test_other_boxes_are_written_as_edge_vectors(tmp_path):
    matrix = np.array([[4.0, 0.3, 0.2], [0.5, 4.2, -0.1], [0.3, 0.4, 3.9]])
    box = mdapy.Box(matrix, origin=[1.0, -2.0, 0.5])
    system = mdapy.System([[1.0, 2.0, 3.0]], box, elements="Cu")
    back = round_trip(tmp_path, system)
    assert back.box == box
    assert "4.0 0.3 0.2 avec" in (tmp_path / "rt.data").read_text()
    back = round_trip(
        tmp_path,
        mdapy.System([[1.0, 2.0, 3.0]], 5.0, elements="Cu"),
        triclinic="general",
    )
    assert back.box == mdapy.Box(5.0)
    assert "5.0 0.0 0.0 avec" in (tmp_path / "rt.data").read_text()


def test_left_handed_boxes_without_images(tmp_path):
    box = mdapy.Box([[4.0, 0, 0], [0, 0, 5.0], [0, 6.0, 0]])
    back = round_trip(tmp_path, mdapy.System([[1.0, 2.0, 3.0]], box, elements="Cu"))
    np.testing.assert_array_equal(back.box.matrix, box.matrix[[0, 2, 1]])
    assert "image" not in back.table.columns


def test_left_handed_boxes_swap_b_and_c(tmp_path):
    box = mdapy.Box([[4.0, 0, 0], [0, 0, 5.0], [0, 6.0, 0]])
    system = mdapy.System([[1.0, 2.0, 3.0]], box, elements="Cu").with_columns(
        image=[[1, 2, 3]]
    )
    back = round_trip(tmp_path, system)
    np.testing.assert_array_equal(back.box.matrix, box.matrix[[0, 2, 1]])
    assert back.table["image"].to_list() == [[1, 3, 2]]
    np.testing.assert_array_equal(back.positions, system.positions)
    unwrapped = back.positions + back.table["image"].to_numpy() @ back.box.matrix
    expected = system.positions + system.table["image"].to_numpy() @ box.matrix
    np.testing.assert_array_equal(unwrapped, expected)


@pytest.mark.parametrize("left_handed", [False, True])
def test_restricted_option_rotates_vectors(tmp_path, left_handed):
    rng = np.random.default_rng(3)
    matrix = np.array([[4.0, 0.3, 0.2], [0.5, 4.2, -0.1], [0.3, 0.4, 3.9]])
    if left_handed:
        matrix = matrix[[0, 2, 1]]
    box = mdapy.Box(matrix, origin=[1.0, -2.0, 0.5])
    frac = rng.random((10, 3))
    system = mdapy.System(frac @ matrix + box.origin, box, elements="Cu").with_columns(
        velocities=rng.standard_normal((10, 3)), image=rng.integers(-2, 3, (10, 3))
    )
    back = round_trip(tmp_path, system, triclinic="restricted")
    m = back.box.matrix
    assert m[0, 1] == m[0, 2] == m[1, 2] == 0
    np.testing.assert_allclose(back.box.volume, box.volume, rtol=1e-13)
    unwrapped = lambda s: s.positions + s.table["image"].to_numpy() @ s.box.matrix  # noqa: E731
    d_back = np.linalg.norm(unwrapped(back)[:, None] - unwrapped(back)[None], axis=-1)
    d_orig = np.linalg.norm(
        unwrapped(system)[:, None] - unwrapped(system)[None], axis=-1
    )
    np.testing.assert_allclose(d_back, d_orig, atol=1e-12)
    speed = lambda s: np.linalg.norm(s.table["velocities"].to_numpy(), axis=1)  # noqa: E731
    np.testing.assert_allclose(speed(back), speed(system), rtol=1e-13)


@pytest.mark.parametrize(
    "system, options, error, message",
    [
        (mdapy.System(np.zeros((1, 3)), 5.0), {"atom_style": "full"}, MissingProperty, "needs a charge"),
        (mdapy.System(np.zeros((1, 3)), 5.0), {"atom_style": "sphere"}, UnsupportedFormat, "not supported"),
        (mdapy.System(np.zeros((1, 3)), 5.0), {"units": "si"}, InvalidParameter, "units must be"),
        (mdapy.System(np.zeros((1, 3)), 5.0), {"triclinic": "x"}, InvalidParameter, "triclinic must be"),
        (
            mdapy.System(np.zeros((2, 3)), 5.0).with_columns(id=[1, 1]),
            {},
            InvalidParameter,
            "positive and unique",
        ),
        (
            mdapy.System(np.zeros((2, 3)), 5.0).with_columns(type=[1, 1], mass=[1.0, 2.0]),
            {},
            InvalidParameter,
            "type 1 have different mass",
        ),
        (
            mdapy.System(np.zeros((2, 3)), 5.0, info={"lammps": {"atom_types": 2, "masses": [1.0]}}),
            {},
            InvalidParameter,
            r"no mass is known for atom types \[2\]",
        ),
        (
            mdapy.System(np.zeros((2, 3)), 5.0, types=[1, 2]).with_columns(type_label=["A", "1B"]),
            {},
            InvalidParameter,
            "invalid labels",
        ),
        (
            mdapy.System(np.zeros((2, 3)), 5.0, types=[1, 3]).with_columns(type_label=["A", "B"]),
            {},
            InvalidParameter,
            r"types \[2\] have none",
        ),
        (
            mdapy.System(np.zeros((2, 3)), 5.0, types=[1, 2]).with_columns(type_label=["A", "A"]),
            {},
            InvalidParameter,
            "labels are repeated",
        ),
        (
            mdapy.System(np.zeros((2, 3)), 5.0, topology=mdapy.Topology(bonds=[[0, 1]])),
            {"atom_style": "atomic"},
            InvalidParameter,
            "cannot hold bonds",
        ),
        (
            mdapy.System(np.zeros((1, 3)), 5.0, info={"lammps": {"bond_types": 1}}),
            {"atom_style": "charge"},
            MissingProperty,
            "needs a charge",
        ),
        (
            mdapy.System(np.zeros((1, 3)), 5.0, info={"lammps": {"bond_types": 1}}),
            {"atom_style": "atomic"},
            InvalidParameter,
            "cannot hold bonds",
        ),
        (
            mdapy.System(np.zeros((1, 3)), 5.0, info={"lammps": "full"}),
            {},
            InvalidParameter,
            "must be a mapping",
        ),
        (
            mdapy.System(
                np.zeros((1, 3)), 5.0,
                info={"lammps": {"coeffs": {"Pair Coeffs": {"lines": ["1 1", "2 2"]}}}},
            ),
            {},
            InvalidParameter,
            "has 2 lines, but 1 are needed",
        ),
        (
            mdapy.System(
                np.zeros((1, 3)), 5.0,
                info={"lammps": {"coeffs": {"Foo Coeffs": {"lines": []}}}},
            ),
            {},
            InvalidParameter,
            "unknown coefficient section",
        ),
    ],
)  # fmt: skip
def test_write_errors(tmp_path, system, options, error, message):
    with pytest.raises(error, match=message):
        mdapy.write(tmp_path / "e.data", system, **options)
    assert not (tmp_path / "e.data").exists()


def test_format_detection():
    assert mdapy.detect_format("water.data") == "lammps-data"
    assert mdapy.detect_format("model.lmp") == "lammps-data"
    assert mdapy.detect_format("data.CuZr") == "lammps-data"
    assert mdapy.detect_format("data.xyz") == "extxyz"  # the extension wins
    assert mdapy.detect_format("data.lmp.gz") == "lammps-data"


def test_write_one_frame_only(tmp_path):
    system = mdapy.System(np.zeros((1, 3)), 5.0)
    with pytest.raises(InvalidParameter, match="hold one frame"):
        mdapy.write(tmp_path / "two.data", [system, system])
