# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest

import mdapy
from mdapy.errors import FileFormatError, InvalidParameter, MissingProperty

CONTCAR = """\
Cubic BN
   3.57
 0.0 0.5 0.5
 0.5 0.0 0.5
 0.5 0.5 0.0
   B_s/1a2b3c   N
   1 1
Selective dynamics
Direct
 0.00 0.00 0.00 T T F  B1
 0.25 0.25 0.25 F F F  ! nitrogen

  0.01  0.02  0.03
  0.00  0.00  -0.004
"""

CELL = 3.57 * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])


def read_text(tmp_path, text, name="POSCAR", **options):
    path = tmp_path / name
    path.write_text(text)
    return mdapy.read(path, **options)


def poscar(
    scale="1.0",
    species="Cu",
    counts="2",
    mode="Direct",
    lines="0 0 0\n0.5 0.5 0.5",
    tail="",
):
    return f"comment\n{scale}\n4 0 0\n0 4 0\n0 0 4\n{species}\n{counts}\n{mode}\n{lines}\n{tail}"


def test_read_contcar(tmp_path):
    system = read_text(tmp_path, CONTCAR, name="CONTCAR")
    np.testing.assert_array_equal(system.box.matrix, CELL)
    assert system.box.pbc.all()
    assert system.table["element"].to_list() == ["B", "N"]
    np.testing.assert_allclose(
        system["positions"][1], [0.25, 0.25, 0.25] @ CELL, rtol=1e-15
    )
    assert system.table["movable"].to_list() == [
        [True, True, False],
        [False, False, False],
    ]
    # Å/fs in the file, Å/ps in mdapy, scaled exactly in decimal
    assert system.table["velocities"].to_list() == [
        [10.0, 20.0, 30.0],
        [0.0, 0.0, -4.0],
    ]
    assert system.info == {"comment": "Cubic BN"}


@pytest.mark.parametrize(
    "scale, factors",
    [("2.0", [2.0, 2.0, 2.0]), ("-512", [2.0, 2.0, 2.0]), ("1 2 0.5", [1.0, 2.0, 0.5])],
)
def test_scaling_factors(tmp_path, scale, factors):
    direct = read_text(tmp_path, poscar(scale=scale))
    np.testing.assert_allclose(
        direct.box.matrix, np.diag(4.0 * np.array(factors)), rtol=1e-15
    )
    np.testing.assert_allclose(
        direct["positions"][1], 2.0 * np.array(factors), rtol=1e-15
    )
    cartesian = read_text(
        tmp_path, poscar(scale=scale, mode="Cartesian", lines="0 0 0\n1 1 1")
    )
    np.testing.assert_allclose(cartesian["positions"][1], factors, rtol=1e-15)


@pytest.mark.parametrize("mode", ["Cartesian", "cart", "K", "k"])
def test_cartesian_modes(tmp_path, mode):
    system = read_text(tmp_path, poscar(mode=mode, lines="0 0 0\n1 2 3"))
    assert system["positions"][1].tolist() == [1.0, 2.0, 3.0]


def test_vasp4_files_need_elements(tmp_path):
    text = "comment\n1.0\n4 0 0\n0 4 0\n0 0 4\n1 2\nDirect\n0 0 0\n0.5 0 0\n0 0.5 0\n"
    with pytest.raises(FileFormatError, match="pass elements="):
        read_text(tmp_path, text)
    system = read_text(tmp_path, text, elements=["Cu", "Zr"])
    assert system.table["element"].to_list() == ["Cu", "Zr", "Zr"]
    with pytest.raises(InvalidParameter, match="unknown symbol 'Qq'"):
        read_text(tmp_path, text, elements=["Cu", "Qq"])


def test_velocity_blocks(tmp_path):
    lines = "0 0 0\n0.5 0.5 0.5"
    plain = read_text(tmp_path, poscar(lines=lines, tail="\n1 2 3\n4 5 6\n"))
    assert plain.table["velocities"].to_list() == [
        [1000.0, 2000.0, 3000.0],
        [4000.0, 5000.0, 6000.0],
    ]
    # ASE writes a mode line instead of the blank line
    labelled = read_text(
        tmp_path, poscar(lines=lines, tail="Cartesian\n1 2 3\n4 5 6\n")
    )
    assert labelled.table["velocities"].to_list() == plain.table["velocities"].to_list()
    direct = read_text(tmp_path, poscar(lines=lines, tail="Direct\n0.25 0 0\n0 0 0\n"))
    assert direct.table["velocities"].to_list()[0] == [
        1000.0,
        0.0,
        0.0,
    ]  # 0.25 * 4 Å/fs
    # the lattice block follows the coordinates directly
    lattice = (
        "Lattice velocities and vectors\n1\n"
        + "0 0 0\n" * 3
        + "4 0 0\n0 4 0\n0 0 4\n\n1 2 3\n4 5 6\n"
    )
    after_lattice = read_text(tmp_path, poscar(lines=lines, tail=lattice))
    assert (
        after_lattice.table["velocities"].to_list()
        == plain.table["velocities"].to_list()
    )
    only_lattice = (
        "Lattice velocities and vectors\n1\n" + "0 0 0\n" * 3 + "4 0 0\n0 4 0\n0 0 4\n"
    )
    assert (
        "velocities"
        not in read_text(tmp_path, poscar(lines=lines, tail=only_lattice)).table.columns
    )
    md_block = read_text(tmp_path, poscar(lines=lines, tail="\n1\n0.5\n"))
    assert "velocities" not in md_block.table.columns
    assert (
        "velocities"
        not in read_text(tmp_path, poscar(lines=lines, tail="\n\n")).table.columns
    )
    assert (
        "velocities"
        not in read_text(
            tmp_path, poscar(lines=lines, tail="Cartesian\n")
        ).table.columns
    )


def test_coordinate_lines_with_extra_words(tmp_path):
    lines = "0 0 0 T F T site-a\n0.5 0.5 0.5 F F F site b ! comment"
    system = read_text(tmp_path, poscar(mode="Selective dynamics\nDirect", lines=lines))
    assert system.table["movable"].to_list() == [
        [True, False, True],
        [False, False, False],
    ]
    np.testing.assert_array_equal(system["positions"][1], [2.0, 2.0, 2.0])
    with pytest.raises(FileFormatError, match="must be T or F"):
        read_text(
            tmp_path,
            poscar(
                mode="Selective dynamics\nDirect",
                lines="0 0 0 T F T a\n0 0 0 T x T b c",
            ),
        )
    with pytest.raises(FileFormatError, match="expected 6 values, found 4") as info:
        read_text(
            tmp_path,
            poscar(mode="Selective dynamics\nDirect", lines="0 0 0 T F T a\n0 0 0 T"),
        )
    assert info.value.line == 11


@pytest.mark.parametrize(
    "text, line, message",
    [
        (poscar(scale="x"), 2, "expected the scaling factor"),
        (poscar(scale="0"), 2, "one number or three positive"),
        (poscar(scale="1 -1 1"), 2, "one number or three positive"),
        (poscar(scale="1 1"), 2, "one number or three positive"),
        (poscar(scale="-10").replace("0 0 4\n", "0 0 0\n", 1), 5, "linearly dependent"),
        ("c\n1\n1 0\n0 1 0\n0 0 1\nCu\n1\nDirect\n0 0 0\n", 3, "three numbers"),
        (
            "c\n1\n1 0 x\n0 1 0\n0 0 1\nCu\n1\nDirect\n0 0 0\n",
            3,
            "expected a lattice vector",
        ),
        (poscar(species="Qq"), 6, "unknown element 'Qq'"),
        (poscar(counts="2 1"), 7, "expected 1 atom counts"),
        (poscar(counts="x"), 7, "expected 1 atom counts"),
        (poscar(mode=""), 8, "expected Direct or Cartesian"),
        (
            poscar(lines="0 0\n0 0 0"),
            9,
            "expected 3 values per coordinate line, found 2",
        ),
        (poscar(lines="0 0 0"), 10, "expected 2 lines"),
        (poscar(lines="0 0 0\n0 0 x"), 10, "cannot read 'x'"),
        (poscar(lines="0 0 0\n0 0 nan"), 3, "finite"),
    ],
)
def test_errors_name_the_line(tmp_path, text, line, message):
    with pytest.raises(FileFormatError, match=message) as info:
        read_text(tmp_path, text)
    assert info.value.line == line


def test_counts_line_comment(tmp_path):
    system = read_text(tmp_path, poscar(counts="2 ! two atoms"))
    assert len(system) == 2


# ---------------------------------------------------------------- writing


def test_write_groups_elements_in_order_of_first_appearance(tmp_path):
    rng = np.random.default_rng(0)
    box = mdapy.Box(
        [[4.0, 0, 0], [1.0, 4.0, 0], [0.5, -0.5, 4.0]], origin=[1.0, 2.0, 3.0]
    )
    elements = ["Zr", "Cu", "Zr", "Al", "Cu"]
    system = mdapy.System(box.to_cartesian(rng.random((5, 3))), box, elements=elements)
    system = system.with_columns(
        movable=rng.random((5, 3)) < 0.5, velocities=rng.standard_normal((5, 3))
    )
    path = tmp_path / "POSCAR"
    mdapy.write(path, system)
    text = path.read_text().splitlines()
    assert text[0] == "written by mdapy"
    assert text[5:9] == ["Zr Cu Al", "2 2 1", "Selective dynamics", "Direct"]
    back = mdapy.read(path)
    order = [0, 2, 1, 4, 3]
    assert back.table["element"].to_list() == [elements[i] for i in order]
    np.testing.assert_array_equal(back.box.matrix, box.matrix)
    np.testing.assert_array_equal(
        back.box.origin, [0.0, 0.0, 0.0]
    )  # no origin in POSCAR
    np.testing.assert_allclose(
        back["positions"], system["positions"][order] - box.origin, atol=1e-14
    )
    assert (
        back.table["movable"].to_list()
        == system.table["movable"].to_numpy()[order].tolist()
    )
    np.testing.assert_array_equal(
        back.table["velocities"].to_numpy(),
        system.table["velocities"].to_numpy()[order],
    )


def test_write_cartesian_and_comments(tmp_path):
    box = mdapy.Box(5.0, origin=[1.0, 1.0, 1.0])
    system = mdapy.System(
        [[1.5, 2.0, 2.5]], box, elements="Si", info={"comment": "silicon"}
    )
    path = tmp_path / "x.vasp"
    mdapy.write(path, system, direct=False)
    lines = path.read_text().splitlines()
    assert lines[0] == "silicon" and lines[7] == "Cartesian" and lines[8] == "0.5 1 1.5"
    assert mdapy.read(path).info == {"comment": "silicon"}


@pytest.mark.parametrize(
    "system, error, message",
    [
        (mdapy.System([[0.0, 0, 0]], 5.0), MissingProperty, "needs element symbols"),
        (mdapy.System([[0.0, 0, 0]], 5.0, elements="H", info={"comment": "a\nb"}), InvalidParameter, "one line"),
        (mdapy.System([[0.0, 0, 0]], 5.0, elements="H", info={"comment": 3}), InvalidParameter, "one line"),
        (mdapy.System([[0.0, 0, 0]], 5.0, elements="H").with_columns(movable=[1]), InvalidParameter, "3 booleans"),
    ],
)  # fmt: skip
def test_write_errors(tmp_path, system, error, message):
    with pytest.raises(error, match=message):
        mdapy.write(tmp_path / "POSCAR", system)
    assert not (tmp_path / "POSCAR").exists()


def test_format_detection():
    for name in (
        "POSCAR",
        "CONTCAR",
        "POSCAR_relaxed",
        "CONTCAR.gz",
        "si.vasp",
        "a.poscar",
    ):
        assert mdapy.detect_format(name) == "poscar"


def test_extra_words_without_selective_dynamics(tmp_path):
    system = read_text(tmp_path, poscar(lines="0 0 0 a\n0.5 0.5 0.5 b c"))
    np.testing.assert_array_equal(system["positions"][1], [2.0, 2.0, 2.0])
    assert "movable" not in system.table.columns


def test_no_atoms(tmp_path):
    system = read_text(tmp_path, poscar(counts="0", lines="", tail="\n"))
    assert len(system) == 0
