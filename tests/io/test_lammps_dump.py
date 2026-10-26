# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import polars as pl
import pytest

import mdapy
from mdapy import units
from mdapy.errors import FileFormatError, InvalidParameter, UnsupportedFormat

# Two frames as LAMMPS writes them with "dump_modify units yes time yes":
# UNITS only in the first frame, a triclinic box as its bounding box.
TWO_FRAMES = """\
ITEM: UNITS
real
ITEM: TIME
0
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS xy xz yz pp pp ff
0.0000000000000000e+00 9.5000000000000000e+00 1.0000000000000000e+00
-5.0000000000000000e-01 8.0000000000000000e+00 5.0000000000000000e-01
0.0000000000000000e+00 8.0000000000000000e+00 -5.0000000000000000e-01
ITEM: ATOMS id type element x y z vx vy vz fx fy fz c_s[1] c_s[2] c_s[3] i_flag
2 2 Zr 1.5 2.5 3.5 0.001 0.002 -0.003 1 0 0 10 20 30 7
1 1 Cu 4.25 0.5 1.0 0 0 0 0 0 0 1 2 3 8
ITEM: TIME
1.5
ITEM: TIMESTEP
1500
ITEM: NUMBER OF ATOMS
1
ITEM: BOX BOUNDS xy xz yz pp pp ff
0.0000000000000000e+00 9.5000000000000000e+00 1.0000000000000000e+00
-5.0000000000000000e-01 8.0000000000000000e+00 5.0000000000000000e-01
0.0000000000000000e+00 8.0000000000000000e+00 -5.0000000000000000e-01
ITEM: ATOMS id type element x y z vx vy vz fx fy fz c_s[1] c_s[2] c_s[3] i_flag
1 1 Cu 4.0 0.5 1.0 0.004 0 0 0 0 0 1 2 3 8
"""

ORTHOGONAL = """\
ITEM: TIMESTEP
100
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
-1.0 4.0
0.0 5.0
1.0 7.0
ITEM: ATOMS {columns}
{atoms}
"""


def write_text(tmp_path, text, name="t.dump"):
    path = tmp_path / name
    path.write_text(text)
    return path


def read_text(tmp_path, text, name="t.dump", **options):
    return mdapy.read(write_text(tmp_path, text, name), **options)


def orthogonal(columns, atoms):
    return ORTHOGONAL.format(columns=columns, atoms=atoms)


def round_trip(tmp_path, systems, **options):
    path = tmp_path / "rt.dump"
    mdapy.write(path, systems, **options)
    return path


# ---------------------------------------------------------------- reading


def test_read_lammps_frames(tmp_path):
    path = write_text(tmp_path, TWO_FRAMES)
    first = mdapy.read(path)
    table = first.table
    assert table.columns == [
        "id", "type", "element", "positions", "velocities", "forces", "c_s", "i_flag",
    ]  # fmt: skip
    assert table["id"].to_list() == [2, 1]  # the order of the file
    assert table["element"].to_list() == ["Zr", "Cu"]
    assert table["positions"].to_list()[0] == [1.5, 2.5, 3.5]
    # real units: Å/fs and kcal/mol/Å in the file
    assert table["velocities"].to_list()[0] == [1.0, 2.0, -3.0]
    assert table["forces"].to_list()[0] == [units.KCAL_PER_MOL_TO_EV, 0.0, 0.0]
    assert table["c_s"].dtype == pl.Array(pl.Float64, 3)
    assert table["c_s"].to_list()[0] == [10.0, 20.0, 30.0]
    assert table["i_flag"].dtype == pl.Int64
    assert first.info == {"timestep": 0, "time": 0.0}
    np.testing.assert_array_equal(
        first.box.matrix, [[8.0, 0, 0], [1.0, 8.0, 0], [0.5, -0.5, 8.0]]
    )
    np.testing.assert_array_equal(first.box.origin, [0.0, 0.0, 0.0])
    assert first.box.pbc.tolist() == [True, True, False]
    second = mdapy.read(path, frame=1)
    assert second.info == {"timestep": 1500, "time": 0.0015}  # fs to ps
    # the units of the first frame apply to every frame
    assert second.table["velocities"].to_list() == [[4.0, 0.0, 0.0]]
    assert mdapy.read(path, frame=-1).info["timestep"] == 1500


def test_coordinates_in_order_of_preference(tmp_path):
    columns = "id type xs ys zs xu yu zu x y z"
    atoms = "1 1 0.5 0.5 0.5 9 9 9 1 2 3\n2 1 0 0 0 9 9 9 4 5 6"
    table = read_text(tmp_path, orthogonal(columns, atoms)).table
    assert table["positions"].to_list() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert table.columns == [
        "id",
        "type",
        "xs",
        "ys",
        "zs",
        "xu",
        "yu",
        "zu",
        "positions",
    ]
    scaled = read_text(
        tmp_path, orthogonal("id type xs ys zs", "1 1 0.5 0.5 0.5\n2 1 0 0 0")
    )
    assert scaled.table["positions"].to_list() == [[1.5, 2.5, 4.0], [-1.0, 0.0, 1.0]]
    unwrapped = read_text(
        tmp_path,
        orthogonal(
            "id type xsu ysu zsu ix iy iz", "1 1 1.5 0 0 1 0 0\n2 1 0 0 0 0 0 0"
        ),
    )
    assert unwrapped.table["positions"].to_list()[0] == [6.5, 0.0, 1.0]
    assert unwrapped.table["image"].dtype == pl.Array(pl.Int32, 3)


def test_incomplete_vectors_and_arrays_stay_columns(tmp_path):
    columns = "id type x y z vx vy c_a[1] c_a[3] d_b[1] d_b[2] mol q mass"
    atoms = "1 1 0 0 0 1 2 3 4 5 6 7 0.5 12\n2 1 0 0 0 1 2 3 4 5 6 7 0.5 12"
    table = read_text(tmp_path, orthogonal(columns, atoms)).table
    assert table.columns == [
        "id", "type", "positions", "vx", "vy", "c_a[1]", "c_a[3]", "d_b",
        "molecule", "charge", "mass",
    ]  # fmt: skip
    assert table["d_b"].to_list()[0] == [5.0, 6.0]
    assert table["molecule"].dtype == pl.Int64


def test_box_forms(tmp_path):
    general = (
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n"
        "ITEM: BOX BOUNDS abc origin pp ff pp\n"
        "4 0.5 0 1\n0 3 1 2\n0.2 0 5 3\n"
        "ITEM: ATOMS id type x y z\n1 1 0 0 0\n"
    )
    box = read_text(tmp_path, general).box
    np.testing.assert_array_equal(box.matrix, [[4, 0.5, 0], [0, 3, 1], [0.2, 0, 5]])
    np.testing.assert_array_equal(box.origin, [1, 2, 3])
    assert box.pbc.tolist() == [True, False, True]
    old = orthogonal("id type x y z", "1 1 0 0 0\n2 1 1 1 1").replace(
        "BOUNDS pp pp pp", "BOUNDS"
    )
    assert read_text(tmp_path, old).box.pbc.all()  # old dumps have no flags
    for flags in ("fs fm ff", "sm ss fs"):
        text = orthogonal("id type x y z", "1 1 0 0 0\n2 1 1 1 1").replace(
            "pp pp pp", flags
        )
        assert not read_text(tmp_path, text).box.pbc.any()


@pytest.mark.parametrize("tilt", [(1.0, 0.5, -0.5), (-2.0, 0.5, 1.5), (0.0, -1.0, 0.0)])
def test_triclinic_bounding_boxes(tmp_path, tilt):
    box = mdapy.Box.from_lammps([0.5, -1, 2], [8.5, 7, 10], tilt)
    path = round_trip(tmp_path, mdapy.System([[1.0, 1.0, 3.0]], box, elements="Cu"))
    back = mdapy.read(path)
    np.testing.assert_allclose(back.box.matrix, box.matrix, rtol=1e-15, atol=1e-14)
    np.testing.assert_allclose(back.box.origin, box.origin, rtol=1e-15, atol=1e-15)


def test_elements_option(tmp_path):
    text = orthogonal("id type x y z", "1 2 0 0 0\n2 1 1 1 1")
    table = read_text(tmp_path, text, elements=["Cu", "Zr"]).table
    assert table["element"].to_list() == ["Zr", "Cu"]
    with pytest.raises(
        InvalidParameter, match="names 1 elements, but the dump has type 2"
    ):
        read_text(tmp_path, text, elements=["Cu"])
    with pytest.raises(InvalidParameter, match="unknown symbol"):
        read_text(tmp_path, text, elements=["Cu", "Qq"])
    with pytest.raises(InvalidParameter, match="needs a type column"):
        read_text(tmp_path, orthogonal("id x y z", "1 0 0 0\n2 1 1 1"), elements=["Cu"])
    # an element column wins over the option
    with_elements = orthogonal("id type element x y z", "1 2 Ni 0 0 0\n2 1 Ni 1 1 1")
    assert read_text(tmp_path, with_elements, elements=["Cu", "Zr"]).table[
        "element"
    ].to_list() == ["Ni", "Ni"]


def test_units(tmp_path):
    lj = "ITEM: UNITS\nlj\n" + orthogonal(
        "id type x y z vx vy vz", "1 1 0 0 0 1 2 3\n2 1 1 1 1 0 0 0"
    )
    with pytest.raises(UnsupportedFormat, match="in lj units"):
        read_text(tmp_path, lj)
    table = read_text(tmp_path, lj, units="metal").table
    assert table["velocities"].to_list()[0] == [1.0, 2.0, 3.0]
    table = read_text(tmp_path, lj, units="real").table
    assert table["velocities"].to_list()[0] == [1000.0, 2000.0, 3000.0]
    with pytest.raises(InvalidParameter, match="units must be"):
        read_text(tmp_path, lj, units="si")


def test_blank_lines_between_and_after_frames(tmp_path):
    frame = orthogonal("id type x y z", "1 1 0 0 0\n2 1 1 1 1")
    path = write_text(tmp_path, "\n" + frame + "\n\n" + frame + "\n  \n")
    assert len(mdapy.read(path, frame=1)) == 2
    with pytest.raises(InvalidParameter, match="the file has 2 frames"):
        mdapy.read(path, frame=2)


# ----------------------------------------------------------------- errors


@pytest.mark.parametrize(
    "text, line, message",
    [
        ("hello\n", 1, "expected an ITEM line, found 'hello'"),
        ("ITEM: FOO\n", 1, "unknown item 'FOO'"),
        ("ITEM: TIMESTEP\nx\n", 2, "expected an integer, found 'x'"),
        ("ITEM: NUMBER OF ATOMS\n-1\n", 2, "negative"),
        ("ITEM: BOX BOUNDS pp pp\n", 1, "invalid boundary flags"),
        ("ITEM: BOX BOUNDS pp pp pp\n0 1\n0 x\n", 3, "expected 2 numbers"),
        ("ITEM: BOX BOUNDS pp pp pp\n0 1\n0 1\n0 0\n", 4, "linearly dependent"),
        ("ITEM: NUMBER OF ATOMS\n1\nITEM: ATOMS id\n", 3, "comes before"),
        (
            orthogonal("id id x y z", "1 1 0 0 0\n2 2 0 0 0"),
            9,
            "a column name is repeated",
        ),
        (
            orthogonal("id type x y z", "1 1 0 0 0"),
            11,
            "the frame has 2 atoms, but the file ends after 1",
        ),
        (orthogonal("id type x y z", "1 1 0 0 0\n2 1 x 0 0"), 11, "cannot read 'x'"),
        (
            orthogonal("id type element x y z", "1 1 Cu 0 0 0\n2 1 Qq 0 0 0"),
            11,
            "unknown element 'Qq'",
        ),
        (orthogonal("id type x y z", "1 1 0 0 0\n2 1 nan 0 0"), 9, "finite"),
    ],
)
def test_errors_name_the_line(tmp_path, text, line, message):
    with pytest.raises(FileFormatError, match=message) as info:
        read_text(tmp_path, text)
    assert info.value.line == line


def test_unsupported_dumps(tmp_path):
    path = tmp_path / "b.dump"
    path.write_bytes(b"\x05\x00\x00\x00\x00\x00\x00\x00DUMPCUSTOM")
    with pytest.raises(UnsupportedFormat, match="binary dump"):
        mdapy.read(path)
    local = "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ENTRIES\n3\n"
    with pytest.raises(UnsupportedFormat, match="dump local"):
        read_text(tmp_path, local)


# ---------------------------------------------------------------- writing


def rich_system(n=20, seed=0, box=None):
    rng = np.random.default_rng(seed)
    box = box or mdapy.Box.from_lammps([0, -1, 2], [9, 8, 12], (1.0, 0.5, -0.5))
    system = mdapy.System(
        box.to_cartesian(rng.random((n, 3))),
        box,
        elements=rng.choice(["Cu", "Zr"], n),
        info={"timestep": 42, "time": 0.042},
    )
    return system.with_columns(
        id=rng.permutation(n) + 10,
        type=rng.integers(1, 4, n),
        type_label=rng.choice(["A", "B"], n),
        molecule=rng.integers(0, 5, n),
        charge=rng.standard_normal(n),
        mass=rng.random(n) + 1,
        image=rng.integers(-2, 3, (n, 3)),
        velocities=rng.standard_normal((n, 3)),
        forces=rng.standard_normal((n, 3)),
        c_pe=rng.standard_normal(n),
        c_stress=rng.standard_normal((n, 6)),
        counts=rng.integers(-5, 5, n),
        flags=rng.random(n) < 0.5,
        tensors=rng.standard_normal((n, 3, 3)),
    )


def test_round_trip(tmp_path):
    system = rich_system()
    back = mdapy.read(round_trip(tmp_path, system))
    expected = system.table.select(
        "id", "type", "element", "type_label", "positions", "molecule", "charge",
        "mass", "image", "velocities", "forces", "c_pe", "c_stress", "counts",
        pl.col("flags").cast(pl.Int64), pl.col("tensors").reshape((-1, 9)),
    )  # fmt: skip
    assert back.table.equals(expected)
    assert back.info == {"timestep": 42, "time": 0.042}
    assert back.box.allclose(system.box, rtol=1e-15, atol=1e-14)


def test_round_trip_in_real_units(tmp_path):
    system = rich_system()
    back = mdapy.read(round_trip(tmp_path, system, units="real"))
    # velocities are scaled in decimal and come back exactly
    np.testing.assert_array_equal(
        back.table["velocities"].to_numpy(), system.table["velocities"].to_numpy()
    )
    np.testing.assert_allclose(
        back.table["forces"].to_numpy(), system.table["forces"].to_numpy(), rtol=1e-15
    )
    assert back.info["time"] == 0.042
    assert "ITEM: TIME\n42\n" in (tmp_path / "rt.dump").read_text()


def test_written_text(tmp_path):
    system = mdapy.System(
        [[0.0, 0.0, 0.0], [1.5, 2.0, 2.5]],
        mdapy.Box(5.0, pbc=[True, True, False]),
        elements=["Cu", "Zr"],
    )
    path = round_trip(tmp_path, system)
    assert path.read_text() == (
        "ITEM: UNITS\nmetal\n"
        "ITEM: TIMESTEP\n0\n"
        "ITEM: NUMBER OF ATOMS\n2\n"
        "ITEM: BOX BOUNDS pp pp ff\n0.0 5.0\n0.0 5.0\n0.0 5.0\n"
        "ITEM: ATOMS id type element x y z\n"
        "1 1 Cu 0 0 0\n2 2 Zr 1.5 2 2.5\n"
    )


def test_many_frames_from_a_generator(tmp_path):
    system = rich_system(5)
    frames = (
        system.with_info(timestep=k).with_columns(
            positions=system["positions"] + 0.1 * k
        )
        for k in range(4)
    )
    path = round_trip(tmp_path, frames)
    text = path.read_text()
    assert text.count("ITEM: UNITS") == 1 and text.count("ITEM: TIMESTEP") == 4
    third = mdapy.read(path, frame=2)
    assert third.info["timestep"] == 2
    np.testing.assert_allclose(
        third["positions"], system["positions"] + 0.2, rtol=1e-15
    )


def test_columns_option(tmp_path):
    system = rich_system(3)
    back = mdapy.read(round_trip(tmp_path, system, columns=["velocities", "c_pe"]))
    assert back.table.columns == [
        "id",
        "type",
        "element",
        "type_label",
        "positions",
        "velocities",
        "c_pe",
    ]
    with pytest.raises(InvalidParameter, match="no column named 'nope'"):
        round_trip(tmp_path, system, columns=["nope"])


def test_types_from_elements(tmp_path):
    system = mdapy.System(np.zeros((3, 3)), 5.0, elements=["Zr", "Cu", "Zr"])
    back = mdapy.read(round_trip(tmp_path, system))
    assert back.table["type"].to_list() == [1, 2, 1]
    back = mdapy.read(round_trip(tmp_path, system, elements=["Cu", "Zr"]))
    assert back.table["type"].to_list() == [2, 1, 2]


@pytest.mark.parametrize("triclinic", ["auto", "general"])
def test_general_boxes(tmp_path, triclinic):
    matrix = np.array([[4.0, 0.3, 0.2], [0.5, 4.2, -0.1], [0.3, 0.4, 3.9]])
    box = mdapy.Box(matrix, origin=[1.0, -2.0, 0.5], pbc=[True, False, True])
    system = mdapy.System([[1.0, 2.0, 3.0]], box, elements="Cu")
    path = round_trip(tmp_path, system, triclinic=triclinic)
    assert "ITEM: BOX BOUNDS abc origin pp ff pp\n4.0 0.3 0.2 1.0\n" in path.read_text()
    assert mdapy.read(path).box == box
    orthogonal_box = mdapy.Box(5.0)
    path = round_trip(
        tmp_path,
        mdapy.System([[1.0, 2.0, 3.0]], orthogonal_box, elements="Cu"),
        triclinic=triclinic,
    )
    assert mdapy.read(path).box == orthogonal_box


def test_left_handed_boxes(tmp_path):
    box = mdapy.Box([[4.0, 0, 0], [0, 0, 5.0], [0, 6.0, 0]], pbc=[True, True, False])
    system = mdapy.System([[1.0, 2.0, 3.0]], box, elements="Cu").with_columns(
        image=[[1, 2, 3]]
    )
    back = mdapy.read(round_trip(tmp_path, system))
    np.testing.assert_array_equal(back.box.matrix, box.matrix[[0, 2, 1]])
    assert back.box.pbc.tolist() == [True, False, True]
    assert back.table["image"].to_list() == [[1, 3, 2]]
    without_images = mdapy.System([[1.0, 2.0, 3.0]], box, elements="Cu")
    back = mdapy.read(round_trip(tmp_path, without_images))
    np.testing.assert_array_equal(back.box.matrix, box.matrix[[0, 2, 1]])


@pytest.mark.parametrize("left_handed", [False, True])
def test_restricted_option_rotates_vectors(tmp_path, left_handed):
    matrix = np.array([[4.0, 0.3, 0.2], [0.5, 4.2, -0.1], [0.3, 0.4, 3.9]])
    if left_handed:
        matrix = matrix[[0, 2, 1]]
    box = mdapy.Box(matrix, origin=[1.0, -2.0, 0.5])
    system = rich_system(10, box=box)
    back = mdapy.read(round_trip(tmp_path, system, triclinic="restricted"))
    m = back.box.matrix
    assert m[0, 1] == m[0, 2] == m[1, 2] == 0
    for name in ("velocities", "forces"):
        np.testing.assert_allclose(
            np.linalg.norm(back.table[name].to_numpy(), axis=1),
            np.linalg.norm(system.table[name].to_numpy(), axis=1),
            rtol=1e-13,
        )
    unwrapped = lambda s: s["positions"] + s.table["image"].to_numpy() @ s.box.matrix  # noqa: E731
    d = lambda p: np.linalg.norm(p[:, None] - p[None], axis=-1)  # noqa: E731
    np.testing.assert_allclose(d(unwrapped(back)), d(unwrapped(system)), atol=1e-12)


@pytest.mark.parametrize(
    "system, options, error, message",
    [
        (mdapy.System(np.zeros((1, 3)), 5.0), {"units": "si"}, InvalidParameter, "units must be"),
        (mdapy.System(np.zeros((1, 3)), 5.0), {"triclinic": "x"}, InvalidParameter, "triclinic must be"),
        (mdapy.System(np.zeros((1, 3)), 5.0).with_columns(**{"a b": 1.0}), {}, InvalidParameter, "cannot be written"),
        (mdapy.System(np.zeros((1, 3)), 5.0).with_columns(note=pl.Series(["x"])), {}, UnsupportedFormat, "of type String"),
        (
            mdapy.System(np.zeros((1, 3)), 5.0).with_columns(maybe=pl.Series([None], dtype=pl.Float64)),
            {},
            InvalidParameter,
            "missing values",
        ),
        (mdapy.System(np.zeros((1, 3)), 5.0).with_columns(vx=1.0, velocities=[[0.0, 0, 0]]), {}, InvalidParameter, r"both written as \['vx'\]"),
        (mdapy.System(np.zeros((1, 3)), 5.0).with_columns(type_label=["a b"]), {}, InvalidParameter, "without blanks"),
        (mdapy.System(np.zeros((1, 3)), 5.0, elements="Cu").with_columns(type=[1]), {"elements": ["Cu"]}, InvalidParameter, "has a type column"),
    ],
)  # fmt: skip
def test_write_errors(tmp_path, system, options, error, message):
    with pytest.raises(error, match=message):
        mdapy.write(tmp_path / "e.dump", system, **options)
    assert not (tmp_path / "e.dump").exists()


def test_format_detection():
    for name in ("dump.lammpstrj", "run.dump", "dump.100", "dump.atom.gz"):
        assert mdapy.detect_format(name) == "lammps-dump"


def test_frames_without_coordinates_or_timestep(tmp_path):
    no_timestep = orthogonal("id type x y z", "1 1 0 0 0\n2 1 1 1 1").replace(
        "ITEM: TIMESTEP\n100\n", ""
    )
    assert read_text(tmp_path, no_timestep).info == {}
    with pytest.raises(FileFormatError, match="positions"):
        read_text(tmp_path, orthogonal("id type", "1 1\n2 1"))


def test_restricted_option_for_left_handed_boxes_without_images(tmp_path):
    box = mdapy.Box([[4.0, 0, 0], [0, 0.5, 5.0], [0.3, 6.0, 0]])
    system = mdapy.System([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], box, elements="Cu")
    back = mdapy.read(round_trip(tmp_path, system, triclinic="restricted"))
    np.testing.assert_allclose(back.box.volume, box.volume, rtol=1e-13)
    distance = np.linalg.norm(back["positions"][1] - back["positions"][0])
    np.testing.assert_allclose(
        distance,
        np.linalg.norm(system["positions"][1] - system["positions"][0]),
        rtol=1e-13,
    )
