# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import math

import numpy as np
import polars as pl
import pytest

import mdapy
from mdapy import units
from mdapy.elements import ELEMENT_DTYPE
from mdapy.errors import (
    FileFormatError,
    InvalidParameter,
    MissingProperty,
    UnsupportedFormat,
)
from mdapy.io._extxyz import _split_items, _value

ASE_TIME_IN_PS = (
    1e-10 * math.sqrt(units.ATOMIC_MASS_UNIT_SI / units.ELEMENTARY_CHARGE) * 1e12
)


def read_text(tmp_path, text, name="t.xyz", **kwargs):
    path = tmp_path / name
    path.write_text(text)
    return mdapy.read(path, **kwargs)


def round_trip(tmp_path, system, **options):
    path = tmp_path / "rt.xyz"
    mdapy.write(path, system, **options)
    return mdapy.read(path)


def assert_tables_identical(a, b):
    assert a.schema == b.schema
    for name in a.columns:
        left, right = a[name].to_numpy(), b[name].to_numpy()
        if left.dtype.kind == "f":
            np.testing.assert_array_equal(left.view(np.uint64), right.view(np.uint64))
        else:
            np.testing.assert_array_equal(left, right)


def rich_system(n=50, seed=0):
    rng = np.random.default_rng(seed)
    box = mdapy.Box(
        [[10.0, 0.0, 0.0], [1.5, 9.0, 0.0], [0.3, -0.7, 11.0]],
        origin=[-1.0, 2.0, 0.5],
        pbc=[True, True, False],
    )
    system = mdapy.System(
        positions=rng.random((n, 3)) * 10,
        box=box,
        elements=rng.choice(["Cu", "Zr", "Al"], n),
        info={"energy": -3.25 * n, "virial": rng.standard_normal((3, 3))},
    )
    return system.with_columns(
        velocities=rng.standard_normal((n, 3)),
        forces=rng.standard_normal((n, 3)),
        type=rng.integers(1, 4, n),
        id=np.arange(1, n + 1),
        charge=rng.standard_normal(n),
        mass=rng.random(n) * 50 + 1,
        atomic_energies=rng.standard_normal(n),
        atomic_virials=rng.standard_normal((n, 3, 3)),
        selected=rng.random(n) < 0.5,
        fixed=rng.random((n, 3)) < 0.5,
        label=pl.Series(rng.choice(["bulk", "surface", "α-phase"], n)),
        pair=pl.Series(rng.choice(["a", "b"], 2 * n)).reshape((n, 2)),
        image=rng.integers(-2, 3, (n, 3)),
        small=pl.Series(rng.standard_normal(n), dtype=pl.Float32),
    )


# ------------------------------------------------------------ round trips


def test_round_trip_is_exact(tmp_path):
    system = rich_system()
    back = round_trip(tmp_path, system)
    expected = system.table.with_columns(pl.col("small").cast(pl.Float64))
    # element and positions are written first
    expected = expected.select(
        "element", "positions", pl.exclude("element", "positions")
    )
    assert_tables_identical(back.table, expected)
    assert back.box == system.box
    assert back.info["energy"] == system.info["energy"]
    np.testing.assert_array_equal(back.info["virial"], system.info["virial"])


def test_round_trip_of_info_values(tmp_path):
    info = {
        "step": 7,
        "time": 1.25,
        "big": 2**62,
        "converged": True,
        "failed": False,
        "name": "water dimer",
        "quoted": 'say "hi" \\ {x} [y] = z',
        "digits": "123",
        "boolean_text": "T",
        "empty": "",
        "blank": "   ",
        "json_text": "_JSON not json",
        "ints": np.array([1, -2, 3]),
        "floats": np.array([0.1, 2.5e-300, -np.inf]),
        "flags": np.array([True, False, True]),
        "one": np.array([4.5]),
        "matrix": np.arange(6.0).reshape(2, 3),
        "words": ["a", "b c"],
        "nested": {"cutoff": 5.0, "types": ["Cu", "Zr"]},
        "nothing": None,
        "stress": np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]]),
    }
    back = round_trip(
        tmp_path, mdapy.System([[0.0, 0.0, 0.0]], 5.0, elements="H", info=info)
    )
    assert set(back.info) == set(info)
    for key in ("step", "time", "big", "converged", "failed", "name", "quoted"):
        assert back.info[key] == info[key] and type(back.info[key]) is type(info[key])
    for key in ("digits", "boolean_text", "empty", "blank", "json_text"):
        assert back.info[key] == info[key]
    for key in ("ints", "floats", "flags", "one", "matrix"):
        np.testing.assert_array_equal(back.info[key], info[key])
        assert back.info[key].dtype == info[key].dtype
    assert back.info["words"] == ("a", "b c")
    assert back.info["nested"] == {"cutoff": 5.0, "types": ("Cu", "Zr")}
    assert back.info["nothing"] is None
    np.testing.assert_allclose(back.info["stress"], info["stress"], rtol=1e-15)


def test_multiple_frames_may_differ(tmp_path):
    path = tmp_path / "frames.xyz"
    first = rich_system(5, seed=1)
    second = mdapy.System([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 4.0, elements=["H", "H"])
    mdapy.write(path, [first, second, first])
    assert_tables_identical(
        mdapy.read(path, frame=0).table, mdapy.read(path, frame=2).table
    )
    back = mdapy.read(path, frame=1)
    assert back.table.columns == ["element", "positions"] and len(back) == 2


def test_zero_atoms(tmp_path):
    empty = mdapy.System(np.zeros((0, 3)), mdapy.Box(3.0), elements=[])
    back = round_trip(tmp_path, empty)
    assert len(back) == 0 and back.box == empty.box


def test_columns_option_selects_what_is_written(tmp_path):
    system = rich_system(4)
    back = round_trip(tmp_path, system, columns=["forces", "type"])
    assert back.table.columns == ["element", "positions", "forces", "type"]
    back = round_trip(tmp_path, system, columns=["positions", "id", "element"])
    assert back.table.columns == ["element", "positions", "id"]


def test_written_text(tmp_path):
    system = mdapy.System(
        [[0.0, 0.0, 0.0], [0.5, 0.25, 1e-20]],
        mdapy.Box(4.0, pbc=[True, True, False]),
        elements=["Cu", "Zr"],
        info={"energy": -7.5},
    ).with_columns(velocities=[[1000.0, 0.0, -2.0], [0.0, 0.0, 0.0]])
    path = tmp_path / "w.xyz"
    mdapy.write(path, system)
    assert path.read_text() == (
        "2\n"
        'Lattice="4.0 0.0 0.0 0.0 4.0 0.0 0.0 0.0 4.0" '
        "Properties=species:S:1:pos:R:3:vel:R:3 "
        'pbc="T T F" energy=-7.5\n'
        "Cu 0 0 0 1 0 -0.002\n"
        "Zr 0.5 0.25 1e-20 0 0 0\n"
    )


# ------------------------------------------------------------ ASE files


def test_ase_style_file(tmp_path):
    """Momenta in ASE units, stress in eV/Å^3, Lattice rows are vectors."""
    text = (
        "2\n"
        'Lattice="5.44 0.0 0.0 0.0 5.44 0.0 1.0 0.0 5.44" '
        "Properties=species:S:1:pos:R:3:momenta:R:3:forces:R:3:masses:R:1:"
        "initial_charges:R:1 "
        'energy=-10.5 stress="0.01 0.0 0.0 0.0 0.02 0.0 0.0 0.0 -0.03" '
        'free_energy=-10.6 pbc="T T T" Time=0.0\n'
        "Si 0.0 0.0 0.0 2.0 0.0 0.0 0.1 0.2 0.3 28.0 0.5\n"
        "si 1.36 1.36 1.36 0.0 -4.0 0.0 -0.1 -0.2 -0.3 28.0 -0.5\n"
    )
    system = read_text(tmp_path, text)
    assert system.table["element"].to_list() == ["Si", "Si"]
    np.testing.assert_array_equal(system.box.matrix[2], [1.0, 0.0, 5.44])
    np.testing.assert_allclose(
        system.table["velocities"].to_numpy(),
        np.array([[2.0, 0, 0], [0, -4.0, 0]]) / 28.0 / ASE_TIME_IN_PS,
        rtol=1e-12,
    )
    assert 1 / ASE_TIME_IN_PS == pytest.approx(98.22694, rel=1e-6)
    np.testing.assert_allclose(
        system.info["stress"], np.diag([0.01, 0.02, -0.03]) * 160.2176634, rtol=1e-12
    )
    assert system.info["energy"] == -10.5
    assert system.info["free_energy"] == -10.6 and system.info["Time"] == 0.0
    np.testing.assert_array_equal(system.table["mass"], [28.0, 28.0])
    np.testing.assert_array_equal(system.table["charge"], [0.5, -0.5])


def test_momenta_use_element_masses_without_a_mass_column(tmp_path):
    text = "1\nProperties=species:S:1:pos:R:3:momenta:R:3\nH 0 0 0 1.008 0 0\n"
    velocities = read_text(tmp_path, text).table["velocities"].to_numpy()
    np.testing.assert_allclose(velocities[0, 0], 1.0 / ASE_TIME_IN_PS, rtol=1e-12)


def test_default_properties_and_no_lattice(tmp_path):
    system = read_text(tmp_path, "2\nplain xyz comment\nC 0 0 0\nO 1.2 0 0\n")
    assert system.table.columns == ["element", "positions"]
    assert not system.box.pbc.any()
    assert system.info == {"plain": True, "xyz": True, "comment": True}


def test_six_component_tensors_are_voigt_ordered(tmp_path):
    text = '1\nvirial="1 2 3 4 5 6" stress="1 2 3 4 5 6"\nH 0 0 0\n'
    info = read_text(tmp_path, text).info
    expected = np.array([[1.0, 6, 5], [6, 2, 4], [5, 4, 3]])
    np.testing.assert_array_equal(info["virial"], expected)
    np.testing.assert_allclose(info["stress"], expected * units.EV_PER_A3_TO_GPA)


def test_atomic_numbers_give_elements(tmp_path):
    text = "2\nProperties=Z:I:1:pos:R:3\n29 0 0 0\n0 1 1 1\n"
    elements = read_text(tmp_path, text).table["element"]
    assert elements.dtype == ELEMENT_DTYPE and elements.to_list() == ["Cu", "X"]


def test_species_and_numbers_together(tmp_path):
    text = "1\nProperties=species:S:1:pos:R:3:Z:I:1\nCu 0 0 0 29\n"
    table = read_text(tmp_path, text).table
    assert table.columns == ["element", "positions", "Z"]


# ---------------------------------------------------------- GPUMD files


def test_gpumd_training_file(tmp_path):
    """Keys in lower case, force instead of forces, virial in eV."""
    text = (
        "2\n"
        'energy=-8.0 config_type=bulk weight=1.0 virial="1 0 0 0 1 0 0 0 1" '
        'lattice="3 0 0 0 3 0 0 0 3" properties=species:S:1:pos:R:3:force:R:3\n'
        "Pb 0 0 0 0.1 0 0\n"
        "Te 1.5 1.5 1.5 -0.1 0 0\n"
    )
    system = read_text(tmp_path, text, name="train.xyz")
    assert system.box.pbc.all()
    np.testing.assert_array_equal(system.box.matrix, np.eye(3) * 3)
    np.testing.assert_array_equal(system.info["virial"], np.eye(3))
    assert system.info["config_type"] == "bulk" and system.info["weight"] == 1.0
    np.testing.assert_array_equal(system.table["forces"].to_numpy()[:, 0], [0.1, -0.1])


def test_gpumd_dump_file(tmp_path):
    """dump_exyz: velocities in Å/fs, per-atom energies as energy_atom."""
    text = (
        "1\n"
        'Time=10.00000000 pbc="T T T" Lattice="4.00000000 0.00000000 0.00000000 '
        '0.00000000 4.00000000 0.00000000 0.00000000 0.00000000 4.00000000" '
        "energy=-3.50000000 "
        "Properties=species:S:1:pos:R:3:vel:R:3:forces:R:3:energy_atom:R:1\n"
        "C 0.10000000 0.20000000 0.30000000 0.01000000 -0.02000000 0.00000000 "
        "0.0 0.0 1.0 -3.50000000\n"
    )
    system = read_text(tmp_path, text, name="dump.xyz")
    np.testing.assert_allclose(
        system.table["velocities"].to_numpy(), [[10.0, -20.0, 0.0]]
    )
    assert system.table["atomic_energies"].to_list() == [-3.5]
    assert system.info["Time"] == 10.0


# ------------------------------------------------------- comment lines


@pytest.mark.parametrize(
    "line, pairs",
    [
        ("a=1 b=2", [("a", "1"), ("b", "2")]),
        ('a="1 2" b', [("a", "1 2"), ("b", None)]),
        ("a = 1  b= 2 c =3", [("a", "1"), ("b", "2"), ("c", "3")]),
        ("a={1 2} b=[3 4] c='5 6'", [("a", "1 2"), ("b", "3 4"), ("c", "5 6")]),
        (r'a="x \"y\" \\ z"', [("a", 'x "y" \\ z')]),
        (r"a=x\ y", [("a", "x y")]),
        ('a="" b=1', [("a", ""), ("b", "1")]),
        ("a=", [("a", "")]),
        ("a=b=c", [("a", "b=c")]),
        ('"my key"=1', [("my key", "1")]),
        ('a="unterminated b=1', [("a", "unterminated b=1")]),
        ("  ", []),
    ],
)
def test_split_items(line, pairs):
    assert _split_items(line) == pairs


def test_split_items_rejects_a_missing_key():
    with pytest.raises(ValueError, match="'=' without a key"):
        _split_items("=1")


@pytest.mark.parametrize(
    "raw, value",
    [
        (None, True),
        ("7", 7),
        ("-7", -7),
        ("1.5", 1.5),
        ("1e3", 1000.0),
        ("nan", math.nan),
        ("T", True),
        ("false", False),
        ("TRUE", True),
        ("abc", "abc"),
        ("", ""),
        ("1 x", "1 x"),
        ('_JSON {"a": 1}', {"a": 1}),
        ('_JSON ["a", 1]', ["a", 1]),
    ],
)
def test_values(raw, value):
    result = _value(raw)
    if isinstance(value, float) and math.isnan(value):
        assert math.isnan(result)
    else:
        assert result == value and type(result) is type(value)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("1 2 3", np.array([1, 2, 3])),
        ("1,2,3", np.array([1, 2, 3])),
        ("1 2.5", np.array([1.0, 2.5])),
        ("T F", np.array([True, False])),
        ("_JSON [[1, 2], [3, 4]]", np.array([[1, 2], [3, 4]])),
    ],
)
def test_array_values(raw, expected):
    result = _value(raw)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == expected.dtype


def test_box_keys_in_any_case_and_defaults(tmp_path):
    text = '1\nLATTICE="2 0 0 0 2 0 0 0 2" PBC=F Origin="1 1 1"\nH 0 0 0\n'
    box = read_text(tmp_path, text).box
    assert not box.pbc.any() and box.origin.tolist() == [1.0, 1.0, 1.0]
    text = '1\nLattice="2 0 0 0 2 0 0 0 2" pbc\nH 0 0 0\n'
    assert read_text(tmp_path, text).box.pbc.all()


def test_zero_lattice_means_no_box(tmp_path):
    text = '1\nLattice="0 0 0 0 0 0 0 0 0" pbc="F F F"\nH 0 0 0\n'
    box = read_text(tmp_path, text).box
    assert not box.pbc.any() and box.volume > 0


def test_windows_line_endings(tmp_path):
    path = tmp_path / "crlf.xyz"
    path.write_bytes(b'1\r\nLattice="2 0 0 0 2 0 0 0 2" energy=1\r\nH 0 0 0.5\r\n')
    system = mdapy.read(path)
    assert system.info == {"energy": 1.0}
    np.testing.assert_array_equal(system.positions, [[0.0, 0.0, 0.5]])


def test_trailing_blank_lines(tmp_path):
    system = read_text(tmp_path, "1\n\nH 0 0 0\n\n  \n", frame=-1)
    assert len(system) == 1


# ---------------------------------------------------------------- errors


@pytest.mark.parametrize(
    "text, line, message",
    [
        ("x\n", 1, "expected the number of atoms, found 'x'"),
        ("2\n\nH 0 0 0\n", 4, "the frame has 2 atoms, but the file ends after 1"),
        ("2\n", 2, "the frame has 2 atoms, but the file ends after 0"),
        ("1\n\nH 0 0\n", 3, "expected 4 values, found 3"),
        ("1\n\nH 0 x 0\n", 3, "cannot read 'x' as a number"),
        ("1\n\nH 0 0 0\n\n1\n\nH 0 0 0\n", 5, "a frame follows a blank line"),
        ("1\n\nQq 0 0 0\n", 3, "unknown element 'Qq'"),
        (
            "2\nProperties=Z:I:1:pos:R:3\n1 0 0 0\n200 0 0 0\n",
            4,
            "unknown atomic number 200",
        ),
        ("1\na=1 a=2\nH 0 0 0\n", 2, "the key 'a' appears twice"),
        ("1\nenergy=1 Energy=2\nH 0 0 0\n", 2, "the key 'Energy' appears twice"),
        ("1\n=1\nH 0 0 0\n", 2, "'=' without a key"),
        ('1\nLattice="1 2 3"\nH 0 0 0\n', 2, "Lattice must have 9 numbers"),
        ('1\nOrigin="1 2"\nH 0 0 0\n', 2, "Origin must have 3 numbers"),
        ('1\npbc="T F"\nH 0 0 0\n', 2, "pbc must have 1 or 3 booleans"),
        ("1\nenergy=abc\nH 0 0 0\n", 2, "energy must have 1 numbers"),
        ('1\nvirial="1 2"\nH 0 0 0\n', 2, "virial must have 9 or 6 numbers"),
        ('1\nstress="x"\nH 0 0 0\n', 2, "stress must have 9 or 6 numbers"),
        ("1\nProperties\nH 0 0 0\n", 2, "Properties needs a value"),
        ('1\na="_JSON {"\nH 0 0 0\n', 2, "Expecting"),
        ("1\nProperties=species:S:1:pos:R\nH 0 0 0\n", 2, "name:type:width triples"),
        ("1\nProperties=species:S:1:pos:Q:3\nH 0 0 0\n", 2, "invalid property pos:Q:3"),
        ("1\nProperties=species:S:1:pos:R:0\nH 0 0 0\n", 2, "invalid property pos:R:0"),
        ("1\nProperties=species:S:1:pos:R:2\nH 0 0\n", 2, "'pos' must be pos:R:3"),
        ("1\nProperties=species:S:1\nH\n", 2, "Properties has no pos property"),
        (
            "1\nProperties=species:S:1:pos:R:3:force:R:3:forces:R:3\nH 0 0 0 0 0 0 0 0 0\n",
            2,
            "property 'forces' appears twice",
        ),
        (
            "1\nProperties=pos:R:3:momenta:R:3\n0 0 0 1 1 1\n",
            2,
            "neither masses nor elements",
        ),
        (
            "1\nProperties=species:S:1:pos:R:3:momenta:R:3\nX 0 0 0 1 1 1\n",
            2,
            "momenta need masses, but an atom has none",
        ),
        ('1\npbc="T T T"\nH 0 0 0\n', 2, "pbc and Origin need a Lattice"),
        ('1\nLattice="1 0 0 1 0 0 0 0 1"\nH 0 0 0\n', 2, "linearly dependent"),
        ("1\n\nH 0 0 nan\n", 2, "finite"),
    ],
)
def test_errors_name_the_line(tmp_path, text, line, message):
    with pytest.raises(FileFormatError, match=message) as info:
        read_text(tmp_path, text, frame=-1)
    assert info.value.line == line
    assert str(info.value).startswith(f"{tmp_path / 't.xyz'}:{line}: ")


def test_second_velocity_source_is_kept_as_a_column(tmp_path):
    text = (
        "1\nProperties=species:S:1:pos:R:3:vel:R:3:momenta:R:3\nH 0 0 0 1 0 0 5 0 0\n"
    )
    table = read_text(tmp_path, text).table
    assert table.columns == ["element", "positions", "velocities", "momenta"]
    assert table["velocities"].to_list() == [[1000.0, 0.0, 0.0]]


def test_frames_after_a_bad_frame_are_not_reached(tmp_path):
    text = "1\n\nH 0 0 0\n1\n\nH 0 0 x\n"
    assert len(read_text(tmp_path, text, frame=0)) == 1
    with pytest.raises(FileFormatError, match=r":6: cannot read 'x'"):
        read_text(tmp_path, text, frame=1)


# ------------------------------------------------------- writing errors


def h_atom(**info):
    return mdapy.System([[0.0, 0.0, 0.0]], 5.0, elements="H", info=info)


@pytest.mark.parametrize(
    "system, error, message",
    [
        (
            mdapy.System([[0.0, 0.0, 0.0]], 5.0),
            MissingProperty,
            "needs element symbols",
        ),
        (h_atom(Lattice=1), InvalidParameter, "'Lattice' is reserved"),
        (h_atom(PBC=True), InvalidParameter, "'PBC' is reserved"),
        (h_atom(Energy=1.0), InvalidParameter, "would be read back as 'energy'"),
        (h_atom(energy="low"), InvalidParameter, "must be a number"),
        (h_atom(energy=True), InvalidParameter, "must be a number"),
        (h_atom(virial=[1, 2, 3]), InvalidParameter, "must be a 3x3 array"),
        (h_atom(**{"": 1}), InvalidParameter, "cannot be empty"),
        (h_atom(thing=object()), InvalidParameter, "of type object cannot be written"),
        (
            h_atom().with_columns(note=pl.Series(["a b"])),
            InvalidParameter,
            "values with blanks",
        ),
        (h_atom().with_columns(note=pl.Series([""])), InvalidParameter, "empty values"),
        (
            h_atom().with_columns(maybe=pl.Series([None], dtype=pl.Float64)),
            InvalidParameter,
            "has missing values",
        ),
        (
            h_atom().with_columns(items=pl.Series([[1, 2]])),
            UnsupportedFormat,
            "of type List",
        ),
        (
            h_atom().with_columns(**{"a:b": 1.0}),
            InvalidParameter,
            "cannot be an extxyz",
        ),
        (h_atom().with_columns(pos=1.0), InvalidParameter, "both written as 'pos'"),
    ],
)
def test_write_errors(tmp_path, system, error, message):
    with pytest.raises(error, match=message):
        mdapy.write(tmp_path / "e.xyz", system)
    assert not (tmp_path / "e.xyz").exists()


def test_write_rejects_unknown_columns(tmp_path):
    with pytest.raises(InvalidParameter, match="no column named 'nope'"):
        mdapy.write(tmp_path / "e.xyz", h_atom(), columns=["nope"])


def test_numbers_in_info_are_written_exactly(tmp_path):
    value = 0.1 + 0.2
    back = round_trip(tmp_path, h_atom(energy=np.float32(1.5), x=value, n=np.int16(3)))
    assert back.info == {"energy": 1.5, "x": value, "n": 3}
