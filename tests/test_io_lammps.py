# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
LAMMPS data + dump (single-frame) IO tests.

Multi-frame dump files are out of scope for `mdapy.System` — they belong to
`XYZTrajectory` (will be covered separately). The single-frame loader must
*reject* a multi-frame dump rather than silently parse only the first frame.

Sample files live under tests/input_files/lammps/ and are tiny by design;
they each exercise one specific syntax variant.
"""

from pathlib import Path

import gzip
import shutil

import numpy as np
import pytest

import mdapy as mp
from mdapy.box import Box

LAMMPS_DIR = Path(__file__).parent / "input_files" / "lammps"


# ===========================================================================
# read_data — atomic + charge styles, image flags, section ordering
# ===========================================================================

def test_data_atomic_basic():
    s = mp.System(str(LAMMPS_DIR / "atomic_basic.data"))
    assert s.N == 4
    assert set(s.data.columns) >= {"id", "type", "x", "y", "z"}
    assert s.data["type"].to_list() == [1, 2, 1, 2]
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]],
    )
    np.testing.assert_allclose(np.diag(s.box.box), [4, 4, 4])


def test_data_masses_with_elements_populates_element_col():
    """When the Masses block carries `# Element` comments for every type,
    the reader auto-populates an `element` column (round-trips with
    write_data(element_list=...))."""
    s = mp.System(str(LAMMPS_DIR / "atomic_basic.data"))
    assert "element" in s.data.columns
    assert s.data["element"].to_list() == ["Al", "Cu", "Al", "Cu"]


def test_data_no_masses_no_element():
    """When Masses comments are absent, no `element` column is invented."""
    s = mp.System(str(LAMMPS_DIR / "atomic_no_atomstyle_comment.data"))
    assert "element" not in s.data.columns


def test_data_atomic_multispace_data_rows():
    """Atom rows are column-aligned with multiple spaces between fields —
    must trigger the slow-path parser in read_data."""
    s = mp.System(str(LAMMPS_DIR / "atomic_multispace_data.data"))
    assert s.N == 4
    assert s.data["type"].to_list() == [1, 1, 1, 1]
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]],
    )


def test_data_atomic_no_atomstyle_comment():
    """`Atoms` with no '# atomic' suffix — atom_style inferred from columns."""
    s = mp.System(str(LAMMPS_DIR / "atomic_no_atomstyle_comment.data"))
    assert s.N == 4
    assert s.data["type"].to_list() == [1, 2, 1, 2]


def test_data_atomic_with_image_flags():
    """atomic + image flags (8 columns: id type x y z ix iy iz).
    Image flags are stored as ix/iy/iz columns; the wrapped coords stay as x/y/z."""
    s = mp.System(str(LAMMPS_DIR / "atomic_with_image_flags.data"))
    assert s.N == 4
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        [[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]],
    )
    # Image flags should be preserved as int columns (well-known convention).
    for c in ("ix", "iy", "iz"):
        assert c in s.data.columns, f"image flag column {c} missing"
    assert s.data["ix"].to_list() == [0, 1, 0, 0]
    assert s.data["iy"].to_list() == [0, 0, -1, 0]
    assert s.data["iz"].to_list() == [0, 0, 0, 2]


def test_data_charge_with_velocities():
    s = mp.System(str(LAMMPS_DIR / "charge_with_velocities.data"))
    assert s.N == 3
    assert "q" in s.data.columns
    np.testing.assert_allclose(s.data["q"].to_numpy(), [1.0, -1.0, 1.0])
    np.testing.assert_allclose(s.data["vx"].to_numpy(), [0.1, -0.1, 0.0])
    np.testing.assert_allclose(s.data["vy"].to_numpy(), [0.2, -0.2, 0.0])
    np.testing.assert_allclose(s.data["vz"].to_numpy(), [0.3, -0.3, 0.0])
    # Box origin is preserved (xlo = -1 here).
    np.testing.assert_allclose(s.box.origin, [-1, -1, -1])


def test_data_atomic_triclinic():
    s = mp.System(str(LAMMPS_DIR / "atomic_triclinic.data"))
    assert s.N == 2
    np.testing.assert_allclose(
        s.box.box,
        [[4, 0, 0], [1.0, 4, 0], [0.5, 0.3, 4]],
    )


def test_data_masses_after_atoms():
    """`Masses` between `Atoms` and `Velocities` — section order shouldn't matter."""
    s = mp.System(str(LAMMPS_DIR / "atomic_masses_between.data"))
    assert s.N == 2
    np.testing.assert_allclose(s.data["vx"].to_numpy(), [1.0, 4.0])
    np.testing.assert_allclose(s.data["vy"].to_numpy(), [2.0, 5.0])
    np.testing.assert_allclose(s.data["vz"].to_numpy(), [3.0, 6.0])


def test_data_unsupported_atomstyle_errors():
    """atom_style=full must raise a clear error, not silently produce garbage."""
    with pytest.raises((ValueError, NotImplementedError)) as exc:
        mp.System(str(LAMMPS_DIR / "unsupported_full.data"))
    msg = str(exc.value).lower()
    assert "full" in msg or "support" in msg, (
        f"error message should mention the unsupported style; got: {exc.value}"
    )


# ===========================================================================
# read_dump — coord variants, header variants, multi-frame rejection
# ===========================================================================

def test_dump_basic():
    s = mp.System(str(LAMMPS_DIR / "dump_basic.dump"))
    assert s.N == 4
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5], [1.5, 1.5, 0.5]],
    )
    np.testing.assert_allclose(np.diag(s.box.box), [4, 4, 4])


def test_dump_triclinic():
    s = mp.System(str(LAMMPS_DIR / "dump_triclinic.dump"))
    assert s.N == 2
    # The dump file's "BOX BOUNDS xy xz yz" form encodes the bounding box;
    # decoding gives box vectors a=(4,0,0), b=(1,4,0), c=(0.5,0.3,4).
    np.testing.assert_allclose(
        s.box.box,
        [[4.0, 0, 0], [1.0, 4.0, 0], [0.5, 0.3, 4.0]],
        atol=1e-9,
    )


def test_dump_scaled():
    """xs/ys/zs must be converted to absolute Cartesian using box origin."""
    s = mp.System(str(LAMMPS_DIR / "dump_scaled.dump"))
    assert s.N == 4
    expected = np.array([
        [1.0 + 0.00 * 4, 2.0 + 0.00 * 4, 3.0 + 0.00 * 4],
        [1.0 + 0.25 * 4, 2.0 + 0.50 * 4, 3.0 + 0.50 * 4],
        [1.0 + 0.50 * 4, 2.0 + 0.25 * 4, 3.0 + 0.75 * 4],
        [1.0 + 0.75 * 4, 2.0 + 0.75 * 4, 3.0 + 0.25 * 4],
    ])
    np.testing.assert_allclose(s.data.select("x", "y", "z").to_numpy(), expected)
    # Scaled columns should be removed (we converted to x/y/z).
    for c in ("xs", "ys", "zs"):
        assert c not in s.data.columns


def test_dump_unwrapped_kept_as_xyz():
    """xu/yu/zu are unwrapped Cartesian. mdapy's contract: store as x/y/z
    (the user can always wrap later if they want)."""
    s = mp.System(str(LAMMPS_DIR / "dump_unwrapped.dump"))
    assert s.N == 3
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        [[0.5, 0.5, 0.5], [6.0, 0.5, 0.5], [-1.5, 0.5, 0.5]],
    )
    # The unwrapped tag columns should be gone.
    for c in ("xu", "yu", "zu"):
        assert c not in s.data.columns


def test_dump_xyz_takes_priority_over_xs_xu():
    """When a dump has explicit x/y/z together with xs/ys/zs and/or xu/yu/zu,
    the explicit Cartesian columns are kept and the scaled / unwrapped
    variants stay under their original names. Previously, the loader
    unconditionally tried to overwrite x/y/z from xs, raising a duplicate
    column error or losing the user's data."""
    s = mp.System(str(LAMMPS_DIR / "dump_xyz_with_xs_xu.dump"))
    assert s.N == 3
    # Cartesian columns kept verbatim
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        [[0.5, 0.5, 0.5], [2.0, 0.5, 0.5], [0.5, 2.0, 0.5]],
    )
    # the alternate forms remain accessible under their original names
    for c in ("xs", "ys", "zs", "xu", "yu", "zu"):
        assert c in s.data.columns


def test_trajectory_dump_xyz_with_xs_xu(tmp_path):
    """Same priority rule must hold for the multi-frame Trajectory reader."""
    src = LAMMPS_DIR / "dump_xyz_with_xs_xu.dump"
    multi = tmp_path / "multi.dump"
    # concatenate two copies → 2-frame dump
    multi.write_text(src.read_text() + src.read_text())
    traj = mp.Trajectory(str(multi))
    assert len(traj) == 2
    s0 = traj[0]
    np.testing.assert_allclose(
        s0.data.select("x", "y", "z").to_numpy(),
        [[0.5, 0.5, 0.5], [2.0, 0.5, 0.5], [0.5, 2.0, 0.5]],
    )
    for c in ("xs", "ys", "zs", "xu", "yu", "zu"):
        assert c in s0.data.columns


def test_dump_image_flags():
    """When ix/iy/iz are present they are kept verbatim as integer columns
    alongside the wrapped x/y/z coordinates."""
    s = mp.System(str(LAMMPS_DIR / "dump_image_flags.dump"))
    assert s.N == 3
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        [[0.5, 0.5, 0.5], [2.0, 0.5, 0.5], [0.5, 2.0, 0.5]],
    )
    for c in ("ix", "iy", "iz"):
        assert c in s.data.columns
    assert s.data["ix"].to_list() == [0, 1, -1]


def test_dump_with_element():
    """`element` column from `dump_modify ... element X Y Z` — kept as string column."""
    s = mp.System(str(LAMMPS_DIR / "dump_with_element.dump"))
    assert s.N == 4
    assert "element" in s.data.columns
    assert s.data["element"].to_list() == ["Al", "Cu", "Al", "Cu"]
    # Velocities should still come through.
    np.testing.assert_allclose(s.data["vx"].to_numpy(), [0.1, 0.0, 0.0, 0.0])


def test_dump_mixed_pbc_and_forces():
    s = mp.System(str(LAMMPS_DIR / "dump_mixed_pbc.dump"))
    assert s.N == 2
    assert s.box.boundary.tolist() == [1, 1, 0]
    assert "fx" in s.data.columns
    np.testing.assert_allclose(s.data["fx"].to_numpy(), [0.01, -0.01])


def test_dump_abc_origin_general_triclinic():
    """Newer LAMMPS general-triclinic header form: 'BOX BOUNDS abc origin'.
    Each row is (a/b/c vector | origin component)."""
    s = mp.System(str(LAMMPS_DIR / "dump_abc_origin.dump"))
    assert s.N == 2
    np.testing.assert_allclose(
        s.box.box,
        [[4.0, 0, 0], [1.0, 4.0, 0], [0.5, 0.3, 4.0]],
        atol=1e-9,
    )
    np.testing.assert_allclose(s.box.origin, [-1.0, -2.0, -3.0], atol=1e-9)


def test_dump_multiframe_rejected():
    """Single-frame loader must refuse a multi-frame dump file with a clear
    message pointing the user at XYZTrajectory."""
    with pytest.raises((ValueError, RuntimeError)) as exc:
        mp.System(str(LAMMPS_DIR / "dump_multiframe.dump"))
    msg = str(exc.value).lower()
    assert "frame" in msg or "trajectory" in msg or "multi" in msg, (
        f"error should mention multi-frame; got: {exc.value}"
    )


# ===========================================================================
# Compressed file support (.gz)
# ===========================================================================

def test_dump_gz(tmp_path):
    """`.dump.gz` should be auto-detected and decompressed."""
    src = LAMMPS_DIR / "dump_basic.dump"
    gz = tmp_path / "dump_basic.dump.gz"
    with open(src, "rb") as fin, gzip.open(gz, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    s = mp.System(str(gz))
    assert s.N == 4


def test_data_gz(tmp_path):
    src = LAMMPS_DIR / "atomic_basic.data"
    gz = tmp_path / "atomic_basic.data.gz"
    with open(src, "rb") as fin, gzip.open(gz, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    s = mp.System(str(gz))
    assert s.N == 4


# ===========================================================================
# write → read roundtrip
# ===========================================================================

def _make_system_atomic(triclinic=False, with_velocities=False):
    pos = np.array([
        [0.5, 0.5, 0.5],
        [1.5, 1.5, 1.5],
        [2.5, 2.5, 2.5],
    ])
    if triclinic:
        box_mat = np.array([[4.0, 0, 0], [1.0, 4.0, 0], [0.5, 0.3, 4.0]])
    else:
        box_mat = np.eye(3) * 4.0
    s = mp.System(pos=pos, box=Box(box_mat, boundary=[1, 1, 1]))
    import polars as pl
    s.update_data(s.data.with_columns(type=pl.Series([1, 2, 1], dtype=pl.Int32)))
    if with_velocities:
        s.update_data(s.data.with_columns(
            vx=pl.Series([0.1, 0.2, 0.3]),
            vy=pl.Series([0.4, 0.5, 0.6]),
            vz=pl.Series([0.7, 0.8, 0.9]),
        ))
    return s


@pytest.mark.parametrize("triclinic", [False, True])
@pytest.mark.parametrize("with_velocities", [False, True])
def test_data_write_read_roundtrip(tmp_path, triclinic, with_velocities):
    s = _make_system_atomic(triclinic=triclinic, with_velocities=with_velocities)
    out = tmp_path / "rt.data"
    s.write_data(str(out))
    s2 = mp.System(str(out))

    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        s2.data.select("x", "y", "z").to_numpy(),
        atol=1e-9,
    )
    assert s.data["type"].to_list() == s2.data["type"].to_list()
    np.testing.assert_allclose(s.box.box, s2.box.box, atol=1e-9)

    if with_velocities:
        np.testing.assert_allclose(
            s.data.select("vx", "vy", "vz").to_numpy(),
            s2.data.select("vx", "vy", "vz").to_numpy(),
            atol=1e-9,
        )


def test_data_write_num_type_validation(tmp_path):
    """SaveSystem.write_data num_type/element_list interaction:
      * raise when num_type < max(type)
      * accept (with truncation) when len(element_list) > num_type
      * raise when len(element_list) < num_type
    """
    from mdapy.load_save import SaveSystem
    s = _make_system_atomic()
    out = tmp_path / "x.data"

    # num_type smaller than data → ValueError
    with pytest.raises(ValueError):
        SaveSystem.write_data(str(out), s.box, s.data, num_type=1)

    # element_list longer than num_type → file written; only first
    # num_type elements end up in the Masses block. The writer emits a
    # UserWarning naming the dropped trailing entries, which the test
    # explicitly captures so it does not surface as an unhandled
    # warning in pytest's summary.
    with pytest.warns(UserWarning, match="element_list has 3 entries"):
        SaveSystem.write_data(str(out), s.box, s.data,
                              num_type=2, element_list=["Al", "Cu", "Ni"])
    s2 = mp.System(str(out))
    assert set(s2.data["element"].to_list()) == {"Al", "Cu"}

    # element_list shorter than num_type → ValueError
    with pytest.raises(ValueError):
        SaveSystem.write_data(str(out), s.box, s.data,
                              num_type=3, element_list=["Al"])


def test_dump_write_with_element_column(tmp_path):
    """When the data has an `element` string column, write_dump emits it
    alongside the numeric columns (mirrors LAMMPS' `dump_modify element`
    output)."""
    import polars as pl
    s = _make_system_atomic()
    s.update_data(s.data.with_columns(element=pl.Series(["Al", "Cu", "Al"])))
    out = tmp_path / "with_elem.dump"
    s.write_dump(str(out))
    s2 = mp.System(str(out))
    assert "element" in s2.data.columns
    assert s2.data["element"].to_list() == ["Al", "Cu", "Al"]


@pytest.mark.parametrize("triclinic", [False, True])
def test_dump_write_read_roundtrip(tmp_path, triclinic):
    s = _make_system_atomic(triclinic=triclinic)
    out = tmp_path / "rt.dump"
    s.write_dump(str(out), timestep=42)
    s2 = mp.System(str(out))

    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        s2.data.select("x", "y", "z").to_numpy(),
        atol=1e-9,
    )
    np.testing.assert_allclose(s.box.box, s2.box.box, atol=1e-9)
    assert s.data["type"].to_list() == s2.data["type"].to_list()
