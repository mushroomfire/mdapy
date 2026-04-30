# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
Single-frame XYZ IO tests.

Multi-frame XYZ trajectories belong to ``XYZTrajectory`` (covered separately).
This file exercises ``System.from_file`` / ``write_xyz`` only.
"""

from pathlib import Path

import gzip
import shutil

import numpy as np
import polars as pl
import pytest

import mdapy as mp
from mdapy.box import Box

XYZ_DIR = Path(__file__).parent / "input_files" / "xyz"


# ===========================================================================
# Reading
# ===========================================================================

def test_classical_xyz():
    """Legacy XYZ: <N>, comment, then `<element> x y z` rows. Box is
    inferred from the bounding box of the coords; boundary defaults to
    fully open."""
    s = mp.System(str(XYZ_DIR / "classical.xyz"))
    assert s.N == 3
    assert s.data["element"].to_list() == ["Al", "Cu", "Ni"]
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]],
    )
    assert s.box.boundary.tolist() == [0, 0, 0]


def test_extended_xyz():
    """Extended XYZ with all the canonical aliases:
    species:S:1, pos:R:3, vel:R:3, forces:R:3."""
    s = mp.System(str(XYZ_DIR / "extended.xyz"))
    assert s.N == 2
    assert s.data["element"].to_list() == ["Al", "Cu"]
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]],
    )
    np.testing.assert_allclose(
        s.data.select("vx", "vy", "vz").to_numpy(),
        [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        s.data.select("fx", "fy", "fz").to_numpy(),
        [[-0.01, 0, 0], [0.01, 0, 0]],
    )
    np.testing.assert_allclose(np.diag(s.box.box), [4.0, 4.0, 4.0])
    np.testing.assert_allclose(s.box.origin, [-1.0, -2.0, -3.0])
    assert s.box.boundary.tolist() == [1, 1, 0]


def test_extended_xyz_multispace():
    """Multi-space (column-aligned) XYZ — must trigger the slow path
    and still parse correctly."""
    s = mp.System(str(XYZ_DIR / "extended_multispace.xyz"))
    assert s.N == 2
    assert s.data["element"].to_list() == ["Al", "Cu"]
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]],
    )


def test_extended_xyz_crlf():
    """Windows-style CRLF line endings — slow path must strip them
    before splitting tokens."""
    s = mp.System(str(XYZ_DIR / "extended_crlf.xyz"))
    assert s.N == 2
    assert s.data["element"].to_list() == ["Al", "Cu"]


def test_extended_xyz_extra_metadata_kept():
    """Free-form key=value tokens on the comment line beyond the well-known
    `lattice/properties/pbc/origin` ones should be preserved in
    System.global_info."""
    # Construct a test file inline so we don't need another fixture.
    text = (
        "1\n"
        'Lattice="2.0 0 0 0 2.0 0 0 0 2.0" Properties=species:S:1:pos:R:3 '
        'energy=-3.5 timestep=42\n'
        "Al 0.0 0.0 0.0\n"
    )
    p = Path("/tmp/_mdapy_xyz_meta.xyz")
    p.write_text(text)
    try:
        s = mp.System(str(p))
        assert s.global_info.get("energy") == "-3.5"
        assert s.global_info.get("timestep") == "42"
    finally:
        p.unlink(missing_ok=True)


# ===========================================================================
# Compressed (.gz)
# ===========================================================================

def test_xyz_gz(tmp_path):
    src = XYZ_DIR / "extended.xyz"
    gz = tmp_path / "extended.xyz.gz"
    with open(src, "rb") as fin, gzip.open(gz, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    s = mp.System(str(gz))
    assert s.N == 2
    assert s.data["element"].to_list() == ["Al", "Cu"]


# ===========================================================================
# Roundtrip (write → read)
# ===========================================================================

def _make_extended_system():
    pos = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [2.5, 2.5, 2.5]])
    s = mp.System(pos=pos, box=Box(np.eye(3) * 4.0, boundary=[1, 1, 0]))
    s.update_data(s.data.with_columns(
        element=pl.Series(["Al", "Cu", "Ni"]),
        vx=pl.Series([0.1, 0.2, 0.3]),
        vy=pl.Series([0.0, 0.0, 0.0]),
        vz=pl.Series([0.0, 0.0, 0.0]),
    ))
    return s


def test_xyz_extended_roundtrip(tmp_path):
    s = _make_extended_system()
    out = tmp_path / "rt.xyz"
    s.write_xyz(str(out))
    s2 = mp.System(str(out))
    assert s.data["element"].to_list() == s2.data["element"].to_list()
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        s2.data.select("x", "y", "z").to_numpy(),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        s.data.select("vx", "vy", "vz").to_numpy(),
        s2.data.select("vx", "vy", "vz").to_numpy(),
        atol=1e-9,
    )
    np.testing.assert_allclose(s.box.box, s2.box.box, atol=1e-9)
    assert s.box.boundary.tolist() == s2.box.boundary.tolist()


def test_xyz_extended_handles_noncanonical_column_order(tmp_path):
    """The Properties string must remain valid extended-XYZ even when
    x/y/z are NOT consecutive in the column order. The old replace-based
    code silently produced an invalid `:x:R:1:y:R:1:z:R:1:` string in
    that case; the new token-by-token assembly emits proper aliases by
    walking only contiguous runs."""
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    s = mp.System(pos=pos, box=Box(np.eye(3) * 4.0, boundary=[1, 1, 1]))
    # Inject a custom column between x and y so the canonical (x,y,z)
    # alias triplet is broken up.
    s.update_data(
        s.data
        .with_columns(element=pl.Series(["Al", "Cu"]))
        .select("element", "x", pl.lit(7.0).alias("custom"), "y", "z")
    )
    out = tmp_path / "noncanonical.xyz"
    s.write_xyz(str(out))

    # The properties line must be a self-consistent extended XYZ string —
    # i.e. each token is `<name>:<T>:<n>` with name/T/n all parseable.
    with open(out) as f:
        f.readline()                                      # natom
        comment = f.readline()
    import re as _re
    props = _re.search(r'Properties=([^\s]+)', comment).group(1)
    parts = props.split(":")
    assert len(parts) % 3 == 0, (
        f"properties string '{props}' is not a clean :name:T:N: tuple list"
    )

    # And it must read back equivalently.
    s2 = mp.System(str(out))
    assert s.data["element"].to_list() == s2.data["element"].to_list()
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        s2.data.select("x", "y", "z").to_numpy(),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        s.data["custom"].to_numpy(),
        s2.data["custom"].to_numpy(),
        atol=1e-9,
    )


def test_xyz_classical_roundtrip(tmp_path):
    s = _make_extended_system()
    out = tmp_path / "rt_classical.xyz"
    s.write_xyz(str(out), classical=True)
    s2 = mp.System(str(out))
    assert s.data["element"].to_list() == s2.data["element"].to_list()
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        s2.data.select("x", "y", "z").to_numpy(),
        atol=1e-9,
    )
