# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
POSCAR (VASP) IO tests.

The format spec we follow:
  https://www.vasp.at/wiki/index.php/POSCAR

Lines (1-indexed):
  1.  Free-form comment
  2.  Universal scale factor (multiplies both the lattice vectors AND the
      Cartesian atom positions)
  3-5. Lattice vectors (one row per a/b/c)
  6.  Optional element-symbol line (e.g. "Al Cu") — present iff the first
      character of line 6 is non-numeric
  6/7. Per-element atom counts
  +.  Optional "Selective dynamics" header (S/s)
  +.  Coordinate type ("Cartesian"/"Direct"/"K-point") — may have leading
      whitespace from real-world editors
  +.  N atom rows (3 floats each, plus 3 T/F flags if selective dynamics)
  +.  Optional "Lattice velocities …" block, then ion-velocity block
"""

from pathlib import Path

import gzip
import shutil

import numpy as np
import polars as pl
import pytest

import mdapy as mp
from mdapy.box import Box

POSCAR_DIR = Path(__file__).parent / "input_files" / "poscar"


# ===========================================================================
# Read
# ===========================================================================

def test_poscar_cartesian_with_species():
    s = mp.System(str(POSCAR_DIR / "cartesian_with_species.POSCAR"), format="poscar")
    assert s.N == 2
    assert s.data["element"].to_list() == ["Al", "Cu"]
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
    )
    np.testing.assert_allclose(np.diag(s.box.box), [4.0, 4.0, 4.0])


def test_poscar_direct_no_species():
    """Fractional coords (Direct), no species names → atoms get a `type` column
    indexed 1..n_types in the order they appear on the count line."""
    s = mp.System(str(POSCAR_DIR / "direct_no_species.POSCAR"), format="poscar")
    assert s.N == 3
    assert "type" in s.data.columns
    assert s.data["type"].to_list() == [1, 1, 2]
    # Direct (0, 0, 0) → (0, 0, 0); (0.5, 0, 0) → (1.5, 0, 0).
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]],
    )


def test_poscar_selective_dynamics():
    s = mp.System(str(POSCAR_DIR / "selective_dynamics.POSCAR"), format="poscar")
    assert s.N == 2
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )
    # Selective-dynamics flags should be preserved.
    assert "sdx" in s.data.columns
    assert s.data["sdx"].to_list() == ["T", "F"]
    assert s.data["sdy"].to_list() == ["T", "F"]
    assert s.data["sdz"].to_list() == ["T", "F"]


def test_poscar_leading_whitespace_on_cartesian_line():
    """Real POSCAR files often have leading whitespace before 'Cartesian'.
    The reader must strip before checking the type letter, otherwise it
    silently interprets coords as fractional and shifts every atom."""
    s = mp.System(str(POSCAR_DIR / "leading_whitespace.POSCAR"), format="poscar")
    assert s.N == 2
    # If the bug isn't fixed, these come out as (0,0,0) and (8,0,0)
    # because (2.0, 0, 0) gets multiplied by box=4*I.
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )


def test_poscar_scale_factor_applied():
    """Scale = 2 means lattice is 2*input AND Cartesian positions are 2*input."""
    s = mp.System(str(POSCAR_DIR / "scale_factor.POSCAR"), format="poscar")
    assert s.N == 1
    np.testing.assert_allclose(np.diag(s.box.box), [4.0, 4.0, 4.0])
    np.testing.assert_allclose(
        s.data.select("x", "y", "z").to_numpy(),
        [[2.0, 0.0, 0.0]],
    )


# ===========================================================================
# Compressed
# ===========================================================================

def test_poscar_gz(tmp_path):
    src = POSCAR_DIR / "cartesian_with_species.POSCAR"
    gz = tmp_path / "cartesian_with_species.poscar.gz"
    with open(src, "rb") as fin, gzip.open(gz, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    s = mp.System(str(gz))
    assert s.N == 2
    assert s.data["element"].to_list() == ["Al", "Cu"]


# ===========================================================================
# Roundtrip
# ===========================================================================

def _make_system_for_poscar():
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    s = mp.System(pos=pos, box=Box(np.eye(3) * 4.0, boundary=[1, 1, 1]))
    s.update_data(s.data.with_columns(element=pl.Series(["Al", "Cu", "Al"])))
    return s


def test_poscar_roundtrip_cartesian(tmp_path):
    s = _make_system_for_poscar()
    out = tmp_path / "rt.POSCAR"
    s.write_poscar(str(out))
    s2 = mp.System(str(out), format="poscar")
    assert sorted(s.data["element"].to_list()) == sorted(s2.data["element"].to_list())
    # write_poscar sorts by element, so the row order may shift; compare
    # by the sorted set of (element, x, y, z).
    a = sorted(zip(s.data["element"].to_list(),
                   *(s.data[c].to_list() for c in ("x", "y", "z"))))
    b = sorted(zip(s2.data["element"].to_list(),
                   *(s2.data[c].to_list() for c in ("x", "y", "z"))))
    assert a == b


def test_poscar_roundtrip_direct(tmp_path):
    s = _make_system_for_poscar()
    out = tmp_path / "rt_direct.POSCAR"
    s.write_poscar(str(out), reduced_pos=True)
    s2 = mp.System(str(out), format="poscar")
    a = sorted(zip(s.data["element"].to_list(),
                   *(s.data[c].to_list() for c in ("x", "y", "z"))))
    b = sorted(zip(s2.data["element"].to_list(),
                   *(s2.data[c].to_list() for c in ("x", "y", "z"))))
    assert a == b
