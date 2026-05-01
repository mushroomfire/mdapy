# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
Multi-frame Trajectory tests — covers both XYZ and LAMMPS-dump formats,
read + write, with the unified `mdapy.Trajectory` class.
"""

from pathlib import Path

import gzip
import shutil

import numpy as np
import polars as pl
import pytest

import mdapy as mp
from mdapy.box import Box

LAMMPS_DIR = Path(__file__).parent / "input_files" / "lammps"
XYZ_DIR = Path(__file__).parent / "input_files" / "xyz"


# ===========================================================================
# Shared frame factory
# ===========================================================================

def _make_frames(n_frames=3):
    """Build a small list of System frames with displaced positions and a
    monotonic timestep — covers the typical MD trajectory shape."""
    frames = []
    for k in range(n_frames):
        pos = np.array([
            [0.5 + 0.1 * k, 0.5,        0.5],
            [1.5,            1.5 + 0.1 * k, 1.5],
            [2.5,            2.5,        2.5 + 0.1 * k],
        ])
        s = mp.System(pos=pos, box=Box(np.eye(3) * 4.0, boundary=[1, 1, 1]))
        s.update_data(s.data.with_columns(
            type=pl.Series([1, 2, 1], dtype=pl.Int32),
        ))
        s.global_info["timestep"] = 100 * k
        frames.append(s)
    return frames


# ===========================================================================
# Reading
# ===========================================================================

def test_trajectory_format_inference():
    """`.xyz` / `.dump` / `.lammpstrj` (with optional `.gz`) are inferred."""
    from mdapy.trajectory import _infer_trajectory_format
    assert _infer_trajectory_format("foo.xyz") == "xyz"
    assert _infer_trajectory_format("foo.xyz.gz") == "xyz"
    assert _infer_trajectory_format("foo.dump") == "dump"
    assert _infer_trajectory_format("foo.dump.gz") == "dump"
    assert _infer_trajectory_format("foo.lammpstrj") == "dump"
    with pytest.raises(ValueError):
        _infer_trajectory_format("foo.unknown")


def test_trajectory_read_multiframe_dump():
    """The committed dump_multiframe fixture has 2 frames; both must
    parse correctly with the right timestep and atom count."""
    traj = mp.Trajectory(str(LAMMPS_DIR / "dump_multiframe.dump"))
    assert len(traj) == 2
    assert traj[0].global_info.get("timestep") == 0
    assert traj[1].global_info.get("timestep") == 1
    assert traj[0].N == 2 and traj[1].N == 2
    # Frame-1 has displaced atoms relative to frame-0.
    np.testing.assert_allclose(traj[0].data["x"].to_numpy(), [0.0, 2.0])
    np.testing.assert_allclose(traj[1].data["x"].to_numpy(), [0.1, 2.1])


def test_trajectory_read_singleframe_dump():
    """Reading a single-frame dump via Trajectory should yield exactly
    one frame (mirror of the single-frame load_save behaviour)."""
    traj = mp.Trajectory(str(LAMMPS_DIR / "dump_basic.dump"))
    assert len(traj) == 1
    assert traj[0].N == 4


# ===========================================================================
# Dump fast mode
# ===========================================================================

def test_dump_rejects_fast_mode_with_clear_message():
    """`fast_mode=True` is intentionally not supported for LAMMPS dump
    (the serial reader is already vectorised per-frame via pl.read_csv).
    The error must say so explicitly so users know to drop the flag."""
    path = str(LAMMPS_DIR / "dump_multiframe_5x8.dump")
    with pytest.raises(ValueError, match=r"(?i)not supported.*dump"):
        mp.Trajectory(path, fast_mode=True)
    # Serial path works regardless.
    traj = mp.Trajectory(path, verbose=False)
    assert len(traj) == 5


def test_dump_serial_verbose_emits_progress(capsys):
    """``verbose=True`` (the new default) must print at least one
    progress line during the read."""
    mp.Trajectory(
        str(LAMMPS_DIR / "dump_multiframe_5x8.dump"),
        verbose=True,
    )
    captured = capsys.readouterr().out
    assert "[dump.serial]" in captured


def test_dump_serial_verbose_off_is_silent(capsys):
    """``verbose=False`` must produce no progress output at all."""
    mp.Trajectory(
        str(LAMMPS_DIR / "dump_multiframe_5x8.dump"),
        verbose=False,
    )
    captured = capsys.readouterr().out
    assert "[dump.serial]" not in captured


# ===========================================================================
# XYZ fast mode
# ===========================================================================

def test_xyz_fast_matches_serial(tmp_path):
    """Round-trip through the writer normalises spacing, so the saved
    file is fast-mode-readable. Compare the result to the serial path."""
    frames = _make_frames(4)
    out = tmp_path / "uniform.xyz"
    mp.Trajectory(systems=frames).save(str(out))
    serial = mp.Trajectory(str(out))
    fast = mp.Trajectory(str(out), fast_mode=True)
    assert len(serial) == len(fast) == 4
    for k in range(4):
        np.testing.assert_allclose(
            serial[k].data.select("x", "y", "z").to_numpy(),
            fast[k].data.select("x", "y", "z").to_numpy(),
            atol=1e-9,
        )


# ===========================================================================
# Slim XYZTrajectory: still has the list-API via the shared mixin
# ===========================================================================

def test_xyztrajectory_inherits_list_api():
    frames = _make_frames(3)
    traj = mp.XYZTrajectory(systems=frames)
    assert len(traj) == 3
    assert isinstance(traj[1:], mp.XYZTrajectory)
    traj.append(frames[0])
    assert len(traj) == 4
    popped = traj.pop()
    assert popped is frames[0]


# ===========================================================================
# Fancy indexing — bool mask + integer array
# ===========================================================================

def test_trajectory_int_array_indexing():
    frames = _make_frames(5)
    traj = mp.Trajectory(systems=frames)
    sub = traj[[0, 2, 4]]
    assert isinstance(sub, mp.Trajectory)
    assert len(sub) == 3
    assert sub[0] is frames[0]
    assert sub[1] is frames[2]
    assert sub[2] is frames[4]


def test_trajectory_int_array_negative_indices():
    frames = _make_frames(5)
    traj = mp.Trajectory(systems=frames)
    sub = traj[np.array([-1, -2])]
    assert len(sub) == 2
    assert sub[0] is frames[-1]
    assert sub[1] is frames[-2]


def test_trajectory_int_array_out_of_bounds_raises():
    traj = mp.Trajectory(systems=_make_frames(3))
    with pytest.raises(IndexError, match="out of bounds"):
        _ = traj[[0, 7]]


def test_trajectory_bool_mask_indexing():
    frames = _make_frames(5)
    traj = mp.Trajectory(systems=frames)
    mask = np.array([True, False, True, False, True])
    sub = traj[mask]
    assert isinstance(sub, mp.Trajectory)
    assert len(sub) == 3
    assert sub[0] is frames[0]
    assert sub[1] is frames[2]
    assert sub[2] is frames[4]


def test_trajectory_bool_mask_wrong_length_raises():
    traj = mp.Trajectory(systems=_make_frames(3))
    with pytest.raises(IndexError, match="boolean mask"):
        _ = traj[np.array([True, False])]


def test_trajectory_filter_by_atom_count():
    """Real-world use case: filter frames by per-frame atom count
    using the new bool-array indexing + ``get_atoms_count`` array."""
    # Build trajectories with mixed atom counts.
    frames = []
    for n in (1, 5, 8, 2, 10):
        rng = np.random.default_rng(n)
        s = mp.System(pos=rng.uniform(0, 5, (n, 3)), box=Box(np.eye(3) * 5,
                                                              boundary=[1, 1, 1]))
        frames.append(s)
    traj = mp.Trajectory(systems=frames)
    big = traj[traj.get_atoms_count() >= 5]
    assert len(big) == 3
    assert [s.N for s in big] == [5, 8, 10]
    # Compose with another filter.
    biggest = traj[traj.get_atoms_count() == traj.get_atoms_count().max()]
    assert len(biggest) == 1
    assert biggest[0].N == 10


def test_get_atoms_count_returns_ndarray():
    traj = mp.Trajectory(systems=_make_frames(4))
    counts = traj.get_atoms_count()
    assert isinstance(counts, np.ndarray)
    assert counts.dtype == np.int64


# ===========================================================================
# Mixed XYZ trajectories: classical + extended frames in one file, plus
# tiny frames (single atom, dimer) — all must parse cleanly via the
# per-frame uniform-space-detection fast path.
# ===========================================================================

def test_xyz_mixed_classical_and_extended():
    """One file containing six frames of varying shape:
       classical 1-atom, extended 2-atom, extended 3-atom with forces,
       classical 4-atom dimer pair, extended 2-atom alloy, classical 1-atom.
    All must parse without raising and yield the right N + element list
    per frame."""
    traj = mp.XYZTrajectory(str(XYZ_DIR / "mixed_traj.xyz"))
    assert len(traj) == 6
    assert [s.N for s in traj] == [1, 2, 3, 4, 2, 1]
    assert traj[0].data["element"].to_list() == ["C"]
    assert traj[1].data["element"].to_list() == ["C", "N"]
    assert traj[2].data["element"].to_list() == ["C", "H", "H"]
    # Frame 2 is extended XYZ with force columns — check they came through.
    for col in ("fx", "fy", "fz"):
        assert col in traj[2].data.columns
    np.testing.assert_allclose(
        traj[2].data["fx"].to_numpy(), [0.1, -0.05, -0.05], atol=1e-9,
    )
    # Frame 3 is classical (no Lattice) — boundary should be all-open.
    assert list(traj[3].box.boundary) == [0, 0, 0]
    # Frame 4 is extended again — boundary should be all-PBC.
    assert list(traj[4].box.boundary) == [1, 1, 1]
    # Frame 5: single classical Ne — Box() must still construct (zero
    # extents are padded to 1e-9 inside the parser).
    assert traj[5].N == 1
    assert traj[5].data["element"].to_list() == ["Ne"]


def test_xyz_mixed_multispace_falls_back_per_frame(tmp_path):
    """A frame with multi-space separators must parse via the Python
    fallback while frames with uniform spacing in the same file go
    through the polars CSV fast path. The check is per-frame."""
    traj = mp.XYZTrajectory(str(XYZ_DIR / "mixed_multispace.xyz"))
    assert len(traj) == 2
    np.testing.assert_allclose(
        traj[0].data.select("x", "y", "z").to_numpy(),
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        traj[1].data.select("x", "y", "z").to_numpy(),
        [[0.1, 0.0, 0.0], [1.3, 0.0, 0.0]],
        atol=1e-9,
    )


def test_xyz_single_atom_no_box_does_not_singular():
    """Regression: a 1-atom classical frame has zero box extent on
    every axis. The cell matrix must still be invertible (Box() rejects
    singular matrices), so the parser pads zero extents to 1e-9."""
    s = mp.System(str(XYZ_DIR / "classical.xyz"))
    # Classical fixture is 3 atoms in a plane; just sanity-check the
    # path runs. Then parse a 1-atom Ne frame from mixed_traj.
    traj = mp.XYZTrajectory(str(XYZ_DIR / "mixed_traj.xyz"))
    sing = traj[5]
    # Box should be invertible.
    np.linalg.inv(sing.box.box)


# ===========================================================================
# Writing
# ===========================================================================

@pytest.mark.parametrize("ext", ["dump", "xyz"])
def test_trajectory_roundtrip(tmp_path, ext):
    frames = _make_frames(3)
    traj = mp.Trajectory(systems=frames)
    out = tmp_path / f"out.{ext}"
    traj.save(str(out))

    traj2 = mp.Trajectory(str(out))
    assert len(traj2) == 3

    # Per-frame positions must round-trip.
    for k in range(3):
        np.testing.assert_allclose(
            frames[k].data.select("x", "y", "z").to_numpy(),
            traj2[k].data.select("x", "y", "z").to_numpy(),
            atol=1e-9,
        )

    # Box should round-trip too.
    np.testing.assert_allclose(frames[0].box.box, traj2[0].box.box, atol=1e-9)


def test_trajectory_save_subset(tmp_path):
    frames = _make_frames(5)
    traj = mp.Trajectory(systems=frames)
    out = tmp_path / "subset.dump"
    traj.save(str(out), frames=[0, 2, 4])
    traj2 = mp.Trajectory(str(out))
    assert len(traj2) == 3
    np.testing.assert_allclose(
        frames[2].data.select("x", "y", "z").to_numpy(),
        traj2[1].data.select("x", "y", "z").to_numpy(),
        atol=1e-9,
    )


def test_trajectory_append_mode(tmp_path):
    """mode='a' must append frames to an existing file."""
    frames = _make_frames(3)
    out = tmp_path / "appended.dump"
    mp.Trajectory(systems=frames[:1]).save(str(out), mode="w")
    mp.Trajectory(systems=frames[1:]).save(str(out), mode="a")
    traj = mp.Trajectory(str(out))
    assert len(traj) == 3


def test_trajectory_dump_gz_roundtrip(tmp_path):
    """Compressed multi-frame dump round-trip — write to .dump first,
    then gzip externally and read back via Trajectory."""
    frames = _make_frames(2)
    plain = tmp_path / "tmp.dump"
    mp.Trajectory(systems=frames).save(str(plain))
    gz = tmp_path / "tmp.dump.gz"
    with open(plain, "rb") as fin, gzip.open(gz, "wb") as fout:
        shutil.copyfileobj(fin, fout)
    traj = mp.Trajectory(str(gz))
    assert len(traj) == 2


# ===========================================================================
# List-like API
# ===========================================================================

def test_trajectory_list_api():
    frames = _make_frames(4)
    traj = mp.Trajectory(systems=frames[:2])
    assert len(traj) == 2

    traj.append(frames[2])
    assert len(traj) == 3

    traj.extend([frames[3]])
    assert len(traj) == 4

    sub = traj[1:3]
    assert isinstance(sub, mp.Trajectory)
    assert len(sub) == 2

    popped = traj.pop()
    assert popped is frames[3]
    assert len(traj) == 3

    traj.insert(0, frames[3])
    assert traj[0] is frames[3]


def test_trajectory_explicit_format(tmp_path):
    """Format override lets users save a .txt file as dump syntax."""
    frames = _make_frames(2)
    out = tmp_path / "weird.txt"
    mp.Trajectory(systems=frames).save(str(out), format="dump")
    traj = mp.Trajectory(str(out), format="dump")
    assert len(traj) == 2
