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
