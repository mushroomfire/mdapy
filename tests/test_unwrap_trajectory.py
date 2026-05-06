# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Tests for ``mdapy.unwrap_trajectory`` covering the three priority paths
(unwrapped columns, image flags, minimum-image scan) plus edge cases.
"""
import numpy as np
import polars as pl
import pytest

import mdapy as mp
from mdapy.box import Box


def _frame(xyz, *, ids=None, types=None, elements=None,
           xu=None, ix=None, box=None, boundary=None):
    """Helper: build a :class:`mdapy.System` from arrays. ``box`` may be a
    :class:`Box` or a 3-element list (cubic). ``boundary`` is forwarded."""
    cols = {
        "x": pl.Series("x", xyz[:, 0], dtype=pl.Float64),
        "y": pl.Series("y", xyz[:, 1], dtype=pl.Float64),
        "z": pl.Series("z", xyz[:, 2], dtype=pl.Float64),
    }
    if ids is not None:
        cols["id"] = pl.Series("id", ids, dtype=pl.Int32)
    if types is not None:
        cols["type"] = pl.Series("type", types, dtype=pl.Int32)
    if elements is not None:
        cols["element"] = pl.Series("element", elements, dtype=pl.Utf8)
    if xu is not None:
        cols["xu"] = pl.Series("xu", xu[:, 0], dtype=pl.Float64)
        cols["yu"] = pl.Series("yu", xu[:, 1], dtype=pl.Float64)
        cols["zu"] = pl.Series("zu", xu[:, 2], dtype=pl.Float64)
    if ix is not None:
        cols["ix"] = pl.Series("ix", ix[:, 0], dtype=pl.Int32)
        cols["iy"] = pl.Series("iy", ix[:, 1], dtype=pl.Int32)
        cols["iz"] = pl.Series("iz", ix[:, 2], dtype=pl.Int32)
    if isinstance(box, Box):
        b = box
    else:
        if boundary is None:
            boundary = [1, 1, 1]
        b = Box(box, boundary=boundary)
    return mp.System(data=pl.DataFrame(cols), box=b)


def test_unwrapped_columns_take_priority():
    """When xu/yu/zu exist, the function just renames them — the wrapped
    x/y/z is ignored. min-image arithmetic is not invoked."""
    box = Box([10.0, 10.0, 10.0])
    f0 = _frame(np.array([[5.0, 0.0, 0.0]]),
                xu=np.array([[5.0, 0.0, 0.0]]), box=box)
    f1 = _frame(np.array([[1.0, 0.0, 0.0]]),    # wrapped after crossing
                xu=np.array([[11.0, 0.0, 0.0]]),  # already unwrapped
                box=box)
    out = mp.unwrap_trajectory(mp.Trajectory(systems=[f0, f1]))
    assert out._unwrap_method == "unwrapped"
    np.testing.assert_array_equal(out[0].data["x"].to_numpy(), [5.0])
    np.testing.assert_array_equal(out[1].data["x"].to_numpy(), [11.0])


def test_image_flags_combine_with_per_frame_box():
    """ix/iy/iz are combined with each frame's *own* cell vectors so NPT
    breathing is handled correctly."""
    f0 = _frame(np.array([[2.0, 0.0, 0.0]]), ix=np.array([[0, 0, 0]]),
                box=[10.0, 10.0, 10.0])
    f1 = _frame(np.array([[3.0, 0.0, 0.0]]), ix=np.array([[1, 0, 0]]),
                box=[10.0, 10.0, 10.0])
    # NPT: cell shrinks by 1 Å along x at frame 2.
    f2 = _frame(np.array([[3.0, 0.0, 0.0]]), ix=np.array([[2, 0, 0]]),
                box=[9.0, 10.0, 10.0])
    out = mp.unwrap_trajectory(mp.Trajectory(systems=[f0, f1, f2]))
    assert out._unwrap_method == "image"
    np.testing.assert_array_equal(out[0].data["x"].to_numpy(), [2.0])
    # frame 1: 3.0 + 1*10.0 = 13.0
    np.testing.assert_array_equal(out[1].data["x"].to_numpy(), [13.0])
    # frame 2: 3.0 + 2*9.0 = 21.0 (uses frame 2's box, not frame 0's)
    np.testing.assert_array_equal(out[2].data["x"].to_numpy(), [21.0])


def test_min_image_scan_unwraps_simple_crossing():
    """No image flags: an atom that wraps once across the +x face should be
    unwrapped to a strictly increasing x sequence."""
    box = Box([10.0, 10.0, 10.0])
    frames = [
        _frame(np.array([[8.0, 5.0, 5.0]]), ids=[1], box=box),
        _frame(np.array([[1.0, 5.0, 5.0]]), ids=[1], box=box),  # wrapped
        _frame(np.array([[3.0, 5.0, 5.0]]), ids=[1], box=box),
    ]
    out = mp.unwrap_trajectory(mp.Trajectory(systems=frames))
    assert out._unwrap_method == "min_image"
    xs = np.concatenate([f.data["x"].to_numpy() for f in out])
    np.testing.assert_array_equal(xs, [8.0, 11.0, 13.0])


def test_min_image_handles_negative_crossing():
    """An atom crossing the -x face should also be unwrapped."""
    box = Box([10.0, 10.0, 10.0])
    frames = [
        _frame(np.array([[1.0, 0.0, 0.0]]), ids=[1], box=box),
        _frame(np.array([[9.0, 0.0, 0.0]]), ids=[1], box=box),  # wrapped to L-1
    ]
    out = mp.unwrap_trajectory(mp.Trajectory(systems=frames))
    xs = np.concatenate([f.data["x"].to_numpy() for f in out])
    np.testing.assert_array_equal(xs, [1.0, -1.0])


def test_min_image_uses_id_for_reordering():
    """Reorder atoms between frames; the id-based path must keep them
    correctly tracked."""
    box = Box([10.0, 10.0, 10.0])
    # Frame 0: id 1 at x=8, id 2 at x=2.
    f0 = _frame(np.array([[8.0, 0, 0], [2.0, 0, 0]]), ids=[1, 2], box=box)
    # Frame 1: rows reordered (id 2 first), id 1 wraps to 1.0.
    f1 = _frame(np.array([[2.5, 0, 0], [1.0, 0, 0]]), ids=[2, 1], box=box)
    out = mp.unwrap_trajectory(mp.Trajectory(systems=[f0, f1]))
    # Output is in canonical (frame-0) order: [id 1, id 2].
    np.testing.assert_array_equal(out[0].data["id"].to_numpy(), [1, 2])
    np.testing.assert_array_equal(out[1].data["id"].to_numpy(), [1, 2])
    # id 1 went 8 → 11 (unwrapped), id 2 went 2 → 2.5.
    np.testing.assert_array_equal(out[0].data["x"].to_numpy(), [8.0, 2.0])
    np.testing.assert_array_equal(out[1].data["x"].to_numpy(), [11.0, 2.5])


def test_image_path_keeps_id_column():
    """Carryover columns (id, type, element) survive into the output and
    only x/y/z change. The image-flag path is also exercised on a binary
    system."""
    box = Box([10.0, 10.0, 10.0])
    f0 = _frame(np.array([[2.0, 0, 0], [4.0, 0, 0]]),
                ids=[1, 2], types=[1, 2], elements=["Cu", "Ni"],
                ix=np.zeros((2, 3), dtype=int), box=box)
    f1 = _frame(np.array([[3.0, 0, 0], [5.0, 0, 0]]),
                ids=[1, 2], types=[1, 2], elements=["Cu", "Ni"],
                ix=np.array([[1, 0, 0], [0, 0, 0]]), box=box)
    out = mp.unwrap_trajectory(mp.Trajectory(systems=[f0, f1]))
    assert out._unwrap_method == "image"
    # Schema is exactly id, type, element, x, y, z (no ix etc.).
    assert out[1].data.columns == ["id", "type", "element", "x", "y", "z"]
    np.testing.assert_array_equal(out[1].data["x"].to_numpy(), [13.0, 5.0])
    np.testing.assert_array_equal(out[1].data["element"].to_list(), ["Cu", "Ni"])


def test_atom_count_mismatch_errors():
    box = Box([10.0, 10.0, 10.0])
    f0 = _frame(np.array([[0.0, 0, 0], [1.0, 0, 0]]), ids=[1, 2], box=box)
    f1 = _frame(np.array([[0.0, 0, 0]]), ids=[1], box=box)
    with pytest.raises(ValueError, match="same number of atoms"):
        mp.unwrap_trajectory(mp.Trajectory(systems=[f0, f1]))


def test_id_set_mismatch_errors():
    box = Box([10.0, 10.0, 10.0])
    f0 = _frame(np.array([[0.0, 0, 0]]), ids=[1], box=box)
    # Wrong id at the same position
    f1 = _frame(np.array([[0.0, 0, 0]]), ids=[2], box=box)
    with pytest.raises(ValueError, match="different id set"):
        mp.unwrap_trajectory(mp.Trajectory(systems=[f0, f1]))


def test_non_periodic_axis_is_left_alone():
    """When PBC is off along z, large z displacements are NOT
    interpreted as wrap events."""
    box = Box([10.0, 10.0, 10.0], boundary=[1, 1, 0])
    frames = [
        _frame(np.array([[5.0, 5.0, 1.0]]), ids=[1], box=box),
        # Big z jump (4 Å, half the box) but z is non-periodic — keep as is.
        _frame(np.array([[5.0, 5.0, 9.0]]), ids=[1], box=box),
    ]
    out = mp.unwrap_trajectory(mp.Trajectory(systems=frames))
    np.testing.assert_array_equal(out[1].data["z"].to_numpy(), [9.0])


def test_pbc_change_warns():
    box0 = Box([10.0, 10.0, 10.0], boundary=[1, 1, 1])
    box1 = Box([10.0, 10.0, 10.0], boundary=[1, 1, 0])
    f0 = _frame(np.array([[0.0, 0, 0]]), ids=[1], box=box0)
    f1 = _frame(np.array([[0.0, 0, 0]]), ids=[1], box=box1)
    with pytest.warns(RuntimeWarning, match="PBC flags change"):
        mp.unwrap_trajectory(mp.Trajectory(systems=[f0, f1]))


def test_method_attribute():
    """Trajectory.unwrap() exposes the same method tag."""
    box = Box([10.0, 10.0, 10.0])
    f0 = _frame(np.array([[0.0, 0, 0]]), ids=[1],
                xu=np.array([[0.0, 0, 0]]), box=box)
    f1 = _frame(np.array([[0.0, 0, 0]]), ids=[1],
                xu=np.array([[10.0, 0, 0]]), box=box)
    out = mp.Trajectory(systems=[f0, f1]).unwrap()
    assert out._unwrap_method == "unwrapped"


def test_id_sorted_output_when_frame0_has_unsorted_ids(tmp_path):
    """Even if frame 0 stores ids in an unsorted order, the unwrapped
    trajectory is emitted in ascending-id order so cross-frame quantities
    (MSD, drift) line up trivially."""
    box = Box([10.0, 10.0, 10.0])
    # Frame 0: id 5 first, id 2 second.
    f0 = _frame(np.array([[1.0, 0, 0], [9.0, 0, 0]]), ids=[5, 2], box=box)
    # Frame 1: same atoms, swapped row order, atom id=2 wrapped from 9→1.
    f1 = _frame(np.array([[1.0, 0, 0], [2.0, 0, 0]]), ids=[2, 5], box=box)
    out = mp.unwrap_trajectory(mp.Trajectory(systems=[f0, f1]))
    np.testing.assert_array_equal(out[0].data["id"].to_numpy(), [2, 5])
    np.testing.assert_array_equal(out[1].data["id"].to_numpy(), [2, 5])
    # id=2 went 9 → 11 (unwrapped), id=5 went 1 → 2.
    np.testing.assert_array_equal(out[0].data["x"].to_numpy(), [9.0, 1.0])
    np.testing.assert_array_equal(out[1].data["x"].to_numpy(), [11.0, 2.0])


def test_gpumd_unwrapped_position_xyz_maps_to_xu(tmp_path):
    """An XYZ frame whose Properties string mentions
    ``unwrapped_position:R:3`` (the GPUMD convention) is read with those
    columns aliased to ``xu/yu/zu`` so ``unwrap_trajectory`` picks the
    direct-rename branch."""
    path = tmp_path / "gpumd.xyz"
    path.write_text(
        "2\n"
        'Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" '
        'Properties=species:S:1:pos:R:3:unwrapped_position:R:3 '
        'pbc="T T T"\n'
        "Cu 1.0 0.0 0.0 1.0 0.0 0.0\n"
        "Ni 9.0 0.0 0.0 9.0 0.0 0.0\n"
        "2\n"
        'Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" '
        'Properties=species:S:1:pos:R:3:unwrapped_position:R:3 '
        'pbc="T T T"\n'
        "Cu 2.0 0.0 0.0 2.0 0.0 0.0\n"
        "Ni 1.0 0.0 0.0 11.0 0.0 0.0\n"  # wrapped x=1, true xu=11
    )
    traj = mp.Trajectory(str(path), verbose=False)
    assert {"x", "y", "z", "xu", "yu", "zu"}.issubset(traj[0].data.columns)
    out = mp.unwrap_trajectory(traj)
    assert out._unwrap_method == "unwrapped"
    np.testing.assert_array_equal(out[1].data["x"].to_numpy(), [2.0, 11.0])


def test_msd_consistency_via_xu():
    """An MSD-style sum-of-squares is invariant to the unwrap method when
    the user provided xu/yu/zu — exactly the typical use-case the priority
    ordering is designed for."""
    rng = np.random.default_rng(0)
    L = 10.0
    box = Box([L, L, L])
    n = 5
    nframes = 6
    # Generate continuous random walks first, then wrap them.
    xu_traj = rng.normal(size=(nframes, n, 3)).cumsum(axis=0) * 0.4
    xu_traj += np.array([5.0, 5.0, 5.0])
    wrapped = xu_traj % L
    frames = [
        _frame(wrapped[t], ids=np.arange(1, n + 1, dtype=np.int32),
               xu=xu_traj[t], box=box)
        for t in range(nframes)
    ]
    out = mp.unwrap_trajectory(mp.Trajectory(systems=frames))
    for t in range(nframes):
        np.testing.assert_allclose(
            np.column_stack([out[t].data[c].to_numpy() for c in ("x", "y", "z")]),
            xu_traj[t],
        )
