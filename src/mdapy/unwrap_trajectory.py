# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Unwrap periodic-boundary trajectories so that particle paths become
continuous across frames — equivalent to OVITO's
``UnwrapTrajectoriesModifier``.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, List

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from mdapy.trajectory import Trajectory

from mdapy.box import Box
from mdapy.system import System


_METHOD_UNWRAPPED = "unwrapped"  # xu/yu/zu present
_METHOD_IMAGE = "image"          # ix/iy/iz present
_METHOD_MIN_IMAGE = "min_image"  # neither — scan with minimum-image principle


def _choose_method(frame0: System) -> str:
    cols = set(frame0.data.columns)
    if {"xu", "yu", "zu"}.issubset(cols):
        return _METHOD_UNWRAPPED
    if {"ix", "iy", "iz"}.issubset(cols):
        return _METHOD_IMAGE
    return _METHOD_MIN_IMAGE


def _validate_and_canonicalise(traj: "Trajectory") -> List[pl.DataFrame]:
    """Return a list of per-frame DataFrames in canonical (id-sorted) order
    when an ``id`` column exists, or in original row order otherwise.

    Validates: trajectory is non-empty, every frame has the same number of
    atoms, and (if ``id`` is present) every frame has the same id set.
    """
    if len(traj) == 0:
        raise ValueError("unwrap_trajectory: trajectory has no frames.")

    n0 = traj[0].N
    if n0 == 0:
        raise ValueError("unwrap_trajectory: frames contain no atoms.")
    has_id = "id" in traj[0].data.columns
    ids0_sorted = None
    if has_id:
        ids0_sorted = np.sort(traj[0].data["id"].to_numpy())
        if len(np.unique(ids0_sorted)) != n0:
            raise ValueError(
                "unwrap_trajectory: 'id' column in frame 0 contains "
                "duplicates; ids must uniquely label atoms."
            )

    canonical = []
    for fi, frame in enumerate(traj):
        if frame.N != n0:
            raise ValueError(
                "unwrap_trajectory: every frame must contain the same "
                f"number of atoms; frame 0 has {n0}, frame {fi} has "
                f"{frame.N}."
            )
        if has_id:
            if "id" not in frame.data.columns:
                raise ValueError(
                    f"unwrap_trajectory: frame {fi} is missing the 'id' "
                    "column that frame 0 carries."
                )
            df = frame.data.sort("id")
            ids = df["id"].to_numpy()
            if not np.array_equal(np.sort(ids), ids0_sorted):
                raise ValueError(
                    f"unwrap_trajectory: frame {fi} has a different id "
                    "set from frame 0 — atoms must be the same across "
                    "the whole trajectory."
                )
            canonical.append(df)
        else:
            canonical.append(frame.data)
    return canonical


def _check_pbc_consistent(traj: "Trajectory") -> np.ndarray:
    """Return frame 0's boundary flags; warn if any later frame disagrees."""
    pbc0 = np.asarray(traj[0].box.boundary, dtype=int)
    for i, frame in enumerate(traj):
        if not np.array_equal(np.asarray(frame.box.boundary, dtype=int), pbc0):
            warnings.warn(
                f"unwrap_trajectory: PBC flags change between frame 0 "
                f"({pbc0.tolist()}) and frame {i} "
                f"({np.asarray(frame.box.boundary, dtype=int).tolist()}); "
                "using frame 0's flags throughout.",
                RuntimeWarning, stacklevel=3,
            )
            break
    return pbc0


def _carryover_columns(df: pl.DataFrame) -> List[str]:
    return [c for c in ("id", "type", "element") if c in df.columns]


def _build_frame(df: pl.DataFrame, pos: np.ndarray, box: Box) -> System:
    """Assemble a System with only id/type/element + unwrapped x/y/z."""
    cols = {name: df[name] for name in _carryover_columns(df)}
    cols["x"] = pl.Series("x", pos[:, 0], dtype=pl.Float64)
    cols["y"] = pl.Series("y", pos[:, 1], dtype=pl.Float64)
    cols["z"] = pl.Series("z", pos[:, 2], dtype=pl.Float64)
    return System(data=pl.DataFrame(cols), box=Box(box))


def _detect_lammps_tilt_flip(prev_box: np.ndarray, curr_box: np.ndarray) -> bool:
    r"""Return True if a LAMMPS-style cell flip likely occurred between two
    frames. LAMMPS keeps each tilt factor (xy, xz, yz) within
    :math:`\pm 0.5` of the relevant edge length; when a tilt drifts past
    that limit the simulator silently re-folds the cell, producing a
    discontinuous jump close to one full edge in the box matrix. The
    minimum-image heuristic cannot follow that jump, so we issue a
    warning and let the user fall back to the image-flag path.
    """
    a_x = prev_box[0, 0]
    b_y = prev_box[1, 1]
    if a_x <= 0 or b_y <= 0:
        return False
    for prev_t, curr_t, denom in (
        (prev_box[1, 0], curr_box[1, 0], a_x),  # xy
        (prev_box[2, 0], curr_box[2, 0], a_x),  # xz
        (prev_box[2, 1], curr_box[2, 1], b_y),  # yz
    ):
        if abs(curr_t - prev_t) / denom > 0.7:
            return True
    return False


def unwrap_trajectory(traj: "Trajectory") -> "Trajectory":
    r"""Make particle trajectories continuous across periodic boundaries.

    Equivalent to OVITO's ``UnwrapTrajectoriesModifier``. The function
    picks one of three methods, in order of decreasing reliability, by
    inspecting the columns of frame 0:

    1. **Pre-unwrapped coordinates** (``xu``, ``yu``, ``zu``). These come
       straight from the simulation code and are already continuous, so
       the function just renames them as the new ``x``, ``y``, ``z``.
    2. **Periodic-image flags** (``ix``, ``iy``, ``iz``). LAMMPS writes
       these whenever ``dump_modify ... pbc yes`` is set. We compute
       :math:`\vec{r}_\text{unwrapped} = \vec{r} + i_x \vec{a}
       + i_y \vec{b} + i_z \vec{c}` using *each frame's own* cell vectors,
       so trajectories that change box (NPT) are handled correctly.
    3. **Minimum-image heuristic**. Walks the trajectory in time and
       accumulates per-atom integer PBC shift vectors from fractional-
       coordinate jumps greater than half a cell edge. This is the same
       algorithm OVITO uses when no image flags are available.

    When an ``id`` column is present every frame is sorted by ``id``
    before unwrapping, so the output rows line up across frames and
    cross-frame analyses (MSD, drift, ...) are a single subtraction.

    Parameters
    ----------
    traj : mdapy.Trajectory
        Multi-frame trajectory. Every frame must contain the same number
        of atoms and (if an ``id`` column is present) the same id set.

    Returns
    -------
    mdapy.Trajectory
        New trajectory with the same number of frames. Each frame keeps
        only the ``id``, ``type``, ``element`` columns (whichever existed
        in the input) plus the unwrapped ``x``, ``y``, ``z`` and the
        original simulation box. The selected unwrap method is recorded
        on the returned object as ``trajectory._unwrap_method``.

    Raises
    ------
    ValueError
        If the trajectory is empty, atom counts differ between frames,
        ids contain duplicates, or the id set differs between frames.

    Warnings
    --------
    A ``RuntimeWarning`` is issued when boundary flags change across
    frames (the frame-0 flags are used throughout) and when a likely
    LAMMPS triclinic-cell flip is detected during the minimum-image scan.

    Notes
    -----
    The minimum-image heuristic *cannot distinguish* a true periodic-
    boundary crossing from genuine physical motion that exceeds half a
    cell edge in a single frame. If your dump frequency is too coarse,
    the unwrapped trajectory will be silently wrong — re-dump with a
    smaller stride or have LAMMPS write the ``ix iy iz`` image flags so
    this routine takes the second branch.
    """
    from mdapy.trajectory import Trajectory

    canonical = _validate_and_canonicalise(traj)
    pbc = _check_pbc_consistent(traj)
    method = _choose_method(traj[0])

    out_frames: List[System] = []

    if method == _METHOD_UNWRAPPED:
        for df, frame in zip(canonical, traj):
            pos = np.column_stack([df[c].to_numpy() for c in ("xu", "yu", "zu")])
            out_frames.append(_build_frame(df, pos, frame.box))

    elif method == _METHOD_IMAGE:
        for df, frame in zip(canonical, traj):
            cell = np.asarray(frame.box.box, dtype=np.float64)
            pos = np.column_stack([df[c].to_numpy() for c in ("x", "y", "z")])
            img = np.column_stack(
                [df[c].to_numpy().astype(np.int64) for c in ("ix", "iy", "iz")]
            )
            unwrapped = pos + img.astype(np.float64) @ cell
            out_frames.append(_build_frame(df, unwrapped, frame.box))

    else:  # _METHOD_MIN_IMAGE
        n = traj[0].N
        shift = np.zeros((n, 3), dtype=np.int64)
        df0 = canonical[0]
        cell0 = np.asarray(traj[0].box.box, dtype=np.float64)
        pos_prev = np.column_stack([df0[c].to_numpy() for c in ("x", "y", "z")])
        frac_prev = pos_prev @ np.linalg.inv(cell0)
        out_frames.append(_build_frame(df0, pos_prev, traj[0].box))

        prev_box = cell0
        flip_warned = False
        for fi in range(1, len(traj)):
            cell = np.asarray(traj[fi].box.box, dtype=np.float64)
            if not flip_warned and _detect_lammps_tilt_flip(prev_box, cell):
                warnings.warn(
                    f"unwrap_trajectory: detected a possible LAMMPS "
                    f"triclinic cell flip between frame {fi - 1} and "
                    f"frame {fi}. The minimum-image heuristic does not "
                    "unflip the cell — consider re-dumping with "
                    "``dump_modify pbc yes`` so ix/iy/iz are written.",
                    RuntimeWarning, stacklevel=3,
                )
                flip_warned = True
            prev_box = cell

            df = canonical[fi]
            pos = np.column_stack([df[c].to_numpy() for c in ("x", "y", "z")])
            frac = pos @ np.linalg.inv(cell)
            for d in range(3):
                if pbc[d]:
                    shift[:, d] += np.round(frac_prev[:, d] - frac[:, d]).astype(np.int64)
            unwrapped = pos + shift.astype(np.float64) @ cell
            frac_prev = frac
            out_frames.append(_build_frame(df, unwrapped, traj[fi].box))

    new_traj = Trajectory(systems=out_frames)
    new_traj._unwrap_method = method
    return new_traj
