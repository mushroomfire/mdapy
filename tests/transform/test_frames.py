# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy import transform
from mdapy.errors import InvalidParameter, MissingProperty


def moving(count=10, box=100.0, speed=1.0, shuffle=False):
    """Two atoms with ids, the first moving along x at a constant speed and
    wrapped into the box, the second at rest; with shuffle, each frame in
    another order."""
    rng = np.random.default_rng(1)
    frames = []
    for t in range(count):
        positions = np.array([[(speed * t) % box, 5.0, 5.0], [1.0, 1.0, 1.0]])
        frame = mdapy.System(positions, mdapy.Box(box), elements="Cu").with_columns(
            id=[1, 2], value=[10.0 + t, 20.0 + t]
        )
        if shuffle and rng.random() < 0.5:
            frame = frame.with_table(frame.table[[1, 0]])
        frames.append(frame)
    return frames


def x_of_first(frame):
    return float(frame["positions.x"][np.asarray(frame["id"]) == 1][0])


@pytest.mark.parametrize(
    "window, expected",
    [
        (1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (2, [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.0]),
        (3, [1 / 3, 1, 2, 3, 4, 5, 6, 7, 8, 26 / 3]),
        (4, [0.75, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.25, 8.75]),
        (5, [0.6, 1.2, 2, 3, 4, 5, 6, 7, 7.8, 8.4]),
    ],
)
def test_smoothing_windows(window, expected):
    """The frames about each one, the first and last repeated beyond the
    ends: as OVITO gives them for a constant speed."""
    smoothed = transform.smooth_trajectory(moving(shuffle=True), window)
    assert len(smoothed) == 10
    np.testing.assert_allclose([x_of_first(f) for f in smoothed], expected, atol=1e-12)


def test_smoothing_through_the_boundaries():
    """An atom crossing the boundary every few frames: the mean follows it,
    placed about where the atom is in that frame."""
    frames = moving(count=8, box=10.0, speed=3.0)
    smoothed = transform.smooth_trajectory(frames, 3)
    # x = 3t wrapped into 10: the mean of 3(t-1), 3t, 3(t+1) is the atom
    # itself in the middle frames; 1 and 20 at the ends, placed about the
    # atom where it is (0 and 1)
    np.testing.assert_allclose(
        [x_of_first(f) for f in smoothed], [1, 3, 6, 9, 2, 5, 8, 0], atol=1e-12
    )
    # the other columns of real numbers averaged, the ids as they are
    np.testing.assert_allclose(smoothed[5]["value"], [15.0, 25.0])
    np.testing.assert_allclose(smoothed[0]["value"], [10 + 1 / 3, 20 + 1 / 3])
    np.testing.assert_array_equal(smoothed[5]["id"], [1, 2])
    # a slice gives the frames it covers
    assert [x_of_first(f) for f in smoothed[-2:]] == [x_of_first(smoothed[6]), 0.0]


def test_smoothing_reads_each_frame_once_this_one_last():
    read = []

    def noted(k):
        read.append(k)
        return frames[k]

    frames = moving()
    smoothed = transform.smooth_trajectory(mdapy.io.Mapped(range(10), noted), 5)
    smoothed[0]
    assert read == [1, 2, 0]
    read.clear()
    smoothed[5]
    assert read == [3, 4, 6, 7, 5]


def test_smoothing_errors():
    with pytest.raises(InvalidParameter, match="window"):
        transform.smooth_trajectory(moving(), 0)
    with pytest.raises(InvalidParameter, match="no frames"):
        transform.smooth_trajectory([], 3)
    with pytest.raises(IndexError):
        transform.smooth_trajectory(moving(), 3)[10]


def test_freezing_a_column():
    frames = moving(shuffle=True)
    frozen = transform.freeze(frames[6], frames[2], "value")
    ids = np.asarray(frozen["id"])
    np.testing.assert_array_equal(
        frozen["value_frozen"], np.where(ids == 1, 12.0, 22.0)
    )
    np.testing.assert_array_equal(frozen["value"], frames[6]["value"])
    named = transform.freeze(frames[6], frames[0], "positions.x", name="x0")
    np.testing.assert_array_equal(named["x0"], np.where(ids == 1, 0.0, 1.0))
    assert (
        "positions_x_frozen"
        in transform.freeze(frames[1], frames[0], "positions.x").table.columns
    )


def test_freezing_errors():
    frames = moving()
    alone = frames[0].filter(id=2)
    with pytest.raises(MissingProperty, match="atom 1"):
        transform.freeze(frames[3], alone, "value")
    with pytest.raises(InvalidParameter, match="cannot be matched"):
        transform.freeze(frames[3].drop("id"), alone.drop("id"), "value")
