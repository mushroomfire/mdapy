# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest

import mdapy
from mdapy.analysis import LindemannIndex, MeanSquaredDisplacement
from mdapy.errors import InvalidParameter

BOX = mdapy.Box([[10.0, 0, 0], [2.0, 9.0, 0], [-1.0, 1.0, 11.0]])


def walk(frames=120, atoms=60, scale=0.12, seed=0):
    """True positions of a random walk and the frames with them wrapped."""
    rng = np.random.default_rng(seed)
    true = np.cumsum(rng.normal(scale=scale, size=(frames, atoms, 3)), axis=0)
    systems = [mdapy.System(BOX.wrap(x)[0], BOX, elements="Ar") for x in true]
    return true, systems


def test_msd_over_every_origin():
    true, frames = walk()
    result = MeanSquaredDisplacement().compute(frames)
    count = len(true)
    expected = np.array(
        [((true[m:] - true[: count - m]) ** 2).mean(axis=(0, 1)) for m in range(count)]
    )
    np.testing.assert_allclose(result.components, expected, atol=1e-9)
    np.testing.assert_allclose(result.msd, expected.sum(axis=1), atol=1e-9)
    np.testing.assert_array_equal(result.lag, np.arange(count))
    first = MeanSquaredDisplacement(window=False).compute(frames)
    np.testing.assert_allclose(
        first.msd, ((true - true[0]) ** 2).sum(-1).mean(1), atol=1e-12
    )
    assert repr(MeanSquaredDisplacement()) == "MeanSquaredDisplacement(window=True)"


def test_ballistic_motion():
    rng = np.random.default_rng(1)
    start, velocity = rng.random((30, 3)) * 9, rng.normal(scale=0.3, size=(30, 3))
    frames = [
        mdapy.System(BOX.wrap(start + t * velocity)[0], BOX, elements="Ar")
        for t in range(40)
    ]
    speed2 = (velocity**2).sum(1).mean()
    for window in (True, False):
        result = MeanSquaredDisplacement(window=window).compute(frames)
        np.testing.assert_allclose(
            result.msd, speed2 * np.arange(40) ** 2, rtol=1e-9, atol=1e-9
        )


def test_trajectory_files(tmp_path):
    true, frames = walk(frames=20, atoms=10, seed=2)
    mdapy.write(tmp_path / "walk.xyz", frames)
    trajectory = mdapy.read_trajectory(tmp_path / "walk.xyz")
    np.testing.assert_allclose(
        MeanSquaredDisplacement().compute(trajectory).msd,
        MeanSquaredDisplacement().compute(frames).msd,
        atol=1e-9,
    )


def lindemann(true):
    count, n = true.shape[:2]
    d = np.linalg.norm(true[:, :, None] - true[:, None, :], axis=3)  # (T, N, N)
    mean, square = d.mean(0), (d * d).mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.sqrt(np.maximum(square - mean * mean, 0)) / mean
    np.fill_diagonal(ratio, 0.0)
    return ratio.sum() / (n * (n - 1)), ratio.sum(1) / (n - 1)


def test_lindemann_index():
    # a cluster vibrating about its sites, in a box that is not periodic
    rng = np.random.default_rng(3)
    sites = rng.random((25, 3)) * 8 + 20
    true = sites + rng.normal(scale=0.1, size=(30, 25, 3))
    frames = [mdapy.System(x, mdapy.Box(50.0, pbc=False), elements="Ar") for x in true]
    result = LindemannIndex().compute(frames)
    index, atomic = lindemann(true)
    assert result.index == pytest.approx(index, rel=1e-10)
    np.testing.assert_allclose(result.atomic, atomic, rtol=1e-10)
    assert result.running[0] == 0.0 and result.running[-1] == result.index
    assert list(frames[0].with_columns(result).table.columns)[-1] == "lindemann"
    # a rigid motion keeps every distance
    rng = np.random.default_rng(4)
    cluster = rng.random((20, 3)) * 5
    big = mdapy.Box(50.0, pbc=False)
    moving = [
        mdapy.System(cluster + [t * 0.3, 0, 0], big, elements="Ar") for t in range(10)
    ]
    assert LindemannIndex().compute(moving).index == pytest.approx(0.0, abs=1e-12)
    assert repr(LindemannIndex()) == "LindemannIndex()"


def test_errors():
    with pytest.raises(InvalidParameter, match="no frames"):
        MeanSquaredDisplacement().compute([])
    one = mdapy.System([[1.0, 1, 1]], BOX, elements="Ar")
    two = mdapy.System([[1.0, 1, 1], [2.0, 2, 2]], BOX, elements="Ar")
    with pytest.raises(InvalidParameter, match="frame 1 has 2 atoms"):
        MeanSquaredDisplacement().compute([one, two])
    empty = mdapy.System(np.zeros((0, 3)), BOX, elements=[])
    with pytest.raises(InvalidParameter, match="no atoms"):
        MeanSquaredDisplacement().compute([empty, empty])
    with pytest.raises(InvalidParameter, match="two atoms or more"):
        LindemannIndex().compute([one, one])
