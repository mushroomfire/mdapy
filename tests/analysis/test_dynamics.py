# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.analysis import (
    LindemannIndex,
    MeanSquaredDisplacement,
    TimeAverage,
    trajectory_lines,
)
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
    assert (
        repr(MeanSquaredDisplacement())
        == "MeanSquaredDisplacement(window=True, unwrap=True)"
    )


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


def flagged(x):
    """The positions wrapped into BOX, with their image flags."""
    wrapped, image = BOX.wrap(x)
    return mdapy.System(wrapped, BOX, elements="Ar").with_columns(image=image)


def test_image_flags():
    # steps of up to the box length, followed only by the image flags
    rng = np.random.default_rng(5)
    true = np.cumsum(rng.normal(scale=3.0, size=(30, 20, 3)), axis=0)
    result = MeanSquaredDisplacement(window=False).compute([flagged(x) for x in true])
    np.testing.assert_allclose(
        result.msd, ((true - true[0]) ** 2).sum(-1).mean(1), rtol=1e-12
    )
    # a frame without flags is followed by the nearest image instead
    small, frames = walk(frames=10, atoms=8, seed=6)
    mixed = [frames[t] if t == 4 else flagged(x) for t, x in enumerate(small)]
    np.testing.assert_allclose(
        MeanSquaredDisplacement(window=False).compute(mixed).msd,
        ((small - small[0]) ** 2).sum(-1).mean(1),
        atol=1e-12,
    )


def test_unwrapped_positions_as_they_are():
    # long steps of positions already unwrapped, outside the box
    rng = np.random.default_rng(8)
    true = np.cumsum(rng.normal(scale=3.0, size=(25, 15, 3)), axis=0)
    frames = [mdapy.System(x, BOX, elements="Ar") for x in true]
    for window in (True, False):
        given = MeanSquaredDisplacement(window=window, unwrap=False).compute(frames)
        expected = MeanSquaredDisplacement(window=window).compute(
            [flagged(x) for x in true]
        )
        np.testing.assert_allclose(given.msd, expected.msd, rtol=1e-10, atol=1e-9)
    followed = MeanSquaredDisplacement(window=False).compute(frames)
    assert abs(followed.msd[-1] - expected.msd[-1]) > 0.1 * expected.msd[-1]
    result = LindemannIndex(unwrap=False).compute(frames)
    index, atomic = lindemann(true)
    assert result.index == pytest.approx(index, rel=1e-10)
    np.testing.assert_allclose(result.atomic, atomic, rtol=1e-10)
    assert repr(LindemannIndex(unwrap=False)) == "LindemannIndex(unwrap=False)"


def test_atoms_matched_by_id():
    # every frame in another order of the atoms, with ids and image flags
    rng = np.random.default_rng(9)
    true = np.cumsum(rng.normal(scale=3.0, size=(20, 12, 3)), axis=0)
    ids = rng.permutation(np.arange(100, 112))
    ordered = [flagged(x).with_columns(id=ids) for x in true]
    orders = [rng.permutation(12) for _ in ordered]
    shuffled = [f.with_table(f.table[o]) for f, o in zip(ordered, orders)]
    for window in (True, False):
        np.testing.assert_allclose(
            MeanSquaredDisplacement(window=window).compute(shuffled).components,
            MeanSquaredDisplacement(window=window).compute(ordered).components,
            rtol=1e-12,
            atol=1e-12,
        )
    # the atoms of the first frame, in its order
    expected = LindemannIndex().compute(ordered)
    result = LindemannIndex().compute(shuffled)
    np.testing.assert_allclose(result.atomic, expected.atomic[orders[0]], rtol=1e-12)
    assert result.index == pytest.approx(expected.index, rel=1e-12)


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
    assert repr(LindemannIndex()) == "LindemannIndex(unwrap=True)"


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
    numbered = two.with_columns(id=[1, 2])
    with pytest.raises(InvalidParameter, match="frame 1 has no id column"):
        MeanSquaredDisplacement().compute([numbered, two])
    with pytest.raises(InvalidParameter, match="frame 1 has other ids"):
        MeanSquaredDisplacement().compute([numbered, two.with_columns(id=[1, 3])])
    with pytest.raises(InvalidParameter, match="repeated ids"):
        MeanSquaredDisplacement().compute([two.with_columns(id=[4, 4])])


def test_time_average_of_moving_atoms():
    """Atoms moving at constant velocity through the periodic boundaries
    average to their positions at the middle time; atoms matched by id."""
    rng = np.random.default_rng(5)
    start, velocity = rng.random((15, 3)) * 8, rng.normal(scale=1.5, size=(15, 3))
    ids = rng.permutation(np.arange(1, 16))
    true = np.array([start + velocity * t for t in range(9)])
    frames = []
    for x in true:
        frame = mdapy.System(BOX.wrap(x)[0], BOX, elements="Ar").with_columns(
            id=ids, charge=np.linalg.norm(x, axis=1)
        )
        order = rng.permutation(15)
        frames.append(frame.with_table(frame.table[order]) if frames else frame)
    result = TimeAverage("positions").compute(frames)
    # followed from where the first frame has them, in the box
    np.testing.assert_allclose(
        result.value, BOX.wrap(start)[0] + 4 * velocity, atol=1e-10
    )
    assert result.frames == 9 and result.name == "positions"
    # as they are: the mean of the wrapped positions of each atom
    wrapped = TimeAverage("positions", unwrap=False).compute(frames).value
    np.testing.assert_allclose(
        wrapped,
        BOX.wrap(true.reshape(-1, 3))[0].reshape(true.shape).mean(0),
        atol=1e-10,
    )
    charge = TimeAverage("charge").compute(frames)
    np.testing.assert_allclose(charge.value, np.linalg.norm(true, axis=2).mean(0))
    added = frames[0].with_columns(charge)
    np.testing.assert_allclose(added["charge_average"], charge.value)
    assert repr(TimeAverage("csp")) == "TimeAverage('csp', unwrap=True)"


def test_time_average_of_the_box_and_of_functions():
    frames = [
        mdapy.System([[1.0, 1, 1], [2.0, 2, 2]], mdapy.Box(10.0 + t), elements="Ar")
        for t in range(4)
    ]
    box = TimeAverage("box").compute(frames)
    np.testing.assert_allclose(box.value, 11.5 * np.eye(3))
    assert box.name is None and box.columns() == {}
    volume = TimeAverage(lambda f: np.linalg.det(f.box.matrix)).compute(frames)
    assert volume.value == pytest.approx(np.mean([(10.0 + t) ** 3 for t in range(4)]))
    sides = TimeAverage(lambda f: np.diag(f.box.matrix)).compute(iter(frames))
    np.testing.assert_allclose(sides.value, [11.5] * 3)
    with pytest.raises(InvalidParameter, match="frame 1 gives shape"):
        TimeAverage(lambda f: np.zeros(int(f.box.matrix[0, 0]))).compute(frames)
    with pytest.raises(InvalidParameter, match="no frames"):
        TimeAverage("box").compute([])
    with pytest.raises(InvalidParameter, match="no frames"):
        TimeAverage("positions").compute([])
    with pytest.raises(InvalidParameter, match="element is not numbers"):
        TimeAverage("element").compute(frames)
    with pytest.raises(InvalidParameter, match="column name, box or a function"):
        TimeAverage(3)


def test_trajectory_lines():
    """The unwrapped path of each atom, the atoms of the first frame in its
    order, all or some."""
    rng = np.random.default_rng(6)
    start, velocity = rng.random((10, 3)) * 8, rng.normal(scale=2.0, size=(10, 3))
    true = np.array([start + velocity * t for t in range(7)])
    ids = rng.permutation(np.arange(1, 11))
    frames = []
    for x in true:
        frame = mdapy.System(BOX.wrap(x)[0], BOX, elements="Ar").with_columns(id=ids)
        frames.append(
            frame.with_table(frame.table[rng.permutation(10)]) if frames else frame
        )
    lines = trajectory_lines(frames)
    assert lines.shape == (10, 7, 3)
    expected = (
        BOX.wrap(start)[0][:, None] + velocity[:, None] * np.arange(7)[None, :, None]
    )
    np.testing.assert_allclose(lines, expected, atol=1e-10)
    np.testing.assert_allclose(trajectory_lines(frames, [3, 1]), lines[[1, 3]])
    mask = np.arange(10) % 2 == 0
    np.testing.assert_allclose(trajectory_lines(iter(frames), mask), lines[mask])
    wrapped = trajectory_lines(frames, unwrap=False)
    np.testing.assert_allclose(wrapped[:, 0], frames[0]["positions"])
    with pytest.raises(InvalidParameter, match="mask of atoms must have 10"):
        trajectory_lines(frames, np.ones(3, dtype=bool))
    with pytest.raises(InvalidParameter, match="no frames"):
        trajectory_lines([])


def test_msd_of_atoms_in_blocks(monkeypatch):
    """Some atoms at a time give what all at once do."""
    from mdapy.analysis import _dynamics

    _, frames = walk(frames=30, atoms=50, seed=8)
    whole = [MeanSquaredDisplacement(window=w).compute(frames) for w in (True, False)]
    monkeypatch.setattr(_dynamics, "_BLOCK", 7 * 30)  # 7 atoms at a time
    for w, expected in zip((True, False), whole):
        got = MeanSquaredDisplacement(window=w).compute(iter(frames))
        np.testing.assert_allclose(
            got.components, expected.components, rtol=1e-12, atol=1e-12
        )
