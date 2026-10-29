# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.errors import FileFormatError, InvalidParameter, UnsupportedFormat
from mdapy.io import _native


def frames(n=5):
    base = mdapy.System(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], mdapy.Box(5.0), elements=["Cu", "Zr"]
    )
    return [
        base.with_info(timestep=k).with_columns(positions=base["positions"] + 0.1 * k)
        for k in range(n)
    ]


@pytest.fixture(params=["t.xyz", "t.dump", "t.mdapy", "t.xyz.gz"])
def path(request, tmp_path):
    path = tmp_path / request.param
    mdapy.write(path, frames())
    return path


def steps(trajectory):
    return [frame.info["timestep"] for frame in trajectory]


def test_length_indexing_and_iteration(path):
    with mdapy.read_trajectory(path) as trajectory:
        assert trajectory.path == str(path)
        assert trajectory.format == mdapy.detect_format(path)
        assert len(trajectory) == 5
        assert trajectory[0].info["timestep"] == 0
        assert trajectory[-1].info["timestep"] == 4
        assert trajectory[-5].info["timestep"] == 0
        assert steps(trajectory) == [0, 1, 2, 3, 4]
        np.testing.assert_allclose(trajectory[3]["positions"], frames()[3]["positions"])
        for index in (5, -6):
            with pytest.raises(IndexError, match="out of range"):
                trajectory[index]
        with pytest.raises(TypeError):
            trajectory[1.0]


def test_slices(tmp_path):
    path = tmp_path / "t.xyz"
    mdapy.write(path, frames(10))
    trajectory = mdapy.read_trajectory(path)
    assert steps(trajectory[2:8:2]) == [2, 4, 6]
    assert steps(trajectory[::-3]) == [9, 6, 3, 0]
    nested = trajectory[1:9][::2][1:]
    assert len(nested) == 3 and steps(nested) == [3, 5, 7]
    assert nested[-1].info["timestep"] == 7
    with pytest.raises(IndexError):
        nested[3]
    assert len(trajectory[20:]) == 0 and list(trajectory[20:]) == []
    trajectory.close()


def test_frames_are_found_only_as_far_as_needed(tmp_path):
    path = tmp_path / "t.xyz"
    mdapy.write(path, frames(50))
    trajectory = mdapy.read_trajectory(path)
    assert "frames=0+" in repr(trajectory)
    assert trajectory[0].info["timestep"] == 0
    assert len(trajectory._frames.positions) == 1
    iterator = iter(trajectory)
    next(iterator), next(iterator)
    assert len(trajectory._frames.positions) == 2
    assert len(trajectory) == 50
    assert repr(trajectory) == f"Trajectory({str(path)!r}, format='extxyz', frames=50)"
    assert repr(trajectory[:3]).endswith("frames=3)")
    trajectory.close()


def test_iteration_stops_at_a_truncated_frame(tmp_path):
    path = tmp_path / "t.xyz"
    mdapy.write(path, frames(3))
    text = path.read_text()
    path.write_text(text[: text.rindex("\n", 0, len(text) - 1)])  # drop the last line
    trajectory = mdapy.read_trajectory(path)
    seen = []
    with pytest.raises(
        FileFormatError, match="the frame has 2 atoms, but the file ends"
    ):
        for frame in trajectory:
            seen.append(frame.info["timestep"])
    assert seen == [0, 1]


def test_single_frame_formats(tmp_path):
    path = tmp_path / "POSCAR"
    mdapy.write(path, frames(1)[0])
    trajectory = mdapy.read_trajectory(path)
    assert len(trajectory) == 1 and trajectory.format == "poscar"
    assert len(list(trajectory)) == 1 and len(trajectory[-1]) == 2
    with pytest.raises(IndexError):
        trajectory[1]


def test_options_apply_to_every_frame(tmp_path):
    path = tmp_path / "t.dump"
    mdapy.write(path, frames(2))
    text = (
        path.read_text()
        .replace(" element", "")
        .replace(" Cu ", " ")
        .replace(" Zr ", " ")
    )
    path.write_text(text)
    trajectory = mdapy.read_trajectory(path, elements=["Ni", "Al"])
    assert [f.table["element"].to_list() for f in trajectory] == [["Ni", "Al"]] * 2


def test_mdapy_files_read_their_index_once(tmp_path, monkeypatch):
    path = tmp_path / "t.mdapy"
    mdapy.write(path, frames(6))
    calls = []
    original = _native._headers
    monkeypatch.setattr(
        _native, "_headers", lambda archive: calls.append(1) or original(archive)
    )
    assert steps(mdapy.read_trajectory(path)) == list(range(6))
    assert len(calls) == 1


def test_converting_a_trajectory(tmp_path):
    source = tmp_path / "t.xyz"
    mdapy.write(source, frames(4))
    target = tmp_path / "t.mdapy"
    with mdapy.read_trajectory(source) as trajectory:
        mdapy.write(target, trajectory[::2])
    assert steps(mdapy.read_trajectory(target)) == [0, 2]


def test_closed_trajectories_release_the_file(tmp_path):
    path = tmp_path / "t.xyz"
    mdapy.write(path, frames(2))
    with mdapy.read_trajectory(path) as trajectory:
        trajectory[1]
        source = trajectory._frames.source(0)
    assert source._map.closed
    path.unlink()


def test_unknown_formats(tmp_path):
    with pytest.raises(UnsupportedFormat, match="cannot tell the format"):
        mdapy.read_trajectory(tmp_path / "t.unknown")


def test_top_level_names():
    assert mdapy.read_trajectory is mdapy.io.read_trajectory
    assert mdapy.Trajectory is mdapy.io.Trajectory


def test_files_of_a_pattern_or_a_list(tmp_path):
    """The frames of several files, those of a pattern in the order of the
    numbers in their names; any frame read in any order."""
    series = frames(6)
    for step, pair in zip((0, 5, 10), (series[0:2], series[2:4], series[4:6])):
        mdapy.write(tmp_path / f"dump.{step}.lammpstrj", pair)
    pattern = tmp_path / "dump.*.lammpstrj"
    trajectory = mdapy.read_trajectory(pattern)
    assert trajectory.path == str(pattern) and trajectory.format == "lammps-dump"
    assert steps(trajectory) == [0, 1, 2, 3, 4, 5] and len(trajectory) == 6
    # back to the first file, then on to the last
    assert [trajectory[k].info["timestep"] for k in (5, 0, 3, -1)] == [5, 0, 3, 5]
    assert steps(trajectory[::2]) == [0, 2, 4]
    listed = [tmp_path / f"dump.{step}.lammpstrj" for step in (10, 0)]
    assert steps(mdapy.read_trajectory(listed)) == [4, 5, 0, 1]
    assert mdapy.read_trajectory(listed).path == str(listed[0])
    # a format of one frame per file
    for k, frame in enumerate(series[:3]):
        mdapy.write(tmp_path / f"frame{k}.xyz", frame)
    assert steps(mdapy.read_trajectory(tmp_path / "frame*.xyz")) == [0, 1, 2]
    with pytest.raises(FileNotFoundError, match="no file matches"):
        mdapy.read_trajectory(tmp_path / "none.*.xyz")
    with pytest.raises(InvalidParameter, match="at least one file"):
        mdapy.read_trajectory([])


def water(n=4):
    """Water molecules with bonds, angles, charges and molecule ids."""
    rng = np.random.default_rng(0)
    oxygens = rng.random((n, 3)) * 12 + 1
    positions = np.stack(
        [oxygens, oxygens + [0.96, 0, 0], oxygens + [-0.24, 0.93, 0]], axis=1
    ).reshape(-1, 3)
    bonds = [[3 * k, 3 * k + i] for k in range(n) for i in (1, 2)]
    angles = [[3 * k + 1, 3 * k, 3 * k + 2] for k in range(n)]
    system = mdapy.System(
        positions,
        mdapy.Box(15.0),
        elements=["O", "H", "H"] * n,
        topology=mdapy.Topology(bonds=bonds, angles=angles),
    )
    return system.with_columns(
        charge=[-0.8476, 0.4238, 0.4238] * n,
        molecule=np.repeat(np.arange(1, n + 1), 3),
    )


@pytest.fixture
def moved(tmp_path):
    """A data file of water and three dumps of it, the atoms in shuffled
    order, moving in a box that grows; the second frame without atom 5."""
    mdapy.write(tmp_path / "water.data", water(), atom_style="full")
    topology = mdapy.read(tmp_path / "water.data")
    rng = np.random.default_rng(1)
    for step in range(3):
        order = rng.permutation(len(topology))
        if step == 1:
            order = order[np.asarray(topology["id"])[order] != 5]
        frame = topology.gather(order).select("id", "type", "positions")
        frame = frame.with_columns(
            positions=frame["positions"] + rng.normal(0, 0.1, (len(frame), 3))
        )
        box = mdapy.Box(15.0 + step, pbc=(True, True, False))
        mdapy.write(
            tmp_path / f"water.{step}.dump",
            frame.with_box(box).with_info(timestep=step),
        )
    return topology, tmp_path / "water.*.dump"


def test_frames_of_a_topology(moved):
    """Each frame in the order of the topology, with its bonds, angles,
    charges and molecules; atoms missing from a frame left out."""
    topology, pattern = moved
    plain = mdapy.read_trajectory(pattern)
    trajectory = mdapy.read_trajectory(pattern, topology=topology)
    for step, (frame, raw) in enumerate(zip(trajectory, plain)):
        ids = np.asarray(frame["id"])
        expected = np.asarray(topology["id"])
        expected = expected[expected != 5] if step == 1 else expected
        np.testing.assert_array_equal(ids, expected)
        where = np.argsort(np.asarray(raw["id"]))
        rows = where[np.searchsorted(np.asarray(raw["id"])[where], ids)]
        np.testing.assert_array_equal(frame["positions"], raw["positions"][rows])
        by_id = {i: k for k, i in enumerate(np.asarray(topology["id"]))}
        source = [by_id[i] for i in ids]
        for column in ("charge", "molecule", "element"):
            assert (
                frame[column].tolist() == np.asarray(topology[column])[source].tolist()
            )
        np.testing.assert_array_equal(frame.box.matrix, raw.box.matrix)
        assert frame.box.pbc.tolist() == [True, True, True]  # of the topology
        assert frame.info["timestep"] == step
        assert "image" not in frame.table.columns  # stale flags go
        pairs = {tuple(ids[list(b)]) for b in frame.topology.bonds["atoms"].to_list()}
        every = {
            tuple(np.asarray(topology["id"])[list(b)])
            for b in topology.topology.bonds["atoms"].to_list()
        }
        assert pairs == {b for b in every if 5 not in b or step != 1}
        if step != 1:
            assert frame.topology is topology.topology  # shared, not copied
        else:
            assert len(frame.topology.angles) == len(topology.topology.angles) - 1


def test_frames_of_a_topology_as_ovito(moved):
    pytest.importorskip("ovito")
    from ovito.io import import_file
    from ovito.modifiers import LoadTrajectoryModifier

    topology, pattern = moved
    pipeline = import_file(
        str(pattern).replace("water.*.dump", "water.data"), atom_style="full"
    )
    loader = LoadTrajectoryModifier()
    loader.source.load(str(pattern))
    pipeline.modifiers.append(loader)
    ours = mdapy.read_trajectory(pattern, topology=topology)
    assert loader.source.num_frames == len(ours) == 3
    for step, frame in enumerate(ours):
        data = pipeline.compute(step)
        particles = data.particles
        np.testing.assert_array_equal(frame["id"], particles.identifiers)
        np.testing.assert_allclose(frame["positions"], particles.positions, atol=1e-12)
        np.testing.assert_allclose(frame["charge"], particles["Charge"], atol=1e-12)
        np.testing.assert_array_equal(
            frame["molecule"], particles["Molecule Identifier"]
        )
        np.testing.assert_allclose(frame.box.matrix, np.asarray(data.cell)[:, :3].T)
        ours_bonds = sorted(
            map(tuple, np.sort(frame.topology.bonds["atoms"].to_list()))
        )
        theirs = sorted(map(tuple, np.sort(np.asarray(particles.bonds.topology))))
        assert ours_bonds == theirs


def test_frames_that_do_not_fit_the_topology(moved, tmp_path):
    topology, pattern = moved
    extra = mdapy.read(tmp_path / "water.0.dump")
    extra = extra.with_columns(
        id=np.where(np.asarray(extra["id"]) == 3, 99, extra["id"])
    )
    mdapy.write(tmp_path / "extra.dump", extra)
    with pytest.raises(InvalidParameter, match="not in the topology, such as id 99"):
        mdapy.read_trajectory(tmp_path / "extra.dump", topology=topology)[0]
    twice = extra.with_columns(id=np.ones(len(extra), dtype=np.int64))
    mdapy.write(tmp_path / "twice.dump", twice)
    with pytest.raises(InvalidParameter, match="repeat ids"):
        mdapy.read_trajectory(tmp_path / "twice.dump", topology=topology)[0]
    with pytest.raises(InvalidParameter, match="the atoms of the topology repeat"):
        mdapy.read_trajectory(pattern, topology=twice)[0]
    # frames without ids: the same number of atoms, the topology numbered
    dumped = mdapy.read(tmp_path / "water.0.dump")
    plain = mdapy.System(dumped["positions"], dumped.box, elements=["O", "H", "H"] * 4)
    mdapy.write(tmp_path / "plain.xyz", plain)
    frame = mdapy.read_trajectory(tmp_path / "plain.xyz", topology=topology)[0]
    np.testing.assert_array_equal(frame["positions"], plain["positions"])
    assert frame.topology is topology.topology
    mdapy.write(tmp_path / "short.xyz", plain.gather(np.arange(5)))
    with pytest.raises(InvalidParameter, match="without ids they must be the same"):
        mdapy.read_trajectory(tmp_path / "short.xyz", topology=topology)[0]
    renumbered = topology.with_columns(id=np.asarray(topology["id"]) + 10)
    with pytest.raises(InvalidParameter, match="not 1 to N in order"):
        mdapy.read_trajectory(tmp_path / "plain.xyz", topology=renumbered)[0]
    with pytest.raises(InvalidParameter, match="those of the topology not"):
        mdapy.read_trajectory(pattern, topology=topology.drop("id"))[0]


def test_frames_through_a_function(tmp_path):
    """map: each frame through the function when it is used, as indices,
    slices, iteration and the analyses of trajectories take it."""
    mdapy.write(tmp_path / "t.dump", frames(6))
    trajectory = mdapy.read_trajectory(tmp_path / "t.dump")
    calls = []

    def shifted(frame):
        calls.append(frame.info["timestep"])
        return frame.with_columns(positions=frame["positions"] + 1.0)

    mapped = trajectory.map(shifted)
    assert len(mapped) == 6 and calls == []
    np.testing.assert_allclose(mapped[2]["positions"], trajectory[2]["positions"] + 1)
    np.testing.assert_allclose(mapped[-1]["positions"], trajectory[5]["positions"] + 1)
    assert [f.info["timestep"] for f in mapped[1::2]] == [1, 3, 5]
    assert isinstance(mapped[1::2], mdapy.io.Mapped)
    calls.clear()
    assert len(list(mapped)) == 6 and calls == [0, 1, 2, 3, 4, 5]
    twice = mapped.map(shifted)
    np.testing.assert_allclose(twice[0]["positions"], trajectory[0]["positions"] + 2)
    mean = mdapy.analysis.TimeAverage("positions", unwrap=False).compute(mapped)
    np.testing.assert_allclose(
        mean.value,
        mdapy.analysis.TimeAverage("positions", unwrap=False).compute(trajectory).value
        + 1,
    )
    assert len(mdapy.transform.smooth_trajectory(mapped, 3)) == 6
    assert len(mdapy.transform.remove_drift(mapped)) == 6
    assert "shifted" in repr(mapped)
    with pytest.raises(IndexError):
        mapped[6]
