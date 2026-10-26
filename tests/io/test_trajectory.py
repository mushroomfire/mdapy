# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.errors import FileFormatError, UnsupportedFormat
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
    assert trajectory._frames.source._map.closed
    path.unlink()


def test_unknown_formats(tmp_path):
    with pytest.raises(UnsupportedFormat, match="cannot tell the format"):
        mdapy.read_trajectory(tmp_path / "t.unknown")


def test_top_level_names():
    assert mdapy.read_trajectory is mdapy.io.read_trajectory
    assert mdapy.Trajectory is mdapy.io.Trajectory
