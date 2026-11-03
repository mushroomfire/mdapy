# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
"""write_movie against a stand-in for FFmpeg, and the real one where there is one."""

import json
import shutil
import sys

import numpy as np
import pytest

from mdapy.errors import CalculationFailed, InvalidParameter
from mdapy.visualization import write_movie

STAND_IN = """
import json, os, sys
if os.environ.get("STAND_IN") == "fail":
    sys.stderr.write("no such codec")
    sys.exit(3)
data = sys.stdin.buffer.read()
with open(sys.argv[-1], "w") as out:
    json.dump({"arguments": sys.argv[1:], "bytes": len(data)}, out)
"""


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    """A program called ffmpeg on the PATH, which notes what it was given."""
    folder = tmp_path / "bin"
    folder.mkdir()
    script = folder / "stand_in.py"
    script.write_text(STAND_IN)
    if sys.platform == "win32":
        program = folder / "ffmpeg.bat"
        program.write_text(f'@"{sys.executable}" "{script}" %*\n')
    else:
        program = folder / "ffmpeg"
        program.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        program.chmod(0o755)
    monkeypatch.setenv("PATH", str(folder))


def picture(channels=4, height=4, width=6):
    return np.zeros((height, width, channels), dtype=np.uint8)


@pytest.mark.parametrize("channels, encoding", [(4, "rgba"), (3, "rgb24")])
def test_pictures_go_to_ffmpeg(stand_in, tmp_path, channels, encoding):
    out = tmp_path / "movie.mp4"
    write_movie(out, (picture(channels) for _ in range(3)), fps=12)
    noted = json.loads(out.read_text())
    assert noted["bytes"] == 3 * 4 * 6 * channels
    arguments = noted["arguments"]
    assert arguments[arguments.index("-pix_fmt") + 1] == encoding
    assert arguments[arguments.index("-s") + 1] == "6x4"
    assert arguments[arguments.index("-framerate") + 1] == "12"
    assert arguments[-1] == str(out)


def test_a_list_of_pictures_at_the_default_rate(stand_in, tmp_path):
    out = tmp_path / "movie.mp4"
    write_movie(out, [picture(), picture()])
    arguments = json.loads(out.read_text())["arguments"]
    assert arguments[arguments.index("-framerate") + 1] == "30"


def test_a_picture_of_another_size_stops_the_movie(stand_in, tmp_path):
    out = tmp_path / "movie.mp4"
    with pytest.raises(InvalidParameter, match="after those of"):
        write_movie(out, [picture(), picture(width=8)])
    assert not out.exists()


def test_an_error_while_drawing_stops_the_movie(stand_in, tmp_path):
    def pictures():
        yield picture()
        raise ValueError("no more")

    out = tmp_path / "movie.mp4"
    with pytest.raises(ValueError, match="no more"):
        write_movie(out, pictures())
    assert not out.exists()


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((4, 6, 4), dtype=float),
        np.zeros((4, 6), dtype=np.uint8),
        np.zeros((4, 6, 2), dtype=np.uint8),
    ],
)
def test_what_is_no_picture(stand_in, tmp_path, bad):
    with pytest.raises(InvalidParameter, match="a picture must be"):
        write_movie(tmp_path / "movie.mp4", [bad])


def test_no_pictures(stand_in, tmp_path):
    with pytest.raises(InvalidParameter, match="no pictures"):
        write_movie(tmp_path / "movie.mp4", [])


@pytest.mark.parametrize("fps", [0, -1, float("nan")])
def test_a_rate_that_is_no_rate(stand_in, tmp_path, fps):
    with pytest.raises(InvalidParameter, match="fps"):
        write_movie(tmp_path / "movie.mp4", [picture()], fps=fps)


def test_what_ffmpeg_says_when_it_fails(stand_in, tmp_path, monkeypatch):
    monkeypatch.setenv("STAND_IN", "fail")
    # larger than a pipe holds, so that writing finds it gone
    large = picture(height=2000, width=2000)
    with pytest.raises(CalculationFailed, match="no such codec"):
        write_movie(tmp_path / "movie.mp4", [large, large])


def test_without_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="conda install ffmpeg"):
        write_movie(tmp_path / "movie.mp4", [picture()])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")
def test_a_real_movie_of_odd_size(tmp_path):
    out = tmp_path / "movie.mp4"
    rng = np.random.default_rng(1)
    write_movie(
        out,
        (rng.integers(0, 255, (33, 47, 4), dtype=np.uint8) for _ in range(5)),
        fps=5,
    )
    assert out.read_bytes()[4:8] == b"ftyp"
