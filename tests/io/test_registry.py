# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import bz2
import gzip
import lzma
import sys

import numpy as np
import polars as pl
import pytest

import mdapy
from mdapy.errors import FileFormatError, InvalidParameter, UnsupportedFormat
from mdapy.io import _registry
from mdapy.io._files import Source, open_output
from mdapy.io._registry import FORMATS, Format, register


def water(shift=0.0):
    return mdapy.System(
        positions=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
        + shift,
        box=mdapy.Box(10.0),
        elements=["O", "H", "H"],
        info={"energy": -14.2 - shift},
    )


@pytest.fixture
def single_frame_format(monkeypatch):
    """A format that holds one frame: the text 'water'."""

    def read(source, position):
        assert position == (0, 0)
        assert source.reader().readline() == "water"
        return water()

    def write(file, systems):
        for _ in systems:
            file.write(b"water\n")

    found = Format(
        "single", "test format", (".single",), ("WATER*",), read, None, write
    )
    monkeypatch.setitem(FORMATS, "single", found)
    return found


def test_top_level_names():
    assert mdapy.read is mdapy.io.read
    assert mdapy.write is mdapy.io.write
    assert mdapy.formats is mdapy.io.formats
    assert mdapy.detect_format is mdapy.io.detect_format


def test_formats_table():
    table = mdapy.formats()
    assert table.schema == pl.Schema(
        {
            "name": pl.String,
            "description": pl.String,
            "extensions": pl.List(pl.String),
            "read": pl.Boolean,
            "write": pl.Boolean,
            "multiple_frames": pl.Boolean,
        }
    )
    row = table.filter(pl.col("name") == "extxyz").row(0, named=True)
    assert row["extensions"] == [".xyz", ".extxyz"]
    assert row["read"] and row["write"] and row["multiple_frames"]


@pytest.mark.parametrize(
    "name",
    [
        "a.xyz",
        "A.XYZ",
        "dir/a.extxyz",
        "a.xyz.gz",
        "a.xyz.BZ2",
        "a.xyz.xz",
        "a.xyz.zst",
    ],
)
def test_detect_format_from_extensions(name):
    assert mdapy.detect_format(name) == "extxyz"


def test_detect_format_from_patterns(single_frame_format):
    assert mdapy.detect_format("WATER_1") == "single"
    assert mdapy.detect_format("dir/WATER.gz") == "single"
    assert mdapy.detect_format("b.single") == "single"
    with pytest.raises(UnsupportedFormat, match="cannot tell the format"):
        mdapy.detect_format("water")  # patterns are case-sensitive


@pytest.mark.parametrize("name", ["a.unknown", "noextension", ".gz", "a.gz"])
def test_detect_format_unknown(name):
    with pytest.raises(UnsupportedFormat, match="pass format=, one of cif, extxyz"):
        mdapy.detect_format(name)


def test_detect_format_ambiguous(monkeypatch):
    monkeypatch.setitem(FORMATS, "other", Format("other", "", (".xyz",)))
    with pytest.raises(UnsupportedFormat, match="fits the formats extxyz, other"):
        mdapy.detect_format("a.xyz")


def test_register_rejects_duplicates():
    with pytest.raises(ValueError, match="already registered"):
        register(FORMATS["extxyz"])


def test_register_adds_a_format(monkeypatch):
    monkeypatch.setattr(_registry, "FORMATS", {})
    found = register(Format("new", "a new format"))
    assert _registry.FORMATS == {"new": found}


def test_unknown_or_unsupported_formats(tmp_path, monkeypatch):
    path = tmp_path / "a.xyz"
    mdapy.write(path, water())
    with pytest.raises(UnsupportedFormat, match="unknown format 'nope'"):
        mdapy.read(path, format="nope")
    with pytest.raises(UnsupportedFormat, match="unknown format 'nope'"):
        mdapy.write(path, water(), format="nope")
    monkeypatch.setitem(
        FORMATS, "readonly", Format("readonly", "", read=lambda s, p: 1)
    )
    monkeypatch.setitem(FORMATS, "writeonly", Format("writeonly", "", write=print))
    with pytest.raises(UnsupportedFormat, match="mdapy cannot write readonly files"):
        mdapy.write(path, water(), format="readonly")
    with pytest.raises(UnsupportedFormat, match="mdapy cannot read writeonly files"):
        mdapy.read(path, format="writeonly")


def test_frame_selection(tmp_path):
    path = tmp_path / "t.xyz"
    mdapy.write(path, [water(k) for k in range(3)])
    for frame, shift in [(0, 0), (1, 1), (2, 2), (-1, 2), (-3, 0)]:
        assert mdapy.read(path, frame=frame).info["energy"] == -14.2 - shift
    assert mdapy.read(path, frame=np.int64(1)).info["energy"] == -15.2
    for frame in (3, -4):
        with pytest.raises(InvalidParameter, match=f"frame {frame} does not exist"):
            mdapy.read(path, frame=frame)
    with pytest.raises(TypeError):
        mdapy.read(path, frame=1.0)


def test_empty_file_has_no_frames(tmp_path):
    path = tmp_path / "empty.xyz"
    path.write_bytes(b"")
    with pytest.raises(
        InvalidParameter, match="frame 0 does not exist: the file has 0"
    ):
        mdapy.read(path)


def test_single_frame_formats(tmp_path, single_frame_format):
    path = tmp_path / "a.single"
    mdapy.write(path, water())
    assert path.read_bytes() == b"water\n"
    mdapy.write(path, (s for s in [water()]))
    assert mdapy.read(path).info["energy"] == -14.2
    assert mdapy.read(path, frame=-1).info["energy"] == -14.2
    with pytest.raises(InvalidParameter, match="hold one frame, so frame must be 0"):
        mdapy.read(path, frame=1)
    with pytest.raises(InvalidParameter, match="hold one frame, but 2 systems"):
        mdapy.write(path, [water(), water()])


def test_write_checks_the_systems(tmp_path):
    path = tmp_path / "a.xyz"
    with pytest.raises(TypeError, match="not str"):
        mdapy.write(path, ["water"])
    assert not path.exists()


def test_failed_writes_leave_no_file(tmp_path):
    path = tmp_path / "a.xyz"
    path.write_text("old")
    no_elements = mdapy.System([[0.0, 0.0, 0.0]], mdapy.Box(5.0))
    with pytest.raises(mdapy.errors.MissingProperty):
        mdapy.write(path, [water(), no_elements])
    assert not path.exists()


def test_options_reach_the_format(tmp_path):
    path = tmp_path / "a.xyz"
    mdapy.write(path, water().with_columns(charge=[-0.8, 0.4, 0.4]), columns=[])
    assert "charge" not in mdapy.read(path).table.columns
    with pytest.raises(TypeError, match="unexpected keyword"):
        mdapy.read(path, columns=[])


# --------------------------------------------------------------- files


@pytest.mark.parametrize("suffix", [".gz", ".bz2", ".xz", ".GZ"])
def test_compressed_round_trip(tmp_path, suffix):
    path = tmp_path / f"a.xyz{suffix}"
    systems = [water(k) for k in range(3)]
    mdapy.write(path, systems)
    opener = {".gz": gzip, ".bz2": bz2, ".xz": lzma}[suffix.lower()]
    with opener.open(path, "rb") as f:
        assert f.read().startswith(b"3\nLattice=")
    assert mdapy.read(path, frame=2).info["energy"] == -16.2
    # Compression is recognized from the content, not the name.
    renamed = path.rename(tmp_path / "renamed.xyz")
    assert mdapy.read(renamed, frame=1).info["energy"] == -15.2


@pytest.mark.skipif(sys.version_info < (3, 14), reason="zstd needs Python 3.14")
def test_zstd_round_trip(tmp_path):
    path = tmp_path / "a.xyz.zst"
    mdapy.write(path, [water(1)])
    assert path.read_bytes()[:4] == b"\x28\xb5\x2f\xfd"
    assert mdapy.read(path).info["energy"] == -15.2


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="zstd is available")
def test_zstd_needs_python_3_14(tmp_path):
    with pytest.raises(UnsupportedFormat, match="Python 3.14"):
        mdapy.write(tmp_path / "a.xyz.zst", water())
    assert not (tmp_path / "a.xyz.zst").exists()
    path = tmp_path / "a.xyz"
    path.write_bytes(b"\x28\xb5\x2f\xfd" + b"\0" * 20)
    with pytest.raises(UnsupportedFormat, match="Python 3.14"):
        mdapy.read(path)


def test_corrupt_compressed_files(tmp_path):
    path = tmp_path / "a.xyz"
    path.write_bytes(b"\x1f\x8b" + b"not gzip data")
    with pytest.raises(FileFormatError, match="cannot decompress the gzip data") as e:
        mdapy.read(path)
    assert e.value.path == str(path) and e.value.line is None


def test_source_maps_plain_files_and_closes_them(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"one\ntwo\n")
    with Source(path) as source:
        assert source.name == str(path)
        reader = source.reader(4, 1)
        assert reader.readline() == "two" and reader.line == 2
        mapped = source._map
        assert not mapped.closed
    assert mapped.closed and source._file.closed
    path.unlink()  # possible on every platform once the file is closed
    empty = tmp_path / "empty.txt"
    empty.write_bytes(b"")
    with Source(empty) as source:
        assert source.buffer == b"" and source.reader().at_end()


def test_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        mdapy.read(tmp_path / "missing.xyz")


def test_open_output_removes_the_file_on_errors(tmp_path):
    path = tmp_path / "out.txt.gz"
    with pytest.raises(RuntimeError):
        with open_output(path) as f:
            f.write(b"partial")
            raise RuntimeError("stop")
    assert not path.exists()
    with open_output(path) as f:
        f.write(b"done")
    assert gzip.decompress(path.read_bytes()) == b"done"
