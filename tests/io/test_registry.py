# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import bz2
import errno
import gzip
import io
import lzma
import shutil
import struct
import subprocess
import sys
import zlib

import numpy as np
import polars as pl
import pytest

import mdapy
from mdapy.core.threads import get_num_threads, set_num_threads
from mdapy.errors import FileFormatError, InvalidParameter, UnsupportedFormat
from mdapy.io import _files, _registry
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
    assert mdapy.io.read is mdapy.io.read
    assert mdapy.io.write is mdapy.io.write
    assert mdapy.io.formats is mdapy.io.formats
    assert mdapy.io.detect_format is mdapy.io.detect_format


def test_formats_table():
    table = mdapy.io.formats()
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
    assert mdapy.io.detect_format(name) == "extxyz"


def test_detect_format_from_patterns(single_frame_format):
    assert mdapy.io.detect_format("WATER_1") == "single"
    assert mdapy.io.detect_format("dir/WATER.gz") == "single"
    assert mdapy.io.detect_format("b.single") == "single"
    with pytest.raises(UnsupportedFormat, match="cannot tell the format"):
        mdapy.io.detect_format("water")  # patterns are case-sensitive


@pytest.mark.parametrize("name", ["a.unknown", "noextension", ".gz", "a.gz"])
def test_detect_format_unknown(name):
    with pytest.raises(UnsupportedFormat, match="pass format=, one of cif, extxyz"):
        mdapy.io.detect_format(name)


def test_detect_format_ambiguous(monkeypatch):
    monkeypatch.setitem(FORMATS, "other", Format("other", "", (".xyz",)))
    with pytest.raises(UnsupportedFormat, match="fits the formats extxyz, other"):
        mdapy.io.detect_format("a.xyz")


def test_register_rejects_duplicates():
    with pytest.raises(ValueError, match="already registered"):
        register(FORMATS["extxyz"])


def test_register_adds_a_format(monkeypatch):
    monkeypatch.setattr(_registry, "FORMATS", {})
    found = register(Format("new", "a new format"))
    assert _registry.FORMATS == {"new": found}


def test_unknown_or_unsupported_formats(tmp_path, monkeypatch):
    path = tmp_path / "a.xyz"
    mdapy.io.write(path, water())
    with pytest.raises(UnsupportedFormat, match="unknown format 'nope'"):
        mdapy.io.read(path, format="nope")
    with pytest.raises(UnsupportedFormat, match="unknown format 'nope'"):
        mdapy.io.write(path, water(), format="nope")
    monkeypatch.setitem(
        FORMATS, "readonly", Format("readonly", "", read=lambda s, p: 1)
    )
    monkeypatch.setitem(FORMATS, "writeonly", Format("writeonly", "", write=print))
    with pytest.raises(UnsupportedFormat, match="MDAPY cannot write readonly files"):
        mdapy.io.write(path, water(), format="readonly")
    with pytest.raises(UnsupportedFormat, match="MDAPY cannot read writeonly files"):
        mdapy.io.read(path, format="writeonly")


def test_frame_selection(tmp_path):
    path = tmp_path / "t.xyz"
    mdapy.io.write(path, [water(k) for k in range(3)])
    for frame, shift in [(0, 0), (1, 1), (2, 2), (-1, 2), (-3, 0)]:
        assert mdapy.io.read(path, frame=frame).info["energy"] == -14.2 - shift
    assert mdapy.io.read(path, frame=np.int64(1)).info["energy"] == -15.2
    for frame in (3, -4):
        with pytest.raises(InvalidParameter, match=f"frame {frame} does not exist"):
            mdapy.io.read(path, frame=frame)
    with pytest.raises(TypeError):
        mdapy.io.read(path, frame=1.0)


def test_empty_file_has_no_frames(tmp_path):
    path = tmp_path / "empty.xyz"
    path.write_bytes(b"")
    with pytest.raises(
        InvalidParameter, match="frame 0 does not exist: the file has 0"
    ):
        mdapy.io.read(path)


def test_single_frame_formats(tmp_path, single_frame_format):
    path = tmp_path / "a.single"
    mdapy.io.write(path, water())
    assert path.read_bytes() == b"water\n"
    mdapy.io.write(path, (s for s in [water()]))
    assert mdapy.io.read(path).info["energy"] == -14.2
    assert mdapy.io.read(path, frame=-1).info["energy"] == -14.2
    with pytest.raises(InvalidParameter, match="hold one frame, so frame must be 0"):
        mdapy.io.read(path, frame=1)
    with pytest.raises(InvalidParameter, match="hold one frame, but 2 systems"):
        mdapy.io.write(path, [water(), water()])


def test_write_checks_the_systems(tmp_path):
    path = tmp_path / "a.xyz"
    with pytest.raises(TypeError, match="not str"):
        mdapy.io.write(path, ["water"])
    assert not path.exists()


def test_failed_writes_keep_the_file_there(tmp_path):
    """A write that fails leaves a file already at the path as it was, and
    no file where there was none."""
    no_elements = [water(), mdapy.System([[0.0, 0.0, 0.0]], mdapy.Box(5.0))]
    same_ids = water().with_columns(id=[1, 1, 1])
    for name, failing, error in [
        ("a.xyz", no_elements, mdapy.errors.MissingProperty),
        ("a.xyz.gz", no_elements, mdapy.errors.MissingProperty),
        ("a.xyz.bz2", no_elements, mdapy.errors.MissingProperty),
        ("a.data", same_ids, InvalidParameter),
    ]:
        path = tmp_path / name
        path.write_bytes(b"old")
        with pytest.raises(error):
            mdapy.io.write(path, failing)
        assert path.read_bytes() == b"old"
        with pytest.raises(error):
            mdapy.io.write(tmp_path / ("b" + name[1:]), failing)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["a.data", "a.xyz", "a.xyz.bz2", "a.xyz.gz"]


def test_options_reach_the_format(tmp_path):
    path = tmp_path / "a.xyz"
    mdapy.io.write(path, water().with_columns(charge=[-0.8, 0.4, 0.4]), columns=[])
    assert "charge" not in mdapy.io.read(path).table.columns
    with pytest.raises(TypeError, match="unexpected keyword"):
        mdapy.io.read(path, columns=[])


# --------------------------------------------------------------- files


@pytest.mark.parametrize("suffix", [".gz", ".bz2", ".xz", ".GZ"])
def test_compressed_round_trip(tmp_path, suffix):
    path = tmp_path / f"a.xyz{suffix}"
    systems = [water(k) for k in range(3)]
    mdapy.io.write(path, systems)
    opener = {".gz": gzip, ".bz2": bz2, ".xz": lzma}[suffix.lower()]
    with opener.open(path, "rb") as f:
        assert f.read().startswith(b"3\nLattice=")
    assert mdapy.io.read(path, frame=2).info["energy"] == -16.2
    # Compression is recognized from the content, not the name.
    renamed = path.rename(tmp_path / "renamed.xyz")
    assert mdapy.io.read(renamed, frame=1).info["energy"] == -15.2


@pytest.mark.skipif(sys.version_info < (3, 14), reason="zstd needs Python 3.14")
def test_zstd_round_trip(tmp_path):
    path = tmp_path / "a.xyz.zst"
    mdapy.io.write(path, [water(1)])
    assert path.read_bytes()[:4] == b"\x28\xb5\x2f\xfd"
    assert mdapy.io.read(path).info["energy"] == -15.2


@pytest.mark.skipif(sys.version_info >= (3, 14), reason="zstd is available")
def test_zstd_needs_python_3_14(tmp_path):
    with pytest.raises(UnsupportedFormat, match="Python 3.14"):
        mdapy.io.write(tmp_path / "a.xyz.zst", water())
    assert not (tmp_path / "a.xyz.zst").exists()
    path = tmp_path / "a.xyz"
    path.write_bytes(b"\x28\xb5\x2f\xfd" + b"\0" * 20)
    with pytest.raises(UnsupportedFormat, match="Python 3.14"):
        mdapy.io.read(path)


def test_corrupt_compressed_files(tmp_path):
    path = tmp_path / "a.xyz"
    path.write_bytes(b"\x1f\x8b" + b"not gzip data")
    with pytest.raises(FileFormatError, match="cannot decompress the gzip data") as e:
        mdapy.io.read(path)
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
        mdapy.io.read(tmp_path / "missing.xyz")


def one_stream(data):
    """The text of a gzip file that holds exactly one stream."""
    stream = zlib.decompressobj(31)
    text = stream.decompress(data)
    assert stream.eof and stream.unused_data == b""
    return text


def many_blocks(blocks=12):
    rng = np.random.default_rng(3)
    numbers = rng.normal(0.0, 100.0, int(blocks * _files._GZIP_BLOCK / 20))
    return "\n".join(map(repr, numbers.tolist())).encode()


@pytest.mark.parametrize("threads", [1, 4])
def test_gzip_is_one_stream_of_many_blocks(tmp_path, threads):
    # some programs stop at the end of the first stream of a file
    text = many_blocks()
    assert len(text) > 8 * _files._GZIP_BLOCK
    path = tmp_path / "a.txt.gz"
    before = get_num_threads()
    try:
        set_num_threads(threads)
        with open_output(path) as file:
            file.write(text[:100])  # a header, as the formats write one
            for at in range(100, len(text), 700000):
                assert file.write(text[at : at + 700000]) == len(text[at : at + 700000])
    finally:
        set_num_threads(before)
    data = path.read_bytes()
    assert one_stream(data) == text
    assert gzip.decompress(data) == text
    assert data[:4] == b"\x1f\x8b\x08\x00"
    crc, size = struct.unpack("<II", data[-8:])
    assert crc == zlib.crc32(text) and size == len(text)


def test_gzip_has_the_same_bytes_for_any_number_of_threads(tmp_path):
    text = many_blocks()
    written = []
    before = get_num_threads()
    try:
        for threads in (1, 2, 4):
            set_num_threads(threads)
            path = tmp_path / f"{threads}.gz"
            with open_output(path) as file:
                file.write(text)
            written.append(path.read_bytes())
    finally:
        set_num_threads(before)
    assert written[0] == written[1] == written[2]


def test_gzip_is_as_small_as_with_one_thread(tmp_path):
    # each block looks back into the one before it
    path = tmp_path / "a.gz"
    text = (b"1 1 0.5 0.25 3.615\n" * 200000)[: 3 * _files._GZIP_BLOCK + 5]
    with open_output(path) as file:
        file.write(text)
    assert path.stat().st_size < 1.01 * len(gzip.compress(text, 6))


@pytest.mark.parametrize("size", [0, 1, 5, 2**15, 2**20 - 1, 2**20, 2**20 + 1, 2**21])
def test_gzip_of_any_size(tmp_path, size):
    text = many_blocks(3)[:size]
    path = tmp_path / "a.gz"
    with open_output(path) as file:
        file.write(text)
    assert one_stream(path.read_bytes()) == text


def test_gzip_is_read_by_the_gzip_program(tmp_path):
    program = shutil.which("gzip")
    if program is None:
        pytest.skip("no gzip program")
    text = many_blocks(3)
    path = tmp_path / "a.gz"
    with open_output(path) as file:
        file.write(text)
    result = subprocess.run(
        [program, "-dc", str(path)], capture_output=True, check=True
    )
    assert result.stdout == text


class _FailingFile(io.BytesIO):
    """A file that takes a few writes, as a disk that fills up."""

    def __init__(self, writes):
        super().__init__()
        self.writes = writes

    def write(self, data):
        self.writes -= 1
        if self.writes < 0:
            raise OSError(errno.ENOSPC, "No space left on device")
        return super().write(data)


@pytest.mark.parametrize("threads", [1, 4])
def test_gzip_that_cannot_be_written_raises(threads):
    text = many_blocks(40)
    before = get_num_threads()
    try:
        set_num_threads(threads)
        target = _FailingFile(3)
        writer = _files._GzipWriter(target)
        with pytest.raises(OSError, match="No space left"):
            writer.write(text)
        with pytest.raises(OSError, match="No space left"):
            writer.close()
        assert writer.closed and target.closed
        writer.close()  # closing twice does nothing
    finally:
        set_num_threads(before)


def test_open_output_removes_a_file_that_cannot_be_finished(tmp_path, monkeypatch):
    path = tmp_path / "a.gz"

    close = _files._GzipWriter.close

    def fail(self):
        close(self)
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(_files._GzipWriter, "close", fail)
    with pytest.raises(OSError, match="No space left"):
        with open_output(path) as file:
            file.write(b"text")
    assert not path.exists()


def test_formats_are_written_through_gzip(tmp_path):
    system = water(1)
    for name in ("a.xyz.gz", "a.dump.gz", "a.data.gz", "a.mdapy.gz"):
        path = tmp_path / name
        mdapy.io.write(path, [system, system] if "data" not in name else system)
        assert one_stream(path.read_bytes())
        np.testing.assert_allclose(
            mdapy.io.read(path)["positions"], system["positions"]
        )


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
