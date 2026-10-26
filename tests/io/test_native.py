# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import gzip
import io
import json
import struct
import zipfile

import numpy as np
import polars as pl
import pytest

import mdapy
from mdapy.errors import FileFormatError, InvalidParameter, UnsupportedFormat
from mdapy.io import _native


def rich_system(seed=0, n=12):
    rng = np.random.default_rng(seed)
    box = mdapy.Box(
        [[5.0, 0, 0], [1.0, 6.0, 0], [0.5, -0.5, 7.0]],
        origin=[1, -2, 3],
        pbc=[True, False, True],
    )
    bonds = pl.DataFrame(
        {
            "atoms": [[0, 1], [1, 2]],
            "type": [1, 2],
            "type_label": ["a", "b"],
            "order": [1.0, 2.0],
        }
    )
    system = mdapy.System(
        box.to_cartesian(rng.random((n, 3))),
        box,
        elements=rng.choice(["Cu", "Zr", "O"], n),
        topology=mdapy.Topology(bonds=bonds, angles=[[0, 1, 2]]),
        info={
            "energy": -1.25,
            "step": 7,
            "converged": True,
            "name": "test",
            "nothing": None,
            "virial": rng.standard_normal((3, 3)),
            "small": np.arange(4, dtype=np.int16),
            "single": np.float32(1.5),
            "flags": np.array([True, False]),
            "words": np.array(["a", "bc"]),
            "scalar_array": np.array(2.5),
            "empty": np.empty((0, 3)),
            "special": [float("nan"), float("inf"), -0.0],
            "nested": {"cutoff": 5.0, "list": [1, [2, 3]], "array": np.ones(2)},
        },
    )
    return system.with_columns(
        velocities=rng.standard_normal((n, 3)),
        type=rng.integers(1, 4, n),
        small=pl.Series(rng.random(n), dtype=pl.Float32),
        tiny=pl.Series(rng.integers(0, 9, n), dtype=pl.UInt8),
        label=pl.Series(rng.choice(["x", "y z"], n)),
        tensors=rng.standard_normal((n, 3, 3)),
        flag=rng.random(n) < 0.5,
    )


def assert_same_info(a, b):
    assert set(a) == set(b)
    for key in a:
        x, y = a[key], b[key]
        if isinstance(x, np.ndarray):
            assert (
                isinstance(y, np.ndarray) and x.dtype == y.dtype and x.shape == y.shape
            )
            np.testing.assert_array_equal(x, y)
        elif key == "special":
            assert np.isnan(y[0]) and y[1:] == (float("inf"), -0.0)
            assert np.copysign(1.0, y[2]) == -1.0
        elif key == "nested":
            assert y["cutoff"] == 5.0 and y["list"] == (1, (2, 3))
            np.testing.assert_array_equal(y["array"], np.ones(2))
        else:
            assert x == y and type(x) is type(y), key


def test_round_trip_is_exact(tmp_path):
    system = rich_system()
    path = tmp_path / "a.mdapy"
    mdapy.write(path, system)
    back = mdapy.read(path)
    assert back.table.equals(system.table)
    assert back.table.schema == system.table.schema  # Enum, Float32, UInt8, Array kept
    assert back.box == system.box
    assert back.topology == system.topology
    assert back.topology.bonds.columns == ["atoms", "type", "type_label", "order"]
    expected = dict(system.info)
    expected["single"] = 1.5  # NumPy scalars become Python numbers
    assert_same_info(expected, back.info)


def test_frames_share_topologies_and_have_typed_columns(tmp_path):
    base = rich_system(n=3)
    other = base.with_topology(mdapy.Topology(bonds=[[0, 2]]))
    frames = [
        base.with_info(energy=-1.0, step=0),
        base.with_info(energy=-2.0, step=1, name=None),
        other.with_info(energy=-3.0, step=2),
        base.with_topology(None).with_info(energy=-4.0, step=3, name=4),
        base.with_info(energy=-5.0, step=4),
    ]
    path = tmp_path / "t.mdapy"
    mdapy.write(path, frames)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        table = pl.read_parquet(io.BytesIO(archive.read("frames.parquet")))
        manifest = json.loads(archive.read("manifest.json"))
    assert sorted(n for n in names if n.endswith("bonds.parquet")) == [
        "topology/00000000/bonds.parquet",
        "topology/00000001/bonds.parquet",
        "topology/00000002/bonds.parquet",  # base again, after other
    ]
    assert table["topology"].to_list() == [
        "topology/00000000", "topology/00000000", "topology/00000001", None,
        "topology/00000002",
    ]  # fmt: skip
    assert table["info.energy"].to_list() == [-1.0, -2.0, -3.0, -4.0, -5.0]
    assert table["info.step"].dtype == pl.Int64
    assert table["info.virial"].dtype == pl.Array(pl.Float64, (3, 3))
    assert "info.name" not in table.columns  # missing in one frame, an int in another
    assert "info.empty" not in table.columns  # no fixed shape to store
    assert manifest["format"] == "mdapy" and manifest["version"] == 1
    assert manifest["frames"] == 5 and manifest["units"]["stress"] == "GPa"
    for k, frame in enumerate(frames):
        back = mdapy.read(path, frame=k)
        assert back.table.equals(frame.table)
        assert back.topology == frame.topology
        assert back.info["energy"] == frame.info["energy"]


def test_mixed_and_irregular_info_values_have_no_typed_columns(tmp_path):
    base = mdapy.System([[0.0, 0, 0]], 5.0, elements="H")
    frames = [
        base.with_info(mixed=1, shape=np.zeros(2), dtype=np.zeros(2), text="a"),
        base.with_info(
            mixed=1.5, shape=np.zeros(3), dtype=np.zeros(2, dtype=int), text="b"
        ),
    ]
    path = tmp_path / "m.mdapy"
    mdapy.write(path, frames)
    with zipfile.ZipFile(path) as archive:
        table = pl.read_parquet(io.BytesIO(archive.read("frames.parquet")))
    assert [c for c in table.columns if c.startswith("info.")] == ["info.text"]


def test_members_are_stored_and_reproducible(tmp_path):
    system = rich_system()
    first, second = tmp_path / "1.mdapy", tmp_path / "2.mdapy"
    mdapy.write(first, [system, system])
    mdapy.write(second, [system, system])
    with zipfile.ZipFile(first) as a, zipfile.ZipFile(second) as b:
        assert all(i.compress_type == zipfile.ZIP_STORED for i in a.infolist())
        for name in a.namelist():
            if name != "manifest.json":  # it holds the time of writing
                assert a.read(name) == b.read(name), name


def test_compressed_files(tmp_path):
    path = tmp_path / "c.mdapy.gz"
    system = rich_system()
    mdapy.write(path, system)
    assert gzip.decompress(path.read_bytes())[:4] == b"PK\x03\x04"
    assert mdapy.read(path).table.equals(system.table)


def truncated(path, before):
    data = path.read_bytes()
    cut = path.with_name("cut.mdapy")
    cut.write_bytes(data[: data.index(before.encode()) - 30])
    return cut


def test_recovery_of_files_that_were_not_closed(tmp_path):
    frames = [rich_system().with_info(step=k) for k in range(3)]
    path = tmp_path / "t.mdapy"
    mdapy.write(path, frames)
    cut = truncated(path, "frames.parquet")  # the writer stopped before closing
    with pytest.raises(zipfile.BadZipFile):
        zipfile.ZipFile(cut)
    for k in range(3):
        back = mdapy.read(cut, frame=k)
        assert back.table.equals(frames[k].table) and back.info["step"] == k
        assert back.topology == frames[k].topology
    cut = truncated(path, "atoms/00000002.parquet")  # before the third frame
    with pytest.raises(InvalidParameter, match="the file has 2 frames"):
        mdapy.read(cut, frame=2)
    data = path.read_bytes()
    middle = data.index(b"atoms/00000002.parquet") + 1000
    cut.write_bytes(data[:middle])  # in the middle of the third frame
    with pytest.raises(InvalidParameter, match="the file has 2 frames"):
        mdapy.read(cut, frame=2)


def test_recovery_of_zip64_members(tmp_path):
    system = rich_system()
    path = tmp_path / "z.mdapy"
    member = io.BytesIO()
    system.table.write_parquet(
        member,
        metadata={
            "mdapy": json.dumps(
                {
                    "frame": 0,
                    "box": _native._box(system.box),
                    "info": {},
                    "topology": None,
                }
            )
        },
    )
    info = zipfile.ZipInfo("atoms/00000000.parquet")
    # another extra field before the ZIP64 one: an extended timestamp
    info.extra = struct.pack("<HHB", 0x5455, 1, 0)
    with zipfile.ZipFile(path, "w") as archive:
        with archive.open(info, "w", force_zip64=True) as f:
            f.write(member.getvalue())
    data = path.read_bytes()
    path.write_bytes(data[: data.rindex(b"PK\x01\x02")])  # drop the central directory
    assert mdapy.read(path).table.equals(system.table)


@pytest.mark.parametrize(
    "content, error, message",
    [
        (b"not a zip file", FileFormatError, "not an mdapy file"),
        ({"manifest.json": b"{"}, FileFormatError, "not JSON"),
        (
            {"manifest.json": b'{"format": "other"}'},
            FileFormatError,
            "does not describe",
        ),
        (
            {"manifest.json": b'{"format": "mdapy", "version": 2}'},
            UnsupportedFormat,
            "version 2",
        ),
        ({"atoms/00000000.parquet": None}, FileFormatError, "has no mdapy metadata"),
    ],
)
def test_invalid_files(tmp_path, content, error, message):
    path = tmp_path / "bad.mdapy"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in content.items():
                if data is None:
                    buffer = io.BytesIO()
                    pl.DataFrame({"positions": [[0.0, 0, 0]]}).write_parquet(buffer)
                    data = buffer.getvalue()
                archive.writestr(name, data)
    with pytest.raises(error, match=message):
        mdapy.read(path)


@pytest.mark.parametrize(
    "info, message",
    [
        ({"objects": np.array([object()])}, "array of object"),
        ({"keys": {1: 2}}, "not strings"),
        ({"thing": object()}, "of type object cannot be written"),
    ],
)
def test_info_that_cannot_be_written(tmp_path, info, message):
    system = mdapy.System([[0.0, 0, 0]], 5.0, elements="H", info=info)
    with pytest.raises(InvalidParameter, match=message):
        mdapy.write(tmp_path / "i.mdapy", system)
    assert not (tmp_path / "i.mdapy").exists()


def test_format_detection():
    assert mdapy.detect_format("run.mdapy") == "mdapy"
    assert mdapy.detect_format("run.mdapy.zst") == "mdapy"


def test_buffer_file():
    file = _native._BufferFile(b"0123456789")
    assert file.seekable() and file.readable()
    assert file.seek(-3, 2) == 7 and file.read() == b"789"
    assert file.seek(2) == 2 and file.read(3) == b"234" and file.tell() == 5
    assert file.seek(1, 1) == 6 and file.read(10) == b"6789"
    with pytest.raises(OSError, match="negative"):
        file.seek(-11, 2)
