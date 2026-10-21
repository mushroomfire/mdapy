# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""The mdapy file format: a ZIP container of Parquet tables."""

import datetime
import io
import json
import struct
import zipfile
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, BinaryIO

import numpy as np
import polars as pl

from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.core.topology import KINDS, Topology
from mdapy.errors import FileFormatError, InvalidParameter, UnsupportedFormat
from mdapy.io._files import Source
from mdapy.io._registry import Format, FramePosition, register

#: Version of the container layout; readers accept files up to this version.
VERSION = 1
_MANIFEST = "manifest.json"
_FRAMES = "frames.parquet"
_METADATA_KEY = "mdapy"
# Fixed member dates make files written from the same systems identical.
_DATE = (1980, 1, 1, 0, 0, 0)
_UNITS = {
    "length": "Å",
    "time": "ps",
    "energy": "eV",
    "mass": "amu",
    "charge": "e",
    "stress": "GPa",
}


# ------------------------------------------------------------ info values


def _encode(value: Any, key: str) -> Any:
    """JSON-compatible form of an info value; _decode inverts it exactly."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, np.generic):
        return _encode(value.item(), key)
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufU":
            raise InvalidParameter(
                f"info[{key!r}] is an array of {value.dtype}, which mdapy files cannot hold"
            )
        return {
            "__ndarray__": {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "data": value.ravel().tolist(),
            }
        }
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise InvalidParameter(f"info[{key!r}] has keys that are not strings")
        return {k: _encode(v, key) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v, key) for v in value]
    raise InvalidParameter(
        f"info[{key!r}] of type {type(value).__name__} cannot be written to an mdapy file"
    )


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__ndarray__"}:
            spec = value["__ndarray__"]
            data = np.array(spec["data"], dtype=np.dtype(spec["dtype"]))
            return data.reshape(spec["shape"])
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def _info_columns(infos: list[Mapping[str, Any]]) -> list[pl.Series]:
    """Typed columns info.<key> for the info values that every frame has with
    the same simple type: a scalar or a numeric array of one shape."""
    columns = []
    keys = dict.fromkeys(k for info in infos for k in info)
    for key in keys:
        values = [info.get(key) for info in infos]
        if any(v is None for v in values):
            continue
        name = f"info.{key}"
        kinds = {type(v.item() if isinstance(v, np.generic) else v) for v in values}
        if len(kinds) == 1 and kinds <= {bool, int, float, str}:
            values = [v.item() if isinstance(v, np.generic) else v for v in values]
            columns.append(
                pl.Series(
                    name,
                    values,
                    dtype={
                        bool: pl.Boolean,
                        int: pl.Int64,
                        float: pl.Float64,
                        str: pl.String,
                    }[kinds.pop()],
                )
            )
        elif all(isinstance(v, np.ndarray) for v in values):
            shapes = {v.shape for v in values}
            dtypes = {v.dtype for v in values}
            if len(shapes) == 1 and len(dtypes) == 1 and values[0].dtype.kind in "biuf":
                shape = values[0].shape
                if shape and 0 not in shape:
                    columns.append(pl.Series(name, np.stack(values)))
    return columns


# ---------------------------------------------------------------- writing


def _parquet(table: pl.DataFrame, metadata: dict[str, str] | None = None) -> bytes:
    buffer = io.BytesIO()
    table.write_parquet(buffer, compression="zstd", metadata=metadata)
    return buffer.getvalue()


def _box(box: Box) -> dict[str, Any]:
    return {
        "matrix": box.matrix.ravel().tolist(),
        "origin": box.origin.tolist(),
        "pbc": box.pbc.tolist(),
    }


def _store(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    archive.writestr(
        zipfile.ZipInfo(name, _DATE), data, compress_type=zipfile.ZIP_STORED
    )


class _Stream:
    """A file that cannot seek back, such as a compressed one. Without
    tell(), zipfile writes its members one after the other."""

    def __init__(self, file: BinaryIO):
        self._file = file

    def write(self, data: bytes) -> int:
        return self._file.write(data)

    def flush(self) -> None:
        self._file.flush()


def _write(file: BinaryIO, systems: Iterable[System]) -> None:
    from mdapy import __version__

    if not isinstance(file, (io.BufferedWriter, io.BytesIO)):
        file = _Stream(file)  # compressed output
    with zipfile.ZipFile(file, "w", allowZip64=True) as archive:
        rows: list[dict[str, Any]] = []
        infos: list[Mapping[str, Any]] = []
        topologies: list[Topology] = []
        for index, system in enumerate(systems):
            topology_id = None
            if system.topology is not None:
                if not topologies or system.topology != topologies[-1]:
                    topologies.append(system.topology)
                    folder = f"topology/{len(topologies) - 1:08d}"
                    for kind in KINDS:
                        _store(
                            archive,
                            f"{folder}/{kind}.parquet",
                            _parquet(getattr(system.topology, kind)),
                        )
                topology_id = f"topology/{len(topologies) - 1:08d}"
            info = {k: _encode(v, k) for k, v in system.info.items()}
            header = {
                "frame": index,
                "box": _box(system.box),
                "info": info,
                "topology": topology_id,
            }
            member = f"atoms/{index:08d}.parquet"
            metadata = {_METADATA_KEY: json.dumps(header)}
            _store(archive, member, _parquet(system.table, metadata))
            rows.append({**header, "member": member, "atoms": len(system)})
            infos.append(system.info)
        frames = pl.DataFrame(
            [
                pl.Series("frame", [r["frame"] for r in rows], dtype=pl.Int64),
                pl.Series("atoms", [r["atoms"] for r in rows], dtype=pl.Int64),
                pl.Series("member", [r["member"] for r in rows], dtype=pl.String),
                pl.Series("topology", [r["topology"] for r in rows], dtype=pl.String),
                pl.Series(
                    "box_matrix",
                    np.array([r["box"]["matrix"] for r in rows]).reshape(-1, 3, 3),
                    dtype=pl.Array(pl.Float64, (3, 3)),
                ),
                pl.Series(
                    "box_origin",
                    np.array([r["box"]["origin"] for r in rows]).reshape(-1, 3),
                    dtype=pl.Array(pl.Float64, 3),
                ),
                pl.Series(
                    "box_pbc",
                    np.array([r["box"]["pbc"] for r in rows], dtype=bool).reshape(
                        -1, 3
                    ),
                    dtype=pl.Array(pl.Boolean, 3),
                ),
                pl.Series(
                    "info", [json.dumps(r["info"]) for r in rows], dtype=pl.String
                ),
                *_info_columns(infos),
            ]
        )
        _store(archive, _FRAMES, _parquet(frames))
        manifest = {
            "format": "mdapy",
            "version": VERSION,
            "mdapy": __version__,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "frames": len(rows),
            "units": _UNITS,
            "members": {
                _FRAMES: "one row per frame: box, info and the member of its atoms",
                "atoms/": "one Parquet table of atoms per frame",
                "topology/": "bonds, angles, dihedrals and impropers, shared by frames",
            },
        }
        _store(archive, _MANIFEST, json.dumps(manifest, indent=2).encode())


# ---------------------------------------------------------------- reading


class _BufferFile(io.RawIOBase):
    """A read-only file over bytes or an mmap, without copying them. (An mmap
    itself has no seekable() before Python 3.13, which zipfile needs.)"""

    def __init__(self, buffer):
        self._buffer = buffer
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        start = {
            io.SEEK_SET: 0,
            io.SEEK_CUR: self._position,
            io.SEEK_END: len(self._buffer),
        }
        position = start[whence] + offset
        if position < 0:
            raise OSError("negative seek position")  # as for a real file
        self._position = position
        return position

    def readinto(self, target) -> int:
        data = self._buffer[self._position : self._position + len(target)]
        target[: len(data)] = data
        self._position += len(data)
        return len(data)


class _Archive:
    """The members of an mdapy file, also when it was not closed properly."""

    def __init__(self, source: Source):
        self.name = source.name
        buffer = source.buffer
        self._buffer = buffer
        try:
            self._zip = zipfile.ZipFile(_BufferFile(buffer))
            self.members = None
        except zipfile.BadZipFile:
            # No central directory: the writer stopped. Read the local headers.
            self._zip = None
            self.members = _local_members(buffer)
            if not self.members:
                raise FileFormatError("not an mdapy file", self.name) from None

    def names(self) -> list[str]:
        return list(self.members) if self._zip is None else self._zip.namelist()

    def read(self, name: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(name)
        start, size = self.members[name]
        return bytes(self._buffer[start : start + size])


def _local_members(buffer) -> dict[str, tuple[int, int]]:
    """Name -> (data offset, size) of the complete stored members."""
    members = {}
    position = 0
    size = len(buffer)
    while (
        position + 30 <= size
        and bytes(buffer[position : position + 4]) == b"PK\x03\x04"
    ):
        _, _, flags, method, _, _, _, compressed, _, name_length, extra_length = (
            struct.unpack("<IHHHHHIIIHH", bytes(buffer[position : position + 30]))
        )
        name_end = position + 30 + name_length
        name = bytes(buffer[position + 30 : name_end]).decode()
        extra = bytes(buffer[name_end : name_end + extra_length])
        if compressed == 0xFFFFFFFF:  # ZIP64: the sizes are in an extra field
            offset = 0
            while offset + 4 <= len(extra):
                tag, length = struct.unpack("<HH", extra[offset : offset + 4])
                if tag == 1:
                    _, compressed = struct.unpack(
                        "<QQ", extra[offset + 4 : offset + 20]
                    )
                offset += 4 + length
        start = name_end + extra_length
        if method != zipfile.ZIP_STORED or flags & 0x08 or start + compressed > size:
            break
        members[name] = (start, compressed)
        position = start + compressed
    return members


def _check_manifest(archive: _Archive) -> None:
    if _MANIFEST not in archive.names():
        return
    try:
        manifest = json.loads(archive.read(_MANIFEST))
    except ValueError:
        raise FileFormatError("the manifest is not JSON", archive.name) from None
    if manifest.get("format") != "mdapy":
        raise FileFormatError(
            "the manifest does not describe an mdapy file", archive.name
        )
    if not isinstance(manifest.get("version"), int) or manifest["version"] > VERSION:
        raise UnsupportedFormat(
            f"{archive.name} has version {manifest.get('version')} of the mdapy "
            f"format; this mdapy reads versions up to {VERSION}"
        )


def _headers(archive: _Archive) -> list[dict[str, Any]]:
    """Box, info and members of every frame."""
    if _FRAMES in archive.names():
        frames = pl.read_parquet(io.BytesIO(archive.read(_FRAMES)))
        return [
            {
                "member": row["member"],
                "topology": row["topology"],
                "box": {
                    "matrix": np.array(row["box_matrix"]).ravel().tolist(),
                    "origin": list(row["box_origin"]),
                    "pbc": list(row["box_pbc"]),
                },
                "info": json.loads(row["info"]),
            }
            for row in frames.iter_rows(named=True)
        ]
    # Recovery: the metadata of each member of atoms.
    headers = []
    for name in sorted(n for n in archive.names() if n.startswith("atoms/")):
        metadata = pl.read_parquet_metadata(io.BytesIO(archive.read(name)))
        if _METADATA_KEY not in metadata:
            raise FileFormatError(f"{name} has no mdapy metadata", archive.name)
        header = json.loads(metadata[_METADATA_KEY])
        header["member"] = name
        headers.append(header)
    return headers


def _open(source: Source) -> tuple[_Archive, list[dict[str, Any]]]:
    """The archive and the frame headers of a file, read once per file."""
    if "mdapy" not in source.cache:
        archive = _Archive(source)
        _check_manifest(archive)
        source.cache["mdapy"] = (archive, _headers(archive))
    return source.cache["mdapy"]


def _scan(source: Source) -> Iterator[FramePosition]:
    for index in range(len(_open(source)[1])):
        yield (index, 0)


def _read(source: Source, position: FramePosition) -> System:
    archive, headers = _open(source)
    header = headers[position[0]]
    table = pl.read_parquet(io.BytesIO(archive.read(header["member"])))
    box = header["box"]
    topology = None
    if header.get("topology"):
        folder = header["topology"]
        topology = Topology(
            **{
                kind: pl.read_parquet(
                    io.BytesIO(archive.read(f"{folder}/{kind}.parquet"))
                )
                for kind in KINDS
            }
        )
    return System(
        table=table,
        box=Box(
            np.reshape(box["matrix"], (3, 3)), origin=box["origin"], pbc=box["pbc"]
        ),
        info={k: _decode(v) for k, v in header["info"].items()},
        topology=topology,
    )


register(
    Format(
        name="mdapy",
        description="mdapy file: Parquet tables in a ZIP container",
        extensions=(".mdapy",),
        read=_read,
        scan=_scan,
        write=_write,
    )
)
