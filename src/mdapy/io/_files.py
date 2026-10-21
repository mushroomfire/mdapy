# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Opening files for reading and writing, with transparent compression."""

import bz2
import gzip
import lzma
import mmap
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import BinaryIO

from mdapy.errors import FileFormatError, UnsupportedFormat
from mdapy.io._text import TextReader

#: File name suffix of each compression format.
COMPRESSION_SUFFIXES = {".gz": "gzip", ".bz2": "bzip2", ".xz": "xz", ".zst": "zstd"}

# First bytes of each compression format; reading detects compression from
# the content, not the name.
_MAGIC = {
    b"\x1f\x8b": "gzip",
    b"BZh": "bzip2",
    b"\xfd7zXZ\x00": "xz",
    b"\x28\xb5\x2f\xfd": "zstd",
}


def _module(compression: str) -> ModuleType:
    """The standard library module of a compression format."""
    if compression == "zstd":
        try:
            from compression import zstd
        except ImportError:
            raise UnsupportedFormat(
                "zstd-compressed files need Python 3.14 or newer"
            ) from None
        return zstd
    return {"gzip": gzip, "bzip2": bz2, "xz": lzma}[compression]


class Source:
    """A file opened for reading: its name and its contents.

    Uncompressed files are memory-mapped, so reading one frame of a large
    file does not read the whole file. Compressed files are decompressed
    into memory.

    Parameters
    ----------
    path : str or os.PathLike
        The file.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        #: File name used in error messages.
        self.name = os.fspath(path)
        #: Values a format computes once per file, such as an index of frames.
        self.cache: dict = {}
        self._file = None
        self._map = None
        with open(self.path, "rb") as f:
            head = f.read(6)
        compression = next(
            (name for magic, name in _MAGIC.items() if head.startswith(magic)), None
        )
        if compression is not None:
            module = _module(compression)
            try:
                with module.open(self.path, "rb") as f:
                    self.buffer = f.read()
            except Exception as e:  # each library raises its own error types
                raise FileFormatError(
                    f"cannot decompress the {compression} data: {e}", self.name
                ) from e
            return
        self._file = open(self.path, "rb")
        if os.fstat(self._file.fileno()).st_size == 0:
            self.buffer = b""
        else:
            self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            self.buffer = self._map

    def reader(self, offset: int = 0, line: int = 0) -> TextReader:
        """A TextReader that starts at a byte offset, after the given line."""
        return TextReader(self.buffer, self.name, offset, line)

    def close(self) -> None:
        """Releases the file."""
        if self._map is not None:
            self._map.close()
        if self._file is not None:
            self._file.close()

    def __enter__(self) -> "Source":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@contextmanager
def open_output(path: str | os.PathLike) -> Iterator[BinaryIO]:
    """Opens a file for writing in binary mode.

    The suffix .gz, .bz2, .xz or .zst selects compression. If writing fails,
    the incomplete file is removed.
    """
    path = Path(path)
    compression = COMPRESSION_SUFFIXES.get(path.suffix.lower())
    if compression is None:
        file = open(path, "wb")
    elif compression == "gzip":
        # Level 6, as in the gzip program: much faster than the default 9.
        file = gzip.open(path, "wb", compresslevel=6)
    else:
        file = _module(compression).open(path, "wb")
    try:
        yield file
    except BaseException:
        file.close()
        path.unlink(missing_ok=True)
        raise
    file.close()
