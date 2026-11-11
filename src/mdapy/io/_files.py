# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Opening files for reading and writing, with transparent compression."""

import bz2
import gzip
import io
import lzma
import mmap
import os
import struct
import zlib
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import BinaryIO

from mdapy.core.threads import get_num_threads
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


# Level 6, as in the gzip program: much faster than the default 9.
_GZIP_LEVEL = 6
# Text compressed at a time: small enough for the threads to share a file, large
# enough that compressing it takes far longer than handing it over.
_GZIP_BLOCK = 1 << 20
_GZIP_WINDOW = 1 << 15  # how far back deflate looks
# Deflate, no flags, no time, no extra flags, unknown system.
_GZIP_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff"


def _deflate(block: bytes, before: bytes, last: bool) -> bytes:
    """A block of a deflate stream, which looks back into the text before it as
    the stream goes on. It ends on a whole byte, where the next block joins, and
    the last block ends the stream."""
    options = {"zdict": before} if before else {}
    compressor = zlib.compressobj(
        _GZIP_LEVEL, zlib.DEFLATED, -zlib.MAX_WBITS, **options
    )
    return compressor.compress(block) + compressor.flush(
        zlib.Z_FINISH if last else zlib.Z_SYNC_FLUSH
    )


class _GzipWriter(io.BufferedIOBase):
    """A gzip file compressed on all threads, in blocks of one stream, as pigz
    does: some programs stop at the end of the first stream of a file. It is
    as small as that of one thread, and has the same bytes for any number of
    threads.

    Parameters
    ----------
    file : BinaryIO
        The file open for writing, which is closed with this one.
    """

    def __init__(self, file: BinaryIO):
        self.__file = file
        self.__file.write(_GZIP_HEADER)
        self.__text = bytearray()
        self.__before = b""
        self.__crc = 0
        self.__size = 0
        self.__threads = get_num_threads()
        self.__pool = None
        self.__pending = deque()

    def writable(self) -> bool:
        return True

    def write(self, data: bytes) -> int:
        self.__text += data
        # the last block is written on closing, even if it holds nothing
        while len(self.__text) > _GZIP_BLOCK:
            block = bytes(self.__text[:_GZIP_BLOCK])
            del self.__text[:_GZIP_BLOCK]
            self.__send(block, False)
        return len(data)

    def __send(self, block: bytes, last: bool) -> None:
        self.__crc = zlib.crc32(block, self.__crc)
        self.__size += len(block)
        before, self.__before = self.__before, block[-_GZIP_WINDOW:]
        if self.__threads == 1 or (last and self.__pool is None):
            self.__file.write(_deflate(block, before, last))
            return
        if self.__pool is None:
            self.__pool = ThreadPoolExecutor(self.__threads)
        self.__pending.append(self.__pool.submit(_deflate, block, before, last))
        # a few blocks for each thread wait in memory, no more
        while self.__pending and (
            self.__pending[0].done() or len(self.__pending) > 2 * self.__threads
        ):
            self.__file.write(self.__pending.popleft().result())

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.__send(bytes(self.__text), True)
            while self.__pending:
                self.__file.write(self.__pending.popleft().result())
            self.__file.write(struct.pack("<II", self.__crc, self.__size & 0xFFFFFFFF))
        finally:
            if self.__pool is not None:
                self.__pool.shutdown(cancel_futures=True)
            self.__file.close()
            super().close()


@contextmanager
def open_output(path: str | os.PathLike) -> Iterator[BinaryIO]:
    """Opens a file for writing in binary mode.

    The suffix .gz, .bz2, .xz or .zst selects compression. The file is written
    beside the target under another name and takes its place when complete;
    if writing fails, it is removed and a file already at the target stays.
    """
    path = Path(path)
    compression = COMPRESSION_SUFFIXES.get(path.suffix.lower())
    part = path.with_name(f".{path.name}.{os.getpid()}.{id(path):x}.part")
    if compression is None:
        file = open(part, "wb")
    elif compression == "gzip":
        file = _GzipWriter(open(part, "wb"))
    else:
        file = _module(compression).open(part, "wb")
    try:
        yield file
        file.close()
        os.replace(part, path)
    except BaseException:
        try:
            file.close()
        finally:
            part.unlink(missing_ok=True)
        raise
