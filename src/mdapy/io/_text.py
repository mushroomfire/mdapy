# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Line and table reading and writing shared by the text formats."""

from collections.abc import Iterator, Sequence
from typing import BinaryIO

import numpy as np
import polars as pl

from mdapy._core import io as _native
from mdapy.errors import FileFormatError

#: Rows formatted per call when writing, which bounds the memory used.
WRITE_BLOCK_ROWS = 1 << 17

_DTYPES = {"f": np.float64, "i": np.int64, "b": np.bool_, "s": np.uint32}


class TextReader:
    """Reads a text buffer line by line and table by table.

    Args:
        buffer: The text, as bytes or an mmap.
        name: File name used in error messages.
        offset: Byte offset to start reading at.
        line: Number of lines before offset.
    """

    def __init__(self, buffer, name: str, offset: int = 0, line: int = 0):
        self.buffer = buffer
        self.name = name
        self.offset = offset
        #: Number of lines read so far; the last line read has this number.
        self.line = line

    def at_end(self) -> bool:
        """Whether all text has been read."""
        return self.offset >= len(self.buffer)

    def readline(self) -> str:
        """Returns the next line without its line ending."""
        if self.at_end():
            raise FileFormatError("unexpected end of file", self.name, self.line + 1)
        end = self.buffer.find(b"\n", self.offset)
        stop = len(self.buffer) if end < 0 else end
        raw = self.buffer[self.offset : stop]
        self.offset = stop + 1
        self.line += 1
        try:
            return raw.decode("utf-8").rstrip("\r")
        except UnicodeDecodeError:
            raise self.error("the line is not UTF-8 text") from None

    def skip(self, n: int) -> None:
        """Skips n lines."""
        try:
            self.offset = _native.skip_lines(self.buffer, self.offset, n)
        except ValueError as e:
            raise self._parse_error(e) from None
        self.line += n

    def table(
        self, rows: int, columns: Sequence[tuple[str, int]], comment: str = ""
    ) -> list:
        """Reads rows lines of values.

        Args:
            rows: Number of lines.
            columns: (kind, width) of each column: kind "f" float, "i" integer,
                "b" boolean (T or F), "s" string, "x" ignored.
            comment: Character that starts a comment, or "".

        Returns:
            One entry per column: an array of shape (rows,) for width 1 or
            (rows, width), a (codes, categories) pair for strings, and None
            for ignored columns.
        """
        try:
            arrays, self.offset = _native.parse_table(
                self.buffer, self.offset, rows, list(columns), comment
            )
        except ValueError as e:
            raise self._parse_error(e) from None
        self.line += rows
        return arrays

    def error(self, message: str, line: int | None = None) -> FileFormatError:
        """Error at a line, by default the last line read."""
        return FileFormatError(message, self.name, self.line if line is None else line)

    def _parse_error(self, error: ValueError) -> Exception:
        if len(error.args) != 2:
            return error
        message, row = error.args
        return self.error(message, self.line + row + 1)


def strings(
    codes: np.ndarray,
    categories: list[str],
    name: str = "",
    dtype: pl.DataType = pl.String,
) -> pl.Series:
    """Series of the strings given by codes into categories.

    The categories are converted to dtype first, so each distinct value is
    checked once.
    """
    values = pl.Series(name, categories, dtype=pl.String).cast(dtype)
    return values.gather(codes)


def categorize(values: pl.Series) -> tuple[np.ndarray, list[str]]:
    """Codes and categories of a string or enum Series."""
    if isinstance(values.dtype, pl.Enum):
        categories = values.dtype.categories.to_list()
    else:
        categories = values.unique(maintain_order=True).to_list()
        values = values.cast(pl.Enum(categories))
    codes = values.to_physical().to_numpy().astype(np.uint32)
    return codes, categories


def format_rows(columns: Sequence[tuple[str, object]]) -> Iterator[bytes]:
    """Formats rows of values, in blocks of at most WRITE_BLOCK_ROWS lines.

    Args:
        columns: (kind, values) of each column: kind "f", "i" or "b" with an
            array of shape (rows,) or (rows, width), or "s" with a
            (codes, categories) pair.
    """
    prepared = []
    for kind, values in columns:
        if kind == "s":
            codes, categories = values
            array = np.ascontiguousarray(codes, dtype=np.uint32)
        else:
            array = np.ascontiguousarray(values, dtype=_DTYPES[kind])
            categories = []
        prepared.append((kind, array, categories))
    rows = len(prepared[0][1]) if prepared else 0
    for start in range(0, rows, WRITE_BLOCK_ROWS):
        stop = start + WRITE_BLOCK_ROWS
        yield _native.format_table([(k, a[start:stop], c) for k, a, c in prepared])


def write_rows(file: BinaryIO, columns: Sequence[tuple[str, object]]) -> None:
    """Writes rows of values to a binary file; see format_rows."""
    for block in format_rows(columns):
        file.write(block)
