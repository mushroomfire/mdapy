# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""The file formats mdapy knows, and reading and writing by format."""

import fnmatch
import operator
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from mdapy.core.system import System
from mdapy.errors import InvalidParameter, UnsupportedFormat
from mdapy.io._files import COMPRESSION_SUFFIXES, Source, open_output

#: Byte offset and number of preceding lines of the start of a frame.
FramePosition = tuple[int, int]


@dataclass(frozen=True)
class Format:
    """A file format and the functions that read and write it.

    Parameters
    ----------
    name : str
        Name passed as ``format=``.
    description : str
        One line shown by :func:`formats`.
    extensions : tuple of str
        Lowercase file name extensions, such as ``".xyz"``.
    patterns : tuple of str
        Shell patterns of file names, such as ``"POSCAR*"``.
    read : callable, optional
        ``read(source, position, **options) -> System`` reads the frame that
        starts at ``position``.
    scan : callable, optional
        ``scan(source)`` yields the position of each frame. Only formats that
        hold several frames have it.
    write : callable, optional
        ``write(file, systems, **options)`` writes an iterable of systems to
        a binary file.
    """

    name: str
    description: str
    extensions: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    read: Callable[..., System] | None = None
    scan: Callable[[Source], Iterator[FramePosition]] | None = None
    write: Callable[..., None] | None = None


FORMATS: dict[str, Format] = {}


def register(format: Format) -> Format:
    """Adds a format to the registry."""
    if format.name in FORMATS:
        raise ValueError(f"format {format.name!r} is already registered")
    FORMATS[format.name] = format
    return format


def formats() -> pl.DataFrame:
    """The file formats mdapy can read or write.

    Returns
    -------
    polars.DataFrame
        One row per format, with columns ``name``, ``description``,
        ``extensions``, ``read``, ``write`` and ``multiple_frames``.
    """
    return pl.DataFrame(
        {
            "name": [f.name for f in FORMATS.values()],
            "description": [f.description for f in FORMATS.values()],
            "extensions": [list(f.extensions) for f in FORMATS.values()],
            "read": [f.read is not None for f in FORMATS.values()],
            "write": [f.write is not None for f in FORMATS.values()],
            "multiple_frames": [f.scan is not None for f in FORMATS.values()],
        },
        schema={
            "name": pl.String,
            "description": pl.String,
            "extensions": pl.List(pl.String),
            "read": pl.Boolean,
            "write": pl.Boolean,
            "multiple_frames": pl.Boolean,
        },
    )


def detect_format(path: str | os.PathLike) -> str:
    """The format of a file, from its name.

    A compression suffix (.gz, .bz2, .xz, .zst) is ignored.

    Parameters
    ----------
    path : str or os.PathLike
        The file name; the file need not exist.

    Returns
    -------
    str
        The format name.

    Raises
    ------
    UnsupportedFormat
        If no format, or more than one, matches the name.
    """
    name = Path(path).name
    for suffix in COMPRESSION_SUFFIXES:
        if name.lower().endswith(suffix) and len(name) > len(suffix):
            name = name[: -len(suffix)]
            break
    extension = Path(name).suffix.lower()
    matches = [
        f.name
        for f in FORMATS.values()
        if extension in f.extensions
        or any(fnmatch.fnmatchcase(name, p) for p in f.patterns)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise UnsupportedFormat(
            f"cannot tell the format of {os.fspath(path)!r} from its name; "
            f"pass format=, one of {', '.join(FORMATS)}"
        )
    raise UnsupportedFormat(
        f"the name {os.fspath(path)!r} fits the formats {', '.join(matches)}; "
        "pass format="
    )


def _format(path: str | os.PathLike, format: str | None, action: str) -> Format:
    name = detect_format(path) if format is None else format
    if name not in FORMATS:
        raise UnsupportedFormat(
            f"unknown format {name!r}; known formats: {', '.join(FORMATS)}"
        )
    found = FORMATS[name]
    if getattr(found, action) is None:
        raise UnsupportedFormat(f"mdapy cannot {action} {name} files")
    return found


def read(
    path: str | os.PathLike, format: str | None = None, *, frame: int = 0, **options
) -> System:
    """Reads one frame of a file.

    Parameters
    ----------
    path : str or os.PathLike
        The file. Compressed files (gzip, bzip2, xz, zstd) are recognized
        from their content.
    format : str, optional
        Format name; by default it is found from the file name with
        :func:`detect_format`.
    frame : int, optional
        Index of the frame to read, for files that hold several frames.
        Negative values count from the end. Default 0, the first frame.
    **options
        Options of the format; see the page of each format.

    Returns
    -------
    System
        The frame.

    Raises
    ------
    UnsupportedFormat
        If the format is unknown or cannot be read.
    InvalidParameter
        If the frame does not exist.
    FileFormatError
        If the file does not follow its format.
    """
    found = _format(path, format, "read")
    frame = operator.index(frame)
    with Source(path) as source:
        if found.scan is None:
            if frame not in (0, -1):
                raise InvalidParameter(
                    f"{found.name} files hold one frame, so frame must be 0 or -1, "
                    f"not {frame}"
                )
            return found.read(source, (0, 0), **options)
        positions = found.scan(source)
        if frame >= 0:
            count = 0
            for position in positions:
                if count == frame:
                    return found.read(source, position, **options)
                count += 1
        else:
            every = list(positions)
            count = len(every)
            if frame >= -count:
                return found.read(source, every[frame], **options)
    raise InvalidParameter(f"frame {frame} does not exist: the file has {count} frames")


def write(
    path: str | os.PathLike,
    systems: System | Iterable[System],
    format: str | None = None,
    **options,
) -> None:
    """Writes systems to a file.

    Parameters
    ----------
    path : str or os.PathLike
        The file; it is replaced if it exists. The suffix .gz, .bz2, .xz or
        .zst compresses it.
    systems : System or iterable of System
        One system, or several for formats that hold several frames. An
        iterator, such as a generator, is written one system at a time.
    format : str, optional
        Format name; by default it is found from the file name with
        :func:`detect_format`.
    **options
        Options of the format; see the page of each format.

    Raises
    ------
    UnsupportedFormat
        If the format is unknown or cannot be written.
    InvalidParameter
        If the format holds one frame and not exactly one system is given,
        or if a system cannot be written in the format.
    """
    found = _format(path, format, "write")
    if isinstance(systems, System):
        systems = [systems]
    elif found.scan is None:
        systems = list(systems)
        if len(systems) != 1:
            raise InvalidParameter(
                f"{found.name} files hold one frame, but {len(systems)} systems "
                "were given"
            )
    with open_output(path) as file:
        found.write(file, _checked(systems), **options)


def _checked(systems: Iterable[System]) -> Iterator[System]:
    for system in systems:
        if not isinstance(system, System):
            raise TypeError(
                f"write takes a System or an iterable of System, not "
                f"{type(system).__name__}"
            )
        yield system
