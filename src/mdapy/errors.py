# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Exception types raised by mdapy."""

__all__ = [
    "MdapyError",
    "InvalidParameter",
    "MissingProperty",
    "UnsupportedFormat",
    "FileFormatError",
    "Cancelled",
    "CalculationFailed",
]


class MdapyError(Exception):
    """Base class of every exception raised by mdapy."""


class InvalidParameter(MdapyError, ValueError):
    """A parameter has a value that mdapy cannot accept.

    The message names the parameter and the accepted values.
    """


class MissingProperty(MdapyError, LookupError):
    """A required per-atom or global property is not present."""


class UnsupportedFormat(MdapyError, ValueError):
    """A file format is unknown, or an operation is not supported for it."""


class FileFormatError(MdapyError, ValueError):
    """A file does not follow its format.

    The message starts with the file name and, when known, the line number.

    Attributes
    ----------
    path : str or None
        Name of the file.
    line : int or None
        Line number, counting from 1.
    """

    def __init__(self, message: str, path: str | None = None, line: int | None = None):
        location = "" if path is None else path if line is None else f"{path}:{line}"
        super().__init__(f"{location}: {message}" if location else message)
        self.path = path
        self.line = line


class Cancelled(MdapyError):
    """A computation was stopped through a :class:`mdapy.CancelToken`."""


class CalculationFailed(MdapyError, RuntimeError):
    """An external program behind a calculator, such as VASP, did not finish
    a calculation.

    The message holds the end of what the program wrote.
    """
