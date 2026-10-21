# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Exception types raised by mdapy."""

__all__ = [
    "MdapyError",
    "InvalidParameter",
    "MissingProperty",
    "UnsupportedFormat",
    "Cancelled",
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


class Cancelled(MdapyError):
    """A computation was stopped through a :class:`mdapy.CancelToken`."""
