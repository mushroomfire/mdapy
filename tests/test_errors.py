# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import pytest

from mdapy import errors


@pytest.mark.parametrize(
    "cls, builtin",
    [
        (errors.InvalidParameter, ValueError),
        (errors.MissingProperty, LookupError),
        (errors.UnsupportedFormat, ValueError),
        (errors.FileFormatError, ValueError),
    ],
)
def test_errors_are_also_builtin_exceptions(cls, builtin):
    """Code that catches the built-in type keeps working."""
    assert issubclass(cls, errors.MdapyError)
    assert issubclass(cls, builtin)
    with pytest.raises(builtin, match="bad"):
        raise cls("bad")


@pytest.mark.parametrize(
    "path, line, text",
    [
        (None, None, "bad"),
        ("a.xyz", None, "a.xyz: bad"),
        ("a.xyz", 12, "a.xyz:12: bad"),
    ],
)
def test_file_format_error_names_the_location(path, line, text):
    error = errors.FileFormatError("bad", path, line)
    assert str(error) == text
    assert (error.path, error.line) == (path, line)


def test_cancelled_is_not_a_value_error():
    assert issubclass(errors.Cancelled, errors.MdapyError)
    assert not issubclass(errors.Cancelled, ValueError)


def test_all_names_are_exception_classes():
    for name in errors.__all__:
        assert issubclass(getattr(errors, name), Exception)
