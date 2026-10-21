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
    ],
)
def test_errors_are_also_builtin_exceptions(cls, builtin):
    """Code that catches the built-in type keeps working."""
    assert issubclass(cls, errors.MdapyError)
    assert issubclass(cls, builtin)
    with pytest.raises(builtin, match="bad"):
        raise cls("bad")


def test_cancelled_is_not_a_value_error():
    assert issubclass(errors.Cancelled, errors.MdapyError)
    assert not issubclass(errors.Cancelled, ValueError)


def test_all_names_are_exception_classes():
    for name in errors.__all__:
        assert issubclass(getattr(errors, name), Exception)
