# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import re

import mdapy


def test_version_is_pep440():
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.dev\d+)?", mdapy.__version__)


def test_public_names_exist():
    for name in mdapy.__all__:
        assert hasattr(mdapy, name), name


def test_errors_and_units_are_importable_as_modules():
    from mdapy.errors import Cancelled
    from mdapy.units import BOLTZMANN

    assert mdapy.errors.Cancelled is Cancelled
    assert mdapy.units.BOLTZMANN == BOLTZMANN
