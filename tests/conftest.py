# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import pytest

import mdapy


@pytest.fixture(autouse=True)
def _restore_num_threads():
    """Every test starts and ends with the same global thread setting."""
    before = mdapy.get_num_threads()
    yield
    mdapy.set_num_threads(before)
