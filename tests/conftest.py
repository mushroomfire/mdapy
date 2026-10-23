# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import importlib.util

import pytest

import mdapy

# OVITO and freud both use voro++: the Voronoi tessellation of whichever is
# loaded second crashes. OVITO is loaded first, before any test module
# imports freud, and freud's Voronoi tessellation runs in processes of its own.
if importlib.util.find_spec("ovito") is not None:
    import ovito.modifiers  # noqa: F401


@pytest.fixture(autouse=True)
def _restore_num_threads():
    """Every test starts and ends with the same global thread setting."""
    before = mdapy.get_num_threads()
    yield
    mdapy.set_num_threads(before)
