# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

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
