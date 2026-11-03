# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import importlib.util
import re

import pytest

import mdapy

# OVITO and freud both use voro++: the Voronoi tessellation of whichever is
# loaded second crashes. OVITO is loaded first, before any test module
# imports freud, and freud's Voronoi tessellation runs in processes of its own.
if importlib.util.find_spec("ovito") is not None:
    import ovito.modifiers  # noqa: F401


def lammps_package_missing(error: BaseException) -> str | None:
    """The package a LAMMPS binary said it lacks in this error, or None."""
    found = re.search(r"is part of the (\S+) package which is not", str(error))
    return found[1] if found else None


@pytest.hookimpl(wrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    """A LAMMPS without a package that a test needs skips the test. CI checks
    that its LAMMPS has the packages, so that nothing is skipped there."""
    try:
        return (yield)
    except Exception as error:
        package = lammps_package_missing(error)
        if package is None:
            raise
        pytest.skip(f"LAMMPS without the {package} package")


@pytest.fixture(autouse=True)
def _restore_num_threads():
    """Every test starts and ends with the same global thread setting."""
    before = mdapy.get_num_threads()
    yield
    mdapy.set_num_threads(before)
