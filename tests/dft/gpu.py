# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The module _dft of the GPU part for the tests of the DFT, which are skipped
where it cannot be loaded. A file of tests that all need it calls module() at
its top; a file that also tests the parts on the processor calls module() or
device() in the tests that need them, so that the others run everywhere."""

from typing import Any

import pytest

from mdapy.calculators._dft import loader
from mdapy.errors import InvalidParameter


def module() -> Any:
    """The module _dft, or the test (or the module of tests) skipped, with
    the reason the loader gives."""
    try:
        return loader.module()
    except InvalidParameter as error:
        pytest.skip(str(error), allow_module_level=True)


def device() -> Any:
    """The module _dft with a GPU to run on, or the test skipped."""
    found = module()
    if found.device_count() == 0:
        pytest.skip("no GPU")
    return found
