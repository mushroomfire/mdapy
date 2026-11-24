# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The test of a built mdapy-rocm6 where there is no GPU: its modules load,
_hip and _dft, and refuse to run without a device. Run it with python -I, so
that the installed package is imported, not the sources beside it."""

import mdapy_rocm6
from mdapy_rocm6 import _dft, _hip

try:
    name = _hip.device_name()
except RuntimeError as error:
    assert str(error).startswith("no HIP device"), error
    print(f"mdapy-rocm6 {mdapy_rocm6.__version__}: {error}")
else:
    print(f"mdapy-rocm6 {mdapy_rocm6.__version__} on {name}")
assert _dft.api == "HIP", _dft.api
print(f"its DFT: {_dft.device_count()} devices")
