# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The test of a built mdapy-cuda12 where there is no GPU, as on the runners of
CI: its modules load, _cuda and _dft (with NVIDIA's cuBLAS, cuFFT and cuSOLVER
installed), and refuse to run without a device. Run it with python -I, so that
the installed package is imported, not the sources beside it."""

import mdapy_cuda12
from mdapy_cuda12 import _cuda, _dft

try:
    name = _cuda.device_name()
except RuntimeError as error:
    assert str(error).startswith("no CUDA device"), error
    print(f"mdapy-cuda12 {mdapy_cuda12.__version__}: {error}")
else:
    print(f"mdapy-cuda12 {mdapy_cuda12.__version__} on {name}")
assert _dft.api == "CUDA", _dft.api
print(f"its DFT: {_dft.device_count()} devices")
