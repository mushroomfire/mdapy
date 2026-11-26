# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The test of a built mdapy-cuda12 where there is no GPU, as on the runners of
CI: its modules load, _cuda and _dft, and refuse to run without a device; and
NVIDIA's cuBLAS, cuFFT and cuSOLVER, which _dft loads for kernels="vendor":
with the argument "nvidia" from the folders of NVIDIA's packages beside the
package, with "any" from wherever the system has them, or missing with an
error that names the extra that installs them.
Run it with python -I, so that the installed package is imported, not the
sources beside it."""

import sys

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
expected = sys.argv[1] if len(sys.argv) > 1 else "nvidia"
try:
    files = _dft.vendor_libraries()
except RuntimeError as error:
    assert expected == "any", error
    assert 'pip install "mdapy[cuda12,dft]"' in str(error), error
    print(f"the libraries of the vendor are missing: {error}")
else:
    assert len(files) == 6, files
    assert expected == "any" or all("/nvidia/" in f for f in files), files
    print("the libraries of the vendor:", *files, sep="\n  ")
