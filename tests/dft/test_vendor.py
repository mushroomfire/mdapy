# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The libraries of the vendor where they are asked for, NVIDIA's cuBLAS, cuFFT
and cuSOLVER with CUDA, hipBLAS, hipFFT and hipSOLVER of ROCm with HIP: the
module _dft names none of them, a computation with MDAPY's kernels runs
without loading them, and kernels="vendor" loads them (with cuSOLVER's own
libraries, or those of ROCm they need). In a process of its own, as the other
tests load them."""

import json
import subprocess
import sys

from dft import espresso, gpu

# Al4 with MDAPY's kernels, then the libraries of the vendor asked for: those
# loaded before and after, from the maps of the process
CHILD = r"""
import json, re, sys
import numpy as np
from mdapy.calculators._dft import loader, scf
from mdapy.calculators._dft.upf import read_upf

def loaded():
    found = set()
    with open("/proc/self/maps") as maps:
        for line in maps:
            name = re.search(r"lib(cublasLt|cublas|cufft|cusolver|cusparse|nvJitLink|hipblas|hipfft|hipsolver"
                             r"|rocblas|rocfft|rocsolver|rocsparse)\.so", line)
            if name:
                found.add(name.group(1))
    return sorted(found)

a = 4.05 / 0.529177210903
fractions = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
r = scf.run(np.diag([a, a, a]), fractions, [0, 0, 0, 0], [read_upf(sys.argv[1])], 10.0, kgrid=(2, 2, 2),
            smearing="gaussian", sigma=0.02, symmetry=False, kernels="mdapy")
before = loaded()
files = loader.vendor_libraries(loader.module())
print(json.dumps({"converged": bool(r["converged"]), "before": before, "after": loaded(), "files": files}))
"""


def test_the_libraries_of_the_vendor_load_where_asked_for():
    dft = gpu.device()
    out = subprocess.run(
        [sys.executable, "-c", CHILD, str(espresso.pseudo("Al_std.upf"))],
        capture_output=True, text=True, check=True, timeout=600,
    )  # fmt: skip
    found = json.loads(out.stdout.strip().splitlines()[-1])
    assert found["converged"]
    assert found["before"] == []
    if dft.api == "CUDA":
        names = ["cublas", "cublasLt", "cufft", "cusolver", "cusparse", "nvJitLink"]
        assert found["after"] == sorted(names)
        assert len(found["files"]) == len(names)
    else:  # ROCm's own libraries come with them
        rocm = {"hipblas", "hipfft", "hipsolver", "rocblas", "rocfft", "rocsolver"}
        assert rocm <= set(found["after"])
        assert len(found["files"]) == 3
    # the same files again, which stay loaded
    assert dft.vendor_libraries() == dft.vendor_libraries()
