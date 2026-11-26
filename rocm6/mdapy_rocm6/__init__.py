# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The GPU part of MDAPY for AMD GPUs, built with HIP of ROCm 6 for Linux on
x86-64: MDAPY loads its module ``_hip`` for calculators given
``device="cuda"``, and its module ``_dft`` for the plane-wave DFT, which loads
hipBLAS, hipFFT and hipSOLVER of ROCm only for ``kernels="vendor"``. Install it with
``pip install "mdapy[rocm6]"``, at the version of MDAPY."""

from mdapy_rocm6._version import __version__

__all__ = ["__version__"]
