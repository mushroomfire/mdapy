# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The GPU part of MDAPY, built with CUDA 12 for NVIDIA GPUs from the V100 on,
for Linux on x86-64 and ARM:
MDAPY loads its module ``_cuda`` for calculators given ``device="cuda"``.
Install it with ``pip install "mdapy[cuda12]"``, at the version of MDAPY."""

from mdapy_cuda12._version import __version__

__all__ = ["__version__"]
