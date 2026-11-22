# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The plane-wave DFT of MDAPY on NVIDIA GPUs, built with CUDA 12: MDAPY loads
its module ``_dft`` for its DFT calculator. Under development."""

from mdapy_dft_cuda12._version import __version__

__all__ = ["__version__"]
