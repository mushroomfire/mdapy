# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The plane-wave DFT of MDAPY: its parts that run on the processor, the
pseudopotentials, their radial transforms and the Ewald sum, in hartree atomic
units; the rest runs on the GPU in mdapy_dft_cuda12. Under development."""
