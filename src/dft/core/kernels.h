// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Whose kernels a calculation runs: MDAPY's own where it has them (today the
// FFTs, dft/fft/box_fft.h, and the dense linear algebra, dft/blas/blas.h), or
// the GPU libraries of the vendor throughout
// (cuFFT, cuBLAS and cuSOLVER of NVIDIA; hipFFT, hipBLAS and hipSOLVER of AMD).
// What MDAPY has no kernel for runs in the libraries of the vendor with either.

#pragma once

#include <stdexcept>
#include <string>

namespace mdapy::dft {

enum class Kernels { Mdapy, Vendor };

inline Kernels kernels_of(const std::string& name) {
    if (name == "mdapy") return Kernels::Mdapy;
    if (name == "vendor") return Kernels::Vendor;
    throw std::invalid_argument("kernels must be 'mdapy' or 'vendor'");
}

}  // namespace mdapy::dft
