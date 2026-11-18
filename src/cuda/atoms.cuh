// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "cuda/device.cuh"

namespace mdapy::cuda {

struct Atoms::Impl {
    std::int64_t n = 0;
    // positions in double precision, the species of each atom as w
    Buffer<double4> x;
    // the same in single precision for mixed precision, made from x when
    // read after x changed
    Buffer<float4> x_single;
    std::uint64_t version = 0;         // of x
    std::uint64_t single_version = 0;  // of x_single, 0 for none
    Buffer<std::int32_t> species;
    Buffer<double> forces;    // (n, 3)
    Buffer<double> energies;  // (n,)
    Buffer<double> virials;   // 9 columns of n, xx xy xz yx yy yz zx zy zz
    bool has_energies = false;
    bool has_virials = false;

    // The positions in a precision, the species as w.
    template <class Real>
    const Real4<Real>* positions();
};

}  // namespace mdapy::cuda
