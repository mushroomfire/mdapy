// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/core/check.h"
#include "dft/xc/xc.h"

namespace mdapy::dft {

namespace {

__global__ void xc_kernel(Functional functional, std::size_t count, const double* rho,
                          const double* sigma, double threshold, double* f, double* v,
                          double* vs) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= count) return;
    const double up = rho[2 * i], down = rho[2 * i + 1];
    if (!(up + down >= threshold) || !(up > 0.0) || !(down > 0.0)) {
        f[i] = 0.0;
        v[2 * i] = v[2 * i + 1] = 0.0;
        vs[3 * i] = vs[3 * i + 1] = vs[3 * i + 2] = 0.0;
        return;
    }
    using D = Dual<5>;
    const D e = xc::energy(functional, D::input(up, 0), D::input(down, 1),
                           D::input(sigma[3 * i], 2), D::input(sigma[3 * i + 1], 3),
                           D::input(sigma[3 * i + 2], 4));
    f[i] = e.v;
    v[2 * i] = e.d[0];
    v[2 * i + 1] = e.d[1];
    vs[3 * i] = e.d[2];
    vs[3 * i + 1] = e.d[3];
    vs[3 * i + 2] = e.d[4];
}

}  // namespace

void evaluate_xc(Functional functional, std::size_t count, const double* rho, const double* sigma,
                 double threshold, double* f, double* v, double* vs) {
    if (count == 0) return;
    const unsigned threads = 128;
    const unsigned blocks = static_cast<unsigned>((count + threads - 1) / threads);
    xc_kernel<<<blocks, threads>>>(functional, count, rho, sigma, threshold, f, v, vs);
    check_cuda(cudaGetLastError(), "the exchange-correlation kernel");
}

}  // namespace mdapy::dft
