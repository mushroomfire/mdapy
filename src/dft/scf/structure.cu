// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/scf/structure.h"

#include <stdexcept>

namespace mdapy::dft {

namespace {

constexpr int kThreads = 256;

__global__ void structure_kernel(std::size_t n, int atoms, const int* miller, const double* tau, const int* kinds,
                                 const double* weights, const double* factors, Complex* out) {
    const std::size_t g = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (g >= n) return;
    const double m0 = miller[3 * g], m1 = miller[3 * g + 1], m2 = miller[3 * g + 2];
    double re = 0.0, im = 0.0;
    for (int i = 0; i < atoms; ++i) {
        const double f = weights[i] * factors[static_cast<std::size_t>(kinds[i]) * n + g];
        // exp(-2 pi i m.tau), the phase reduced to [-1, 1) turns first
        double x = m0 * tau[3 * i] + m1 * tau[3 * i + 1] + m2 * tau[3 * i + 2];
        x = 2.0 * (x - floor(x + 0.5));
        double s = 0.0, c = 0.0;
        sincospi(x, &s, &c);
        re += f * c;
        im -= f * s;
    }
    out[g] = make_cuDoubleComplex(re, im);
}

}  // namespace

void structure_sum(const int* miller, std::size_t n, const std::vector<double>& positions,
                   const std::vector<int>& kinds, const std::vector<double>& weights, const double* factors,
                   Complex* out) {
    const std::size_t atoms = kinds.size();
    if (positions.size() != 3 * atoms || weights.size() != atoms)
        throw std::invalid_argument("positions (atoms, 3), a kind and a weight for each atom");
    if (n == 0) return;
    if (atoms == 0) {
        check_cuda(cudaMemset(out, 0, n * sizeof(Complex)), "cudaMemset");
        return;
    }
    DeviceArray<double> tau, w;
    DeviceArray<int> k;
    tau.upload(positions.data(), positions.size());
    w.upload(weights.data(), weights.size());
    k.upload(kinds.data(), kinds.size());
    structure_kernel<<<static_cast<unsigned>((n + kThreads - 1) / kThreads), kThreads>>>(
        n, static_cast<int>(atoms), miller, tau.data(), k.data(), w.data(), factors, out);
    check_cuda(cudaGetLastError(), "the structure sums");
}

}  // namespace mdapy::dft
