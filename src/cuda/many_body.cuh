// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The second pass of many-body potentials on the device, as
// calculators/many_body.h gathers: the force on each atom from the
// derivatives of its pairs and of their reverses, and, when the atoms' own
// energies and virials are asked for, the shares of the terms of others.

#pragma once

#include "cuda/device.cuh"

namespace mdapy::cuda {

// derivative: 3 planes of plane values, c plane + k n + i for the pair k of
// atom i; share_energy 1 plane, share_virial 9. The energies and virials of
// the centres are in energies and virials already.
template <class Real, bool Energy, bool Virial, bool Atomic>
__global__ void gather_shares(std::int64_t n, std::int64_t plane, const std::int32_t* indices,
                              const std::int32_t* counts, const std::int32_t* reverse,
                              const Real* derivative, const double* share_energy,
                              const double* share_virial, double* forces, double* energies,
                              double* virials) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double force[3] = {0.0, 0.0, 0.0};
    double energy = 0.0;
    double w[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const int count = counts[i];
    for (int k = 0; k < count; ++k) {
        const std::int64_t p = static_cast<std::int64_t>(k) * n + i;
        const std::int64_t j = static_cast<std::uint32_t>(indices[p]) & kIndexMask;
        const std::int64_t q = static_cast<std::int64_t>(reverse[p]) * n + j;
        for (int c = 0; c < 3; ++c) {
            force[c] += static_cast<double>(derivative[c * plane + p]) -
                        static_cast<double>(derivative[c * plane + q]);
        }
        if (Atomic && Energy) energy += share_energy[q];
        if (Atomic && Virial) {
            for (int c = 0; c < 9; ++c) w[c] += share_virial[c * plane + q];
        }
    }
    for (int c = 0; c < 3; ++c) forces[3 * i + c] = force[c];
    if (Atomic && Energy) energies[i] += energy;
    if (Atomic && Virial) {
        for (int c = 0; c < 9; ++c) virials[static_cast<std::int64_t>(c) * n + i] += w[c];
    }
}

}  // namespace mdapy::cuda
