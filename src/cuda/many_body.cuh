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

#include <stdexcept>

#include "cuda/device.cuh"
#include "cuda/neighbors.cuh"

namespace mdapy::cuda {

// The pairs of each atom closer than a cutoff, copied from the neighbor list
// with its skin, in the same columns.
template <class Real>
__global__ void short_pairs(std::int64_t n, const Real4<Real>* x, Images<Real> images,
                            const std::int32_t* indices, const std::int32_t* counts, Real cutsq,
                            std::int32_t* short_indices, std::int32_t* short_counts) {
    __shared__ Real steps[81];
    fill_steps(images, steps);
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Real4<Real> xi = x[i];
    const int count = counts[i];
    int kept = 0;
    for (int k = 0; k < count; ++k) {
        const std::int32_t entry = indices[static_cast<std::int64_t>(k) * n + i];
        const auto packed = static_cast<std::uint32_t>(entry);
        const Real4<Real> xj = x[packed & kIndexMask];
        Real r[3] = {xj.x - xi.x, xj.y - xi.y, xj.z - xi.z};
        add_step(steps, packed >> kIndexBits, r[0], r[1], r[2]);
        if (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] < cutsq) {
            short_indices[static_cast<std::int64_t>(kept) * n + i] = entry;
            ++kept;
        }
    }
    short_counts[i] = kept;
}

// For each pair (i, j) of the short lists, the place of (j, i) in the list
// of j; -1 where it is missing, and missing set. A template, so that only
// the files that use it compile it.
template <class Index>
__global__ void short_reverse(std::int64_t n, const Index* indices, const Index* counts,
                              Index* reverse, int* missing) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int count = counts[i];
    for (int k = 0; k < count; ++k) {
        const auto packed = static_cast<std::uint32_t>(indices[static_cast<std::int64_t>(k) * n + i]);
        const std::int64_t j = packed & kIndexMask;
        const std::uint32_t wanted =
            static_cast<std::uint32_t>(i) | ((26u - (packed >> kIndexBits)) << kIndexBits);
        int found = -1;
        const int others = counts[j];
        for (int m = 0; m < others; ++m) {
            if (static_cast<std::uint32_t>(indices[static_cast<std::int64_t>(m) * n + j]) == wanted) {
                found = m;
                break;
            }
        }
        reverse[static_cast<std::int64_t>(k) * n + i] = found;
        if (found < 0) atomicExch(missing, 1);
    }
}

// The pairs of each atom within the cutoff of a many-body potential, made
// again at every step from the neighbor list and its skin, so that the loops
// over the third atoms of a centre run over the pairs that count: for
// silicon 4 of the 16 within the cutoff and the skin. Both ends of a pair
// compute the same r^2, x_i - x_j being exactly -(x_j - x_i), so the short
// lists are symmetric and each pair has its reverse; checked once for each
// build of the neighbor list.
class ShortPairs {
public:
    template <class Real>
    void build(const Neighbors::Impl& l, const Real4<Real>* x, const Images<Real>& images,
               Real cutsq) {
        const std::int64_t n = l.n;
        const auto size = static_cast<std::size_t>(l.most) * static_cast<std::size_t>(n);
        indices_.resize(size);
        reverse_.resize(size);
        counts_.resize(static_cast<std::size_t>(n));
        const unsigned g = blocks(n);
        short_pairs<Real><<<g, kBlock>>>(n, x, images, l.indices.data(), l.counts.data(), cutsq,
                                         indices_.data(), counts_.data());
        MDAPY_LAUNCHED();
        const bool check = checked_build_ != l.builds;
        if (check) {
            missing_.resize(1);
            missing_.zero();
        }
        short_reverse<std::int32_t><<<g, kBlock>>>(n, indices_.data(), counts_.data(),
                                                   reverse_.data(), missing_.data());
        MDAPY_LAUNCHED();
        if (check) {
            int missing = 0;
            missing_.download(&missing, 1);
            if (missing) {
                throw std::logic_error("the pairs within the cutoff are not symmetric");
            }
            checked_build_ = l.builds;
        }
    }

    const std::int32_t* indices() const { return indices_.data(); }
    const std::int32_t* counts() const { return counts_.data(); }
    const std::int32_t* reverse() const { return reverse_.data(); }

private:
    Buffer<std::int32_t> indices_, counts_, reverse_;
    Buffer<int> missing_;
    std::int64_t checked_build_ = -1;
};

// derivative: 3 planes of plane values, c plane + k n + i for the pair k of
// atom i; share_energy 1 plane, share_virial 9. The energies and virials of
// the centres are in energies and virials already; own_force, (n, 3) or
// null, the forces atoms found on themselves.
template <class Real, bool Energy, bool Virial, bool Atomic>
__global__ void gather_shares(std::int64_t n, std::int64_t plane, const std::int32_t* indices,
                              const std::int32_t* counts, const std::int32_t* reverse,
                              const Real* derivative, const double* share_energy,
                              const double* share_virial, const double* own_force,
                              double* forces, double* energies, double* virials) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double force[3] = {0.0, 0.0, 0.0};
    if (own_force != nullptr) {
        for (int c = 0; c < 3; ++c) force[c] = own_force[3 * i + c];
    }
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
