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
#include <type_traits>

#include "cuda/device.cuh"
#include "cuda/neighbors.cuh"

namespace mdapy::cuda {

// The pairs of each atom closer than a cutoff, copied from the neighbor list
// with its skin, in the same columns.
// widest: the most pairs an atom keeps, reduced over each warp first.
template <class Real>
__global__ void short_pairs(std::int64_t n, const Real4<Real>* x, Images<Real> images,
                            const std::int32_t* indices, const std::int32_t* counts, Real cutsq,
                            std::int32_t* short_indices, std::int32_t* short_counts,
                            int* widest) {
    __shared__ Real steps[81];
    fill_steps(images, steps);
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int kept = 0;
    if (i < n) {
        const Real4<Real> xi = x[i];
        const int count = counts[i];
        // which of the first 64 pairs are kept, then those in order, so that
        // the threads of a warp write their k-th pairs together
        std::uint64_t mask = 0;
        const int first = count < 64 ? count : 64;
        for (int k = 0; k < count; ++k) {
            const std::int32_t entry = indices[static_cast<std::int64_t>(k) * n + i];
            const auto packed = static_cast<std::uint32_t>(entry);
            const Real4<Real> xj = x[packed & kIndexMask];
            Real r[3] = {xj.x - xi.x, xj.y - xi.y, xj.z - xi.z};
            add_step(steps, packed >> kIndexBits, r[0], r[1], r[2]);
            if (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] < cutsq) {
                if (k < first) {
                    mask |= std::uint64_t{1} << k;
                } else {
                    // beyond the mask: after the masked ones, at once
                    short_indices[static_cast<std::int64_t>(kept) * n + i] = entry;
                }
                ++kept;
            }
        }
        int place = 0;
        while (mask != 0) {
            const int k = __ffsll(static_cast<long long>(mask)) - 1;
            mask &= mask - 1;
            short_indices[static_cast<std::int64_t>(place) * n + i] =
                indices[static_cast<std::int64_t>(k) * n + i];
            ++place;
        }
        short_counts[i] = kept;
    }
    // every lane of the warp takes part: blocks are whole warps
    int most = kept;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const int other = __shfl_down_sync(0xffffffffu, most, offset);
        most = other > most ? other : most;
    }
    if ((threadIdx.x & 31u) == 0u) atomicMax(widest, most);
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
        widest_.resize(1);
        widest_.zero();
        const unsigned g = blocks(n);
        short_pairs<Real><<<g, kBlock>>>(n, x, images, l.indices.data(), l.counts.data(), cutsq,
                                         indices_.data(), counts_.data(), widest_.data());
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
        widest_.download(&widest_count_, 1);
    }

    const std::int32_t* indices() const { return indices_.data(); }
    const std::int32_t* counts() const { return counts_.data(); }
    const std::int32_t* reverse() const { return reverse_.data(); }
    // The values of planes indexed by pair, k n + i, after build: the most
    // pairs an atom has times the atoms, at least one.
    std::int64_t plane(std::int64_t n) const {
        return static_cast<std::int64_t>(widest_count_ > 0 ? widest_count_ : 1) * n;
    }

private:
    Buffer<std::int32_t> indices_, counts_, reverse_;
    Buffer<int> missing_, widest_;
    int widest_count_ = 0;
    std::int64_t checked_build_ = -1;
};

// derivative: by pair, k n + i for the pair k of atom i, its x, y and z in
// one Real4, one load for the reverse of a pair; share_energy 1 plane of
// plane values, share_virial 9. The energies and virials of the centres are
// in energies and virials already; own_force, (n, 3) or null, the forces
// atoms found on themselves. Mine: the derivatives of each atom's own pairs
// are read here too; else own_force holds their sum already.
template <class Real, bool Energy, bool Virial, bool Atomic, bool Mine>
__global__ void gather_shares(std::int64_t n, std::int64_t plane, const std::int32_t* indices,
                              const std::int32_t* counts, const std::int32_t* reverse,
                              const Real4<Real>* derivative, const double* share_energy,
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
        const Real4<Real> theirs = derivative[q];
        if (Mine) {
            const Real4<Real> mine = derivative[p];
            force[0] += static_cast<double>(mine.x);
            force[1] += static_cast<double>(mine.y);
            force[2] += static_cast<double>(mine.z);
        }
        force[0] -= static_cast<double>(theirs.x);
        force[1] -= static_cast<double>(theirs.y);
        force[2] -= static_cast<double>(theirs.z);
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

// The virial of each atom for the heat flux, that of Fan et al., PRB 92,
// 094301 (2015): the sum over its pairs (i, j) of r_ij (x) the derivative of
// the terms of j by the reverse pair, which is their derivative by x_i, as
// the processor's gather computes it with heat.
template <class Real>
__global__ void heat_virials(std::int64_t n, const Real4<Real>* x, Images<Real> images,
                             const std::int32_t* indices, const std::int32_t* counts,
                             const std::int32_t* reverse, const Real4<Real>* derivative,
                             double* virials) {
    __shared__ Real steps[81];
    fill_steps(images, steps);
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Real4<Real> xi = x[i];
    double w[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const int count = counts[i];
    for (int k = 0; k < count; ++k) {
        const std::int64_t p = static_cast<std::int64_t>(k) * n + i;
        const auto packed = static_cast<std::uint32_t>(indices[p]);
        const std::int64_t j = packed & kIndexMask;
        const Real4<Real> xj = x[j];
        Real r[3] = {xj.x - xi.x, xj.y - xi.y, xj.z - xi.z};
        add_step(steps, packed >> kIndexBits, r[0], r[1], r[2]);
        const Real4<Real> theirs = derivative[static_cast<std::int64_t>(reverse[p]) * n + j];
        const double d[3] = {static_cast<double>(theirs.x), static_cast<double>(theirs.y),
                             static_cast<double>(theirs.z)};
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) w[3 * a + b] += static_cast<double>(r[a]) * d[b];
        }
    }
    for (int c = 0; c < 9; ++c) virials[static_cast<std::int64_t>(c) * n + i] = w[c];
}

template <class Real>
void heat_virials(std::int64_t n, const Real4<Real>* x, const Images<Real>& images,
                  const ShortPairs& pairs, const Real4<Real>* derivative, double* virials) {
    heat_virials<Real><<<blocks(n), kBlock>>>(n, x, images, pairs.indices(), pairs.counts(),
                                              pairs.reverse(), derivative, virials);
    MDAPY_LAUNCHED();
}

// A vector as the Real4 of a precision, w zero.
template <class Real>
__device__ inline Real4<Real> four(double x, double y, double z) {
    Real4<Real> out;
    out.x = static_cast<Real>(x);
    out.y = static_cast<Real>(y);
    out.z = static_cast<Real>(z);
    out.w = Real(0);
    return out;
}

// Adds a vector to the derivative of a pair.
template <class Real>
__device__ inline void add_to(Real4<Real>* derivative, std::int64_t p, Real x, Real y, Real z) {
    Real4<Real> d = derivative[p];
    d.x += x;
    d.y += y;
    d.z += z;
    derivative[p] = d;
}

// A buffer of at least size elements, which keeps its memory when smaller
// ones are asked for.
template <class T>
void at_least(Buffer<T>& buffer, std::size_t size) {
    if (buffer.size() < size) buffer.resize(size + size / 8);
}

// The gather of a many-body potential: the force of each atom, and the
// shares of the atoms' own energies and virials when atomic.
// mine: read the derivatives of each atom's own pairs, else own_force holds
// their sum.
template <class Real>
void gather(std::int64_t n, std::int64_t plane, const ShortPairs& pairs,
            const Real4<Real>* derivative, const double* share_energy,
            const double* share_virial, const double* own_force, bool mine, bool energy,
            bool virial, bool atomic, double* forces, double* energies, double* virials) {
    const auto run = [&](auto kernel) {
        kernel<<<blocks(n), kBlock>>>(n, plane, pairs.indices(), pairs.counts(), pairs.reverse(),
                                      derivative, share_energy, share_virial, own_force, forces,
                                      energies, virials);
    };
    const auto pick = [&](auto with_mine) {
        constexpr bool m = decltype(with_mine)::value;
        if (atomic && energy && virial) run(gather_shares<Real, true, true, true, m>);
        else if (atomic && energy) run(gather_shares<Real, true, false, true, m>);
        else if (atomic && virial) run(gather_shares<Real, false, true, true, m>);
        else run(gather_shares<Real, false, false, false, m>);
    };
    if (mine) pick(std::true_type{});
    else pick(std::false_type{});
    MDAPY_LAUNCHED();
}

}  // namespace mdapy::cuda
