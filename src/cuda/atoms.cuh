// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <vector>

#include "cuda/device.cuh"

namespace mdapy::cuda {

struct Atoms::Impl {
    std::int64_t n = 0;
    // Under MPI the atoms of this process come first, then its ghosts: the
    // steps of the integrator and the sums over the atoms go through the
    // first owned; 0 while every atom is this process's.
    std::int64_t owned = 0;
    std::int64_t active() const { return owned > 0 ? owned : n; }
    // The index among all the atoms of the atom at each place, by which it
    // draws its noise; empty while that is its index in the caller's order.
    Buffer<std::int64_t> ids;
    // positions in double precision, the species of each atom as w
    Buffer<double4a> x;
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
    Buffer<double> velocities;  // (n, 3)
    Buffer<double> masses;      // (n,)
    Buffer<double> scratch;     // 6 columns of n, for the kinetic tensor
    Buffer<int3> images;        // (n,), the boxes the wrapping moved each atom by
    // The directions in which each atom may move, bit a for axis a of the
    // system, whose vectors are rotation times those of the atoms; empty when
    // every atom moves freely.
    Buffer<std::int32_t> movable;
    double rotation[9] = {};
    // Of the controls of a run, empty for none: the velocities the movable
    // mask sets (zero where atoms move freely) and the forces added to the
    // atoms, (n, 3) in the frame of the atoms; the groups of each atom, bit k
    // for group k.
    Buffer<double> prescribed;
    Buffer<double> added;
    Buffer<std::int32_t> groups;
    // Of a Switch: the forces of the potential while the other computes, and
    // the sites (n, 3) and constants (n,) of springs, empty for none.
    Buffer<double> stash;
    Buffer<double> sites;
    Buffer<double> spring_constants;
    // The atoms are sorted by the cells of the neighbor list when it is built,
    // so that neighbors are near in memory: original holds the atom of the
    // caller at each place; what comes in and goes out is in the caller's
    // order. Empty while the order is the caller's.
    Buffer<std::int32_t> original;
    mutable std::vector<std::int32_t> host_original;  // its copy, when not stale
    mutable bool stale = false;

    // The places of the caller's atoms, empty for the caller's own order.
    const std::vector<std::int32_t>& order() const;
    // out[p] = in[original[p]] and out[original[p]] = in[p], width values each.
    template <class T>
    void to_places(const T* in, T* out, int width) const;
    template <class T>
    void to_caller(const T* in, T* out, int width) const;
    // The atoms at their new places: the one at place order[p] goes to p.
    void permute(const std::int32_t* order);

    // The positions in a precision, the species as w.
    template <class Real>
    const Real4<Real>* positions();
};

}  // namespace mdapy::cuda
