// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "cuda/atoms.cuh"
#include "cuda/device.cuh"

namespace mdapy::cuda {

struct Neighbors::Impl {
    double cutoff = 0.0;
    double skin = 0.0;
    std::int64_t n = 0;
    std::int64_t builds = 0;
    int most = 0;                    // columns of the list
    Buffer<std::int32_t> indices;    // (most, n): neighbor k of atom i at k * n + i
    Buffer<std::int32_t> rows;       // (n, most): the same by atom, as a build writes it
    Buffer<std::int32_t> counts;     // (n,)
    // the state of the last build, to tell when to build again
    Buffer<double4a> x_built;
    Box box_built{};
    std::uint64_t atoms_version = 0;
    // the cells
    Buffer<std::int32_t> keys, keys_sorted, order, order_sorted;
    Buffer<std::int32_t> cell_begin, cell_end;
    Buffer<unsigned char> sort_scratch;
    Buffer<double> displacement;     // (n,) for the test of the skin
    Buffer<int> found;               // the most neighbors an atom has
    // How the pairs are reached. With cells, the code of a pair is the cell
    // of its neighbor around its own, d of -1, 0 or 1 along each vector, and
    // x_j - x_i = l_j - l_i + sum d_k h_k / n_k, l the position of each atom
    // from the corner of its cell: small numbers, which single precision
    // holds to a fraction of a micrometre whatever the size of the box.
    // Without (fewer than 3 cells along a periodic direction), the code is
    // the image of the box, n_k = 1 and l from the corner of the box.
    bool with_cells = false;
    int3 cells{1, 1, 1};
    Buffer<float4> local_single;
    Buffer<double4a> local_double;
    std::uint64_t local_version[2] = {0, 0};  // of the atoms they were made from
    Box local_box[2]{};

    // For each pair (i, j) of the list, the place of (j, i) in the list of j,
    // for many-body potentials, found once for each build.
    Buffer<std::int32_t> reverse;
    std::int64_t reverse_build = -1;
    const std::int32_t* reversed();

    // The positions l of the atoms for the potentials, in a precision.
    template <class Real>
    const Real4<Real>* local(const Atoms::Impl& atoms, const Box& box);
    // The vectors h_k / n_k of the codes of the pairs.
    template <class Real>
    Images<Real> steps(const Box& box) const;
};

}  // namespace mdapy::cuda
