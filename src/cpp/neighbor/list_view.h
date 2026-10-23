// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// A neighbor list read where it is, without a copy.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "core/index.h"
#include "neighbor/frame.h"

namespace mdapy::neighbor {

// The pairs of atom i are offsets[i] to offsets[i + 1] of a list; with
// count > 0 only the first count of them, the nearest once sort_nearest has
// put them first; with limits only those closer than the limit of the types
// of their atoms, and otherwise with cutoff > 0 only those closer than it,
// as within keeps them. Their vectors come from pair_vector, as the searches
// compute them, so an analysis of a view gives what it gives for a list of
// just these pairs. When the first sorted pairs of each atom are its
// nearest, nearest first, a pair among them beyond the cutoff ends the
// pairs within it: the others are farther.
struct ListView {
    const double* positions;  // n x 3
    const double* matrix;     // rows a, b, c
    const std::int64_t* offsets;
    const index_t* neighbors;
    const std::int8_t* shifts;
    const double* distances;  // read with a cutoff or limits only
    std::int64_t count = 0;
    double cutoff = 0.0;
    std::int64_t sorted = 0;  // the nearest pairs of each atom first, in order
    const std::int32_t* type = nullptr;  // of each atom, with limits
    int types = 0;
    const double* limits = nullptr;  // types x types

    // Whether pair p of atom i counts by its distance.
    bool close(std::int64_t i, std::int64_t p) const {
        if (limits != nullptr) {
            return distances[p] < limits[static_cast<std::size_t>(type[i]) * types +
                                         static_cast<std::size_t>(type[neighbors[p]])];
        }
        return !(cutoff > 0.0) || distances[p] < cutoff;
    }

    // Calls visit(p, j, r) for each pair p of atom i that counts, j its
    // second atom and r its vector; returns how many count.
    template <class Visit>
    std::int64_t for_each(std::int64_t i, Visit&& visit) const {
        const std::int64_t begin = offsets[i];
        const std::int64_t end = count > 0 ? std::min(offsets[i + 1], begin + count) : offsets[i + 1];
        // with one cutoff, the sorted pairs end at the first beyond it
        const std::int64_t ordered =
            limits == nullptr && cutoff > 0.0 ? std::min(end, begin + sorted) : begin;
        std::int64_t m = 0;
        for (std::int64_t p = begin; p < end; ++p) {
            if (!close(i, p)) {
                if (p < ordered) {
                    break;
                }
                continue;
            }
            double r[3];
            pair_vector(positions + 3 * i, positions + 3 * static_cast<std::int64_t>(neighbors[p]),
                        shifts + 3 * p, matrix, r);
            visit(p, neighbors[p], static_cast<const double*>(r));
            ++m;
        }
        return m;
    }

    // The vectors (3 each) of the first most pairs of atom i that count into
    // vectors, and their second atoms into atoms when not null; returns how
    // many count, which may be more than most.
    std::int64_t gather(std::int64_t i, double* vectors, std::int64_t most,
                        index_t* atoms = nullptr) const {
        return for_each(i, [&, m = std::int64_t{0}](std::int64_t, index_t j, const double* r) mutable {
            if (m < most) {
                std::copy_n(r, 3, vectors + 3 * m);
                if (atoms != nullptr) {
                    atoms[m] = j;
                }
            }
            ++m;
        });
    }
};

}  // namespace mdapy::neighbor
