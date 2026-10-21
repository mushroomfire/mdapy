// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "analysis/distribution.h"

#include <algorithm>
#include <cmath>

#include "neighbor/cell_list.h"

namespace mdapy::analysis {

std::vector<std::int64_t> pair_histogram(const double* positions, std::int64_t n,
                                         const double* matrix, const double* origin,
                                         const bool* pbc, double r_max, int bins,
                                         const std::int32_t* type, int types,
                                         const Control& control) {
    const neighbor::CellGrid grid(positions, n, matrix, origin, pbc, r_max);
    using Histogram = std::vector<std::int64_t>;
    const std::size_t size = static_cast<std::size_t>(types) * types * bins;
    return parallel_reduce(
        n, 0, Histogram(size, 0),
        [&](std::int64_t begin, std::int64_t end) {
            Histogram local(size, 0);
            for (std::int64_t i = begin; i < end; ++i) {
                const std::size_t row = static_cast<std::size_t>(type[i]) * types;
                grid.visit(i, [&](std::int32_t j, const std::int32_t*, double r2) {
                    const int bin =
                        std::min(static_cast<int>(std::sqrt(r2) / r_max * bins), bins - 1);
                    ++local[(row + type[j]) * bins + bin];
                });
            }
            return local;
        },
        [](Histogram a, const Histogram& b) {
            for (std::size_t k = 0; k < a.size(); ++k) {
                a[k] += b[k];
            }
            return a;
        },
        control);
}

}  // namespace mdapy::analysis
