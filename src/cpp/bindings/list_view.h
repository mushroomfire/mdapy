// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// A neighbor list given to a kernel as its arrays, read where it is.

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <stdexcept>

#include "core/index.h"
#include "neighbor/list_view.h"

namespace mdapy::bindings::lists {

namespace nb = nanobind;

using Positions = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Matrix = nb::ndarray<const double, nb::shape<3, 3>, nb::c_contig, nb::device::cpu>;
using Offsets = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Neighbors = nb::ndarray<const index_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Shifts = nb::ndarray<const std::int8_t, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Distances = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Types = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Limits = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// The list of the atoms at positions, its first sorted pairs per atom its
// nearest in order, as a view of its first count pairs per atom (all when
// 0) closer than cutoff (all when 0). Throws std::invalid_argument when the
// arrays do not fit each other.
inline neighbor::ListView view(Positions positions, Matrix matrix, Offsets offsets,
                               Neighbors neighbors, Shifts shifts, Distances distances,
                               std::int64_t sorted, std::int64_t count, double cutoff) {
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    const std::size_t pairs = neighbors.shape(0);
    if (offsets.shape(0) != static_cast<std::size_t>(n) + 1 || shifts.shape(0) != pairs ||
        distances.shape(0) != pairs || offsets.data()[n] != static_cast<std::int64_t>(pairs)) {
        throw std::invalid_argument("the pairs do not fit the atoms");
    }
    if (count < 0 || cutoff < 0.0 || sorted < 0) {
        throw std::invalid_argument("sorted, count and cutoff must not be negative");
    }
    return {positions.data(), matrix.data(), offsets.data(), neighbors.data(),
            shifts.data(),    distances.data(), count, cutoff, sorted};
}

// The view with a limit per pair of types (types x types) instead of its
// cutoff, type giving each atom's type.
inline neighbor::ListView with_limits(neighbor::ListView list, std::int64_t n, Types type,
                                      Limits limits) {
    const auto types = static_cast<int>(limits.shape(0));
    if (types < 1 || limits.shape(1) != limits.shape(0) ||
        type.shape(0) != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("limits must be a square table and type one per atom");
    }
    for (std::int64_t i = 0; i < n; ++i) {
        if (type.data()[i] < 0 || type.data()[i] >= types) {
            throw std::invalid_argument("a type is outside the table of limits");
        }
    }
    list.type = type.data();
    list.types = types;
    list.limits = limits.data();
    return list;
}

}  // namespace mdapy::bindings::lists
