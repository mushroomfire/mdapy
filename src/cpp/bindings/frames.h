// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Many frames one after another given to a kernel as their arrays.

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <stdexcept>

#include "neighbor/frames.h"

namespace mdapy::bindings::frames {

namespace nb = nanobind;

using Starts = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Positions = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Matrices = nb::ndarray<const double, nb::shape<-1, 3, 3>, nb::c_contig, nb::device::cpu>;
using Origins = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Flags = nb::ndarray<const bool, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

// The frames of the arrays: starts (frames + 1) from 0 to the atoms of
// positions, a box each. Throws std::invalid_argument when the arrays do not
// fit each other.
inline neighbor::Frames of(Starts starts, Positions positions, Matrices matrices, Origins origins,
                           Flags pbc) {
    const auto count = static_cast<std::int64_t>(matrices.shape(0));
    const std::int64_t* s = starts.data();
    if (static_cast<std::int64_t>(starts.shape(0)) != count + 1 ||
        static_cast<std::int64_t>(origins.shape(0)) != count ||
        static_cast<std::int64_t>(pbc.shape(0)) != count || s[0] != 0 ||
        s[count] != static_cast<std::int64_t>(positions.shape(0))) {
        throw std::invalid_argument("the arrays of the frames do not match");
    }
    for (std::int64_t k = 0; k < count; ++k) {
        if (s[k + 1] < s[k]) {
            throw std::invalid_argument("the starts of the frames decrease");
        }
    }
    neighbor::Frames frames;
    frames.count = count;
    frames.starts = s;
    frames.positions = positions.data();
    frames.matrices = matrices.data();
    frames.origins = origins.data();
    frames.pbc = pbc.data();
    return frames;
}

}  // namespace mdapy::bindings::frames
