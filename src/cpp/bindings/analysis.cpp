// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Bindings of the analyses.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "analysis/steinhardt.h"
#include "bindings/control.h"
#include "core/parallel.h"

namespace mdapy::bindings {
namespace {

using Offsets = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Neighbors = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Vectors = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;

nb::object steinhardt(Offsets offsets, Neighbors neighbors, Vectors vectors,
                      const std::vector<int>& degrees, bool average, nb::handle progress,
                      const CancelToken* cancel) {
    if (offsets.shape(0) < 1 || neighbors.shape(0) != vectors.shape(0) ||
        offsets.data()[offsets.shape(0) - 1] != static_cast<std::int64_t>(vectors.shape(0))) {
        throw std::invalid_argument("the offsets do not fit the vectors");
    }
    const auto n = static_cast<std::int64_t>(offsets.shape(0)) - 1;
    auto* q = new double[static_cast<std::size_t>(n) * degrees.size()];
    nb::capsule owner(q, [](void* p) noexcept { delete[] static_cast<double*>(p); });
    const std::size_t shape[2] = {static_cast<std::size_t>(n), degrees.size()};
    nb::object result = nb::ndarray<nb::numpy, double>(q, 2, shape, owner).cast();
    run_released(
        [&](const Control& control) {
            analysis::steinhardt(n, offsets.data(), neighbors.data(), vectors.data(), degrees,
                                 average, q, control);
        },
        progress, cancel);
    return result;
}

}  // namespace

void bind_analysis(nb::module_& m) {
    m.def("steinhardt", &steinhardt, nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("vectors"), nb::arg("degrees"), nb::arg("average"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
}

}  // namespace mdapy::bindings
