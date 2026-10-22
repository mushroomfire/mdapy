// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Bindings of the structure builder.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "bindings/control.h"
#include "build/sqs.h"
#include "core/index.h"

namespace mdapy::bindings {
namespace {

using Offsets = nb::ndarray<const std::int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Neighbors = nb::ndarray<const index_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Shells = nb::ndarray<const std::int8_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Types = nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Values = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

void check_pairs(const Offsets& offsets, const Neighbors& neighbors, const Shells& shell,
                 const Types& type, const Values& weights, int shells) {
    const auto n = static_cast<std::int64_t>(type.shape(0));
    if (static_cast<std::int64_t>(offsets.shape(0)) != n + 1 ||
        offsets.data()[n] != static_cast<std::int64_t>(neighbors.shape(0)) ||
        shell.shape(0) != neighbors.shape(0) ||
        static_cast<int>(weights.shape(0)) != shells) {
        throw std::invalid_argument("the offsets, pairs, shells and weights do not fit");
    }
    for (std::size_t p = 0; p < neighbors.shape(0); ++p) {
        if (neighbors.data()[p] < 0 || neighbors.data()[p] >= n) {
            throw std::invalid_argument("a neighbor index is outside the atoms");
        }
    }
}

double sqs_objective(Offsets offsets, Neighbors neighbors, Shells shell, int shells, Types type,
                     int elements, Values weights) {
    check_pairs(offsets, neighbors, shell, type, weights, shells);
    return build::sqs_objective(static_cast<std::int64_t>(type.shape(0)), offsets.data(),
                                neighbors.data(), shell.data(), shells, elements, type.data(),
                                weights.data());
}

nb::tuple sqs(Offsets offsets, Neighbors neighbors, Shells shell, int shells, Types type,
              int elements, Values weights, std::int64_t steps, double temperature,
              std::uint64_t seed, nb::handle progress, const CancelToken* cancel) {
    check_pairs(offsets, neighbors, shell, type, weights, shells);
    const auto n = static_cast<std::int64_t>(type.shape(0));
    auto* out = new std::vector<std::int32_t>(type.data(), type.data() + n);
    nb::capsule owner(out, [](void* p) noexcept { delete static_cast<std::vector<std::int32_t>*>(p); });
    build::SqsResult result{};
    run_released(
        [&](const Control& control) {
            result = build::sqs(n, offsets.data(), neighbors.data(), shell.data(), shells, elements,
                                out->data(), weights.data(), steps, temperature, seed, control);
        },
        progress, cancel);
    const std::size_t shape[1] = {static_cast<std::size_t>(n)};
    return nb::make_tuple(nb::ndarray<nb::numpy, std::int32_t>(out->data(), 1, shape, owner),
                          result.objective, result.start, result.accepted);
}

}  // namespace

void bind_build(nb::module_& m) {
    m.def("sqs_objective", &sqs_objective, nb::arg("offsets"), nb::arg("neighbors"),
          nb::arg("shell"), nb::arg("shells"), nb::arg("type"), nb::arg("elements"),
          nb::arg("weights"));
    m.def("sqs", &sqs, nb::arg("offsets"), nb::arg("neighbors"), nb::arg("shell"),
          nb::arg("shells"), nb::arg("type"), nb::arg("elements"), nb::arg("weights"),
          nb::arg("steps"), nb::arg("temperature"), nb::arg("seed"),
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
}

}  // namespace mdapy::bindings
