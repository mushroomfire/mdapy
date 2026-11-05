// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <cstdint>

#include "bindings/control.h"
#include "core/buffer.h"
#include "core/hardware.h"
#include "core/parallel.h"
#include "core/wrap.h"

namespace mdapy::bindings {
namespace {

using Positions = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Matrix = nb::ndarray<const double, nb::shape<3, 3>, nb::c_contig, nb::device::cpu>;
using Vector3 = nb::ndarray<const double, nb::shape<3>, nb::c_contig, nb::device::cpu>;
using Flags = nb::ndarray<const bool, nb::shape<3>, nb::c_contig, nb::device::cpu>;

template <class T>
nb::object owned_array(Buffer<T>* values, std::size_t n) {
    nb::capsule owner(values, [](void* p) noexcept { delete static_cast<Buffer<T>*>(p); });
    const std::size_t shape[2] = {n, 3};
    return nb::ndarray<nb::numpy, T>(values->data(), 2, shape, owner).cast();
}

nb::tuple wrap_positions(Positions positions, Matrix matrix, Matrix inverse, Vector3 origin,
                         Flags pbc) {
    const std::size_t n = positions.shape(0);
    auto* wrapped = new Buffer<double>(3 * n);
    auto* image = new Buffer<std::int64_t>(3 * n);
    nb::object first = owned_array(wrapped, n), second = owned_array(image, n);
    run_released(
        [&](const Control& control) {
            wrap(positions.data(), static_cast<std::int64_t>(n), matrix.data(), inverse.data(),
                 origin.data(), pbc.data(), wrapped->data(), image->data(), control);
        },
        nb::none(), nullptr);
    return nb::make_tuple(first, second);
}

}  // namespace

void bind_core(nb::module_& m) {
    m.def("get_num_threads", &get_num_threads);
    m.def("set_num_threads", &set_num_threads, nb::arg("n"));
    m.def("hardware_threads", &hardware_threads);
    m.def("cpu_name", &cpu_name);
    m.def("memory_bytes", &memory_bytes);
    m.def("default_grain", &default_grain, nb::arg("n"));
    m.def("wrap", &wrap_positions, nb::arg("positions"), nb::arg("matrix"), nb::arg("inverse"),
          nb::arg("origin"), nb::arg("pbc"));

    nb::class_<CancelToken>(m, "CancelToken")
        .def(nb::init<>())
        .def("cancel", &CancelToken::cancel)
        .def("reset", &CancelToken::reset)
        .def_prop_ro("cancelled", &CancelToken::cancelled);
}

}  // namespace mdapy::bindings
