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
#include "core/philox.h"
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

using Indices = nb::ndarray<const std::int64_t, nb::shape<-1>, nb::c_contig, nb::device::cpu>;

// The four words of Philox4x64-10 for a counter and a key, for the tests.
nb::tuple philox_block(std::uint64_t c0, std::uint64_t c1, std::uint64_t c2, std::uint64_t c3,
                       std::uint64_t k0, std::uint64_t k1) {
    const random::Block out = random::philox({{c0, c1, c2, c3}}, k0, k1);
    return nb::make_tuple(out.v[0], out.v[1], out.v[2], out.v[3]);
}

// Three Gaussian numbers for each atom of indices, at a draw of a stream.
nb::object normal_noise(Indices indices, std::uint64_t seed, std::uint64_t stream,
                        std::uint64_t draw) {
    const std::size_t n = indices.shape(0);
    auto* out = new Buffer<double>(3 * n);
    nb::object array = owned_array(out, n);
    const std::int64_t* index = indices.data();
    double* values = out->data();
    run_released(
        [&](const Control& control) {
            parallel_for(
                static_cast<std::int64_t>(n), 0,
                [&](std::int64_t begin, std::int64_t end) {
                    for (std::int64_t i = begin; i < end; ++i) {
                        random::normal3(seed, stream, static_cast<std::uint64_t>(index[i]), draw,
                                        values + 3 * i);
                    }
                },
                control);
        },
        nb::none(), nullptr);
    return array;
}

}  // namespace

void bind_core(nb::module_& m) {
    m.def("philox", &philox_block, nb::arg("c0"), nb::arg("c1"), nb::arg("c2"), nb::arg("c3"),
          nb::arg("k0"), nb::arg("k1"));
    m.def("normals", &normal_noise, nb::arg("indices"), nb::arg("seed"), nb::arg("stream"),
          nb::arg("draw"));
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
