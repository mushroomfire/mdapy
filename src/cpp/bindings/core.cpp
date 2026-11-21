// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

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

using Triples = nb::ndarray<double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using Masses = nb::ndarray<const double, nb::shape<-1>, nb::c_contig, nb::device::cpu>;

// The steps of the velocity Verlet integrator, atom by atom as NumPy computes
// them, in one pass: velocities += (c forces) / masses, and with dt
// positions += dt velocities after.
void kick_drift(Triples velocities, Positions forces, Masses masses, double c, Triples* positions,
                double dt) {
    const auto n = static_cast<std::int64_t>(velocities.shape(0));
    if (forces.shape(0) != velocities.shape(0) || masses.shape(0) != velocities.shape(0) ||
        (positions != nullptr && positions->shape(0) != velocities.shape(0))) {
        throw std::invalid_argument("the arrays of the atoms differ in length");
    }
    double* v = velocities.data();
    const double* f = forces.data();
    const double* m = masses.data();
    double* x = positions != nullptr ? positions->data() : nullptr;
    nb::gil_scoped_release release;
    parallel_for(n, 0, [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
            for (int k = 0; k < 3; ++k) {
                v[3 * i + k] += (c * f[3 * i + k]) / m[i];
                if (x != nullptr) x[3 * i + k] += dt * v[3 * i + k];
            }
        }
    });
}

// The largest over the atoms of the square of how far each moved from built,
// as NumPy sums the squares of an atom and takes the largest.
double largest_step(Positions positions, Positions built) {
    if (positions.shape(0) != built.shape(0)) {
        throw std::invalid_argument("the arrays of the atoms differ in length");
    }
    const auto n = static_cast<std::int64_t>(positions.shape(0));
    const double* x = positions.data();
    const double* b = built.data();
    nb::gil_scoped_release release;
    return parallel_reduce(
        n, 0, 0.0,
        [&](std::int64_t begin, std::int64_t end) {
            double most = 0.0;
            for (std::int64_t i = begin; i < end; ++i) {
                const double d0 = x[3 * i] - b[3 * i], d1 = x[3 * i + 1] - b[3 * i + 1],
                             d2 = x[3 * i + 2] - b[3 * i + 2];
                const double r2 = d0 * d0 + d1 * d1 + d2 * d2;
                // a step that is not a number wins, as NumPy's max gives it
                if (std::isnan(r2)) return r2;
                if (r2 > most) most = r2;
            }
            return most;
        },
        [](double a, double b2) { return std::isnan(a) || std::isnan(b2) ? a + b2 : (b2 > a ? b2 : a); });
}

}  // namespace

void bind_core(nb::module_& m) {
    m.def(
        "kick", [](Triples v, Positions f, Masses masses, double c) {
            kick_drift(v, f, masses, c, nullptr, 0.0);
        },
        nb::arg("velocities"), nb::arg("forces"), nb::arg("masses"), nb::arg("c"));
    m.def(
        "kick_drift",
        [](Triples v, Positions f, Masses masses, double c, Triples x, double dt) {
            kick_drift(v, f, masses, c, &x, dt);
        },
        nb::arg("velocities"), nb::arg("forces"), nb::arg("masses"), nb::arg("c"),
        nb::arg("positions"), nb::arg("dt"));
    m.def("largest_step", &largest_step, nb::arg("positions"), nb::arg("built"));
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
