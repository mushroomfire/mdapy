// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Small kernels that exercise the thread pool from Python. They are internal
// and exist only for the test suite.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>

#include "bindings/control.h"
#include "core/parallel.h"
#include "io/powers_of_five.h"

namespace mdapy::bindings {
namespace {

using ConstVector = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using Vector = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Sum of values, reduced chunk by chunk in a fixed order.
double parallel_sum(ConstVector values, std::int64_t grain, nb::handle progress,
                    const CancelToken* cancel) {
    const double* v = values.data();
    const auto n = static_cast<std::int64_t>(values.shape(0));
    double result = 0.0;
    run_released(
        [&](const Control& control) {
            result = parallel_reduce(
                n, grain, 0.0,
                [&](std::int64_t begin, std::int64_t end) {
                    double sum = 0.0;
                    for (std::int64_t i = begin; i < end; ++i) {
                        sum += v[i];
                    }
                    return sum;
                },
                std::plus<double>(), control);
        },
        progress, cancel);
    return result;
}

// Squares values in place.
void parallel_square(Vector values, std::int64_t grain) {
    double* v = values.data();
    const auto n = static_cast<std::int64_t>(values.shape(0));
    run_released(
        [&](const Control& control) {
            parallel_for(
                n, grain,
                [&](std::int64_t begin, std::int64_t end) {
                    for (std::int64_t i = begin; i < end; ++i) {
                        v[i] *= v[i];
                    }
                },
                control);
        },
        nb::none(), nullptr);
}

// Sleeps for the given time per item; used to test progress and cancellation.
void sleep_items(std::int64_t n, double seconds_per_item, std::int64_t grain,
                 nb::handle progress, const CancelToken* cancel) {
    const auto pause = std::chrono::duration<double>(seconds_per_item);
    run_released(
        [&](const Control& control) {
            parallel_for(
                n, grain,
                [&](std::int64_t begin, std::int64_t end) {
                    for (std::int64_t i = begin; i < end; ++i) {
                        std::this_thread::sleep_for(pause);
                    }
                },
                control);
        },
        progress, cancel);
}

// Throws std::runtime_error when item `index` is processed.
void fail_at(std::int64_t n, std::int64_t index, std::int64_t grain) {
    run_released(
        [&](const Control& control) {
            parallel_for(
                n, grain,
                [&](std::int64_t begin, std::int64_t end) {
                    if (begin <= index && index < end) {
                        throw std::runtime_error("failure at item " + std::to_string(index));
                    }
                },
                control);
        },
        nb::none(), nullptr);
}

// Sum of all i * j for 0 <= i < n_outer, 0 <= j < n_inner, with the inner
// loop as a nested parallel region.
std::int64_t nested_sum(std::int64_t n_outer, std::int64_t n_inner) {
    std::int64_t result = 0;
    run_released(
        [&](const Control& control) {
            result = parallel_reduce(
                n_outer, 1, std::int64_t{0},
                [&](std::int64_t begin, std::int64_t end) {
                    std::int64_t sum = 0;
                    for (std::int64_t i = begin; i < end; ++i) {
                        sum += parallel_reduce(
                            n_inner, 1, std::int64_t{0},
                            [&](std::int64_t b, std::int64_t e) {
                                std::int64_t s = 0;
                                for (std::int64_t j = b; j < e; ++j) {
                                    s += i * j;
                                }
                                return s;
                            },
                            std::plus<std::int64_t>());
                    }
                    return sum;
                },
                std::plus<std::int64_t>(), control);
        },
        nb::none(), nullptr);
    return result;
}

// Number of distinct threads that ran chunks.
std::int64_t count_threads(std::int64_t n, double seconds_per_item) {
    std::mutex mutex;
    std::set<std::thread::id> ids;
    const auto pause = std::chrono::duration<double>(seconds_per_item);
    run_released(
        [&](const Control& control) {
            parallel_for(
                n, 1,
                [&](std::int64_t, std::int64_t) {
                    std::this_thread::sleep_for(pause);
                    std::lock_guard<std::mutex> lock(mutex);
                    ids.insert(std::this_thread::get_id());
                },
                control);
        },
        nb::none(), nullptr);
    return static_cast<std::int64_t>(ids.size());
}

// The table used by the float parser, as (high, low) pairs.
nb::list powers_of_five() {
    nb::list table;
    for (const auto& entry : io::kPowersOfFive) {
        table.append(nb::make_tuple(entry[0], entry[1]));
    }
    return table;
}

}  // namespace

void bind_testing(nb::module_& m) {
    m.def("powers_of_five", &powers_of_five);
    m.def("parallel_sum", &parallel_sum, nb::arg("values"), nb::arg("grain") = 0,
          nb::arg("progress") = nb::none(), nb::arg("cancel").none() = nullptr);
    m.def("parallel_square", &parallel_square, nb::arg("values"), nb::arg("grain") = 0);
    m.def("sleep_items", &sleep_items, nb::arg("n"), nb::arg("seconds_per_item"),
          nb::arg("grain") = 1, nb::arg("progress") = nb::none(),
          nb::arg("cancel").none() = nullptr);
    m.def("fail_at", &fail_at, nb::arg("n"), nb::arg("index"), nb::arg("grain") = 1);
    m.def("nested_sum", &nested_sum, nb::arg("n_outer"), nb::arg("n_inner"));
    m.def("count_threads", &count_threads, nb::arg("n"), nb::arg("seconds_per_item"));
}

}  // namespace mdapy::bindings
