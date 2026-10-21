// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Glue between Python and mdapy::parallel_for: every kernel binding runs its
// work through run_released so that the GIL is released, progress callbacks
// and cancellation tokens work, and Ctrl-C stops the computation.

#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <optional>
#include <utility>

#include "core/parallel.h"

namespace mdapy::bindings {

namespace nb = nanobind;

// Runs work(control) with the GIL released.
//
// progress: optional Python callable progress(done, total), or None.
// cancel:   optional CancelToken, or nullptr.
//
// While the work runs, the calling thread periodically takes the GIL back to
// call progress and to check for pending signals (Ctrl-C). An exception raised
// by progress, or a KeyboardInterrupt, cancels the work and is re-raised here.
template <class Work>
void run_released(Work&& work, nb::handle progress, const CancelToken* cancel) {
    std::optional<nb::python_error> pending;
    Control control;
    control.cancel = cancel;
    control.monitor = [&](std::int64_t done, std::int64_t total) -> bool {
        nb::gil_scoped_acquire gil;
        if (pending) {
            return false;
        }
        try {
            if (PyErr_CheckSignals() != 0) {
                throw nb::python_error();
            }
            if (progress.is_valid() && !progress.is_none()) {
                progress(done, total);
            }
            return true;
        } catch (nb::python_error& error) {
            pending.emplace(std::move(error));
            return false;
        }
    };
    try {
        nb::gil_scoped_release release;
        work(static_cast<const Control&>(control));
    } catch (const Cancelled&) {
        if (pending) {
            throw nb::python_error(std::move(*pending));
        }
        throw;
    }
    if (pending) {
        throw nb::python_error(std::move(*pending));
    }
}

void bind_core(nb::module_& m);
void bind_io(nb::module_& m);
void bind_testing(nb::module_& m);

}  // namespace mdapy::bindings
