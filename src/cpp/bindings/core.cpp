// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>

#include "bindings/control.h"
#include "core/parallel.h"

namespace mdapy::bindings {

void bind_core(nb::module_& m) {
    m.def("get_num_threads", &get_num_threads);
    m.def("set_num_threads", &set_num_threads, nb::arg("n"));
    m.def("hardware_threads", &hardware_threads);
    m.def("default_grain", &default_grain, nb::arg("n"));

    nb::class_<CancelToken>(m, "CancelToken")
        .def(nb::init<>())
        .def("cancel", &CancelToken::cancel)
        .def("reset", &CancelToken::reset)
        .def_prop_ro("cancelled", &CancelToken::cancelled);
}

}  // namespace mdapy::bindings
