// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// The single compiled extension module mdapy._core. Each functional area
// registers its own submodule.

#include <nanobind/nanobind.h>

#include "bindings/control.h"
#include "core/index.h"
#include "core/parallel.h"

namespace nb = nanobind;

NB_MODULE(_core, m) {
    m.doc() = "Compiled kernels of mdapy (internal; use the Python API).";
    // 32, or 64 when built with MDAPY_INDEX64
    m.attr("index_bits") = 8 * sizeof(mdapy::index_t);

    nb::register_exception_translator([](const std::exception_ptr& p, void*) {
        try {
            std::rethrow_exception(p);
        } catch (const mdapy::Cancelled& e) {
            nb::object cls = nb::module_::import_("mdapy.errors").attr("Cancelled");
            PyErr_SetString(cls.ptr(), e.what());
        }
    });

    nb::module_ core = m.def_submodule("core", "Thread pool control.");
    mdapy::bindings::bind_core(core);

    nb::module_ io = m.def_submodule("io", "Text table parsing and formatting.");
    mdapy::bindings::bind_io(io);

    nb::module_ neighbor = m.def_submodule("neighbor", "Neighbor search.");
    mdapy::bindings::bind_neighbor(neighbor);

    nb::module_ calculators = m.def_submodule("calculators", "Compiled calculators.");
    mdapy::bindings::bind_calculators(calculators);

    nb::module_ analysis = m.def_submodule("analysis", "Analyses of the properties of systems.");
    mdapy::bindings::bind_analysis(analysis);

    nb::module_ build = m.def_submodule("build", "Structure builder.");
    mdapy::bindings::bind_build(build);

    nb::module_ identify = m.def_submodule("identify", "Structure identification.");
    mdapy::bindings::bind_identify(identify);

    nb::module_ visualization = m.def_submodule("visualization", "Rendering of pictures.");
    mdapy::bindings::bind_visualization(visualization);

    nb::module_ testing = m.def_submodule("testing", "Kernels used only by the test suite.");
    mdapy::bindings::bind_testing(testing);
}
