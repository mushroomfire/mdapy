// Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
// This file is from the mdapy project, released under the BSD 3-Clause License.

#include "cluster.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <deque>

namespace py = pybind11;
using namespace std;

PYBIND11_PLUGIN(_cluster_analysis)
{
    pybind11::module m("_cluster_analysis", "auto-compiled c++ extension for computing the atomic cluster.");
    m.def("get_cluster", &get_cluster);
    return m.ptr();
}