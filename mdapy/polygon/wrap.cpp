// Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
// This file is from the mdapy project, released under the BSD 3-Clause License.

#include "polygon.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
//#include <tuple>
#include <pybind11/stl.h>
#include <cmath>
//#include <iostream>

namespace py = pybind11;
using namespace std;

PYBIND11_PLUGIN(poly)
{
    pybind11::module m("poly", "auto-compiled c++ extension");
    m.def("get_cell_info", &get_cell_info);
    return m.ptr();
}