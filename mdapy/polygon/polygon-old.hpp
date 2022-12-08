// Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
// This file is from the mdapy project, released under the BSD 3-Clause License.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <tuple>
#include <pybind11/stl.h>
#include <cmath>

namespace py = pybind11;
using namespace std;

tuple<vector<vector<vector<int>>>, vector<vector<vector<double>>>, vector<double>, vector<double>> get_cell_info(py::array pos, py::array box, py::array boundary);