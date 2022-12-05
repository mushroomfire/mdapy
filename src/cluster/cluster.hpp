// Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
// This file is from the mdapy project, released under the BSD 3-Clause License.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <deque>

namespace py = pybind11;
using namespace std;

int get_cluster(py::array verlet_list, py::array distance_list, double rc, py::array particleClusters);