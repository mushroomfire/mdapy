#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <pybind11/stl.h>
//#include <iostream>
#include <cmath>

namespace py = pybind11;
using namespace std;

void get_cell_info(py::array pos, py::array box, py::array boundary, py::array volume_radius, py::array face_vertices, py::array vertices_pos, py::array face_areas);