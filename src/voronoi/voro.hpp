#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <pybind11/stl.h>
#include <cmath>

namespace py = pybind11;
using namespace std;

void get_voronoi_volume(py::array pos, py::array box, py::array boundary, py::array vol, py::array neighbor_number, py::array cavity_radius);