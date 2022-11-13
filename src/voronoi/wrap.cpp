#include "voro.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <pybind11/stl.h>
#include <cmath>

namespace py = pybind11;
using namespace std;

PYBIND11_PLUGIN(_voronoi_analysis)
{
    pybind11::module m("_voronoi_analysis", "auto-compiled c++ extension for computing the atomic voronoi volume.");
    m.def("get_voronoi_volume", &get_voronoi_volume);
    return m.ptr();
}