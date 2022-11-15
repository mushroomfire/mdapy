#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <deque>

namespace py = pybind11;
using namespace std;

int get_cluster(py::array verlet_list, py::array distance_list, double rc, py::array particleClusters);