#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <tuple>
#include <pybind11/stl.h>
#include <cmath>

namespace py = pybind11;
using namespace std;

tuple<vector<vector< vector<int> >>, vector<vector< vector<double> >>, vector<double>, vector<double>> get_cell_info(py::array pos, py::array box, py::array boundary);