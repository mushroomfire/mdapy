#include "type.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "spline.h"

namespace nb = nanobind;

NB_MODULE(_spline, m)
{
     m.doc() = "Cubic spline interpolation module with OpenMP acceleration";

     nb::class_<CubicSpline>(m, "CubicSpline")
         .def(nb::init<const ROneArrayD, const ROneArrayD>(),
              nb::arg("x"), nb::arg("y"),
              "Create a cubic spline from numpy arrays (clamped boundary)")
         .def("__call__",
              nb::overload_cast<double>(&CubicSpline::evaluate, nb::const_),
              nb::arg("x"),
              "Evaluate the spline at point x")
         // 单点计算
         .def("evaluate",
              nb::overload_cast<double>(&CubicSpline::evaluate, nb::const_),
              nb::arg("x"),
              "Evaluate the spline at point x")
         .def("derivative",
              nb::overload_cast<double>(&CubicSpline::derivative, nb::const_),
              nb::arg("x"),
              "Evaluate the first derivative at point x")
         // 批量计算（numpy array）
         .def("evaluate",
              nb::overload_cast<const ROneArrayD &>(&CubicSpline::evaluate, nb::const_),
              nb::arg("x_vals"),
              "Evaluate the spline at multiple points (returns numpy array)")
         .def("derivative",
              nb::overload_cast<const ROneArrayD &>(&CubicSpline::derivative, nb::const_),
              nb::arg("x_vals"),
              "Evaluate the first derivative at multiple points (returns numpy array)");
}