// Copyright (c) 2022-2026, Yongchao Wu in Aalto University
// This file is from the mdapy project, released under the BSD 3-Clause License.
#include "type.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include "spline.h"

namespace nb = nanobind;

NB_MODULE(_spline, m)
{
     m.doc() = "Cubic spline interpolation module with OpenMP acceleration";

     nb::enum_<CubicSpline::BCType>(m, "BCType")
         .value("NotAKnot", CubicSpline::BCType::NotAKnot)
         .value("Natural", CubicSpline::BCType::Natural)
         .value("Clamped", CubicSpline::BCType::Clamped);

     nb::class_<CubicSpline>(m, "CubicSpline")
         .def(nb::init<const ROneArrayD, const ROneArrayD, CubicSpline::BCType>(),
              nb::arg("x"), nb::arg("y"),
              nb::arg("bc_type") = CubicSpline::BCType::NotAKnot,
              "Create a cubic spline with the chosen boundary condition")
         .def(nb::init<const ROneArrayD, const ROneArrayD, double, double>(),
              nb::arg("x"), nb::arg("y"), nb::arg("dy0"), nb::arg("dyn"),
              "Create a clamped cubic spline with user-supplied endpoint derivatives")
         .def("__call__",
              nb::overload_cast<double>(&CubicSpline::evaluate, nb::const_),
              nb::arg("x"),
              "Evaluate the spline at point x")
         // Single-point
         .def("evaluate",
              nb::overload_cast<double>(&CubicSpline::evaluate, nb::const_),
              nb::arg("x"), "Evaluate the spline at point x")
         .def("derivative",
              nb::overload_cast<double>(&CubicSpline::derivative, nb::const_),
              nb::arg("x"), "Evaluate the first derivative at point x")
         .def("second_derivative",
              nb::overload_cast<double>(&CubicSpline::second_derivative, nb::const_),
              nb::arg("x"), "Evaluate the second derivative at point x")
         // Batch (numpy array). Disambiguate by explicit member-pointer cast
         // (can't use nb::overload_cast here because the array overloads
         // return nb::ndarray, while the scalar ones return double).
         .def("evaluate",
              static_cast<nb::ndarray<nb::numpy, double> (CubicSpline::*)(const ROneArrayD &) const>(&CubicSpline::evaluate),
              nb::arg("x_vals"),
              "Evaluate the spline at multiple points (returns numpy array)")
         .def("derivative",
              static_cast<nb::ndarray<nb::numpy, double> (CubicSpline::*)(const ROneArrayD &) const>(&CubicSpline::derivative),
              nb::arg("x_vals"),
              "Evaluate the first derivative at multiple points (returns numpy array)")
         .def("second_derivative",
              static_cast<nb::ndarray<nb::numpy, double> (CubicSpline::*)(const ROneArrayD &) const>(&CubicSpline::second_derivative),
              nb::arg("x_vals"),
              "Evaluate the second derivative at multiple points (returns numpy array)");
}
