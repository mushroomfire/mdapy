# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Curves and surfaces through data: cubic splines along a line and on a
grid, least-squares polynomials and surfaces, functions with parameters fitted
by nonlinear least squares, and equations of state."""

from mdapy.fitting._eos import FORMS, EquationOfState, fit_eos
from mdapy.fitting._least_squares import (
    MODELS,
    CurveFit,
    Model,
    PolynomialFit,
    SurfaceFit,
    arrhenius,
    exponential,
    fit_curve,
    fit_polynomial,
    fit_surface,
    gaussian,
    lorentzian,
    power_law,
    stretched_exponential,
)
from mdapy.fitting._spline import BOUNDARIES, GridSpline, Spline

__all__ = [
    "BOUNDARIES",
    "FORMS",
    "MODELS",
    "CurveFit",
    "EquationOfState",
    "GridSpline",
    "Model",
    "PolynomialFit",
    "Spline",
    "SurfaceFit",
    "arrhenius",
    "exponential",
    "fit_curve",
    "fit_eos",
    "fit_polynomial",
    "fit_surface",
    "gaussian",
    "lorentzian",
    "power_law",
    "stretched_exponential",
]
