#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
// ReadOnly
using ROneArrayD = nb::ndarray<double, nb::ro, nb::ndim<1>>;
using RTwoArrayD = nb::ndarray<double, nb::ro, nb::ndim<2>>;
using RThreeArrayD = nb::ndarray<double, nb::ro, nb::ndim<3>>;
using ROneArrayI = nb::ndarray<int, nb::ro, nb::ndim<1>>;
using RTwoArrayI = nb::ndarray<int, nb::ro, nb::ndim<2>>;
using RThreeArrayI = nb::ndarray<int, nb::ro, nb::ndim<3>>;
// WriteRead
using OneArrayI = nb::ndarray<int, nb::ndim<1>>;
using TwoArrayI = nb::ndarray<int, nb::ndim<2>>;
using ThreeArrayI = nb::ndarray<int, nb::ndim<3>>;
using OneArrayD = nb::ndarray<double, nb::ndim<1>>;
using TwoArrayD = nb::ndarray<double, nb::ndim<2>>;
using ThreeArrayD = nb::ndarray<double, nb::ndim<3>>;