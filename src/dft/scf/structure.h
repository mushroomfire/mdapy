// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Sums of form factors over the atoms on a sphere of reciprocal space,
//   out(G) = sum_i w_i f_{s_i}(G) exp(-2 pi i m.tau_i),
// m the Miller indices of G and tau_i the fractional positions, as the first
// density and magnetization of the free atoms take them.

#pragma once

#include <cstddef>
#include <vector>

#include "dft/core/device.h"

namespace mdapy::dft {

// miller (n, 3), factors (species, n) and out (n) on the device; positions
// (atoms, 3), kinds and weights (atoms) on the host
void structure_sum(const int* miller, std::size_t n, const std::vector<double>& positions,
                   const std::vector<int>& kinds, const std::vector<double>& weights, const double* factors,
                   Complex* out);

}  // namespace mdapy::dft
