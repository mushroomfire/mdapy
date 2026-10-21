// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Geometry of bonds: clusters of bonded atoms and the distribution of bond
// angles.

#pragma once

#include <cstdint>

#include "core/parallel.h"

namespace mdapy::analysis {

// The cluster of each of n atoms, given its bonds in CSR form: clusters are
// numbered from 0 by decreasing size, ties by their first atom. Returns the
// number of clusters; sizes (at least n long) gets the size of each.
std::int64_t clusters(std::int64_t n, const std::int64_t* offsets, const std::int32_t* neighbors,
                      std::int64_t* cluster, std::int64_t* sizes);

// Counts of the angles between the bonds of each atom, given as vectors in
// CSR form, in bins of equal width from 0 to 180 degrees.
void bond_angles(std::int64_t n, const std::int64_t* offsets, const double* vectors, int bins,
                 std::int64_t* counts, const Control& control = {});

}  // namespace mdapy::analysis
