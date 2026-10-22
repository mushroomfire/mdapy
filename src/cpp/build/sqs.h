// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Special quasirandom structures: elements arranged on fixed sites so that
// the pair correlations of the nearest shells are those of a random alloy.

#pragma once

#include <cstdint>
#include <vector>

#include "core/index.h"
#include "core/parallel.h"

namespace mdapy::build {

struct SqsResult {
    double objective;       // of the configuration returned
    double start;           // of the configuration given
    std::int64_t accepted;  // swaps accepted
};

// The deviation of the pair correlations from those of a random alloy:
// sum over shells s of weights[s] times the sum over elements a, b of
// (C_sab / P_s - x_a x_b)^2, with C_sab the ordered pairs of shell s from an
// atom of a to one of b, P_s all pairs of shell s and x the fractions.
double sqs_objective(std::int64_t n, const std::int64_t* offsets, const index_t* neighbors,
                     const std::int8_t* shell, int shells, int elements, const std::int32_t* type,
                     const double* weights);

// Simulated annealing of swaps of two atoms of different elements, from the
// elements type (from 0 to elements - 1, changed in place to the best
// configuration found), with the pairs of each atom in CSR form and the
// shell of each pair (from 0 to shells - 1; the list must hold both
// directions of every pair). The temperature falls linearly from
// temperature times the typical change of a swap to zero over steps.
SqsResult sqs(std::int64_t n, const std::int64_t* offsets, const index_t* neighbors,
              const std::int8_t* shell, int shells, int elements, std::int32_t* type,
              const double* weights, std::int64_t steps, double temperature, std::uint64_t seed,
              const Control& control = {});

}  // namespace mdapy::build
