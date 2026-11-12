// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Three-phonon scattering: the rates at which the modes of one wave vector
// scatter with the pairs of modes of the wave vectors q' of a mesh and
// q'' = q - q', for the thermal conductivity in the relaxation-time
// approximation.

#pragma once

#include <complex>
#include <cstdint>

#include "core/parallel.h"

namespace mdapy::analysis {

// The modes and the third order force constants of one wave vector q and of
// count wave vectors q' with their q''. b = 3n is the number of modes.
//
// first (b x b): the polarization vectors of the modes of q over the square
//   roots of the masses, mode by mode, components (atom, axis) last.
// second, third (count x b x b): the same of q' and of q''.
// constants (count x b x b x b): the third order force constants of q' and
//   q'', components (atom, axis) of q, of q' and of q''.
// frequencies0 (b), frequencies1, frequencies2 (count x b): the angular
//   frequencies of the modes; modes at or below live carry nothing.
// occupation1, occupation2 (temperatures x count x b): the Bose-Einstein
//   occupations of the modes of q' and q''.
// joined, parted: with sigma <= 0, (count x b x b x b) weights of the deltas
//   of the frequencies, joined of w - w' - w'' and parted of
//   (w + w' - w'') - (w - w' + w''), in seconds per radian; with sigma > 0,
//   null and Gaussians of that width in rad/s.
struct ScatteringInput {
    std::int64_t count;
    std::int64_t modes;
    std::int64_t temperatures;
    const std::complex<double>* first;
    const std::complex<double>* second;
    const std::complex<double>* third;
    const std::complex<double>* constants;
    const double* frequencies0;
    const double* frequencies1;
    const double* frequencies2;
    const double* occupation1;
    const double* occupation2;
    const double* joined;
    const double* parted;
    double sigma;
    double live;
};

// Adds to rates (temperatures x b) the sum over q', q'' and their modes of
// |Phi|^2 / (w w' w'') times the deltas weighted by the occupations: the
// rate of each mode of q is this sum times a constant.
void three_phonon_rates(const ScatteringInput& input, double* rates,
                        const Control& control = {});

// The weights of linear tetrahedra in the conservation of energy of the
// scatterings of the modes of one wave vector, of frequencies levels (b),
// with the modes of count wave vectors q' and their q'' = q - q':
// first and second (count x g x 4 x b) hold the frequencies of the modes at
// the corners of the g tetrahedra around each q', the point first, and at
// the opposite steps around q''. Writes joined, the weights of
// w - w' - w'', and parted, those of (w + w' - w'') - (w - w' + w''), both
// (count x b x b x b) by the mode of q, of q' and of q'', per unit of the
// frequencies. Corners closer than same are one value.
void tetrahedron_triplet_weights(std::int64_t count, std::int64_t g, std::int64_t b,
                                 const double* first, const double* second, const double* levels,
                                 double same, double* joined, double* parted,
                                 const Control& control = {});

}  // namespace mdapy::analysis
