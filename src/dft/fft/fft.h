// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Batched 3D FFTs of cuFFT on boxes of n1 x n2 x n3 points, the third index the
// fastest, and the maps between the coefficients of a sphere of plane waves and
// a box. The conventions:
//   to_real:        f(r_j) = sum_G c_G exp(+i G.r_j), unnormalized;
//   to_reciprocal:  c_G = sum_j f(r_j) exp(-i G.r_j), unnormalized: the round
//                   trip multiplies by N = n1 n2 n3, which gather divides out.
// The coefficients of a block of bands are column-major, a band after the other:
// band b of a sphere of n plane waves starts at b n; the boxes of a block of
// bands follow one another the same way. A real box (Gamma only) holds n1 n2 n3
// real values in real space and n1 n2 (n3 / 2 + 1) complex ones in reciprocal
// space, as cuFFT's real transforms.

#pragma once

#include <cufft.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>

#include "dft/core/device.h"
#include "dft/lattice/lattice.h"

namespace mdapy::dft {

class Fft {
public:
    explicit Fft(const Int3& grid);
    ~Fft();
    Fft(const Fft&) = delete;
    Fft& operator=(const Fft&) = delete;

    const Int3& grid() const noexcept { return grid_; }
    std::size_t points() const noexcept;    // n1 n2 n3
    std::size_t spectrum() const noexcept;  // n1 n2 (n3 / 2 + 1), of a real box

    // Complex boxes, in place.
    void to_real(Complex* boxes, int count);
    void to_reciprocal(Complex* boxes, int count);
    // Real boxes, out of place; to_real overwrites its spectrum, as cuFFT's
    // complex-to-real transforms do.
    void to_real(Complex* spectrum, double* values, int count);
    void to_reciprocal(const double* values, Complex* spectrum, int count);

private:
    cufftHandle plan(cufftType type, int count);

    Int3 grid_;
    std::map<std::pair<int, int>, cufftHandle> plans_;  // (type, count) -> plan
};

// The coefficients of count bands of a sphere of n plane waves placed in their
// boxes of size points, the rest zero: boxes[b points + box[g]] = c[b n + g].
// With partner (real boxes), the conjugate also at partner[g] where it is not
// negative, which the complex-to-real transform needs in its plane m3 = 0.
void scatter(const Complex* coefficients, std::size_t n, const std::int64_t* box,
             const std::int64_t* partner, int count, Complex* boxes, std::size_t points);

// The coefficients of a sphere read from boxes, times scale: c[b n + g] =
// scale boxes[b points + box[g]].
void gather(const Complex* boxes, std::size_t points, const std::int64_t* box, std::size_t n,
            int count, double scale, Complex* coefficients);

// The Fourier components of a coarse box carried to a fine box and back, for
// densities and potentials: to_fine places each component of |m_i| < n_i / 2 of
// the coarse box at its place in the fine box, the rest of the fine box zero;
// to_coarse takes them back. Both on complex boxes of count functions.
class Transfer {
public:
    Transfer(const Int3& coarse, const Int3& fine);
    void to_fine(const Complex* coarse, Complex* fine, int count) const;
    void to_coarse(const Complex* fine, Complex* coarse, int count) const;
    std::size_t size() const noexcept { return coarse_.size(); }

private:
    std::size_t coarse_points_ = 0, fine_points_ = 0;
    DeviceArray<std::int64_t> coarse_, fine_;  // the places of the components kept
};

}  // namespace mdapy::dft
