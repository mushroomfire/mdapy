// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Batched 3D FFTs, MDAPY's own or those of cuFFT, on boxes of n1 x n2 x n3 points, the third index the
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

#include "dft/core/portable.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>

#include "dft/core/kernels.h"
#include "dft/core/device.h"
#include "dft/lattice/lattice.h"

namespace mdapy::dft {

class BoxFft;

class Fft {
public:
    // With Kernels::Mdapy the transforms are MDAPY's own (BoxFft) when every
    // size is a product of 2, 3, 5 and 7, as the grids of the DFT are; other
    // sizes, and Kernels::Vendor take those of the library of the vendor.
    explicit Fft(const Int3& grid, Kernels kernels = Kernels::Mdapy);
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
    // Complex boxes of single precision, in place, for the bands of the
    // single-precision steps.
    void to_real(cuComplex* boxes, int count);
    void to_reciprocal(cuComplex* boxes, int count);
    // Real boxes of single precision, out of place, as those of double.
    void to_real(cuComplex* spectrum, float* values, int count);
    void to_reciprocal(const float* values, cuComplex* spectrum, int count);

private:
    cufftHandle plan(cufftType type, int count);

    Int3 grid_;
    std::map<std::pair<int, int>, cufftHandle> plans_;  // (type, count) -> plan
    std::unique_ptr<BoxFft> own_;  // MDAPY's own transforms, when they are the ones used
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

// The same, added to what the coefficients hold: c[b n + g] += scale boxes[...].
void gather_add(const Complex* boxes, std::size_t points, const std::int64_t* box, std::size_t n,
                int count, double scale, Complex* coefficients);

// The same of single-precision coefficients and boxes.
void scatter(const cuComplex* coefficients, std::size_t n, const std::int64_t* box, int count, cuComplex* boxes,
             std::size_t points);
void scatter(const cuComplex* coefficients, std::size_t n, const std::int64_t* box, const std::int64_t* partner,
             int count, cuComplex* boxes, std::size_t points);
void gather_add(const cuComplex* boxes, std::size_t points, const std::int64_t* box, std::size_t n, int count,
                float scale, cuComplex* coefficients);

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
