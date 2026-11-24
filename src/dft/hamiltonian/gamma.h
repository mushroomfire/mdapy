// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Real wavefunctions of Gamma on half a sphere of plane waves, c(-G) = conj
// c(G), G = 0 first and real. A block (n, m) of complex coefficients is also a
// real matrix (2 n, m) of rows Re c_0, Im c_0, Re c_1, ...; real linear
// combinations of the columns are those of that matrix, and the product over
// the whole sphere is
//   <a|b> = 2 sum_G Re conj(a_G) b_G - a_0 b_0 = 2 A[2:]^T B[2:] + A[0]^T B[0],
// which never reads Im c_0. The projections <beta|psi> of real functions are
// real, (projectors, m) doubles. Each helper has its counterpart of single
// precision (cuComplex blocks, real matrices of floats) for the bands of the
// single-precision steps; gamma_dots sums those in double too.

#pragma once

#include <cublas_v2.h>

#include <cstddef>

#include "dft/core/device.h"

namespace mdapy::dft {

// out (ka, kb, leading dimension ldo) = <a|b> of the ka columns of a and the
// kb of b (n, .) over the whole sphere, plus scale times out
void gamma_inner(cublasHandle_t blas, std::size_t n, const Complex* a, int ka, const Complex* b, int kb,
                 double* out, int ldo, double scale = 0.0);

// out (n, m) = alpha a y + beta out, a (n, k) complex, y (k, m) real of
// leading dimension ldy
void gamma_combine(cublasHandle_t blas, std::size_t n, const Complex* a, int k, const double* y, int ldy, int m,
                   double alpha, double beta, Complex* out);

// out (n, m) = out r^-1, r (m, m) upper triangular, real
void gamma_solve(cublasHandle_t blas, std::size_t n, int m, const double* r, Complex* out);

// Im c_0 = 0 of each of count columns of n
void gamma_clean(std::size_t n, int count, Complex* c);

// out[c] = <a_c|b_c> over the whole sphere of each of count columns of n
void gamma_dots(std::size_t n, int count, const Complex* a, const Complex* b, double* out);

// The same in single precision.
void gamma_inner(cublasHandle_t blas, std::size_t n, const cuComplex* a, int ka, const cuComplex* b, int kb,
                 float* out, int ldo, float scale = 0.0f);
void gamma_combine(cublasHandle_t blas, std::size_t n, const cuComplex* a, int k, const float* y, int ldy, int m,
                   float alpha, float beta, cuComplex* out);
void gamma_solve(cublasHandle_t blas, std::size_t n, int m, const float* r, cuComplex* out);
void gamma_clean(std::size_t n, int count, cuComplex* c);
void gamma_dots(std::size_t n, int count, const cuComplex* a, const cuComplex* b, double* out);

}  // namespace mdapy::dft
