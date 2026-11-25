// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The dense linear algebra of the DFT on the GPU, of matrices stored by
// columns: the products, the triangular solves and the small level-1 and
// level-2 routines that the solvers and the projectors take, with the
// arguments of BLAS, in double and single precision, complex and real. With
// the kernels of MDAPY they are its own; with those of the vendor, and for what
// MDAPY has no kernel for, they are cuBLAS's (hipBLAS's on AMD GPUs).
//
// MDAPY's product computes a tile of the result per block (64 x 64 to 128 x
// 128 values), each thread 8 x 8 real or 4 x 4 complex ones in registers, from
// 8 or 16 columns of op(a) and rows of op(b) at a time in shared memory. On
// the NVIDIA GPUs of full double precision (compute capability 8.0, 9.0 and
// 10.0) the products of double precision run on the tensor cores instead
// (dmma.cuh): the same sums in another order. A product whose result is small and
// whose sum is long (the inner products of the bands: a few hundred squared
// values, each a sum over 1e4 to 1e5 plane waves) is split along the sum: each
// block sums a segment into a partial result, and a second kernel adds the
// partial results in their order, so that the result does not depend on the
// GPU or its load.
//
// A Blas belongs to one host thread at a time and works on its default stream.

#pragma once

#include "dft/core/portable.h"

#include <cstddef>

#include "dft/core/device.h"
#include "dft/core/kernels.h"

namespace mdapy::dft {

// op(a) of a product: a itself, its transpose, or its conjugate transpose.
enum class Op { N, T, C };

class Blas {
public:
    explicit Blas(Kernels kernels = Kernels::Mdapy);
    ~Blas();
    Blas(const Blas&) = delete;
    Blas& operator=(const Blas&) = delete;

    Kernels kernels() const noexcept { return kernels_; }

    // c = alpha op(a) op(b) + beta c: c is (m, n), op(a) (m, k), op(b) (k, n).
    // c is not read when beta is zero.
    void gemm(Op ta, Op tb, int m, int n, int k, Complex alpha, const Complex* a, int lda, const Complex* b, int ldb,
              Complex beta, Complex* c, int ldc);
    void gemm(Op ta, Op tb, int m, int n, int k, cuComplex alpha, const cuComplex* a, int lda, const cuComplex* b,
              int ldb, cuComplex beta, cuComplex* c, int ldc);
    void gemm(Op ta, Op tb, int m, int n, int k, double alpha, const double* a, int lda, const double* b, int ldb,
              double beta, double* c, int ldc);
    void gemm(Op ta, Op tb, int m, int n, int k, float alpha, const float* a, int lda, const float* b, int ldb,
              float beta, float* c, int ldc);

    // The same of batch products: product i takes a + i stride_a, the one b and
    // c + i stride_c. c = a op(b) + beta c, tb N or T.
    void gemm_batched(Op tb, int m, int n, int k, const double* a, int lda, long long stride_a, const double* b,
                      int ldb, double beta, double* c, int ldc, long long stride_c, int batch);

    // b = b r^-1: b is (m, n), r (n, n) upper triangular (its lower part is
    // not read).
    void trsm(int m, int n, const Complex* r, int ldr, Complex* b, int ldb);
    void trsm(int m, int n, const cuComplex* r, int ldr, cuComplex* b, int ldb);
    void trsm(int m, int n, const double* r, int ldr, double* b, int ldb);
    void trsm(int m, int n, const float* r, int ldr, float* b, int ldb);

    // y[i incy] = x[i incx], i < n.
    void copy(int n, const Complex* x, int incx, Complex* y, int incy);
    void copy(int n, const cuComplex* x, int incx, cuComplex* y, int incy);
    void copy(int n, const double* x, int incx, double* y, int incy);
    void copy(int n, const float* x, int incx, float* y, int incy);

    // x[i] = alpha x[i], i < n.
    void scal(int n, double alpha, double* x);
    void scal(int n, float alpha, float* x);

    // a[i, j] += alpha x[i incx] y[j incy]: a is (m, n).
    void ger(int m, int n, double alpha, const double* x, int incx, const double* y, int incy, double* a, int lda);
    void ger(int m, int n, float alpha, const float* x, int incx, const float* y, int incy, float* a, int lda);

    // Products with a long sum are split into at most this many segments
    // (0: the default); 1 for none. For measurements and tests.
    void set_segments(int most) noexcept { most_segments_ = most; }

private:
    template <class S>
    void own_gemm(Op ta, Op tb, int m, int n, int k, S alpha, const S* a, long long lda, long long stride_a,
                  const S* b, long long ldb, S beta, S* c, long long ldc, long long stride_c, int batch);
    template <class S>
    void own_trsm(int m, int n, const S* r, int ldr, S* b, int ldb);

    cublasHandle_t vendor();
    // finds what the products ask of the GPU, at the first of them
    void look();

    Kernels kernels_;
    cublasHandle_t vendor_ = nullptr;  // of the library of the vendor, made at its first use
    DeviceArray<double> partial_;  // the partial results of a split product
    int most_segments_ = 0;
    int processors_ = 0;   // multiprocessors of the GPU; 0 before look()
    // the products of double precision run on the tensor cores of this GPU, and
    // their tiles are copied whole by one instruction (compute capability 9.0);
    // never with HIP
    [[maybe_unused]] bool tensor_ = false;
    [[maybe_unused]] bool maps_ = false;
    // the kernels of those products that were let take their shared memory, by
    // type, transpositions and way of copying
    [[maybe_unused]] bool asked_[2][2][2][3] = {};
};

}  // namespace mdapy::dft
