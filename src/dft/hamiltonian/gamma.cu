// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/hamiltonian/gamma.h"

#include <new>
#include <stdexcept>
#include <string>

namespace mdapy::dft {

namespace {

constexpr int kThreads = 256;

void check_blas(cublasStatus_t status, const char* call) {
    if (status == CUBLAS_STATUS_SUCCESS) return;
    if (status == CUBLAS_STATUS_ALLOC_FAILED) throw std::bad_alloc();
    throw std::runtime_error(std::string(call) + " failed with cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
}

__global__ void clean_kernel(std::size_t n, int count, Complex* c) {
    const int b = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (b < count) c[static_cast<std::size_t>(b) * n].y = 0.0;
}

// one block per column: 2 sum_G Re conj(a) b - a_0 b_0
__global__ void dots_kernel(std::size_t n, const Complex* a, const Complex* b, double* out) {
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * n;
    double s = 0.0;
    for (std::size_t g = threadIdx.x + 1; g < n; g += blockDim.x)
        s += a[base + g].x * b[base + g].x + a[base + g].y * b[base + g].y;
    __shared__ double partial[kThreads];
    partial[threadIdx.x] = s;
    __syncthreads();
    for (int k = kThreads / 2; k > 0; k >>= 1) {
        if (static_cast<int>(threadIdx.x) < k) partial[threadIdx.x] += partial[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = 2.0 * partial[0] + (n > 0 ? a[base].x * b[base].x : 0.0);
}

}  // namespace

void gamma_inner(cublasHandle_t blas, std::size_t n, const Complex* a, int ka, const Complex* b, int kb,
                 double* out, int ldo, double scale) {
    if (ka == 0 || kb == 0) return;
    const int rows = static_cast<int>(2 * n);
    const double two = 2.0, one = 1.0;
    const double* x = reinterpret_cast<const double*>(a);
    const double* y = reinterpret_cast<const double*>(b);
    if (n > 1) {
        check_blas(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, ka, kb, rows - 2, &two, x + 2, rows, y + 2, rows,
                               &scale, out, ldo),
                   "cublasDgemm");
    } else if (scale != 1.0) {
        // only G = 0: out = scale out before the term of G = 0
        for (int c = 0; c < kb; ++c)
            check_blas(cublasDscal(blas, ka, &scale, out + static_cast<std::size_t>(c) * ldo, 1), "cublasDscal");
    }
    if (n > 0)
        check_blas(cublasDger(blas, ka, kb, &one, x, rows, y, rows, out, ldo), "cublasDger");
}

void gamma_combine(cublasHandle_t blas, std::size_t n, const Complex* a, int k, const double* y, int ldy, int m,
                   double alpha, double beta, Complex* out) {
    if (m == 0 || n == 0) return;
    const int rows = static_cast<int>(2 * n);
    check_blas(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, rows, m, k, &alpha, reinterpret_cast<const double*>(a),
                           rows, y, ldy, &beta, reinterpret_cast<double*>(out), rows),
               "cublasDgemm");
}

void gamma_solve(cublasHandle_t blas, std::size_t n, int m, const double* r, Complex* out) {
    if (m == 0 || n == 0) return;
    const int rows = static_cast<int>(2 * n);
    const double one = 1.0;
    check_blas(cublasDtrsm(blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, rows,
                           m, &one, r, m, reinterpret_cast<double*>(out), rows),
               "cublasDtrsm");
}

void gamma_clean(std::size_t n, int count, Complex* c) {
    if (count == 0 || n == 0) return;
    clean_kernel<<<static_cast<unsigned>((count + kThreads - 1) / kThreads), kThreads>>>(n, count, c);
    check_cuda(cudaGetLastError(), "Im c_0 of Gamma");
}

void gamma_dots(std::size_t n, int count, const Complex* a, const Complex* b, double* out) {
    if (count == 0) return;
    dots_kernel<<<static_cast<unsigned>(count), kThreads>>>(n, a, b, out);
    check_cuda(cudaGetLastError(), "the products of Gamma");
}

}  // namespace mdapy::dft
