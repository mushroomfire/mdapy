// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Nothing here depends on the size of a warp or of a wavefront: the threads of
// a block (64, or 256 in the small routines) share the work by their index and
// meet at barriers of the block.

#include "dft/blas/blas.h"

#include <new>
#include <stdexcept>
#include <string>

#ifdef MDAPY_DFT_BLAS_PROFILE
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>
#endif

#include "dft/core/check.h"

namespace mdapy::dft {

namespace {

// ---------------------------------------------------------------- scalars

template <class R>
struct Cx {
    R x, y;
};

__device__ __forceinline__ float zero_of(float) { return 0.0f; }
__device__ __forceinline__ double zero_of(double) { return 0.0; }
template <class R>
__device__ __forceinline__ Cx<R> zero_of(Cx<R>) {
    return {static_cast<R>(0), static_cast<R>(0)};
}

__host__ __device__ __forceinline__ bool is_zero(float a) { return a == 0.0f; }
__host__ __device__ __forceinline__ bool is_zero(double a) { return a == 0.0; }
template <class R>
__host__ __device__ __forceinline__ bool is_zero(Cx<R> a) {
    return a.x == static_cast<R>(0) && a.y == static_cast<R>(0);
}

__device__ __forceinline__ float conj_of(float a) { return a; }
__device__ __forceinline__ double conj_of(double a) { return a; }
template <class R>
__device__ __forceinline__ Cx<R> conj_of(Cx<R> a) {
    return {a.x, -a.y};
}

__device__ __forceinline__ float sum(float a, float b) { return a + b; }
__device__ __forceinline__ double sum(double a, double b) { return a + b; }
template <class R>
__device__ __forceinline__ Cx<R> sum(Cx<R> a, Cx<R> b) {
    return {a.x + b.x, a.y + b.y};
}

__device__ __forceinline__ float product(float a, float b) { return a * b; }
__device__ __forceinline__ double product(double a, double b) { return a * b; }
template <class R>
__device__ __forceinline__ Cx<R> product(Cx<R> a, Cx<R> b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

// s + a b
__device__ __forceinline__ float plus(float s, float a, float b) { return s + a * b; }
__device__ __forceinline__ double plus(double s, double a, double b) { return s + a * b; }
template <class R>
__device__ __forceinline__ Cx<R> plus(Cx<R> s, Cx<R> a, Cx<R> b) {
    return {s.x + a.x * b.x - a.y * b.y, s.y + a.x * b.y + a.y * b.x};
}

// s - a b
__device__ __forceinline__ float minus(float s, float a, float b) { return s - a * b; }
__device__ __forceinline__ double minus(double s, double a, double b) { return s - a * b; }
template <class R>
__device__ __forceinline__ Cx<R> minus(Cx<R> s, Cx<R> a, Cx<R> b) {
    return {s.x - a.x * b.x + a.y * b.y, s.y - a.x * b.y - a.y * b.x};
}

__device__ __forceinline__ float quotient(float a, float b) { return a / b; }
__device__ __forceinline__ double quotient(double a, double b) { return a / b; }
template <class R>
__device__ __forceinline__ Cx<R> quotient(Cx<R> a, Cx<R> b) {
    // scaled by the larger part of b, as Smith's division
    const R size_x = b.x < static_cast<R>(0) ? -b.x : b.x, size_y = b.y < static_cast<R>(0) ? -b.y : b.y;
    if (size_x >= size_y) {
        const R t = b.y / b.x, d = b.x + b.y * t;
        return {(a.x + a.y * t) / d, (a.y - a.x * t) / d};
    }
    const R t = b.x / b.y, d = b.x * t + b.y;
    return {(a.x * t + a.y) / d, (a.y * t - a.x) / d};
}

// ---------------------------------------------------------------- product

constexpr int kBlock = 64;    // threads of a block of a product, 8 x 8
constexpr int kSide = 8;
constexpr int kThreads = 256;  // of the small routines

// What a thread and a block of a product hold: each thread MM x MN values of
// the result in registers, the block (8 MM) x (8 MN) of them, from depth terms
// of the sum in shared memory at a time. Real numbers take more values a
// thread than complex ones, which are two each. Measured on an RTX A2000
// (tests/cpp/dft_blas_check.cu).
#ifndef MDAPY_BLAS_REAL
#define MDAPY_BLAS_REAL 8, 8, 16
#endif
#ifndef MDAPY_BLAS_COMPLEX
#define MDAPY_BLAS_COMPLEX 4, 4, 16
#endif
template <int MM, int MN, int Depth>
struct Form {
    static constexpr int mm = MM, mn = MN, depth = Depth;
};
template <class S>
struct Tiles : Form<MDAPY_BLAS_REAL> {};
template <class R>
struct Tiles<Cx<R>> : Form<MDAPY_BLAS_COMPLEX> {};

struct GemmArgs {
    int m, n, k;
    long long lda, ldb, ldc;
    int conj_a, conj_b;
    // split: block z sums the terms [z length, (z + 1) length) into
    // partial + z m n, (m, n) with no gaps
    int segments, length;
    // not split: block z is product z of a batch
    long long stride_a, stride_b, stride_c;
};

// See the top of blas.h. op(a)[i, t] is a[i + t lda], or a[t + i lda] with
// TransA; op(b)[t, j] is b[t + j ldb], or b[j + t ldb] with TransB: each tile
// is read along the columns of its matrix. Values outside the matrices are
// zeros in shared memory, so that the sums take no bounds.
template <class S, bool TransA, bool TransB>
__global__ void gemm_kernel(const GemmArgs g, const S alpha, const S beta, const S* __restrict__ a,
                            const S* __restrict__ b, S* c, S* partial) {
    constexpr int kMM = Tiles<S>::mm, kMN = Tiles<S>::mn, kDepth = Tiles<S>::depth;
    constexpr int kTM = kSide * kMM, kTN = kSide * kMN;  // values of the result of a block along each index
    constexpr int kLoadsA = kTM * kDepth / kBlock, kLoadsB = kTN * kDepth / kBlock;  // of a thread at a time
    __shared__ S as[kDepth * kTM], bs[kDepth * kTN];
    const unsigned tid = threadIdx.x;
    const unsigned i0 = blockIdx.x * kTM, j0 = blockIdx.y * kTN, z = blockIdx.z;
    const unsigned m = static_cast<unsigned>(g.m), n = static_cast<unsigned>(g.n);
    unsigned first = 0, last = static_cast<unsigned>(g.k);
    if (g.segments > 1) {
        first = z * static_cast<unsigned>(g.length);
        last = first + static_cast<unsigned>(g.length) < last ? first + static_cast<unsigned>(g.length) : last;
    } else {
        a += z * g.stride_a;
        b += z * g.stride_b;
        c += z * g.stride_c;
    }
    const unsigned ti = (tid % kSide) * kMM, tj = (tid / kSide) * kMN;
    const bool active = i0 + ti < m && j0 + tj < n;
    S acc[kMM * kMN];
#pragma unroll
    for (int e = 0; e < kMM * kMN; ++e) acc[e] = zero_of(S{});
    for (unsigned t0 = first; t0 < last; t0 += kDepth) {
#ifdef MDAPY_BLAS_SAME_CHUNK
        // an experiment of tests/cpp/dft_blas_check.sh, not a product: every
        // chunk reads the terms of the first, which stay in the cache, to
        // tell the bandwidth from the arithmetic
        const unsigned t0_read = first;
#else
        const unsigned t0_read = t0;
#endif
        S va[kLoadsA], vb[kLoadsB];
#pragma unroll
        for (int q = 0; q < kLoadsA; ++q) {
            const unsigned e = tid + q * kBlock;
            const unsigned i = TransA ? e / kDepth : e % kTM, t = TransA ? e % kDepth : e / kTM;
            const bool in = i0 + i < m && t0 + t < last;
            const long long at = TransA ? (t0_read + t) + (i0 + i) * g.lda : (i0 + i) + (t0_read + t) * g.lda;
            va[q] = in ? a[at] : zero_of(S{});
        }
#pragma unroll
        for (int q = 0; q < kLoadsB; ++q) {
            const unsigned e = tid + q * kBlock;
            const unsigned j = TransB ? e % kTN : e / kDepth, t = TransB ? e / kTN : e % kDepth;
            const bool in = j0 + j < n && t0 + t < last;
            const long long at = TransB ? (j0 + j) + (t0_read + t) * g.ldb : (t0_read + t) + (j0 + j) * g.ldb;
            vb[q] = in ? b[at] : zero_of(S{});
        }
#pragma unroll
        for (int q = 0; q < kLoadsA; ++q) {
            const unsigned e = tid + q * kBlock;
            const unsigned i = TransA ? e / kDepth : e % kTM, t = TransA ? e % kDepth : e / kTM;
            as[t * kTM + i] = g.conj_a ? conj_of(va[q]) : va[q];
        }
#pragma unroll
        for (int q = 0; q < kLoadsB; ++q) {
            const unsigned e = tid + q * kBlock;
            const unsigned j = TransB ? e % kTN : e / kDepth, t = TransB ? e / kTN : e % kDepth;
            bs[t * kTN + j] = g.conj_b ? conj_of(vb[q]) : vb[q];
        }
        __syncthreads();
        if (active) {
#pragma unroll
            for (int t = 0; t < kDepth; ++t) {
                S x[kMM], y[kMN];
#pragma unroll
                for (int p = 0; p < kMM; ++p) x[p] = as[t * kTM + ti + p];
#pragma unroll
                for (int q = 0; q < kMN; ++q) y[q] = bs[t * kTN + tj + q];
#pragma unroll
                for (int q = 0; q < kMN; ++q)
#pragma unroll
                    for (int p = 0; p < kMM; ++p) acc[p + q * kMM] = plus(acc[p + q * kMM], x[p], y[q]);
            }
        }
        __syncthreads();
    }
    if (!active) return;
#pragma unroll
    for (int q = 0; q < kMN; ++q)
#pragma unroll
        for (int p = 0; p < kMM; ++p) {
            const unsigned i = i0 + ti + p, j = j0 + tj + q;
            if (i >= m || j >= n) continue;
            if (g.segments > 1) {
                partial[(static_cast<long long>(z) * n + j) * m + i] = acc[p + q * kMM];
            } else {
                S* to = c + i + j * g.ldc;
                const S value = product(alpha, acc[p + q * kMM]);
                *to = is_zero(beta) ? value : plus(value, beta, *to);
            }
        }
}

// c = alpha (the sum of the partial results, in their order) + beta c
template <class S>
__global__ void gather_kernel(int m, int n, int segments, const S alpha, const S beta, const S* __restrict__ partial,
                              S* c, long long ldc) {
    const long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    const long long size = static_cast<long long>(m) * n;
    if (e >= size) return;
    S total = partial[e];
    for (int s = 1; s < segments; ++s) total = sum(total, partial[s * size + e]);
    S* to = c + e % m + (e / m) * ldc;
    const S value = product(alpha, total);
    *to = is_zero(beta) ? value : plus(value, beta, *to);
}

// ---------------------------------------------------------------- triangular solve

constexpr int kSolve = 16;  // columns solved at a time, a row to a thread in registers

// x <- x r^-1 of w <= kSolve columns of the rows of b, r (w, w) upper triangular:
// x_j = (b_j - sum_(i < j) x_i r_ij) / r_jj
template <class S>
__global__ void solve_kernel(int rows, int w, const S* __restrict__ r, long long ldr, S* b, long long ldb) {
    __shared__ S rs[kSolve * kSolve];
#pragma unroll
    for (int q = 0; q < kSolve * kSolve / kBlock; ++q) {
        const unsigned e = threadIdx.x + q * kBlock, i = e % kSolve, j = e / kSolve;
        if (i <= j && j < static_cast<unsigned>(w)) rs[e] = r[i + j * ldr];
    }
    __syncthreads();
    const long long row = blockIdx.x * static_cast<long long>(kBlock) + threadIdx.x;
    if (row >= rows) return;
    S x[kSolve];
#pragma unroll
    for (int j = 0; j < kSolve; ++j)
        if (j < w) x[j] = b[row + j * ldb];
#pragma unroll
    for (int j = 0; j < kSolve; ++j)
        if (j < w) {
            S value = x[j];
#pragma unroll
            for (int i = 0; i < j; ++i) value = minus(value, x[i], rs[i + j * kSolve]);
            x[j] = quotient(value, rs[j + j * kSolve]);
        }
#pragma unroll
    for (int j = 0; j < kSolve; ++j)
        if (j < w) b[row + j * ldb] = x[j];
}

// ---------------------------------------------------------------- small routines

template <class S>
__global__ void copy_kernel(int n, const S* __restrict__ x, long long incx, S* __restrict__ y, long long incy) {
    const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i < n) y[i * incy] = x[i * incx];
}

template <class S>
__global__ void scal_kernel(int n, const S alpha, S* x) {
    const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i < n) x[i] = product(alpha, x[i]);
}

template <class S>
__global__ void ger_kernel(int m, int n, const S alpha, const S* __restrict__ x, long long incx,
                           const S* __restrict__ y, long long incy, S* __restrict__ a, long long lda) {
    const long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (e >= static_cast<long long>(m) * n) return;
    const long long i = e % m, j = e / m;
    a[i + j * lda] = plus(a[i + j * lda], product(alpha, x[i * incx]), y[j * incy]);
}

unsigned blocks(long long work, int threads) {
    return static_cast<unsigned>((work + threads - 1) / threads);
}

// ---------------------------------------------------------------- the vendor's

void check_blas(cublasStatus_t status, const char* call) {
    if (status == CUBLAS_STATUS_SUCCESS) return;
    if (status == CUBLAS_STATUS_ALLOC_FAILED) throw std::bad_alloc();
    throw std::runtime_error(std::string(call) + " of the library of the vendor failed with status " +
                             std::to_string(static_cast<int>(status)));
}

cublasOperation_t vendor_op(Op op) {
    return op == Op::N ? CUBLAS_OP_N : op == Op::T ? CUBLAS_OP_T : CUBLAS_OP_C;
}

cublasStatus_t vendor_gemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                           Complex alpha, const Complex* a, int lda, const Complex* b, int ldb, Complex beta,
                           Complex* c, int ldc) {
    return cublasZgemm(h, ta, tb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
}
cublasStatus_t vendor_gemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                           cuComplex alpha, const cuComplex* a, int lda, const cuComplex* b, int ldb, cuComplex beta,
                           cuComplex* c, int ldc) {
    return cublasCgemm(h, ta, tb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
}
cublasStatus_t vendor_gemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                           double alpha, const double* a, int lda, const double* b, int ldb, double beta, double* c,
                           int ldc) {
    return cublasDgemm(h, ta, tb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
}
cublasStatus_t vendor_gemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                           float alpha, const float* a, int lda, const float* b, int ldb, float beta, float* c,
                           int ldc) {
    return cublasSgemm(h, ta, tb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
}

cublasStatus_t vendor_trsm(cublasHandle_t h, int m, int n, const Complex* r, int ldr, Complex* b, int ldb) {
    const Complex one = make_cuDoubleComplex(1.0, 0.0);
    return cublasZtrsm(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &one, r, ldr, b, ldb);
}
cublasStatus_t vendor_trsm(cublasHandle_t h, int m, int n, const cuComplex* r, int ldr, cuComplex* b, int ldb) {
    const cuComplex one = make_cuComplex(1.0f, 0.0f);
    return cublasCtrsm(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &one, r, ldr, b, ldb);
}
cublasStatus_t vendor_trsm(cublasHandle_t h, int m, int n, const double* r, int ldr, double* b, int ldb) {
    const double one = 1.0;
    return cublasDtrsm(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &one, r, ldr, b, ldb);
}
cublasStatus_t vendor_trsm(cublasHandle_t h, int m, int n, const float* r, int ldr, float* b, int ldb) {
    const float one = 1.0f;
    return cublasStrsm(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &one, r, ldr, b, ldb);
}

// the scalars of the kernels have the layout of those of the libraries
Cx<double> own(Complex a) { return {a.x, a.y}; }
Cx<float> own(cuComplex a) { return {a.x, a.y}; }
const Cx<double>* own(const Complex* p) { return reinterpret_cast<const Cx<double>*>(p); }
Cx<double>* own(Complex* p) { return reinterpret_cast<Cx<double>*>(p); }
const Cx<float>* own(const cuComplex* p) { return reinterpret_cast<const Cx<float>*>(p); }
Cx<float>* own(cuComplex* p) { return reinterpret_cast<Cx<float>*>(p); }

// plus or minus one
float unit(float, int sign) { return static_cast<float>(sign); }
double unit(double, int sign) { return static_cast<double>(sign); }
template <class R>
Cx<R> unit(Cx<R>, int sign) {
    return {static_cast<R>(sign), static_cast<R>(0)};
}

template <class S>
void launch_copy(int n, const S* x, int incx, S* y, int incy) {
    if (n < 1) return;
    copy_kernel<S><<<blocks(n, kThreads), kThreads>>>(n, x, incx, y, incy);
    check_cuda(cudaGetLastError(), "a copy");
}

template <class S>
void launch_scal(int n, S alpha, S* x) {
    if (n < 1) return;
    scal_kernel<S><<<blocks(n, kThreads), kThreads>>>(n, alpha, x);
    check_cuda(cudaGetLastError(), "a scaling");
}

template <class S>
void launch_ger(int m, int n, S alpha, const S* x, int incx, const S* y, int incy, S* a, int lda) {
    if (m < 1 || n < 1) return;
    ger_kernel<S><<<blocks(static_cast<long long>(m) * n, kThreads), kThreads>>>(m, n, alpha, x, incx, y, incy, a, lda);
    check_cuda(cudaGetLastError(), "a rank-one update");
}

template <class S, bool TransA, bool TransB>
void launch_gemm(const GemmArgs& g, dim3 grid, S alpha, S beta, const S* a, const S* b, S* c, S* partial) {
    gemm_kernel<S, TransA, TransB><<<grid, kBlock>>>(g, alpha, beta, a, b, c, partial);
    check_cuda(cudaGetLastError(), "a product of matrices");
}

}  // namespace

Blas::Blas(Kernels kernels) : kernels_(kernels) {}

// made at its first use: a calculation with MDAPY's kernels takes none
cublasHandle_t Blas::vendor() {
    if (vendor_ == nullptr) {
        check_blas(cublasCreate(&vendor_), "cublasCreate");
        check_blas(cublasSetStream(vendor_, cudaStreamPerThread), "cublasSetStream");
    }
    return vendor_;
}

Blas::~Blas() {
    if (vendor_ != nullptr) cublasDestroy(vendor_);
}

// A grid of blocks has at most 65535 along its second and third index.
template <class S>
void Blas::own_gemm(Op ta, Op tb, int m, int n, int k, S alpha, const S* a, long long lda, long long stride_a,
                    const S* b, long long ldb, S beta, S* c, long long ldc, long long stride_c, int batch) {
    GemmArgs g{};
    g.m = m;
    g.n = n;
    g.k = k;
    g.lda = lda;
    g.ldb = ldb;
    g.ldc = ldc;
    g.conj_a = ta == Op::C;
    g.conj_b = tb == Op::C;
    g.segments = 1;
    g.length = k;
    g.stride_a = stride_a;
    g.stride_b = 0;
    g.stride_c = stride_c;
    constexpr int kTM = kSide * Tiles<S>::mm, kTN = kSide * Tiles<S>::mn, kDepth = Tiles<S>::depth;
    const long long tiles = static_cast<long long>((m + kTM - 1) / kTM) * ((n + kTN - 1) / kTN);
    // a small result of long sums: segments of at least 512 terms, as many as
    // give about 2048 blocks
    constexpr long long kWanted = 2048;
    if (batch == 1 && most_segments_ != 1 && tiles < kWanted && k >= 1024) {
        long long segments = (kWanted + tiles - 1) / tiles;
        if (segments > k / 512) segments = k / 512;
        const long long most = most_segments_ > 0 ? most_segments_ : 256;
        if (segments > most) segments = most;
        if (segments > 1) {
            long long length = (k + segments - 1) / segments;
            length = kDepth * ((length + kDepth - 1) / kDepth);
            g.length = static_cast<int>(length);
            g.segments = static_cast<int>((k + length - 1) / length);
        }
    }
    S* partial = nullptr;
    if (g.segments > 1) {
        const std::size_t values = static_cast<std::size_t>(g.segments) * m * n;
        partial_.reserve(values * sizeof(S) / sizeof(double) + 1);
        partial = reinterpret_cast<S*>(partial_.data());
    }
    const dim3 grid(static_cast<unsigned>((m + kTM - 1) / kTM), static_cast<unsigned>((n + kTN - 1) / kTN),
                    static_cast<unsigned>(g.segments > 1 ? g.segments : batch));
    const bool trans_a = ta != Op::N, trans_b = tb != Op::N;
    if (trans_a && trans_b)
        launch_gemm<S, true, true>(g, grid, alpha, beta, a, b, c, partial);
    else if (trans_a)
        launch_gemm<S, true, false>(g, grid, alpha, beta, a, b, c, partial);
    else if (trans_b)
        launch_gemm<S, false, true>(g, grid, alpha, beta, a, b, c, partial);
    else
        launch_gemm<S, false, false>(g, grid, alpha, beta, a, b, c, partial);
    if (g.segments > 1) {
        gather_kernel<S><<<blocks(static_cast<long long>(m) * n, kThreads), kThreads>>>(m, n, g.segments, alpha, beta,
                                                                                       partial, c, ldc);
        check_cuda(cudaGetLastError(), "the sum of the partial products");
    }
}

// Blocked: kSolve columns solved in registers, then taken out of the columns
// to their right by a product.
template <class S>
void Blas::own_trsm(int m, int n, const S* r, int ldr, S* b, int ldb) {
    const S one = unit(S{}, 1), minus_one = unit(S{}, -1);
    for (int j0 = 0; j0 < n; j0 += kSolve) {
        const int w = n - j0 < kSolve ? n - j0 : kSolve;
        S* columns = b + static_cast<long long>(j0) * ldb;
        solve_kernel<S><<<blocks(m, kBlock), kBlock>>>(m, w, r + j0 + static_cast<long long>(j0) * ldr, ldr, columns,
                                                       ldb);
        check_cuda(cudaGetLastError(), "a triangular solve");
        const int rest = n - j0 - w;
        if (rest > 0)
            own_gemm<S>(Op::N, Op::N, m, rest, w, minus_one, columns, ldb, 0,
                        r + j0 + static_cast<long long>(j0 + w) * ldr, ldr, one, columns + static_cast<long long>(w) * ldb,
                        ldb, 0, 1);
    }
}

namespace {

#ifdef MDAPY_DFT_BLAS_PROFILE
// Built with -DMDAPY_DFT_BLAS_PROFILE, every product and solve waits for the
// GPU before and after, and the process prints at its end the calls of each
// shape and their seconds, the longest first: the shapes of a calculation.
struct Probe {
    using Key = std::tuple<std::string, int, int, int>;
    struct Table {
        std::mutex lock;
        std::map<Key, std::pair<long long, double>> rows;
        ~Table() {
            std::vector<std::pair<double, Key>> order;
            double total = 0.0;
            for (const auto& [key, value] : rows) {
                order.push_back({value.second, key});
                total += value.second;
            }
            std::sort(order.rbegin(), order.rend());
            std::fprintf(stderr, "BLAS %.3f s in all\n", total);
            for (std::size_t i = 0; i < order.size() && i < 60; ++i) {
                const Key& key = order[i].second;
                std::fprintf(stderr, "BLAS %-10s %7d x %7d, %7d  calls %8lld  %9.3f s  %5.1f %%\n",
                             std::get<0>(key).c_str(), std::get<1>(key), std::get<2>(key), std::get<3>(key),
                             rows[key].first, order[i].first, 100.0 * order[i].first / total);
            }
        }
    };
    static Table& table() {
        static Table t;
        return t;
    }
    Key key;
    std::chrono::steady_clock::time_point start;
    Probe(const char* routine, Op ta, Op tb, int m, int n, int k)
        : key(std::string(routine) + " " + "NTC"[static_cast<int>(ta)] + "NTC"[static_cast<int>(tb)], m, n, k) {
        cudaStreamSynchronize(cudaStreamPerThread);
        start = std::chrono::steady_clock::now();
    }
    ~Probe() {
        cudaStreamSynchronize(cudaStreamPerThread);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        Table& t = table();
        const std::lock_guard<std::mutex> guard(t.lock);
        auto& row = t.rows[key];
        ++row.first;
        row.second += seconds;
    }
};
#define MDAPY_PROBE(routine, ta, tb, m, n, k) const Probe probe(routine, ta, tb, m, n, k)
#else
#define MDAPY_PROBE(routine, ta, tb, m, n, k)
#endif

// whether the grid of a product fits: see own_gemm
bool fits(int n, int batch) {
    return n / kSide <= 65535 && batch <= 65535;
}

}  // namespace

#define MDAPY_GEMM(T, CAST, NAME)                                                                                       \
    void Blas::gemm(Op ta, Op tb, int m, int n, int k, T alpha, const T* a, int lda, const T* b, int ldb, T beta,   \
                    T* c, int ldc) {                                                                                \
        if (m < 1 || n < 1) return;                                                                                 \
        MDAPY_PROBE(NAME "gemm", ta, tb, m, n, k);                                                                  \
        if (kernels_ == Kernels::Mdapy && fits(n, 1))                                                               \
            own_gemm(ta, tb, m, n, k, CAST(alpha), CAST(a), lda, 0, CAST(b), ldb, CAST(beta), CAST(c), ldc, 0, 1);  \
        else                                                                                                        \
            check_blas(vendor_gemm(vendor(), vendor_op(ta), vendor_op(tb), m, n, k, alpha, a, lda, b, ldb, beta, c,  \
                                   ldc),                                                                            \
                       "gemm");                                                                                     \
    }                                                                                                               \
    void Blas::trsm(int m, int n, const T* r, int ldr, T* b, int ldb) {                                             \
        if (m < 1 || n < 1) return;                                                                                 \
        MDAPY_PROBE(NAME "trsm", Op::N, Op::N, m, n, 0);                                                            \
        if (kernels_ == Kernels::Mdapy)                                                                             \
            own_trsm(m, n, CAST(r), ldr, CAST(b), ldb);                                                             \
        else                                                                                                        \
            check_blas(vendor_trsm(vendor(), m, n, r, ldr, b, ldb), "trsm");                                         \
    }

namespace {
double own(double a) { return a; }
float own(float a) { return a; }
const double* own(const double* p) { return p; }
double* own(double* p) { return p; }
const float* own(const float* p) { return p; }
float* own(float* p) { return p; }
}  // namespace

MDAPY_GEMM(Complex, own, "Z")
MDAPY_GEMM(cuComplex, own, "C")
MDAPY_GEMM(double, own, "D")
MDAPY_GEMM(float, own, "S")
#undef MDAPY_GEMM

void Blas::gemm_batched(Op tb, int m, int n, int k, const double* a, int lda, long long stride_a, const double* b,
                        int ldb, double beta, double* c, int ldc, long long stride_c, int batch) {
    if (m < 1 || n < 1 || batch < 1) return;
    MDAPY_PROBE("Dbatched", Op::N, tb, m, n, k);
    if (kernels_ == Kernels::Mdapy && fits(n, batch)) {
        own_gemm<double>(Op::N, tb, m, n, k, 1.0, a, lda, stride_a, b, ldb, beta, c, ldc, stride_c, batch);
        return;
    }
    const double one = 1.0;
    check_blas(cublasDgemmStridedBatched(vendor(), CUBLAS_OP_N, vendor_op(tb), m, n, k, &one, a, lda, stride_a, b, ldb,
                                         0, &beta, c, ldc, stride_c, batch),
               "gemmStridedBatched");
}

#define MDAPY_COPY(T, VENDOR)                                                      \
    void Blas::copy(int n, const T* x, int incx, T* y, int incy) {                 \
        if (kernels_ == Kernels::Mdapy)                                            \
            launch_copy(n, own(x), incx, own(y), incy);                            \
        else if (n > 0)                                                            \
            check_blas(VENDOR(vendor(), n, x, incx, y, incy), "copy");              \
    }
MDAPY_COPY(Complex, cublasZcopy)
MDAPY_COPY(cuComplex, cublasCcopy)
MDAPY_COPY(double, cublasDcopy)
MDAPY_COPY(float, cublasScopy)
#undef MDAPY_COPY

#define MDAPY_REAL(T, SCAL, GER)                                                                                   \
    void Blas::scal(int n, T alpha, T* x) {                                                                        \
        if (kernels_ == Kernels::Mdapy)                                                                            \
            launch_scal(n, alpha, x);                                                                              \
        else if (n > 0)                                                                                            \
            check_blas(SCAL(vendor(), n, &alpha, x, 1), "scal");                                                    \
    }                                                                                                              \
    void Blas::ger(int m, int n, T alpha, const T* x, int incx, const T* y, int incy, T* a, int lda) {             \
        if (kernels_ == Kernels::Mdapy)                                                                            \
            launch_ger(m, n, alpha, x, incx, y, incy, a, lda);                                                     \
        else if (m > 0 && n > 0)                                                                                   \
            check_blas(GER(vendor(), m, n, &alpha, x, incx, y, incy, a, lda), "ger");                               \
    }
MDAPY_REAL(double, cublasDscal, cublasDger)
MDAPY_REAL(float, cublasSscal, cublasSger)
#undef MDAPY_REAL

}  // namespace mdapy::dft
