// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/lapack/lapack.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cmath>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "dft/core/check.h"
#include "dft/core/profile.h"
#include "dft/lapack/divide.h"
#include "dft/lapack/host.h"

namespace mdapy::dft {

namespace {

void check_solver(cusolverStatus_t status, const char* call) {
    if (status == CUSOLVER_STATUS_SUCCESS) return;
    if (status == CUSOLVER_STATUS_ALLOC_FAILED) throw std::bad_alloc();
    throw std::runtime_error(std::string(call) + " of the library of the vendor failed with status " +
                             std::to_string(static_cast<int>(status)));
}

// the scalars of the host in double precision of those of the GPU
std::complex<double> wide(Complex x) { return {x.x, x.y}; }
std::complex<double> wide(cuComplex x) { return {x.x, x.y}; }
double wide(double x) { return x; }
double wide(float x) { return x; }
void narrow(const std::complex<double>& x, Complex& out) { out = make_cuDoubleComplex(x.real(), x.imag()); }
void narrow(const std::complex<double>& x, cuComplex& out) {
    out = make_cuComplex(static_cast<float>(x.real()), static_cast<float>(x.imag()));
}
void narrow(double x, double& out) { out = x; }
void narrow(double x, float& out) { out = static_cast<float>(x); }

template <class T>
void download(const T* device, std::size_t count, T* host) {
    check_cuda(cudaMemcpy(host, device, count * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy from the GPU");
}

template <class T>
void upload(const T* host, std::size_t count, T* device) {
    check_cuda(cudaMemcpy(device, host, count * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy to the GPU");
}

#define kUpper CUBLAS_FILL_MODE_UPPER

// cuSOLVER's eigensolvers and factorizations by the type of the matrix; W the
// type of its work, R that of its eigenvalues
struct Z {
    using Work = Complex;
    using Real = double;
    static cusolverStatus_t size(cusolverDnHandle_t h, int n, Complex* a, double* w, int* lwork) {
        return cusolverDnZheevd_bufferSize(h, CUSOLVER_EIG_MODE_VECTOR, kUpper, n, a, n, w, lwork);
    }
    static cusolverStatus_t solve(cusolverDnHandle_t h, int n, Complex* a, double* w, Complex* work, int lwork, int* info) {
        return cusolverDnZheevd(h, CUSOLVER_EIG_MODE_VECTOR, kUpper, n, a, n, w, work, lwork, info);
    }
    static cusolverStatus_t factor_size(cusolverDnHandle_t h, int n, Complex* a, int* lwork) {
        return cusolverDnZpotrf_bufferSize(h, kUpper, n, a, n, lwork);
    }
    static cusolverStatus_t factor(cusolverDnHandle_t h, int n, Complex* a, Complex* work, int lwork, int* info) {
        return cusolverDnZpotrf(h, kUpper, n, a, n, work, lwork, info);
    }
};
struct C {
    using Work = cuComplex;
    using Real = float;
    static cusolverStatus_t size(cusolverDnHandle_t h, int n, cuComplex* a, float* w, int* lwork) {
        return cusolverDnCheevd_bufferSize(h, CUSOLVER_EIG_MODE_VECTOR, kUpper, n, a, n, w, lwork);
    }
    static cusolverStatus_t solve(cusolverDnHandle_t h, int n, cuComplex* a, float* w, cuComplex* work, int lwork, int* info) {
        return cusolverDnCheevd(h, CUSOLVER_EIG_MODE_VECTOR, kUpper, n, a, n, w, work, lwork, info);
    }
    static cusolverStatus_t factor_size(cusolverDnHandle_t h, int n, cuComplex* a, int* lwork) {
        return cusolverDnCpotrf_bufferSize(h, kUpper, n, a, n, lwork);
    }
    static cusolverStatus_t factor(cusolverDnHandle_t h, int n, cuComplex* a, cuComplex* work, int lwork, int* info) {
        return cusolverDnCpotrf(h, kUpper, n, a, n, work, lwork, info);
    }
};
struct D {
    using Work = double;
    using Real = double;
    static cusolverStatus_t size(cusolverDnHandle_t h, int n, double* a, double* w, int* lwork) {
        return cusolverDnDsyevd_bufferSize(h, CUSOLVER_EIG_MODE_VECTOR, kUpper, n, a, n, w, lwork);
    }
    static cusolverStatus_t solve(cusolverDnHandle_t h, int n, double* a, double* w, double* work, int lwork, int* info) {
        return cusolverDnDsyevd(h, CUSOLVER_EIG_MODE_VECTOR, kUpper, n, a, n, w, work, lwork, info);
    }
    static cusolverStatus_t factor_size(cusolverDnHandle_t h, int n, double* a, int* lwork) {
        return cusolverDnDpotrf_bufferSize(h, kUpper, n, a, n, lwork);
    }
    static cusolverStatus_t factor(cusolverDnHandle_t h, int n, double* a, double* work, int lwork, int* info) {
        return cusolverDnDpotrf(h, kUpper, n, a, n, work, lwork, info);
    }
};
struct S {
    using Work = float;
    using Real = float;
    static cusolverStatus_t size(cusolverDnHandle_t h, int n, float* a, float* w, int* lwork) {
        return cusolverDnSsyevd_bufferSize(h, CUSOLVER_EIG_MODE_VECTOR, kUpper, n, a, n, w, lwork);
    }
    static cusolverStatus_t solve(cusolverDnHandle_t h, int n, float* a, float* w, float* work, int lwork, int* info) {
        return cusolverDnSsyevd(h, CUSOLVER_EIG_MODE_VECTOR, kUpper, n, a, n, w, work, lwork, info);
    }
    static cusolverStatus_t factor_size(cusolverDnHandle_t h, int n, float* a, int* lwork) {
        return cusolverDnSpotrf_bufferSize(h, kUpper, n, a, n, lwork);
    }
    static cusolverStatus_t factor(cusolverDnHandle_t h, int n, float* a, float* work, int lwork, int* info) {
        return cusolverDnSpotrf(h, kUpper, n, a, n, work, lwork, info);
    }
};

// room for count values of T in an array of complex numbers of double precision
template <class T>
T* room(DeviceArray<Complex>& work, std::size_t count) {
    work.reserve((count * sizeof(T) + sizeof(Complex) - 1) / sizeof(Complex) + 1);
    return reinterpret_cast<T*>(work.data());
}

template <class V, class T, class R>
void vendor_eigh(cusolverDnHandle_t h, T* a, int n, R* w, DeviceArray<Complex>& work, DeviceArray<int>& info_device,
                 std::vector<double>& values) {
    int lwork = 0;
    check_solver(V::size(h, n, a, w, &lwork), "the size of the work of heevd");
    check_solver(V::solve(h, n, a, w, room<T>(work, static_cast<std::size_t>(std::max(lwork, 1))), lwork,
                          info_device.data()),
                 "heevd");
    int info = 0;
    info_device.download(&info, 1);
    if (info != 0)
        throw std::runtime_error("the eigenvalues of a subspace did not converge (heevd of the library of the vendor, "
                                 "info " + std::to_string(info) + ")");
    std::vector<R> host(static_cast<std::size_t>(n));
    download(w, host.size(), host.data());
    values.assign(host.begin(), host.end());
}

template <class V, class T>
int vendor_potrf(cusolverDnHandle_t h, T* a, int n, DeviceArray<Complex>& work, DeviceArray<int>& info_device) {
    int lwork = 0;
    check_solver(V::factor_size(h, n, a, &lwork), "the size of the work of potrf");
    check_solver(V::factor(h, n, a, room<T>(work, static_cast<std::size_t>(std::max(lwork, 1))), lwork,
                           info_device.data()),
                 "potrf");
    int info = 0;
    info_device.download(&info, 1);
    if (info < 0) throw std::runtime_error("potrf of the library of the vendor refused its argument " + std::to_string(-info));
    return info;
}


// ---------------------------------------------------------------- on the GPU

constexpr int kThreads = 256;

unsigned blocks(long long work, int threads = kThreads) {
    return static_cast<unsigned>((work + threads - 1) / threads);
}

template <class R>
struct Cx {
    R x, y;
};

__device__ __forceinline__ float re(float a) { return a; }
__device__ __forceinline__ double re(double a) { return a; }
template <class R>
__device__ __forceinline__ R re(Cx<R> a) {
    return a.x;
}
__device__ __forceinline__ float im(float) { return 0.0f; }
__device__ __forceinline__ double im(double) { return 0.0; }
template <class R>
__device__ __forceinline__ R im(Cx<R> a) {
    return a.y;
}
__device__ __forceinline__ float cj(float a) { return a; }
__device__ __forceinline__ double cj(double a) { return a; }
template <class R>
__device__ __forceinline__ Cx<R> cj(Cx<R> a) {
    return {a.x, -a.y};
}
__device__ __forceinline__ float mul(float a, float b) { return a * b; }
__device__ __forceinline__ double mul(double a, double b) { return a * b; }
template <class R>
__device__ __forceinline__ Cx<R> mul(Cx<R> a, Cx<R> b) {
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}
__device__ __forceinline__ float add(float a, float b) { return a + b; }
__device__ __forceinline__ double add(double a, double b) { return a + b; }
template <class R>
__device__ __forceinline__ Cx<R> add(Cx<R> a, Cx<R> b) {
    return {a.x + b.x, a.y + b.y};
}
__device__ __forceinline__ float sub(float a, float b) { return a - b; }
__device__ __forceinline__ double sub(double a, double b) { return a - b; }
template <class R>
__device__ __forceinline__ Cx<R> sub(Cx<R> a, Cx<R> b) {
    return {a.x - b.x, a.y - b.y};
}
__device__ __forceinline__ float from_real(float, float x) { return x; }
__device__ __forceinline__ double from_real(double, double x) { return x; }
template <class R>
__device__ __forceinline__ Cx<R> from_real(Cx<R>, R x) {
    return {x, static_cast<R>(0)};
}
// 1 / a
__device__ __forceinline__ float inverse(float a) { return 1.0f / a; }
__device__ __forceinline__ double inverse(double a) { return 1.0 / a; }
template <class R>
__device__ __forceinline__ Cx<R> inverse(Cx<R> a) {
    const R size = a.x * a.x + a.y * a.y;
    return {a.x / size, -a.y / size};
}

// the real type of a scalar of the kernels
template <class S>
struct RealOf {
    using type = S;
};
template <class R>
struct RealOf<Cx<R>> {
    using type = R;
};

// The lower triangle of a (n, n) from its upper one, and a real diagonal.
template <class S>
__global__ void mirror_kernel(int n, S* a) {
    const long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (e >= static_cast<long long>(n) * n) return;
    const long long i = e % n, j = e / n;
    if (i > j)
        a[e] = cj(a[j + i * n]);
    else if (i == j)
        a[e] = from_real(S{}, re(a[e]));
}

constexpr int kStep = 128;    // threads of a block of the tridiagonalization
constexpr int kFused = 1024;  // the largest matrix of which a step is one launch
constexpr int kFusedBytes = 8192;  // and the most bytes of a column of it (512 complex numbers of double precision: a V100)
constexpr int kPanelsFrom = 1024;  // the smallest matrix reduced in panels, complex numbers of double precision
constexpr int kFusedSingle = 256;  // the same in single precision on a GPU of little double precision

constexpr int kGroups = 16;   // of the threads of a block, for its sums
constexpr int kEarly = 4;     // numbers of a column a thread asks for ahead

// The sum of the values of the threads of a block, in an order that is always
// the same: each of kGroups threads adds those of a group, and every thread
// the sums of the groups. partial holds a value a thread, then one a group.
template <class S>
__device__ __forceinline__ S block_sum(S* partial, S value) {
    constexpr int kGroup = kStep / kGroups;
    const int t = static_cast<int>(threadIdx.x);
    partial[t] = value;
    __syncthreads();
    if (t < kGroups) {
        S sum = partial[t * kGroup];
        for (int i = 1; i < kGroup; ++i) sum = add(sum, partial[t * kGroup + i]);
        partial[kStep + t] = sum;
    }
    __syncthreads();
    S total = partial[kStep];
    for (int i = 1; i < kGroups; ++i) total = add(total, partial[kStep + i]);
    return total;
}

// Step k of the tridiagonalization. The reflector k, h = 1 - tau v v^H with
// v_(k + 1) = 1 and h^H x = beta e_1 (LAPACK's larfg), is that of the column k
// below the diagonal; it is kept as the column k of reflectors (zeros down to
// the diagonal), with tau, the diagonal element and beta, the subdiagonal one.
// The trailing block b (the indices beyond k) becomes b - v w^H - w v^H with
// w = tau p - (tau / 2) ((tau p)^H v) v and p = b v, which the step after
// applies, column by column: a step has two parts,
// - the first, of numbers of the order of n: w of the reflector before from
//   its p (p_in), the column k with that reflector applied, and its reflector;
// - the second, a block to each column j beyond k: the reflector before
//   applied to the column, and p_j = sum_i conj(a_ij) v_i of the new one (p_out).
// phase 2 is both in one launch of n - 1 - k blocks (one for the last column):
// every block works the first part out for itself in its shared memory,
// 2 (n - k) numbers after those of block_sum, all with the same result, and
// the first block writes it. phase 0 is the first part alone, in one block
// that writes w into held and v into reflectors, and phase 1 the second, which
// reads them there: for matrices beyond the shared memory of a block.
template <class S>
__global__ void tridiagonal_kernel(int n, int k, int phase, S* a, S* reflectors, S* tau,
                                   typename RealOf<S>::type* d, typename RealOf<S>::type* e, S* held,
                                   const S* __restrict__ p_in, S* __restrict__ p_out) {
    using R = typename RealOf<S>::type;
    extern __shared__ double shared[];
    S* partial = reinterpret_cast<S*>(shared);
    const int t = static_cast<int>(threadIdx.x);
    S* w = phase == 2 ? partial + kStep + kGroups : held;  // w_i at i - k
    S* v = phase == 2 ? partial + kStep + kGroups + (n - k) : reflectors + static_cast<long long>(k) * n + k + 1;  // v_i at i - k - 1
    const S* before = reflectors + static_cast<long long>(k > 0 ? k - 1 : 0) * n;  // the reflector before, v_i at i
    const int j = k + 1 + static_cast<int>(blockIdx.x);
    // the first numbers of a thread of the columns k and j are asked for at
    // once, so that they are there when their turn comes
    S early_k[kEarly], early_j[kEarly];
    if (phase == 2)
        for (int s = 0; s < kEarly; ++s) {
            const int i = k + t + s * kStep;
            if (i < n) early_k[s] = a[i + static_cast<long long>(k) * n];
            if (i + 1 < n) early_j[s] = a[i + 1 + static_cast<long long>(j) * n];
        }
    if (phase != 1) {
        const S* column = a + static_cast<long long>(k) * n;
        if (k > 0) {
            const S factor = tau[k - 1];
            S sum = from_real(S{}, static_cast<R>(0));
            for (int i = k + t; i < n; i += kStep) {
                const S scaled = mul(factor, p_in[i]);
                w[i - k] = scaled;
                sum = add(sum, mul(cj(scaled), before[i]));
            }
            const S half = mul(from_real(S{}, static_cast<R>(-0.5)), mul(factor, block_sum(partial, sum)));
            for (int i = k + t; i < n; i += kStep) w[i - k] = add(w[i - k], mul(half, before[i]));
            __syncthreads();
        }
        R tail = static_cast<R>(0);
        for (int i = k + t, s = 0; i < n; i += kStep, ++s) {
            S x = phase == 2 && s < kEarly ? early_k[s] : column[i];
            if (k > 0) x = sub(x, add(mul(before[i], cj(w[0])), mul(w[i - k], cj(before[k]))));
            if (i == k) {
                if (blockIdx.x == 0) d[k] = re(x);
            } else {
                v[i - k - 1] = x;
                if (i > k + 1) tail += re(x) * re(x) + im(x) * im(x);
            }
        }
        if (k == n - 1) return;
        tail = re(block_sum(partial, from_real(S{}, tail)));
        const S alpha = v[0];
        S reflection = from_real(S{}, static_cast<R>(0)), scale = from_real(S{}, static_cast<R>(1));
        R beta = re(alpha);
        if (tail != static_cast<R>(0) || im(alpha) != static_cast<R>(0)) {
            const R size = sqrt(re(alpha) * re(alpha) + im(alpha) * im(alpha) + tail);
            beta = re(alpha) >= static_cast<R>(0) ? -size : size;
            // tau = (beta - alpha) / beta, v = x / (alpha - beta)
            reflection = mul(sub(from_real(S{}, beta), alpha), from_real(S{}, static_cast<R>(1) / beta));
            scale = inverse(sub(alpha, from_real(S{}, beta)));
        }
        __syncthreads();
        for (int i = k + 1 + t; i < n; i += kStep)
            v[i - k - 1] = i == k + 1 ? from_real(S{}, static_cast<R>(1)) : mul(v[i - k - 1], scale);
        if (blockIdx.x == 0 && t == 0) {
            tau[k] = reflection;
            e[k] = beta;
        }
        __syncthreads();
        if (phase == 0) return;
        if (blockIdx.x == 0)
            for (int i = k + 1 + t; i < n; i += kStep) reflectors[i + static_cast<long long>(k) * n] = v[i - k - 1];
    }
    S* column = a + static_cast<long long>(j) * n;
    S wj = from_real(S{}, static_cast<R>(0)), vj = wj;
    if (k > 0) {
        wj = cj(w[j - k]);
        vj = cj(before[j]);
    }
    S sum = from_real(S{}, static_cast<R>(0));
    for (int i = k + 1 + t, s = 0; i < n; i += kStep, ++s) {
        S x = phase == 2 && s < kEarly ? early_j[s] : column[i];
        if (k > 0) {
            x = sub(x, add(mul(before[i], wj), mul(w[i - k], vj)));
            column[i] = x;
        }
        sum = add(sum, mul(cj(x), v[i - k - 1]));
    }
    sum = block_sum(partial, sum);
    if (t == 0) p_out[j] = sum;
}

// ------------------------------------------- the tridiagonalization in panels
//
// For the large matrices, as LAPACK's latrd: kColumns columns are reduced
// without touching the trailing matrix, which takes them all at the end as a
// product, b - v w^H - w v^H with v (the reflectors) and w of kColumns columns.
// A column j, the i-th of its panel (i = j - first), costs three launches:
// - by rows (panel_rows_kernel), for the reflector j - 1: its product y with
//   the trailing matrix as it was less v (w^H v) and w (v^H v) of the columns
//   of the panel before it, times tau: the start of its w; and the column j
//   with the reflectors of the panel applied, but for a multiple of the
//   reflector j - 1 that waits for a sum over all rows;
// - one block (panel_column_kernel): that sum, the end of w and of the
//   column, and its reflector;
// - a block to each column beyond j (panel_products_kernel): y = b v of the
//   new reflector, and a block to each column of w and of v of the panel,
//   their products with it.
// w (n, kColumns) holds a column to each reflector of the panel, reflectors
// its v at the column first + c, dots w^H v then v^H v (kColumns each).
constexpr int kColumns = 32;
constexpr int kRows = 32;     // rows of a block of panel_rows_kernel
constexpr int kLanes = 8;     // threads of a row there
constexpr int kColumn = 256;  // threads of the block of a column

// A block to kRows rows from j, kLanes threads a row which share the columns
// of the panel. made (the column i - 1 of w) gets tau p, p = y - v g - w h;
// held the column j less the reflectors before j - 1 and less made; sums, at
// the block, sum_r conj(made_r) v_r, v the reflector j - 1. With finish (j is
// past the panel) only made and sums.
template <class S>
__global__ void panel_rows_kernel(int n, int j, int first, int finish, const S* __restrict__ a,
                                  const S* __restrict__ reflectors, S* w, const S* __restrict__ tau,
                                  const S* __restrict__ y, const S* __restrict__ dots, S* __restrict__ held,
                                  S* __restrict__ sums) {
    using R = typename RealOf<S>::type;
    __shared__ S g[kColumns], h[kColumns], wrow[kColumns], vrow[kColumns], part[2][kRows * kLanes];
    const int t = static_cast<int>(threadIdx.x), i = j - first, before = i - 1;  // columns before the reflector j - 1
    const int row = t % kRows, lane = t / kRows, r = j + static_cast<int>(blockIdx.x) * kRows + row;
    const S zero = from_real(S{}, static_cast<R>(0));
    const S factor = tau[j - 1];
    if (t < before) {
        g[t] = dots[t];
        h[t] = dots[kColumns + t];
        wrow[t] = cj(w[j + static_cast<long long>(t) * n]);
        vrow[t] = cj(reflectors[j + static_cast<long long>(first + t) * n]);
    }
    __syncthreads();
    S p = zero, x = zero;
    if (r < n)
        for (int c = lane; c < before; c += kLanes) {
            const S vc = reflectors[r + static_cast<long long>(first + c) * n], wc = w[r + static_cast<long long>(c) * n];
            p = add(p, add(mul(vc, g[c]), mul(wc, h[c])));
            x = add(x, add(mul(vc, wrow[c]), mul(wc, vrow[c])));
        }
    part[0][t] = p;
    part[1][t] = x;
    __syncthreads();
    if (lane == 0) {
        S product = zero;
        if (r < n) {
            p = part[0][row];
            x = part[1][row];
            for (int l = 1; l < kLanes; ++l) {
                p = add(p, part[0][row + l * kRows]);
                x = add(x, part[1][row + l * kRows]);
            }
            const S made = mul(factor, sub(y[r], p));
            w[r + static_cast<long long>(before) * n] = made;
            if (!finish) held[r] = sub(sub(a[r + static_cast<long long>(j) * n], x), made);
            product = mul(cj(made), reflectors[r + static_cast<long long>(j - 1) * n]);
        }
        part[0][row] = product;
    }
    __syncthreads();
    if (t == 0) {
        S sum = part[0][0];
        for (int q = 1; q < kRows; ++q) sum = add(sum, part[0][q]);
        sums[blockIdx.x] = sum;
    }
}

// The sum of the values of the kColumn threads of a block, in an order that is
// always the same; partial holds a value a thread, then one a group of 16.
template <class S>
__device__ __forceinline__ S column_sum(S* partial, S value) {
    const int t = static_cast<int>(threadIdx.x);
    partial[t] = value;
    __syncthreads();
    if (t < kColumn / 16) {
        S sum = partial[t * 16];
        for (int i = 1; i < 16; ++i) sum = add(sum, partial[t * 16 + i]);
        partial[kColumn + t] = sum;
    }
    __syncthreads();
    S total = partial[kColumn];
    for (int i = 1; i < kColumn / 16; ++i) total = add(total, partial[kColumn + i]);
    return total;
}

// One block, kColumn threads, after panel_rows_kernel (when j is not the first
// of its panel): alpha = -tau / 2 sum_r conj(made_r) v_r from the count sums of
// its blocks; w of the reflector j - 1 is made + alpha v, and the column j
// held - (conj(made_j) + 2 Re(alpha)) v. Then the reflector of the column j, its tau, the
// diagonal element and the subdiagonal one. With finish only w.
template <class S>
__global__ void panel_column_kernel(int n, int j, int first, int finish, const S* __restrict__ a, S* reflectors, S* w,
                                    S* tau, typename RealOf<S>::type* d, typename RealOf<S>::type* e, S* held,
                                    const S* __restrict__ sums, int count) {
    using R = typename RealOf<S>::type;
    __shared__ S partial[kColumn + kColumn / 16];
    const int t = static_cast<int>(threadIdx.x), i = j - first;
    const S zero = from_real(S{}, static_cast<R>(0));
    if (i > 0) {
        const S* before = reflectors + static_cast<long long>(j - 1) * n;
        S* made = w + static_cast<long long>(i - 1) * n;
        S sum = zero;
        for (int q = t; q < count; q += kColumn) sum = add(sum, sums[q]);
        const S at_j = made[j];  // read by all before the sums, changed after them
        const S alpha = mul(from_real(S{}, static_cast<R>(-0.5)), mul(tau[j - 1], column_sum(partial, sum)));
        const S both = add(cj(at_j), from_real(S{}, static_cast<R>(2) * re(alpha)));
        for (int r = j + t; r < n; r += kColumn) {
            const S v = before[r];
            made[r] = add(made[r], mul(alpha, v));
            if (!finish) held[r] = sub(held[r], mul(both, v));
        }
        if (finish) return;
    } else {
        for (int r = j + t; r < n; r += kColumn) held[r] = a[r + static_cast<long long>(j) * n];
    }
    __syncthreads();
    R tail = static_cast<R>(0);
    for (int r = j + 2 + t; r < n; r += kColumn) tail += re(held[r]) * re(held[r]) + im(held[r]) * im(held[r]);
    tail = re(column_sum(partial, from_real(S{}, tail)));
    if (t == 0) d[j] = re(held[j]);
    if (j == n - 1) return;
    const S alpha = held[j + 1];
    S reflection = zero, scale = from_real(S{}, static_cast<R>(1));
    R beta = re(alpha);
    if (tail != static_cast<R>(0) || im(alpha) != static_cast<R>(0)) {
        const R size = sqrt(re(alpha) * re(alpha) + im(alpha) * im(alpha) + tail);
        beta = re(alpha) >= static_cast<R>(0) ? -size : size;
        reflection = mul(sub(from_real(S{}, beta), alpha), from_real(S{}, static_cast<R>(1) / beta));
        scale = inverse(sub(alpha, from_real(S{}, beta)));
    }
    S* v = reflectors + static_cast<long long>(j) * n;
    for (int r = j + 1 + t; r < n; r += kColumn) v[r] = r == j + 1 ? from_real(S{}, static_cast<R>(1)) : mul(held[r], scale);
    if (t == 0) {
        tau[j] = reflection;
        e[j] = beta;
    }
}

// The products with the reflector j, v_r at r: a block to each column beyond
// j, y_c = sum_r conj(a_rc) v_r, then one to each of the count columns of w
// and of v of the panel before it, into dots and dots + kColumns. kStep
// threads, kEarly numbers of the column on their way at a time.
template <class S>
__global__ void panel_products_kernel(int n, int j, int first, int count, const S* __restrict__ a,
                                      const S* __restrict__ reflectors, const S* __restrict__ w, S* __restrict__ y,
                                      S* __restrict__ dots) {
    using R = typename RealOf<S>::type;
    __shared__ S partial[kStep + kGroups];
    const int t = static_cast<int>(threadIdx.x), m = n - 1 - j, b = static_cast<int>(blockIdx.x);
    const S* v = reflectors + static_cast<long long>(j) * n;
    const S* column = b < m ? a + static_cast<long long>(j + 1 + b) * n
                      : b < m + count ? w + static_cast<long long>(b - m) * n
                                      : reflectors + static_cast<long long>(first + b - m - count) * n;
    const S zero = from_real(S{}, static_cast<R>(0));
    S sum = zero;
    for (int r = j + 1 + t; r < n; r += kEarly * kStep) {
        S x[kEarly], u[kEarly];
#pragma unroll
        for (int q = 0; q < kEarly; ++q) {
            const bool in = r + q * kStep < n;
            x[q] = in ? column[r + q * kStep] : zero;
            u[q] = in ? v[r + q * kStep] : zero;
        }
#pragma unroll
        for (int q = 0; q < kEarly; ++q) sum = add(sum, mul(cj(x[q]), u[q]));
    }
    sum = block_sum(partial, sum);
    if (t != 0) return;
    if (b < m)
        y[j + 1 + b] = sum;
    else
        dots[b < m + count ? b - m : kColumns + b - m - count] = sum;
}

constexpr int kReflectors = 64;  // the most of a block of the back-transformation

// The triangular factor t (count, count) of a block of reflectors,
// 1 - v t v^H: t_ii = tau_i and t[:i, i] = -tau_i t[:i, :i] (v^H v)[:i, i],
// from gram = v^H v. A row of t needs only itself: a thread to each.
template <class S>
__global__ void block_factor_kernel(int count, const S* __restrict__ tau, const S* __restrict__ gram, S* __restrict__ t) {
    using R = typename RealOf<S>::type;
    const int r = static_cast<int>(threadIdx.x);
    if (r >= count) return;
    S row[kReflectors];
    for (int i = 0; i < r; ++i) t[r + i * count] = from_real(S{}, static_cast<R>(0));
    row[r] = tau[r];
    t[r + r * count] = row[r];
    for (int i = r + 1; i < count; ++i) {
        S sum = from_real(S{}, static_cast<R>(0));
        for (int c = r; c < i; ++c) sum = add(sum, mul(row[c], gram[c + i * count]));
        row[i] = mul(sub(from_real(S{}, static_cast<R>(0)), tau[i]), sum);
        t[r + i * count] = row[i];
    }
}

// z (n, n) of the type of the matrix from the real q.
template <class S>
__global__ void widen_kernel(long long size, const typename RealOf<S>::type* __restrict__ q, S* __restrict__ z) {
    const long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (e < size) z[e] = from_real(S{}, q[e]);
}

// ------------------------------------------------- the divide and conquer
//
// The merges of one height of the tree at a time. Everything of a merge is
// found by the position p of a row or column of the matrix: the offset, the
// size and the count of kept columns of the merge that holds p (size 0 where
// none does), and at the position offset + c what belongs to its column c in
// the order [kept, deflated]: the column it comes from (source), and of a kept
// one its pole and its component w of the unit vector, of a deflated one its
// eigenvalue (all, where the roots of the kept join them).
struct Level {
    int n;
    const int* offset;
    const int* size;
    const int* kept;
    const int* source;
    const double* pole;
    const double* w;
    const double* rho;
    double* all;
};

constexpr int kSecular = 64;  // threads of the block of a root

// f = 1 + rho (psi + phi) of the secular equation of root i at d_origin + mu:
// psi over the poles up to i, phi over those beyond, their slopes, and the size
// of the terms; the threads of the block share the poles, the first adds their
// sums (row 5 of part) and all read them.
__device__ __forceinline__ double secular_value(int i, int k, const double* __restrict__ d, const double* __restrict__ w,
                                                double rho, int origin, double mu, double (*part)[kSecular],
                                                double& psi, double& dpsi, double& phi, double& dphi, double& size) {
    double sum[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    const double shift = d[origin];
    for (int j = static_cast<int>(threadIdx.x); j < k; j += kSecular) {
        const double inverse = 1.0 / ((d[j] - shift) - mu);
        const double term = w[j] * w[j] * inverse;
        sum[4] += fabs(term);
        if (j <= i) {
            sum[0] += term;
            sum[1] += term * inverse;
        } else {
            sum[2] += term;
            sum[3] += term * inverse;
        }
    }
    for (int q = 0; q < 5; ++q) part[q][threadIdx.x] = sum[q];
    __syncthreads();
    if (threadIdx.x == 0)
        for (int q = 0; q < 5; ++q) {
            double total = part[q][0];
            for (int x = 1; x < kSecular; ++x) total += part[q][x];
            part[5][q] = total;
        }
    __syncthreads();
    for (int q = 0; q < 5; ++q) sum[q] = part[5][q];
    psi = sum[0];
    dpsi = sum[1];
    phi = sum[2];
    dphi = sum[3];
    size = 1.0 + rho * sum[4];
    return 1.0 + rho * (sum[0] + sum[2]);
}

// A block to each position: root i (of k) of 1 + rho sum_j w_j^2 / (d_j - x) = 0,
// rho > 0 and d ascending and distinct: the one between d_i and d_(i + 1), or
// beyond d_(k - 1) for the last. Column i of delta (at the block of the merge)
// holds d_j - x, each from the nearer pole so that it is accurate; the root
// goes to all.
__global__ void secular_kernel(Level a, double* __restrict__ delta) {
    const int p = static_cast<int>(blockIdx.x), offset = a.offset[p], k = a.kept[p], i = p - offset;
    if (a.size[p] == 0 || i >= k) return;
    __shared__ double part[6][kSecular];
    const double* d = a.pole + offset;
    const double* w = a.w + offset;
    const double rho = a.rho[p];
    const bool last = i == k - 1;
    double gap = 0.0;
    if (last) {
        double norm = 0.0;
        for (int j = static_cast<int>(threadIdx.x); j < k; j += kSecular) norm += w[j] * w[j];
        part[0][threadIdx.x] = norm;
        __syncthreads();
        if (threadIdx.x == 0) {
            norm = part[0][0];
            for (int x = 1; x < kSecular; ++x) norm += part[0][x];
            part[5][0] = norm;
        }
        __syncthreads();
        gap = rho * part[5][0];
    } else {
        gap = d[i + 1] - d[i];
    }
    double psi, dpsi, phi, dphi, size;
    // the sign of the function at the middle chooses the pole the root is nearer to
    int origin = i;
    if (!last && secular_value(i, k, d, w, rho, i, 0.5 * gap, part, psi, dpsi, phi, dphi, size) < 0.0) origin = i + 1;
    // the root as d_origin + mu, mu within (lo, hi)
    double lo = origin == i ? 0.0 : -gap, hi = origin == i ? gap : 0.0;
    if (!last) {
        if (origin == i)
            hi = 0.5 * gap;
        else
            lo = -0.5 * gap;
    }
    double mu = 0.5 * (lo + hi);
    const double eps = 2.220446049250313e-16;
    for (int iteration = 0; iteration < 200; ++iteration) {
        const double f = secular_value(i, k, d, w, rho, origin, mu, part, psi, dpsi, phi, dphi, size);
        if (fabs(f) <= 8.0 * eps * size) break;
        if (f < 0.0)
            lo = mu;
        else
            hi = mu;
        if (hi - lo <= 2.0 * eps * fmax(fabs(lo), fabs(hi))) {
            mu = 0.5 * (lo + hi);
            break;
        }
        // psi and phi each as a constant plus a simple pole, at d_i and at
        // d_(i + 1), with their values and slopes here: the root of the
        // resulting quadratic within the bracket, or its middle
        const double di = (d[i] - d[origin]) - mu;                      // d_i - x
        const double dn = last ? 0.0 : (d[i + 1] - d[origin]) - mu;     // d_(i + 1) - x
        const double pp = dpsi * di * di, s = psi - dpsi * di;          // psi ~ s + pp / (d_i - x)
        const double q = last ? 0.0 : dphi * dn * dn, r = last ? 0.0 : phi - dphi * dn;
        const double constant = 1.0 + rho * (s + r);
        double next = 0.5 * (lo + hi);
        bool found = false;
        if (last || q == 0.0) {
            // constant + rho pp / (d_i - x) = 0: d_i - x = -rho pp / constant
            if (constant > 0.0) {
                next = mu + (di + rho * pp / constant);
                found = true;
            }
        } else {
            // constant (di - t)(dn - t) + rho pp (dn - t) + rho q (di - t) = 0 in the step t = x_new - x
            const double qa = constant;
            const double qb = -(constant * (di + dn) + rho * (pp + q));
            const double qc = constant * di * dn + rho * (pp * dn + q * di);
            const double disc = qb * qb - 4.0 * qa * qc;
            if (disc >= 0.0) {
                const double root = sqrt(disc);
                const double big = qb >= 0.0 ? -(qb + root) / 2.0 : -(qb - root) / 2.0;
                const double t1 = qa != 0.0 ? big / qa : qc / (-qb), t2 = big != 0.0 ? qc / big : t1;
                if (mu + t1 > lo && mu + t1 < hi) {
                    next = mu + t1;
                    found = true;
                } else if (mu + t2 > lo && mu + t2 < hi) {
                    next = mu + t2;
                    found = true;
                }
            }
        }
        if (!found || !(next > lo && next < hi)) next = 0.5 * (lo + hi);
        mu = next;
    }
    double* column = delta + static_cast<long long>(p) * a.n + offset;
    for (int j = static_cast<int>(threadIdx.x); j < k; j += kSecular) column[j] = (d[j] - d[origin]) - mu;
    if (threadIdx.x == 0) a.all[p] = d[origin] + mu;
}

// A block to each position: w of the kept column j from the roots,
// w_j^2 = prod_i (root_i - d_j) / (rho prod_(i != j) (d_i - d_j)), with the sign
// it had; the threads share the roots, and the first multiplies their products.
__global__ void fit_kernel(Level a, const double* __restrict__ delta, double* __restrict__ fitted) {
    const int p = static_cast<int>(blockIdx.x), offset = a.offset[p], k = a.kept[p], j = p - offset;
    if (a.size[p] == 0 || j >= k) return;
    __shared__ double part[kSecular];
    const double* d = a.pole + offset;
    const double* row = delta + static_cast<long long>(offset) * a.n + p;  // d_j - root_i, i after i
    double product = 1.0;
    for (int i = static_cast<int>(threadIdx.x); i < k; i += kSecular)
        if (i != j) product *= row[static_cast<long long>(i) * a.n] / (d[j] - d[i]);
    part[threadIdx.x] = product;
    __syncthreads();
    if (threadIdx.x != 0) return;
    product = -row[static_cast<long long>(j) * a.n] / a.rho[p];
    for (int x = 0; x < kSecular; ++x) product *= part[x];
    fitted[p] = copysign(sqrt(fabs(product)), a.w[p]);
}

// A block to each position: the eigenvector of root i of the merge in the
// kept columns, w_j / (d_j - root_i) at unit length, as column i of u (at the
// block of the merge), in the precision of the vectors; the threads share the
// rows, and the first adds their squares.
template <class R>
__global__ void vectors_kernel(Level a, double* __restrict__ delta, const double* __restrict__ fitted, R* __restrict__ u) {
    const int p = static_cast<int>(blockIdx.x), offset = a.offset[p], k = a.kept[p];
    if (a.size[p] == 0 || p - offset >= k) return;
    __shared__ double part[kSecular + 1];
    double* column = delta + static_cast<long long>(p) * a.n + offset;
    R* out = u + static_cast<long long>(p) * a.n + offset;
    double norm = 0.0;
    for (int j = static_cast<int>(threadIdx.x); j < k; j += kSecular) {
        const double x = fitted[offset + j] / column[j];
        column[j] = x;
        norm += x * x;
    }
    part[threadIdx.x] = norm;
    __syncthreads();
    if (threadIdx.x == 0) {
        norm = part[0];
        for (int x = 1; x < kSecular; ++x) norm += part[x];
        part[kSecular] = 1.0 / sqrt(norm);
    }
    __syncthreads();
    norm = part[kSecular];
    for (int j = static_cast<int>(threadIdx.x); j < k; j += kSecular) out[j] = static_cast<R>(column[j] * norm);
}

// A thread to each position: the place of the column c of [roots, deflated]
// among the eigenvalues of its merge ascending (equal ones in their order):
// order at that place is c, and sorted its eigenvalue.
__global__ void order_kernel(Level a, int* __restrict__ order, double* __restrict__ sorted) {
    const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= a.n) return;
    const int offset = a.offset[p], size = a.size[p], c = p - offset;
    if (size == 0) return;
    const double* all = a.all + offset;
    const double value = all[c];
    int place = 0;
    for (int b = 0; b < size; ++b) place += all[b] < value || (all[b] == value && b < c) ? 1 : 0;
    order[offset + place] = c;
    sorted[offset + place] = value;
}

// A thread to each element: the kept columns of the block of each merge, in
// their order, as the first columns of the same block of gathered.
template <class R>
__global__ void gather_kernel(Level a, const R* __restrict__ q, R* __restrict__ gathered) {
    const long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (e >= static_cast<long long>(a.n) * a.n) return;
    const int p = static_cast<int>(e / a.n), r = static_cast<int>(e % a.n);
    const int offset = a.offset[p], size = a.size[p];
    if (size == 0 || p - offset >= a.kept[p] || r < offset || r >= offset + size) return;
    gathered[e] = q[r + static_cast<long long>(offset + a.source[p]) * a.n];
}

// A thread to each element: next is q with the block of each merge replaced by
// the columns of [product, the deflated columns of q] in the order of their
// eigenvalues; the first and the last row of each block go to first and last.
template <class R>
__global__ void assemble_kernel(Level a, const R* __restrict__ q, const R* __restrict__ product,
                                const int* __restrict__ order, R* __restrict__ next, double* __restrict__ first,
                                double* __restrict__ last) {
    const long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (e >= static_cast<long long>(a.n) * a.n) return;
    const int p = static_cast<int>(e / a.n), r = static_cast<int>(e % a.n);
    const int offset = a.offset[p], size = a.size[p];
    if (size == 0 || r < offset || r >= offset + size) {
        next[e] = q[e];
        return;
    }
    const int from = order[p];
    const R value = from < a.kept[p] ? product[r + static_cast<long long>(offset + from) * a.n]
                                     : q[r + static_cast<long long>(offset + a.source[offset + from]) * a.n];
    next[e] = value;
    if (r == offset) first[p] = static_cast<double>(value);
    if (r == offset + size - 1) last[p] = static_cast<double>(value);
}

// A thread to each row of a block: its columns i and j (positions, count pairs
// one after the other in columns) replaced by c q_i + s q_j and c q_j - s q_i,
// in their order.
template <class R>
__global__ void rotate_kernel(int n, int offset, int size, int count, const int* __restrict__ columns,
                              const double* __restrict__ angles, R* q) {
    const int r = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (r >= size) return;
    for (int m = 0; m < count; ++m) {
        R* x = q + static_cast<long long>(columns[2 * m]) * n + offset + r;
        R* y = q + static_cast<long long>(columns[2 * m + 1]) * n + offset + r;
        const R c = static_cast<R>(angles[2 * m]), s = static_cast<R>(angles[2 * m + 1]);
        const R xa = *x, yb = *y;
        *x = c * xa + s * yb;
        *y = c * yb - s * xa;
    }
}

// The upper triangle from the lower one, outside the diagonal blocks of
// block columns: r_ij = conj(l_ji) where i is before the block of j.
template <class S>
__global__ void upper_from_lower_kernel(int n, int block, S* a) {
    const long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (e >= static_cast<long long>(n) * n) return;
    const long long i = e % n, j = e / n;
    if (i < (j / block) * block) a[e] = cj(a[j + i * n]);
}

void launched(const char* what) { check_cuda(cudaGetLastError(), what); }

// The eigenvalues (d on return, ascending) of the tridiagonal matrix of
// diagonal d (n) and subdiagonal e (n - 1), and its eigenvectors as the columns
// of q (n, n), real numbers of the precision of the matrix: the leaves on the
// host, then the merges height by height, each height one transfer to the GPU
// and one from it. With scalars the roots of the merges and their small
// eigenvectors are found on the GPU, else on the host.
template <class R>
void divide_and_conquer(std::vector<double>& d, const std::vector<double>& e, DeviceArray<R>& q, Blas& blas, bool scalars,
                        int leaf = 32) {
    const int n = static_cast<int>(d.size());
    const std::size_t count = static_cast<std::size_t>(n), size = count * count;
    // scaled to a largest element of one, so that the tolerances are relative
    double scale = 0.0;
    for (int i = 0; i < n; ++i) scale = std::max({scale, std::abs(d[i]), i + 1 < n ? std::abs(e[i]) : 0.0});
    if (scale == 0.0) scale = 1.0;
    std::vector<double> cut_e(e);
    for (double& x : d) x /= scale;
    for (double& x : cut_e) x /= scale;
    const host::Tree tree = host::cut(d, cut_e, leaf);
    // the leaves, and of every block the first and the last row of its vectors
    std::vector<R> start(size, static_cast<R>(0));
    std::vector<double> first(count), last(count), values, below, vectors;
    for (std::size_t i = 0; i < tree.leaves.size(); i += 2) {
        const int offset = tree.leaves[i], rows = tree.leaves[i + 1];
        values.assign(d.begin() + offset, d.begin() + offset + rows);
        below.assign(cut_e.begin() + offset, cut_e.begin() + offset + std::max(rows - 1, 0));
        host::tridiagonal_ql(values, below, vectors);
        std::copy(values.begin(), values.end(), d.begin() + offset);
        for (int j = 0; j < rows; ++j) {
            const double* column = vectors.data() + static_cast<std::size_t>(j) * rows;
            for (int r = 0; r < rows; ++r)
                start[static_cast<std::size_t>(offset + j) * count + offset + r] = static_cast<R>(column[r]);
            first[static_cast<std::size_t>(offset + j)] = column[0];
            last[static_cast<std::size_t>(offset + j)] = column[rows - 1];
        }
    }
    q.upload(start.data(), size);
    if (tree.merges.empty()) {
        for (double& x : d) x *= scale;
        return;
    }
    DeviceArray<R> next(size), gathered(size), product(size), u(size);
    DeviceArray<double> delta, numbers(4 * count), fitted(count), results(3 * count), angles;
    std::vector<R> host_u;
    std::vector<double> small;
    if (scalars)
        delta.resize(size);
    else
        host_u.resize(size);
    DeviceArray<int> integers(4 * count), order(count), columns;
    std::vector<double> host_numbers(4 * count), host_results(3 * count), host_angles, z;
    std::vector<int> host_integers(4 * count), host_columns;
    struct Turned {
        int offset, size, first, count;  // the rotations of a merge, from first
    };
    std::vector<Turned> turned;
    const unsigned positions = blocks(n), elements = blocks(static_cast<long long>(size));
    std::size_t at = 0;
    for (int height = 1; height <= tree.height; ++height) {
        std::fill(host_integers.begin(), host_integers.end(), 0);
        host_angles.clear();
        host_columns.clear();
        turned.clear();
        const std::size_t begin = at;
        for (; at < tree.merges.size() && tree.merges[at].height == height; ++at) {
            const host::Merge& m = tree.merges[at];
            const std::size_t o = static_cast<std::size_t>(m.offset);
            // the last row of the first half and, with the sign of the cut, the first of the second
            z.resize(static_cast<std::size_t>(m.size));
            for (int j = 0; j < m.size; ++j) z[static_cast<std::size_t>(j)] = j < m.half ? last[o + j] : m.sign * first[o + j];
            const host::Deflation f = host::deflate(d.data() + m.offset, m.size, z, m.rho);
            const int kept = static_cast<int>(f.kept.size());
            for (int c = 0; c < m.size; ++c) {
                const std::size_t p = o + static_cast<std::size_t>(c);
                host_integers[p] = m.offset;
                host_integers[count + p] = m.size;
                host_integers[2 * count + p] = kept;
                host_numbers[2 * count + p] = f.rho;
                if (c < kept) {
                    host_integers[3 * count + p] = f.kept[static_cast<std::size_t>(c)];
                    host_numbers[p] = f.pole[static_cast<std::size_t>(c)];
                    host_numbers[count + p] = f.w[static_cast<std::size_t>(c)];
                } else {
                    const int from = f.deflated[static_cast<std::size_t>(c - kept)];
                    host_integers[3 * count + p] = from;
                    host_numbers[3 * count + p] = f.values[static_cast<std::size_t>(from)];
                }
            }
            if (!scalars && kept > 0) {
                // the roots beside the eigenvalues of the deflated columns, u at the block of the merge
                small.resize(static_cast<std::size_t>(kept) * kept);
                host::secular(f, host_numbers.data() + 3 * count + o, small.data());
                for (int i = 0; i < kept; ++i)
                    for (int j = 0; j < kept; ++j)
                        host_u[(o + static_cast<std::size_t>(i)) * count + o + static_cast<std::size_t>(j)] =
                            static_cast<R>(small[static_cast<std::size_t>(i) * kept + j]);
            }
            if (!f.rotations.empty()) {
                turned.push_back({m.offset, m.size, static_cast<int>(host_columns.size() / 2), static_cast<int>(f.rotations.size())});
                for (const auto& r : f.rotations) {
                    host_columns.push_back(m.offset + r.i);
                    host_columns.push_back(m.offset + r.j);
                    host_angles.push_back(r.c);
                    host_angles.push_back(r.s);
                }
            }
        }
        integers.assign(host_integers.data(), host_integers.size());
        numbers.assign(host_numbers.data(), host_numbers.size());
        const Level level{n, integers.data(), integers.data() + count, integers.data() + 2 * count,
                          integers.data() + 3 * count, numbers.data(), numbers.data() + count,
                          numbers.data() + 2 * count, numbers.data() + 3 * count};
        if (!turned.empty()) {
            columns.assign(host_columns.data(), host_columns.size());
            angles.assign(host_angles.data(), host_angles.size());
            for (const Turned& t : turned)
                rotate_kernel<R><<<blocks(t.size), kThreads>>>(n, t.offset, t.size, t.count, columns.data() + 2 * t.first,
                                                              angles.data() + 2 * t.first, q.data());
        }
        if (scalars) {
            secular_kernel<<<static_cast<unsigned>(n), kSecular>>>(level, delta.data());
            fit_kernel<<<static_cast<unsigned>(n), kSecular>>>(level, delta.data(), fitted.data());
            vectors_kernel<R><<<static_cast<unsigned>(n), kSecular>>>(level, delta.data(), fitted.data(), u.data());
        } else {
            u.assign(host_u.data(), size);
        }
        order_kernel<<<positions, kThreads>>>(level, order.data(), results.data());
        gather_kernel<R><<<elements, kThreads>>>(level, q.data(), gathered.data());
        launched("the merges of the divide and conquer");
        for (std::size_t i = begin; i < at; ++i) {
            const host::Merge& m = tree.merges[i];
            const int kept = host_integers[2 * count + static_cast<std::size_t>(m.offset)];
            const std::size_t block = static_cast<std::size_t>(m.offset) * count + static_cast<std::size_t>(m.offset);
            if (kept > 0)
                blas.gemm(Op::N, Op::N, m.size, kept, kept, static_cast<R>(1), gathered.data() + block, n,
                          u.data() + block, n, static_cast<R>(0), product.data() + block, n);
        }
        assemble_kernel<R><<<elements, kThreads>>>(level, q.data(), product.data(), order.data(), next.data(),
                                                  results.data() + count, results.data() + 2 * count);
        launched("the eigenvectors of the merges");
        std::swap(q, next);
        results.download(host_results.data(), host_results.size());
        for (std::size_t i = begin; i < at; ++i) {
            const host::Merge& m = tree.merges[i];
            const std::size_t o = static_cast<std::size_t>(m.offset);
            for (std::size_t c = 0; c < static_cast<std::size_t>(m.size); ++c) {
                d[o + c] = host_results[o + c];
                first[o + c] = host_results[count + o + c];
                last[o + c] = host_results[2 * count + o + c];
            }
        }
    }
    for (double& x : d) x *= scale;
}

// the scalars of the kernels have the layout of those of the libraries
Cx<double>* own(Complex* p) { return reinterpret_cast<Cx<double>*>(p); }
Cx<float>* own(cuComplex* p) { return reinterpret_cast<Cx<float>*>(p); }
double* own(double* p) { return p; }
float* own(float* p) { return p; }
std::complex<double> host_of(Complex x) { return {x.x, x.y}; }
std::complex<double> host_of(cuComplex x) { return {x.x, x.y}; }
std::complex<double> host_of(double x) { return x; }
std::complex<double> host_of(float x) { return x; }
void device_of(const std::complex<double>& x, Complex& out) { out = make_cuDoubleComplex(x.real(), x.imag()); }
void device_of(const std::complex<double>& x, cuComplex& out) {
    out = make_cuComplex(static_cast<float>(x.real()), static_cast<float>(x.imag()));
}
void device_of(const std::complex<double>& x, double& out) { out = x.real(); }
void device_of(const std::complex<double>& x, float& out) { out = static_cast<float>(x.real()); }
Complex unit(Complex, double x) { return make_cuDoubleComplex(x, 0.0); }
cuComplex unit(cuComplex, double x) { return make_cuComplex(static_cast<float>(x), 0.0f); }
double unit(double, double x) { return x; }
float unit(float, double x) { return static_cast<float>(x); }

constexpr int kPanel = 64;       // columns of a block of the Cholesky factorization

}  // namespace

Lapack::Lapack(Kernels kernels) : kernels_(kernels), blas_(kernels) { info_.resize(1); }

Lapack::~Lapack() {
    if (vendor_ != nullptr) cusolverDnDestroy(vendor_);
}

// made at its first use
cusolverDnHandle_t Lapack::vendor() {
    if (vendor_ == nullptr) {
        check_solver(cusolverDnCreate(&vendor_), "cusolverDnCreate");
        check_solver(cusolverDnSetStream(vendor_, cudaStreamPerThread), "cusolverDnSetStream");
    }
    return vendor_;
}

// The NVIDIA GPUs of compute capability 6.0 to 10.0, and the AMD GPUs HIP runs on.
bool Lapack::full_double() {
    if (full_double_ < 0) {
#if defined(MDAPY_GPU_HIP)
        full_double_ = 1;
#else
        int device = 0, major = 0, minor = 0;
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        check_cuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "the capability");
        check_cuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device), "the capability");
        full_double_ = minor == 0 && major >= 6 && major <= 10 ? 1 : 0;
#endif
    }
    return full_double_ == 1;
}

// On the host, in double precision, when small; else on the GPU.
template <class T>
void Lapack::own_eigh(T* a, int n, std::vector<double>& values) {
    if (n > host_limit_) return device_eigh(a, n, values);
    const std::size_t size = static_cast<std::size_t>(n) * n;
    std::vector<T> narrow_host(size);
    download(a, size, narrow_host.data());
    std::vector<decltype(wide(T{}))> wide_host(size);
    for (std::size_t i = 0; i < size; ++i) wide_host[i] = wide(narrow_host[i]);
    values.resize(static_cast<std::size_t>(n));
    host::eigh(wide_host.data(), n, values.data());
    for (std::size_t i = 0; i < size; ++i) narrow(wide_host[i], narrow_host[i]);
    upload(narrow_host.data(), size, a);
}

// a = q t q^H on the GPU (reflector after reflector), the eigenpairs of the
// tridiagonal t by divide and conquer, and q applied to them in blocks of
// reflectors.
template <class T>
void Lapack::device_eigh(T* a, int n, std::vector<double>& values) {
    using S = std::remove_pointer_t<decltype(own(a))>;
    using R = typename RealOf<S>::type;
    const std::size_t size = static_cast<std::size_t>(n) * n;
    S* matrix = own(a);
    // the reflectors as columns (zeros down to the diagonal of each), and p of two steps
    DeviceArray<T> tau(static_cast<std::size_t>(n)), reflectors(size), products(2 * static_cast<std::size_t>(n)), held;
    DeviceArray<R> diagonal(static_cast<std::size_t>(n)), subdiagonal(static_cast<std::size_t>(n));
    reflectors.zero();
    mirror_kernel<S><<<blocks(static_cast<long long>(size)), kThreads>>>(n, matrix);
    launched("the lower triangle of a matrix");
    {
        MDAPY_SPAN("tridiagonal");
        // one launch a step repeats its first part in every block: not where the
        // arithmetic of double precision is what the GPU lacks, and there in
        // single precision for the smaller matrices (measured on an RTX A2000)
        const int limit = fused_limit_ >= 0 ? std::min(fused_limit_, kFused)
                          : full_double() ? std::min(kFused, static_cast<int>(kFusedBytes / sizeof(S)))
                          : sizeof(R) == sizeof(float) ? kFusedSingle : 0;
        const bool fused = n <= limit;
        if (!fused) held.resize(static_cast<std::size_t>(n));
        const std::size_t sums = (kStep + kGroups) * sizeof(S);
        // panels from 1024 complex numbers of double precision and 1536 of the
        // other types (the crossovers of a GH200 and a V100 lie around them); on
        // a GPU of little double precision in single precision only (an RTX A2000)
        const int from = panels_from_ >= 0 ? panels_from_
                         : sizeof(R) == sizeof(double) && !full_double() ? n + 1
                         : sizeof(S) == 2 * sizeof(double) ? kPanelsFrom : kPanelsFrom + kPanelsFrom / 2;
        const bool panels = n >= from && !fused;
        if (panels) {
            DeviceArray<T> w(static_cast<std::size_t>(n) * kColumns), dots(2 * kColumns), sums(blocks(n, kRows));
            const T one = unit(T{}, 1.0), minus = unit(T{}, -1.0);
            S* y = own(products.data());
            // w of the reflector before the column j, which is past the panel when finish
            const auto rows = [&](int j, int first, int finish) {
                const int count = static_cast<int>(blocks(n - j, kRows));
                if (j > first)
                    panel_rows_kernel<S><<<static_cast<unsigned>(count), kRows * kLanes>>>(
                        n, j, first, finish, matrix, own(reflectors.data()), own(w.data()), own(tau.data()), y,
                        own(dots.data()), own(held.data()), own(sums.data()));
                panel_column_kernel<S><<<1, kColumn>>>(n, j, first, finish, matrix, own(reflectors.data()), own(w.data()),
                                                      own(tau.data()), diagonal.data(), subdiagonal.data(),
                                                      own(held.data()), own(sums.data()), count);
            };
            for (int first = 0; first < n; first += kColumns) {
                const int width = std::min(kColumns, n - first);
                for (int j = first; j < first + width; ++j) {
                    rows(j, first, 0);
                    if (j + 1 < n)
                        panel_products_kernel<S><<<static_cast<unsigned>(n - 1 - j + 2 * (j - first)), kStep>>>(
                            n, j, first, j - first, matrix, own(reflectors.data()), own(w.data()), y, own(dots.data()));
                }
                launched("a panel of the tridiagonalization");
                const int next = first + width, rest = n - next;
                if (rest < 1) break;
                rows(next, first, 1);
                launched("the last column of a panel");
                T* trailing = a + static_cast<std::size_t>(next) * n + next;
                const T* v2 = reflectors.data() + static_cast<std::size_t>(first) * n + next;
                const T* w2 = w.data() + next;
                blas_.gemm(Op::N, Op::C, rest, rest, width, minus, v2, n, w2, n, one, trailing, n);
                blas_.gemm(Op::N, Op::C, rest, rest, width, minus, w2, n, v2, n, one, trailing, n);
            }
        }
        for (int k = 0; k < n && !panels; ++k) {
            const S* in = own(products.data()) + ((k + 1) & 1) * static_cast<std::size_t>(n);
            S* out = own(products.data()) + (k & 1) * static_cast<std::size_t>(n);
            const unsigned columns = static_cast<unsigned>(std::max(n - 1 - k, 1));
            if (fused) {
                tridiagonal_kernel<S><<<columns, kStep, sums + 2 * static_cast<std::size_t>(n - k) * sizeof(S)>>>(
                    n, k, 2, matrix, own(reflectors.data()), own(tau.data()), diagonal.data(), subdiagonal.data(),
                    nullptr, in, out);
                continue;
            }
            tridiagonal_kernel<S><<<1, kStep, sums>>>(n, k, 0, matrix, own(reflectors.data()), own(tau.data()),
                                                     diagonal.data(), subdiagonal.data(), own(held.data()), in, out);
            if (k + 1 < n)
                tridiagonal_kernel<S><<<columns, kStep, sums>>>(n, k, 1, matrix, own(reflectors.data()),
                                                               own(tau.data()), diagonal.data(), subdiagonal.data(),
                                                               own(held.data()), in, out);
        }
        launched("the tridiagonalization");
    }
    std::vector<R> host(static_cast<std::size_t>(n));
    diagonal.download(host.data(), host.size());
    std::vector<double> d(host.begin(), host.end());
    subdiagonal.download(host.data(), host.size());
    std::vector<double> e(host.begin(), host.end() - 1);
    // the eigenpairs of t
    DeviceArray<R> q;
    {
        MDAPY_SPAN("divide");
        divide_and_conquer(d, e, q, blas_, full_double(), leaf_ > 0 ? leaf_ : n >= 1024 && full_double() ? 16 : 32);
    }
    values = d;
    // z = h_0 h_1 ... q, the last block of reflectors first: a block is
    // 1 - v t v^H with t upper triangular, t_ii = tau_i and
    // t[:i, i] = -tau_i t[:i, :i] (v^H v)[:i, i]
    MDAPY_SPAN("back");
    // blocks of 32 reflectors; of 64 for the large matrices where double
    // precision is full (measured on a GH200 and a V100: 4 to 9 % from 1024)
    const bool large = n >= 1024 && full_double();
    const int per = reflectors_ > 0 ? std::min(reflectors_, kReflectors) : large ? 64 : 32;
    DeviceArray<T> z(size), w(static_cast<std::size_t>(n) * per), tw(static_cast<std::size_t>(n) * per),
        gram(static_cast<std::size_t>(per) * per), small(static_cast<std::size_t>(per) * per);
    widen_kernel<S><<<blocks(static_cast<long long>(size)), kThreads>>>(static_cast<long long>(size), q.data(), own(z.data()));
    launched("the eigenvectors of the tridiagonal matrix");
    const int reflectors_count = n - 1;
    const T one = unit(T{}, 1.0), zero = unit(T{}, 0.0), minus = unit(T{}, -1.0);
    for (int first = reflectors_count > 0 ? ((reflectors_count - 1) / per) * per : -1; first >= 0; first -= per) {
        const int count = std::min(per, reflectors_count - first);
        const T* v = reflectors.data() + static_cast<std::size_t>(first) * n;
        blas_.gemm(Op::C, Op::N, count, count, n, one, v, n, v, n, zero, gram.data(), count);
        block_factor_kernel<S><<<1, kReflectors>>>(count, own(tau.data()) + first, own(gram.data()), own(small.data()));
        launched("the factor of a block of reflectors");
        blas_.gemm(Op::C, Op::N, count, n, n, one, v, n, z.data(), n, zero, w.data(), count);
        blas_.gemm(Op::N, Op::N, count, n, count, one, small.data(), count, w.data(), count, zero, tw.data(), count);
        blas_.gemm(Op::N, Op::N, n, n, count, minus, v, n, tw.data(), count, one, z.data(), n);
    }
    check_cuda(cudaMemcpy(a, z.data(), size * sizeof(T), cudaMemcpyDeviceToDevice), "cudaMemcpy");
}

// Blocks of kPanel columns on the lower triangle, a = l l^H: the diagonal
// block on the host (r of its upper triangle, l = r^H), the panel below it
// l_21 = a_21 r^-1, the trailing matrix less l_21 l_21^H; r = l^H at the end.
template <class T>
int Lapack::device_potrf(T* a, int n) {
    using S = std::remove_pointer_t<decltype(own(a))>;
    const std::size_t size = static_cast<std::size_t>(n) * n;
    const std::size_t ld = static_cast<std::size_t>(n) * sizeof(T);
    mirror_kernel<S><<<blocks(static_cast<long long>(size)), kThreads>>>(n, own(a));
    launched("the lower triangle of a matrix");
    std::vector<T> block(static_cast<std::size_t>(kPanel) * kPanel);
    std::vector<std::complex<double>> wide_block(block.size());
    for (int first = 0; first < n; first += kPanel) {
        const int width = std::min(kPanel, n - first), below = n - first - width;
        T* corner = a + static_cast<std::size_t>(first) * n + first;
        const std::size_t bytes = static_cast<std::size_t>(width) * sizeof(T);
        check_cuda(cudaMemcpy2D(block.data(), bytes, corner, ld, bytes, static_cast<std::size_t>(width), cudaMemcpyDeviceToHost),
                   "cudaMemcpy2D from the GPU");
        const std::size_t count = static_cast<std::size_t>(width) * width;
        for (std::size_t i = 0; i < count; ++i) wide_block[i] = host_of(block[i]);
        const int info = host::potrf(wide_block.data(), width);
        if (info != 0) return first + info;
        for (std::size_t i = 0; i < count; ++i) device_of(wide_block[i], block[i]);
        check_cuda(cudaMemcpy2D(corner, ld, block.data(), bytes, bytes, static_cast<std::size_t>(width), cudaMemcpyHostToDevice),
                   "cudaMemcpy2D to the GPU");
        if (below == 0) break;
        T* panel = corner + width;
        blas_.trsm(below, width, corner, n, panel, n);
        blas_.gemm(Op::N, Op::C, below, below, width, unit(T{}, -1.0), panel, n, panel, n, unit(T{}, 1.0),
                   panel + static_cast<std::size_t>(width) * n, n);
    }
    upper_from_lower_kernel<S><<<blocks(static_cast<long long>(size)), kThreads>>>(n, kPanel, own(a));
    launched("the upper triangle of a factor");
    return 0;
}

template <class T>
int Lapack::own_potrf(T* a, int n) {
    if (n > host_limit_) return device_potrf(a, n);
    const std::size_t size = static_cast<std::size_t>(n) * n;
    std::vector<T> narrow_host(size);
    download(a, size, narrow_host.data());
    std::vector<decltype(wide(T{}))> wide_host(size);
    for (std::size_t i = 0; i < size; ++i) wide_host[i] = wide(narrow_host[i]);
    const int info = host::potrf(wide_host.data(), n);
    if (info != 0) return info;
    // the upper triangle alone goes back
    for (int j = 0; j < n; ++j)
        for (int i = 0; i <= j; ++i) {
            const std::size_t at = static_cast<std::size_t>(j) * n + i;
            narrow(wide_host[at], narrow_host[at]);
        }
    upload(narrow_host.data(), size, a);
    return 0;
}

#define MDAPY_LAPACK(T, V, VALUES)                                                                           \
    void Lapack::eigh(T* a, int n, std::vector<double>& values) {                                            \
        values.clear();                                                                                      \
        if (n < 1) return;                                                                                   \
        MDAPY_SPAN("eigh");                                                                                  \
        if (kernels_ == Kernels::Mdapy) return own_eigh(a, n, values);                                       \
        VALUES.reserve(static_cast<std::size_t>(n));                                                         \
        vendor_eigh<V>(vendor(), a, n, VALUES.data(), work_, info_, values);                                 \
    }                                                                                                        \
    int Lapack::potrf(T* a, int n) {                                                                         \
        if (n < 1) return 0;                                                                                 \
        MDAPY_SPAN("potrf");                                                                                 \
        if (kernels_ == Kernels::Mdapy) return own_potrf(a, n);                                              \
        return vendor_potrf<V>(vendor(), a, n, work_, info_);                                                \
    }
MDAPY_LAPACK(Complex, Z, values_)
MDAPY_LAPACK(cuComplex, C, values_single_)
MDAPY_LAPACK(double, D, values_)
MDAPY_LAPACK(float, S, values_single_)
#undef MDAPY_LAPACK

}  // namespace mdapy::dft
