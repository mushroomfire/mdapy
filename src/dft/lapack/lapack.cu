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

// Step k of the tridiagonalization, one block: the reflector of the column k
// below the diagonal, h = 1 - tau v v^H with v_0 = 1 and h^H x = beta e_1
// (LAPACK's larfg); v replaces the column, tau, the diagonal element and beta
// (the subdiagonal element) are kept.
template <class S>
__global__ void reflector_kernel(int n, int k, S* a, S* tau, typename RealOf<S>::type* d,
                                 typename RealOf<S>::type* e) {
    using R = typename RealOf<S>::type;
    __shared__ R partial[kThreads];
    __shared__ S scale;
    S* column = a + static_cast<long long>(k) * n;
    R sum = static_cast<R>(0);
    for (int i = k + 2 + static_cast<int>(threadIdx.x); i < n; i += kThreads) {
        const S x = column[i];
        sum += re(x) * re(x) + im(x) * im(x);
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int width = kThreads / 2; width > 0; width >>= 1) {
        if (static_cast<int>(threadIdx.x) < width) partial[threadIdx.x] += partial[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const S alpha = column[k + 1];
        const R tail = partial[0];
        d[k] = re(column[k]);
        if (tail == static_cast<R>(0) && im(alpha) == static_cast<R>(0)) {
            tau[k] = from_real(S{}, static_cast<R>(0));
            e[k] = re(alpha);
            scale = from_real(S{}, static_cast<R>(1));
        } else {
            const R size = sqrt(re(alpha) * re(alpha) + im(alpha) * im(alpha) + tail);
            const R beta = re(alpha) >= static_cast<R>(0) ? -size : size;
            // tau = (beta - alpha) / beta, v = x / (alpha - beta)
            const S difference = sub(from_real(S{}, beta), alpha);
            tau[k] = mul(difference, from_real(S{}, static_cast<R>(1) / beta));
            scale = inverse(sub(alpha, from_real(S{}, beta)));
            e[k] = beta;
        }
        column[k + 1] = from_real(S{}, static_cast<R>(1));
    }
    __syncthreads();
    const S factor = scale;
    for (int i = k + 2 + static_cast<int>(threadIdx.x); i < n; i += kThreads) column[i] = mul(column[i], factor);
}

// p_i = sum_j conj(a_ji) v_j over the trailing block (the indices beyond k),
// v the column k: a block to each i, along its column.
template <class S>
__global__ void column_products_kernel(int n, int k, const S* __restrict__ a, S* __restrict__ p) {
    __shared__ S partial[kThreads / 2];
    const int i = k + 1 + static_cast<int>(blockIdx.x);
    const S* column = a + static_cast<long long>(i) * n;
    const S* v = a + static_cast<long long>(k) * n;
    S sum = from_real(S{}, static_cast<typename RealOf<S>::type>(0));
    for (int j = k + 1 + static_cast<int>(threadIdx.x); j < n; j += kThreads / 2) sum = add(sum, mul(cj(column[j]), v[j]));
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int width = kThreads / 4; width > 0; width >>= 1) {
        if (static_cast<int>(threadIdx.x) < width) partial[threadIdx.x] = add(partial[threadIdx.x], partial[threadIdx.x + width]);
        __syncthreads();
    }
    if (threadIdx.x == 0) p[i] = partial[0];
}

// w = tau p - (tau / 2) ((tau p)^H v) v, in p, one block.
template <class S>
__global__ void update_vector_kernel(int n, int k, const S* __restrict__ a, const S* __restrict__ tau, S* p) {
    using R = typename RealOf<S>::type;
    __shared__ S partial[kThreads];
    __shared__ S half;
    const S* v = a + static_cast<long long>(k) * n;
    const S t = tau[k];
    S sum = from_real(S{}, static_cast<R>(0));
    for (int i = k + 1 + static_cast<int>(threadIdx.x); i < n; i += kThreads) {
        const S scaled = mul(t, p[i]);
        p[i] = scaled;
        sum = add(sum, mul(cj(scaled), v[i]));
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int width = kThreads / 2; width > 0; width >>= 1) {
        if (static_cast<int>(threadIdx.x) < width) partial[threadIdx.x] = add(partial[threadIdx.x], partial[threadIdx.x + width]);
        __syncthreads();
    }
    if (threadIdx.x == 0) half = mul(from_real(S{}, static_cast<R>(-0.5)), mul(t, partial[0]));
    __syncthreads();
    const S factor = half;
    for (int i = k + 1 + static_cast<int>(threadIdx.x); i < n; i += kThreads) p[i] = add(p[i], mul(factor, v[i]));
}

// The trailing block b <- b - v w^H - w v^H.
template <class S>
__global__ void rank_two_kernel(int n, int k, S* a, const S* __restrict__ w) {
    const long long m = n - k - 1;
    const long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (e >= m * m) return;
    const long long i = k + 1 + e % m, j = k + 1 + e / m;
    const S* v = a + static_cast<long long>(k) * n;
    S* at = a + i + j * n;
    *at = sub(*at, add(mul(v[i], cj(w[j])), mul(w[i], cj(v[j]))));
}

template <class S>
__global__ void last_diagonal_kernel(int n, const S* a, typename RealOf<S>::type* d) {
    if (threadIdx.x == 0 && blockIdx.x == 0) d[n - 1] = re(a[static_cast<long long>(n) * n - 1]);
}

// The reflectors first .. first + count as the columns of v (n, count): zeros
// down to the diagonal of each, one below it, then what a holds.
template <class S>
__global__ void reflectors_kernel(int n, int first, int count, const S* __restrict__ a, S* __restrict__ v) {
    using R = typename RealOf<S>::type;
    const long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (e >= static_cast<long long>(n) * count) return;
    const int i = static_cast<int>(e % n), c = static_cast<int>(e / n), k = first + c;
    v[e] = i <= k ? from_real(S{}, static_cast<R>(0))
                  : i == k + 1 ? from_real(S{}, static_cast<R>(1)) : a[i + static_cast<long long>(k) * n];
}

// z (n, n) of the type of the matrix from the real q.
template <class S>
__global__ void widen_kernel(long long size, const typename RealOf<S>::type* __restrict__ q, S* __restrict__ z) {
    const long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (e < size) z[e] = from_real(S{}, q[e]);
}

// Columns i and j of rows numbers: c q_i + s q_j and c q_j - s q_i.
template <class R>
__global__ void rotate_kernel(int rows, R* x, R* y, R c, R s) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const R a = x[r], b = y[r];
    x[r] = c * a + s * b;
    y[r] = c * b - s * a;
}

// out (rows, count): column c is column index[c] of the block of q.
template <class R>
__global__ void gather_columns_kernel(int rows, int count, const R* __restrict__ q, long long ld,
                                      const int* __restrict__ index, R* __restrict__ out) {
    const long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (e >= static_cast<long long>(rows) * count) return;
    out[e] = q[e % rows + index[e / rows] * ld];
}

// out (rows, rows): column j is column order[j] of [product (rows, kept),
// then the columns deflated[.] of the block of q].
template <class R>
__global__ void assemble_kernel(int rows, int kept, const R* __restrict__ product, const R* __restrict__ q, long long ld,
                                const int* __restrict__ deflated, const int* __restrict__ order, R* __restrict__ out) {
    const long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (e >= static_cast<long long>(rows) * rows) return;
    const long long r = e % rows;
    const int from = order[e / rows];
    out[e] = from < kept ? product[r + static_cast<long long>(from) * rows] : q[r + deflated[from - kept] * ld];
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

// The eigenvectors of the divide and conquer on the GPU, real numbers of the
// precision of the matrix.
template <class R>
class DeviceVectors : public host::Vectors {
public:
    DeviceVectors(int n, Blas& blas) : n_(n), blas_(blas) {
        q_.resize(static_cast<std::size_t>(n) * n);
        q_.zero();
    }
    R* data() noexcept { return q_.data(); }

    void set(int offset, int size, const double* z) override {
        const std::size_t count = static_cast<std::size_t>(size) * size;
        std::vector<R> host(z, z + count);
        check_cuda(cudaMemcpy2D(block(offset), pitch(), host.data(), static_cast<std::size_t>(size) * sizeof(R),
                                static_cast<std::size_t>(size) * sizeof(R), static_cast<std::size_t>(size),
                                cudaMemcpyHostToDevice),
                   "cudaMemcpy2D to the GPU");
    }
    void rows(int offset, int size, int middle, double* a, double* b) override {
        std::vector<R> host(static_cast<std::size_t>(size));
        for (int which = 0; which < 2; ++which) {
            // one number of each column: a row
            const R* from = q_.data() + static_cast<std::size_t>(offset) * n_ + (middle - 1 + which);
            check_cuda(cudaMemcpy2D(host.data(), sizeof(R), from, pitch(), sizeof(R), static_cast<std::size_t>(size),
                                    cudaMemcpyDeviceToHost),
                       "cudaMemcpy2D from the GPU");
            std::copy(host.begin(), host.end(), which == 0 ? a : b);
        }
    }
    void rotate(int offset, int size, int i, int j, double c, double s) override {
        rotate_kernel<R><<<blocks(size), kThreads>>>(size, block(offset) + static_cast<std::size_t>(i) * n_,
                                                    block(offset) + static_cast<std::size_t>(j) * n_,
                                                    static_cast<R>(c), static_cast<R>(s));
        launched("a rotation of two eigenvectors");
    }
    void merge(int offset, int size, const std::vector<int>& kept, const double* u, const std::vector<int>& deflated,
               const std::vector<int>& order) override {
        const int k = static_cast<int>(kept.size());
        const std::size_t rows = static_cast<std::size_t>(size);
        next_.reserve(rows * rows);
        order_.assign(order.data(), order.size());
        deflated_.reserve(1);
        if (!deflated.empty()) deflated_.assign(deflated.data(), deflated.size());
        if (k > 0) {
            kept_.assign(kept.data(), kept.size());
            gathered_.reserve(rows * k);
            product_.reserve(rows * k);
            const std::vector<R> host(u, u + static_cast<std::size_t>(k) * k);
            u_.assign(host.data(), host.size());
            gather_columns_kernel<R><<<blocks(static_cast<long long>(rows) * k), kThreads>>>(
                size, k, block(offset), n_, kept_.data(), gathered_.data());
            launched("the eigenvectors kept");
            blas_.gemm(Op::N, Op::N, size, k, k, static_cast<R>(1), gathered_.data(), size, u_.data(), k,
                       static_cast<R>(0), product_.data(), size);
        }
        assemble_kernel<R><<<blocks(static_cast<long long>(rows) * rows), kThreads>>>(
            size, k, product_.data(), block(offset), n_, deflated_.data(), order_.data(), next_.data());
        launched("the eigenvectors of a merge");
        check_cuda(cudaMemcpy2D(block(offset), pitch(), next_.data(), rows * sizeof(R), rows * sizeof(R), rows,
                                cudaMemcpyDeviceToDevice),
                   "cudaMemcpy2D");
    }

private:
    R* block(int offset) { return q_.data() + static_cast<std::size_t>(offset) * n_ + offset; }
    std::size_t pitch() const { return static_cast<std::size_t>(n_) * sizeof(R); }

    int n_;
    Blas& blas_;
    DeviceArray<R> q_, next_, gathered_, product_, u_;
    DeviceArray<int> kept_, deflated_, order_;
};

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

constexpr int kReflectors = 32;  // of a block of the back-transformation
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
// tridiagonal t by divide and conquer (scalars on the host, vectors on the
// GPU), and q applied to them in blocks of reflectors.
template <class T>
void Lapack::device_eigh(T* a, int n, std::vector<double>& values) {
    using S = std::remove_pointer_t<decltype(own(a))>;
    using R = typename RealOf<S>::type;
    const std::size_t size = static_cast<std::size_t>(n) * n;
    S* matrix = own(a);
    DeviceArray<T> tau(static_cast<std::size_t>(n)), vector(static_cast<std::size_t>(n));
    DeviceArray<R> diagonal(static_cast<std::size_t>(n)), subdiagonal(static_cast<std::size_t>(n));
    mirror_kernel<S><<<blocks(static_cast<long long>(size)), kThreads>>>(n, matrix);
    launched("the lower triangle of a matrix");
    {
        MDAPY_SPAN("tridiagonal");
        for (int k = 0; k + 1 < n; ++k) {
            const long long m = n - k - 1;
            reflector_kernel<S><<<1, kThreads>>>(n, k, matrix, own(tau.data()), diagonal.data(), subdiagonal.data());
            column_products_kernel<S><<<static_cast<unsigned>(m), kThreads / 2>>>(n, k, matrix, own(vector.data()));
            update_vector_kernel<S><<<1, kThreads>>>(n, k, matrix, own(tau.data()), own(vector.data()));
            rank_two_kernel<S><<<blocks(m * m), kThreads>>>(n, k, matrix, own(vector.data()));
        }
        last_diagonal_kernel<S><<<1, 64>>>(n, matrix, diagonal.data());
        launched("the tridiagonalization");
    }
    std::vector<R> host(static_cast<std::size_t>(n));
    diagonal.download(host.data(), host.size());
    std::vector<double> d(host.begin(), host.end());
    subdiagonal.download(host.data(), host.size());
    std::vector<double> e(host.begin(), host.end() - 1);
    std::vector<T> tau_host(static_cast<std::size_t>(n));
    tau.download(tau_host.data(), tau_host.size());
    // the eigenpairs of t
    DeviceVectors<R> q(n, blas_);
    {
        MDAPY_SPAN("divide");
        host::divide_and_conquer(d, e, q);
    }
    values = d;
    // z = h_0 h_1 ... q, the last block of reflectors first: a block is
    // 1 - v t v^H with t upper triangular, t_ii = tau_i and
    // t[:i, i] = -tau_i t[:i, :i] (v^H v)[:i, i]
    MDAPY_SPAN("back");
    DeviceArray<T> z(size), v(static_cast<std::size_t>(n) * kReflectors), w(static_cast<std::size_t>(n) * kReflectors),
        tw(static_cast<std::size_t>(n) * kReflectors), small(static_cast<std::size_t>(kReflectors) * kReflectors);
    widen_kernel<S><<<blocks(static_cast<long long>(size)), kThreads>>>(static_cast<long long>(size), q.data(), own(z.data()));
    launched("the eigenvectors of the tridiagonal matrix");
    const int reflectors = n - 1;
    const T one = unit(T{}, 1.0), zero = unit(T{}, 0.0), minus = unit(T{}, -1.0);
    for (int first = reflectors > 0 ? ((reflectors - 1) / kReflectors) * kReflectors : -1; first >= 0; first -= kReflectors) {
        const int count = std::min(kReflectors, reflectors - first);
        reflectors_kernel<S><<<blocks(static_cast<long long>(n) * count), kThreads>>>(n, first, count, matrix, own(v.data()));
        launched("a block of reflectors");
        blas_.gemm(Op::C, Op::N, count, count, n, one, v.data(), n, v.data(), n, zero, small.data(), count);
        std::vector<T> gram(static_cast<std::size_t>(count) * count);
        small.download(gram.data(), gram.size());
        std::vector<std::complex<double>> t(static_cast<std::size_t>(count) * count, 0.0);
        for (int i = 0; i < count; ++i) {
            const std::complex<double> ti = host_of(tau_host[static_cast<std::size_t>(first + i)]);
            for (int r = 0; r < i; ++r) {
                std::complex<double> sum = 0.0;
                for (int c = r; c < i; ++c) sum += t[static_cast<std::size_t>(c) * count + r] * host_of(gram[static_cast<std::size_t>(i) * count + c]);
                t[static_cast<std::size_t>(i) * count + r] = -ti * sum;
            }
            t[static_cast<std::size_t>(i) * count + i] = ti;
        }
        for (std::size_t i = 0; i < gram.size(); ++i) device_of(t[i], gram[i]);
        small.assign(gram.data(), gram.size());
        blas_.gemm(Op::C, Op::N, count, n, n, one, v.data(), n, z.data(), n, zero, w.data(), count);
        blas_.gemm(Op::N, Op::N, count, n, count, one, small.data(), count, w.data(), count, zero, tw.data(), count);
        blas_.gemm(Op::N, Op::N, n, n, count, minus, v.data(), n, tw.data(), count, one, z.data(), n);
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
