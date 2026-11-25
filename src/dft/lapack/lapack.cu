// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/lapack/lapack.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <string>

#include "dft/core/check.h"
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

}  // namespace

Lapack::Lapack(Kernels kernels) : kernels_(kernels) { info_.resize(1); }

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

// On the host, in double precision.
template <class T>
void Lapack::own_eigh(T* a, int n, std::vector<double>& values) {
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

template <class T>
int Lapack::own_potrf(T* a, int n) {
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
        if (kernels_ == Kernels::Mdapy && n <= host_limit_) return own_eigh(a, n, values);                   \
        VALUES.reserve(static_cast<std::size_t>(n));                                                         \
        vendor_eigh<V>(vendor(), a, n, VALUES.data(), work_, info_, values);                                 \
    }                                                                                                        \
    int Lapack::potrf(T* a, int n) {                                                                         \
        if (n < 1) return 0;                                                                                 \
        if (kernels_ == Kernels::Mdapy && n <= host_limit_) return own_potrf(a, n);                          \
        return vendor_potrf<V>(vendor(), a, n, work_, info_);                                                \
    }
MDAPY_LAPACK(Complex, Z, values_)
MDAPY_LAPACK(cuComplex, C, values_single_)
MDAPY_LAPACK(double, D, values_)
MDAPY_LAPACK(float, S, values_single_)
#undef MDAPY_LAPACK

}  // namespace mdapy::dft
