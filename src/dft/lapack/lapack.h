// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The eigenpairs of the Hermitian matrices and the Cholesky factors of the
// positive definite ones of the DFT, matrices of the GPU stored by columns, in
// double and single precision, complex and real. With the kernels of MDAPY
// they are its own: matrices up to host_limit() on the host (dft/lapack/host.h;
// they are kilobytes, and one transfer costs less than the many small kernels
// of the GPU would), in double precision whatever the precision of the matrix.
// With the kernels of the vendor, and for larger matrices, cuSOLVER's
// (hipSOLVER's on AMD GPUs).
//
// A Lapack belongs to one host thread at a time and works on its default stream.

#pragma once

#include "dft/core/portable.h"

#include <vector>

#include "dft/core/device.h"
#include "dft/core/kernels.h"

namespace mdapy::dft {

class Lapack {
public:
    explicit Lapack(Kernels kernels = Kernels::Mdapy);
    ~Lapack();
    Lapack(const Lapack&) = delete;
    Lapack& operator=(const Lapack&) = delete;

    Kernels kernels() const noexcept { return kernels_; }

    // The eigenvalues (n, ascending, on the host) and the eigenvectors (the
    // columns of a on return) of a (n, n), of which the upper triangle is
    // read. Throws if they do not converge.
    void eigh(Complex* a, int n, std::vector<double>& values);
    void eigh(cuComplex* a, int n, std::vector<double>& values);
    void eigh(double* a, int n, std::vector<double>& values);
    void eigh(float* a, int n, std::vector<double>& values);

    // The upper triangle of a (n, n) replaced by r, a = r^H r. Returns 0, or j
    // when the leading minor of order j is not positive definite; a is then of
    // no use.
    int potrf(Complex* a, int n);
    int potrf(cuComplex* a, int n);
    int potrf(double* a, int n);
    int potrf(float* a, int n);

    // The largest matrix MDAPY's kernels take on the host. For measurements
    // and tests.
    int host_limit() const noexcept { return host_limit_; }
    void set_host_limit(int n) noexcept { host_limit_ = n; }

private:
    cusolverDnHandle_t vendor();
    template <class T>
    void own_eigh(T* a, int n, std::vector<double>& values);
    template <class T>
    int own_potrf(T* a, int n);

    Kernels kernels_;
    cusolverDnHandle_t vendor_ = nullptr;  // of the library of the vendor, made at its first use
    int host_limit_ = 256;
    DeviceArray<Complex> work_;
    DeviceArray<double> values_;
    DeviceArray<float> values_single_;
    DeviceArray<int> info_;
};

}  // namespace mdapy::dft
