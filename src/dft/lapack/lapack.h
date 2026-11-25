// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The eigenpairs of the Hermitian matrices and the Cholesky factors of the
// positive definite ones of the DFT, matrices of the GPU stored by columns, in
// double and single precision, complex and real. With the kernels of the
// vendor they are cuSOLVER's (hipSOLVER's on AMD GPUs). With those of MDAPY
// they are its own:
// - matrices up to host_limit() on the host (dft/lapack/host.h; they are
//   kilobytes, and one transfer costs less than the many small kernels of the
//   GPU would), in double precision whatever the precision of the matrix;
// - larger ones on the GPU. Eigenpairs: Householder tridiagonalization, a
//   reflector after the other, one launch each for matrices up to 512 to
//   1024 by their type (two beyond, and in double precision on GPUs that have
//   little of it), and in panels whose products are Blas's from 1024 to 1536;
//   divide
//   and conquer on the tridiagonal matrix (dft/lapack/divide.h), the merges of
//   a height of its tree together, their scalar work in double precision on
//   the GPU when it has full double precision and on the host otherwise, and
//   their eigenvectors products of Blas; then the reflectors applied to them
//   in blocks, three products each. Cholesky:
//   blocks of columns, the diagonal block on the host, the panel by the
//   triangular solve of Blas and the trailing matrix by its product.
//
// A Lapack belongs to one host thread at a time and works on its default stream.

#pragma once

#include "dft/core/portable.h"

#include <vector>

#include "dft/blas/blas.h"
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

    // The upper triangle of a (n, n) replaced by r, a = r^H r; the lower one is
    // not read, and holds nothing of use on return. Returns 0, or j when the
    // leading minor of order j is not positive definite; a is then of no use.
    int potrf(Complex* a, int n);
    int potrf(cuComplex* a, int n);
    int potrf(double* a, int n);
    int potrf(float* a, int n);

    // The largest matrix MDAPY's kernels take on the host. For measurements
    // and tests.
    int host_limit() const noexcept { return host_limit_; }
    void set_host_limit(int n) noexcept { host_limit_ = n; }
    // The largest matrix of which a step of the tridiagonalization is one
    // launch (-1: by the GPU and the precision), and whether the GPU is taken
    // as one of full double precision (-1: asked of it). For measurements and
    // tests.
    void set_fused_limit(int n) noexcept { fused_limit_ = n; }
    // The smallest matrix tridiagonalized in panels (-1: MDAPY's choice).
    void set_panels_from(int n) noexcept { panels_from_ = n; }
    // The reflectors of a block of the back-transformation and the rows of a
    // leaf of the divide and conquer (0: MDAPY's choice).
    void set_reflectors(int count) noexcept { reflectors_ = count; }
    void set_leaf(int rows) noexcept { leaf_ = rows; }
    void set_full_double(int full) noexcept { full_double_ = full; }

private:
    cusolverDnHandle_t vendor();
    template <class T>
    void own_eigh(T* a, int n, std::vector<double>& values);
    template <class T>
    int own_potrf(T* a, int n);
    template <class T>
    void device_eigh(T* a, int n, std::vector<double>& values);
    template <class T>
    int device_potrf(T* a, int n);
    bool full_double();

    Kernels kernels_;
    Blas blas_;  // the products and the triangular solves of the large matrices
    cusolverDnHandle_t vendor_ = nullptr;  // of the library of the vendor, made at its first use
    // measured on a V100 (host: one core of its node): the host is the faster
    // up to about 64, the GPU beyond
    int host_limit_ = 64;
    int fused_limit_ = -1;
    int panels_from_ = -1;
    int reflectors_ = 0;
    int leaf_ = 0;
    int full_double_ = -1;  // whether the GPU runs double precision at half the rate of single; -1 before it is asked
    DeviceArray<Complex> work_;
    DeviceArray<double> values_;
    DeviceArray<float> values_single_;
    DeviceArray<int> info_;
};

}  // namespace mdapy::dft
