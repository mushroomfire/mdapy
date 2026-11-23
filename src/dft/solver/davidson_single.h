// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Davidson's block method of dft/solver/davidson.h in single precision, for
// the first steps of a self-consistent field on GPUs of little double
// precision: every product, transform, factorization and eigensolver of the
// subspace in single precision, the norms of the residuals summed in double.
// Complex bands only (not those of Gamma). The bands come and go in double
// precision; a residual norm below about 1e-7 of the largest |H_GG| is not
// reached, so that tolerances are bounded below by the caller.

#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <vector>

#include "dft/core/device.h"
#include "dft/hamiltonian/hamiltonian.h"
#include "dft/solver/davidson.h"

namespace mdapy::dft {

class DavidsonSingle {
public:
    DavidsonSingle();
    ~DavidsonSingle();
    DavidsonSingle(const DavidsonSingle&) = delete;
    DavidsonSingle& operator=(const DavidsonSingle&) = delete;

    // As Davidson::solve, psi (plane waves, count) of double precision on the
    // device, in and out.
    DavidsonResult solve(Hamiltonian& h, Complex* psi, int count, std::vector<double>& eigenvalues,
                         const std::vector<double>& tolerances, int max_iterations, int largest_basis);

private:
    void eigh(cuComplex* a, int n);  // eigenpairs into values_, vectors into a
    // out (ka, kb) = a^H b + pa^H q pb (the second without h)
    void gram(Hamiltonian* h, const cuComplex* a, const cuComplex* pa, int ka, const cuComplex* b,
              const cuComplex* pb, int kb, int rows, cuComplex* out);
    // w (rows, m) orthonormal to v (rows, n) and in itself in the metric of S,
    // projections along; the columns kept
    int orthonormalize(Hamiltonian* h, const cuComplex* v, const cuComplex* pv, int n, cuComplex* w, cuComplex* pw,
                       int m, int rows);

    cublasHandle_t blas_ = nullptr;
    cusolverDnHandle_t solver_ = nullptr;
    int projectors_ = 0;
    DeviceArray<cuComplex> basis_, hbasis_, pbasis_, pvectors_, vectors_, residual_, subspace_, hsub_, overlap_,
        chol_, work_, qp_, picked_, lapack_, diagonal_r_;
    DeviceArray<float> values_, diagonal_, overlap_diagonal_, scale_, energies_;
    DeviceArray<double> sums_;
    DeviceArray<int> index_, info_;
};

}  // namespace mdapy::dft
