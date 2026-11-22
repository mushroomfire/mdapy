// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The lowest eigenpairs of H psi = e S psi of a k-point by block Davidson, on
// the GPU: an orthonormal basis grown by the preconditioned residuals of the
// bands not yet converged (as many as fit), Rayleigh-Ritz in it with cuSOLVER,
// and a restart from the Ritz vectors once the basis is full. The residual of
// a band of eigenvalue e is preconditioned by 1 / (H_GG - e), the difference
// bounded below by a smooth maximum with one rydberg (Davidson's diagonal
// preconditioner). New vectors are scaled to unit norm, made orthogonal to the
// basis and among themselves by Cholesky QR, twice; when the factor is ill
// conditioned, by the eigenvectors of their overlap, dropping the directions of
// eigenvalues below 1e-10. The matrix of H in the basis grows by the columns of
// the new vectors. With an overlap the projections <beta|basis> follow the
// basis through every linear step, so that S never has to be applied: the
// metric is a^H b + (beta^H a)^H q (beta^H b), and H w adds beta D (beta^H w).
// A guess that is orthonormal already, as the bands of the step before, is
// taken as it is; at a restart H in the basis of the Ritz vectors is
// diag(e). Only the bands still open have their residuals computed; a
// band that converges is checked again with all the others before a restart
// and at the end, and opened again if it no longer is.

#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <vector>

#include "dft/core/device.h"
#include "dft/hamiltonian/hamiltonian.h"

namespace mdapy::dft {

struct DavidsonResult {
    int iterations = 0;    // of the subspace
    int converged = 0;     // bands of residual norm below the tolerance
    double residual = 0.0;  // the largest residual norm of the bands
};

class Davidson {
public:
    Davidson();
    ~Davidson();
    Davidson(const Davidson&) = delete;
    Davidson& operator=(const Davidson&) = delete;

    // The count lowest eigenpairs of h: psi (plane waves, count) on the
    // device, the guess in, the eigenvectors out; eigenvalues on the host.
    // A band is converged when the norm of its residual H psi - e psi is
    // below tolerance; the first tight bands take tolerance, the others
    // loose_tolerance (empty bands of metals).
    DavidsonResult solve(Hamiltonian& h, Complex* psi, int count, std::vector<double>& eigenvalues,
                         double tolerance, int tight, double loose_tolerance, int max_iterations,
                         int largest_basis);

private:
    // makes the m columns of w orthonormal to the n columns of v and among
    // themselves; returns how many independent columns are left, moved first.
    // With h, in the metric of S, a^H b + (pa)^H q pb, pw and pv the
    // projections <beta|w> and <beta|v>, which follow w; else the plain one
    int orthonormalize(Hamiltonian* h, const Complex* v, const Complex* pv, int n, Complex* w, Complex* pw, int m,
                       int rows);
    // out (ka, kb) = a^H b, plus pa^H q pb with h
    void gram(Hamiltonian* h, const Complex* a, const Complex* pa, int ka, const Complex* b, const Complex* pb,
              int kb, int rows, Complex* out);
    // the eigenvalues (ascending, into values_) and eigenvectors (into a) of
    // the Hermitian n x n matrix a, of which the upper triangle is read
    void eigh(Complex* a, int n);

    cublasHandle_t blas_ = nullptr;
    cusolverDnHandle_t solver_ = nullptr;
    DeviceArray<Complex> basis_, hbasis_, subspace_, vectors_, work_, residual_, overlap_, lapack_;
    DeviceArray<Complex> hsub_, chol_, diagonal_r_;  // basis^H H basis; Cholesky factors
    DeviceArray<Complex> pbasis_, pvectors_, qp_;     // <beta|basis>, <beta|X>, q P, with an overlap
    DeviceArray<Complex> picked_;                     // the eigenvectors of the open bands
    DeviceArray<double> overlap_diagonal_;
    DeviceArray<double> values_, sums_, energies_, scale_, diagonal_;
    DeviceArray<int> info_, index_;
};

}  // namespace mdapy::dft
