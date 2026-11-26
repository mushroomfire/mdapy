// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The eigenpairs of a Hermitian matrix and the Cholesky factor of a positive
// definite one on the host, in double precision, of matrices stored by
// columns: for the small matrices of the DFT, whose transfer costs less than
// the many small kernels of the GPU would, and as the leaves and the reference
// of what the GPU does for the large ones. Plain C++, no LAPACK.

#pragma once

#include <complex>

namespace mdapy::dft::host {

// The eigenvalues w (n, ascending) and the eigenvectors (the columns of a, on
// return) of a (n, n), of which the upper triangle is read: Householder
// tridiagonalization, the implicit QL iteration on the tridiagonal matrix with
// the rotations applied to the vectors. Throws if an eigenvalue takes more
// than 60 iterations.
void eigh(std::complex<double>* a, int n, double* w);
void eigh(double* a, int n, double* w);

// The upper triangle of a (n, n) replaced by r, a = r^H r, its diagonal
// positive; the lower triangle is not touched. Returns 0, or j when the
// leading minor of order j is not positive definite (as LAPACK's potrf).
int potrf(std::complex<double>* a, int n);
int potrf(double* a, int n);

}  // namespace mdapy::dft::host
