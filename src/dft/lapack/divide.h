// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The eigenpairs of a real symmetric tridiagonal matrix by divide and conquer
// (Cuppen; Gu and Eisenstat): the matrix is cut in two, the halves solved, and
// their eigenpairs merged, which is the eigenproblem of a diagonal matrix plus
// a matrix of rank one. Here is what the host does of it, in double precision:
// the tree of the cuts, the leaves (by the QL iteration), and the deflation of
// each merge (small components, and equal poles rotated together, so that
// equal eigenvalues cost nothing). The rest of a merge: the secular equation
// of each root from its nearer pole, the vector of the rank-one term
// recomputed from the roots (Löwner's formula), which makes the vectors
// orthogonal to rounding, and the eigenvectors, a product of matrices. On the
// GPUs of full double precision all of it is theirs (dft/lapack/lapack.cu),
// every merge of a height of the tree at a time; for the others the roots and
// the small matrix of the product are found here, and the GPU has the product.

#pragma once

#include <vector>

namespace mdapy::dft::host {

// A merge of the tree: the block of size rows and columns from offset, of
// which the first half was solved apart from the rest.
struct Merge {
    int offset, size, half;
    int height;    // one more than the higher of its halves, leaves being at 0
    double rho;    // the size of the element cut
    double sign;   // and its sign
};

struct Tree {
    std::vector<Merge> merges;  // in the order of their heights
    std::vector<int> leaves;    // offset and size of each
    int height = 0;
};

// The tree of a matrix of diagonal d (n) and subdiagonal e, cut down to blocks
// of leaf rows or fewer; d is left as the diagonals of the leaves: at a cut
// t = (t_1, t_2) + |rho| u u^T, u the last unit vector of the first half and,
// with the sign of rho, the first of the second.
Tree cut(std::vector<double>& d, const std::vector<double>& e, int leaf);

// What a merge keeps for its secular equation and what is final already.
struct Deflation {
    struct Rotation {
        int i, j;     // columns i and j replaced by c q_i + s q_j and c q_j - s q_i
        double c, s;
    };
    std::vector<Rotation> rotations;  // in their order
    std::vector<int> kept, deflated;  // the columns, the kept by ascending pole
    std::vector<double> pole, w;      // of the kept: the pole, and the component of the unit vector
    std::vector<double> values;       // (size) the eigenvalues of the deflated columns
    double rho = 0.0;                 // of the unit vector
};

// The deflation of the merge of d (size, the eigenvalues of the two halves,
// those of rotated columns replaced) with the rank-one term rho0 z z^T; z
// (size) is scaled to unit length.
Deflation deflate(double* d, int size, std::vector<double>& z, double rho0);

// The roots (the kept eigenvalues, ascending) of the secular equation of a
// merge, and u (kept, kept; by columns) the eigenvectors of its kept columns.
void secular(const Deflation& f, double* roots, double* u);

// The eigenpairs of the tridiagonal matrix d, e by the QL iteration: d
// ascending and z (n, n) its eigenvectors, by columns.
void tridiagonal_ql(std::vector<double>& d, std::vector<double>& e, std::vector<double>& z);

}  // namespace mdapy::dft::host
