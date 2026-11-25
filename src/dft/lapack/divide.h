// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The eigenpairs of a real symmetric tridiagonal matrix by divide and conquer
// (Cuppen; Gu and Eisenstat): the matrix is cut in two, the halves solved, and
// their eigenpairs merged, which is the eigenproblem of a diagonal matrix plus
// a matrix of rank one. The scalar work is here, on the host, in double
// precision: the tree, the deflation (small components, and equal poles
// rotated together, so that equal eigenvalues cost nothing), the secular
// equation of each root from its nearer pole, and the vector of the rank-one
// term recomputed from the roots (Löwner's formula), which makes the vectors
// orthogonal to rounding. The eigenvectors themselves, an (n, n) matrix, belong
// to whoever implements Vectors: the host for the tests and small matrices, the
// GPU for large ones, where each merge is one product of matrices.

#pragma once

#include <vector>

namespace mdapy::dft::host {

// The matrix q (n, n) of the eigenvectors, block diagonal while the tree is
// climbed: a block is the size rows and columns from offset.
class Vectors {
public:
    virtual ~Vectors() = default;
    // The block set to z (size, size), stored by columns.
    virtual void set(int offset, int size, const double* z) = 0;
    // Row first and row last of the two blocks that meet at middle: the last
    // row of the block [offset, middle) into a, the first of [middle, offset +
    // size) into b, size numbers each (zeros outside their block).
    virtual void rows(int offset, int size, int middle, double* a, double* b) = 0;
    // Columns i and j of the block replaced by c q_i + s q_j and c q_j - s q_i.
    virtual void rotate(int offset, int size, int i, int j, double c, double s) = 0;
    // The block replaced by the columns of [q[:, kept] u, q[:, deflated]] in
    // the order given: u is (k, k) with k kept columns, stored by columns.
    virtual void merge(int offset, int size, const std::vector<int>& kept, const double* u,
                       const std::vector<int>& deflated, const std::vector<int>& order) = 0;
};

// The eigenvalues (d on return, ascending) of the tridiagonal matrix of
// diagonal d (n) and subdiagonal e (n - 1), and its eigenvectors as the columns
// of q. Blocks of leaf rows or fewer are solved by the QL iteration.
void divide_and_conquer(std::vector<double>& d, std::vector<double>& e, Vectors& q, int leaf = 32);

// q on the host: an (n, n) array stored by columns.
class HostVectors : public Vectors {
public:
    explicit HostVectors(int n) : n_(n), q_(static_cast<std::size_t>(n) * n, 0.0) {}
    void set(int offset, int size, const double* z) override;
    void rows(int offset, int size, int middle, double* a, double* b) override;
    void rotate(int offset, int size, int i, int j, double c, double s) override;
    void merge(int offset, int size, const std::vector<int>& kept, const double* u, const std::vector<int>& deflated,
               const std::vector<int>& order) override;
    const std::vector<double>& data() const noexcept { return q_; }

private:
    int n_;
    std::vector<double> q_;
};

// The eigenpairs of the tridiagonal matrix d, e by the QL iteration: d
// ascending and z (n, n) its eigenvectors, by columns.
void tridiagonal_ql(std::vector<double>& d, std::vector<double>& e, std::vector<double>& z);

}  // namespace mdapy::dft::host
