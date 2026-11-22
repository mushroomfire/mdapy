// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/solver/davidson.h"

#include <algorithm>
#include <cmath>
#include <new>
#include <stdexcept>
#include <string>

namespace mdapy::dft {

namespace {

constexpr int kThreads = 256;
constexpr double kDependent = 1e-10;  // overlap eigenvalue of a dropped direction

unsigned blocks(std::size_t work) { return static_cast<unsigned>((work + kThreads - 1) / kThreads); }

void check_blas(cublasStatus_t status, const char* call) {
    if (status == CUBLAS_STATUS_SUCCESS) return;
    if (status == CUBLAS_STATUS_ALLOC_FAILED) throw std::bad_alloc();
    throw std::runtime_error(std::string(call) + " failed with cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
}

void check_solver(cusolverStatus_t status, const char* call) {
    if (status == CUSOLVER_STATUS_SUCCESS) return;
    if (status == CUSOLVER_STATUS_ALLOC_FAILED) throw std::bad_alloc();
    throw std::runtime_error(std::string(call) + " failed with cuSOLVER status " +
                             std::to_string(static_cast<int>(status)));
}

void copy(Complex* to, const Complex* from, std::size_t count) {
    check_cuda(cudaMemcpy(to, from, count * sizeof(Complex), cudaMemcpyDeviceToDevice), "cudaMemcpy");
}

const Complex kOne = make_cuDoubleComplex(1.0, 0.0);
const Complex kZero = make_cuDoubleComplex(0.0, 0.0);
const Complex kMinus = make_cuDoubleComplex(-1.0, 0.0);

// R = HX - X diag(e)
__global__ void residual_kernel(std::size_t n, int count, const Complex* x, const Complex* hx,
                                const double* e, Complex* r) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    const double ev = e[i / n];
    r[i] = make_cuDoubleComplex(hx[i].x - ev * x[i].x, hx[i].y - ev * x[i].y);
}

// one block per column c: dots[c] = Re sum conj(a) b
__global__ void column_dots(std::size_t n, const Complex* a, const Complex* b, double* dots) {
    const Complex* x = a + static_cast<std::size_t>(blockIdx.x) * n;
    const Complex* y = b + static_cast<std::size_t>(blockIdx.x) * n;
    double s = 0.0;
    for (std::size_t g = threadIdx.x; g < n; g += blockDim.x) s += x[g].x * y[g].x + x[g].y * y[g].y;
    __shared__ double partial[kThreads];
    partial[threadIdx.x] = s;
    __syncthreads();
    for (int k = kThreads / 2; k > 0; k >>= 1) {
        if (static_cast<int>(threadIdx.x) < k) partial[threadIdx.x] += partial[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) dots[blockIdx.x] = partial[0];
}

// one block per column c of a (n, columns): norms[c] = sum |a|^2
__global__ void column_norms(std::size_t n, const Complex* a, double* norms) {
    const Complex* column = a + static_cast<std::size_t>(blockIdx.x) * n;
    double s = 0.0;
    for (std::size_t g = threadIdx.x; g < n; g += blockDim.x) s += column[g].x * column[g].x + column[g].y * column[g].y;
    __shared__ double partial[kThreads];
    partial[threadIdx.x] = s;
    __syncthreads();
    for (int k = kThreads / 2; k > 0; k >>= 1) {
        if (static_cast<int>(threadIdx.x) < k) partial[threadIdx.x] += partial[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) norms[blockIdx.x] = partial[0];
}

// column c of t is column index[c] of r over (H_GG - e[c] S_GG), the difference
// in rydberg x bounded below by the smooth maximum (1 + x + sqrt(1 + (x - 1)^2)) / 2
// of x and 1 (Davidson's preconditioner)
__global__ void precondition_kernel(std::size_t n, int count, const int* index, const double* diagonal,
                                    const double* overlap, const double* e, const Complex* r, Complex* t) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    const std::size_t c = i / n, g = i % n;
    const double x = 2.0 * (diagonal[g] - e[c] * overlap[g]);
    const double k = 2.0 / (1.0 + x + sqrt(1.0 + (x - 1.0) * (x - 1.0)));
    const Complex v = r[static_cast<std::size_t>(index[c]) * n + g];
    t[i] = make_cuDoubleComplex(k * v.x, k * v.y);
}

// column c of x times s[c]
__global__ void scale_columns(std::size_t n, int count, const double* s, Complex* x) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    const double f = s[i / n];
    x[i] = make_cuDoubleComplex(f * x[i].x, f * x[i].y);
}

}  // namespace

Davidson::Davidson() {
    check_blas(cublasCreate(&blas_), "cublasCreate");
    check_solver(cusolverDnCreate(&solver_), "cusolverDnCreate");
    info_.resize(1);
}

Davidson::~Davidson() {
    if (solver_ != nullptr) cusolverDnDestroy(solver_);
    if (blas_ != nullptr) cublasDestroy(blas_);
}

void Davidson::eigh(Complex* a, int n) {
    values_.reserve(static_cast<std::size_t>(n));
    int lwork = 0;
    check_solver(cusolverDnZheevd_bufferSize(solver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, a, n,
                                             values_.data(), &lwork),
                 "cusolverDnZheevd_bufferSize");
    lapack_.reserve(static_cast<std::size_t>(std::max(lwork, 1)));
    check_solver(cusolverDnZheevd(solver_, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, a, n,
                                  values_.data(), lapack_.data(), lwork, info_.data()),
                 "cusolverDnZheevd");
    int info = 0;
    info_.download(&info, 1);
    if (info != 0)
        throw std::runtime_error("the eigenvalues of a subspace did not converge (cusolverDnZheevd info " +
                                 std::to_string(info) + ")");
}

int Davidson::orthonormalize(const Complex* v, const Complex* sv, int n, Complex* w, Complex* sw, int m, int rows) {
    if (m == 0) return 0;
    const bool general = sw != nullptr;
    const Complex* svv = general ? sv : v;  // S v
    Complex* sww = general ? sw : w;        // S w
    const std::size_t r = static_cast<std::size_t>(rows);
    overlap_.reserve(static_cast<std::size_t>(std::max(n, m)) * m);
    work_.reserve(r * m);
    std::vector<double> host(static_cast<std::size_t>(m));
    // unit columns in the metric of S, so that the overlap eigenvalues measure
    // independence
    sums_.reserve(static_cast<std::size_t>(m));
    column_dots<<<m, kThreads>>>(r, w, sww, sums_.data());
    check_cuda(cudaGetLastError(), "the column norms");
    sums_.download(host.data(), host.size());
    for (double& x : host) x = x > 0.0 ? 1.0 / std::sqrt(x) : 0.0;
    scale_.assign(host.data(), host.size());
    scale_columns<<<blocks(r * m), kThreads>>>(r, m, scale_.data(), w);
    if (general) scale_columns<<<blocks(r * m), kThreads>>>(r, m, scale_.data(), sw);
    check_cuda(cudaGetLastError(), "the unit columns");
    for (int pass = 0; pass < 2; ++pass) {
        if (n > 0) {  // w -= v (v^H S w), S w likewise
            check_blas(cublasZgemm(blas_, CUBLAS_OP_C, CUBLAS_OP_N, n, m, rows, &kOne, svv, rows, w, rows, &kZero,
                                   overlap_.data(), n),
                       "cublasZgemm");
            check_blas(cublasZgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, rows, m, n, &kMinus, v, rows, overlap_.data(),
                                   n, &kOne, w, rows),
                       "cublasZgemm");
            if (general)
                check_blas(cublasZgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, rows, m, n, &kMinus, sv, rows,
                                       overlap_.data(), n, &kOne, sw, rows),
                           "cublasZgemm");
        }
        check_blas(cublasZgemm(blas_, CUBLAS_OP_C, CUBLAS_OP_N, m, m, rows, &kOne, w, rows, sww, rows, &kZero,
                               overlap_.data(), m),
                   "cublasZgemm");
        // Cholesky QR, w <- w R^-1 with w^H S w = R^H R, while R is well
        // conditioned (its diagonal within 1e-4 of its largest)
        const std::size_t mm = static_cast<std::size_t>(m) * m;
        chol_.reserve(mm);
        copy(chol_.data(), overlap_.data(), mm);
        int lwork = 0;
        check_solver(cusolverDnZpotrf_bufferSize(solver_, CUBLAS_FILL_MODE_UPPER, m, chol_.data(), m, &lwork),
                     "cusolverDnZpotrf_bufferSize");
        lapack_.reserve(static_cast<std::size_t>(std::max(lwork, 1)));
        check_solver(cusolverDnZpotrf(solver_, CUBLAS_FILL_MODE_UPPER, m, chol_.data(), m, lapack_.data(), lwork,
                                      info_.data()),
                     "cusolverDnZpotrf");
        int info = 0;
        info_.download(&info, 1);
        if (info == 0) {
            diagonal_r_.reserve(static_cast<std::size_t>(m));
            check_blas(cublasZcopy(blas_, m, chol_.data(), m + 1, diagonal_r_.data(), 1), "cublasZcopy");
            std::vector<Complex> d(static_cast<std::size_t>(m));
            diagonal_r_.download(d.data(), d.size());
            double low = std::abs(d[0].x), high = low;
            for (const Complex& x : d) {
                low = std::min(low, std::abs(x.x));
                high = std::max(high, std::abs(x.x));
            }
            if (low > 1e-4 * high) {
                check_blas(cublasZtrsm(blas_, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                       CUBLAS_DIAG_NON_UNIT, rows, m, &kOne, chol_.data(), m, w, rows),
                           "cublasZtrsm");
                if (general)
                    check_blas(cublasZtrsm(blas_, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                           CUBLAS_DIAG_NON_UNIT, rows, m, &kOne, chol_.data(), m, sw, rows),
                               "cublasZtrsm");
                continue;
            }
        }
        // else w <- w Q Lambda^-1/2 over the eigenpairs of w^H S w above
        // kDependent of the largest
        eigh(overlap_.data(), m);
        host.resize(static_cast<std::size_t>(m));
        values_.download(host.data(), host.size());
        const double largest = std::max(host.back(), 1e-300);
        int first = 0;
        while (first < m && !(host[static_cast<std::size_t>(first)] > kDependent * largest)) ++first;
        const int kept = m - first;
        if (kept == 0) return 0;
        std::vector<double> inverse(static_cast<std::size_t>(kept));
        for (int i = 0; i < kept; ++i)
            inverse[static_cast<std::size_t>(i)] = 1.0 / std::sqrt(host[static_cast<std::size_t>(first + i)]);
        scale_.assign(inverse.data(), inverse.size());
        for (Complex* target : {w, general ? sw : nullptr}) {
            if (target == nullptr) continue;
            check_blas(cublasZgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, rows, kept, m, &kOne, target, rows,
                                   overlap_.data() + static_cast<std::size_t>(first) * m, m, &kZero, work_.data(),
                                   rows),
                       "cublasZgemm");
            scale_columns<<<blocks(r * kept), kThreads>>>(r, kept, scale_.data(), work_.data());
            check_cuda(cudaGetLastError(), "the orthonormal columns");
            copy(target, work_.data(), r * kept);
        }
        m = kept;
    }
    return m;
}

DavidsonResult Davidson::solve(Hamiltonian& h, Complex* psi, int count, std::vector<double>& eigenvalues,
                               double tolerance, int tight, double loose_tolerance, int max_iterations,
                               int largest_basis) {
    const int rows = static_cast<int>(h.plane_waves());
    if (count < 1 || count > rows) throw std::invalid_argument("from one band to as many as the plane waves");
    if (max_iterations < 1) throw std::invalid_argument("max_iterations must be positive");
    const int most = std::min(std::max(largest_basis, 2 * count), rows);
    const std::size_t n = static_cast<std::size_t>(rows), bands = static_cast<std::size_t>(count);
    const bool general = h.has_overlap();
    basis_.reserve(n * most);
    hbasis_.reserve(n * most);
    if (general) sbasis_.reserve(n * most);
    vectors_.reserve(3 * n * bands);  // X, HX and SX
    residual_.reserve(n * bands);
    subspace_.reserve(static_cast<std::size_t>(most) * most);
    diagonal_.reserve(n);
    h.diagonal(diagonal_.data());
    overlap_diagonal_.reserve(n);
    h.overlap_diagonal(overlap_diagonal_.data());
    copy(basis_.data(), psi, n * bands);
    Complex* sb = general ? sbasis_.data() : nullptr;
    if (general) h.apply_overlap(basis_.data(), sb, count);
    int nbase = orthonormalize(nullptr, nullptr, 0, basis_.data(), sb, count, rows);
    if (nbase < count) throw std::invalid_argument("the guess of the bands is linearly dependent");
    h.apply(basis_.data(), hbasis_.data(), nbase);
    // basis^H H basis, kept between iterations and grown by the columns of
    // the new vectors (the upper triangle, which the eigensolver reads)
    const std::size_t ld = static_cast<std::size_t>(most);
    hsub_.reserve(ld * ld);
    auto grow = [&](int first, int columns) {
        check_blas(cublasZgemm(blas_, CUBLAS_OP_C, CUBLAS_OP_N, first + columns, columns, rows, &kOne,
                               basis_.data(), rows, hbasis_.data() + n * static_cast<std::size_t>(first), rows,
                               &kZero, hsub_.data() + ld * static_cast<std::size_t>(first), most),
                   "cublasZgemm");
    };
    grow(0, nbase);
    Complex* x = vectors_.data();
    Complex* hx = vectors_.data() + n * bands;
    Complex* sx = general ? vectors_.data() + 2 * n * bands : x;
    DavidsonResult result;
    std::vector<double> lambda, residuals(bands);
    std::vector<int> open;
    for (int iteration = 1;; ++iteration) {
        result.iterations = iteration;
        // Rayleigh-Ritz: the count lowest eigenpairs of basis^H H basis
        check_cuda(cudaMemcpy2D(subspace_.data(), static_cast<std::size_t>(nbase) * sizeof(Complex), hsub_.data(),
                                ld * sizeof(Complex), static_cast<std::size_t>(nbase) * sizeof(Complex),
                                static_cast<std::size_t>(nbase), cudaMemcpyDeviceToDevice),
                   "cudaMemcpy2D");
        eigh(subspace_.data(), nbase);
        lambda.resize(static_cast<std::size_t>(nbase));
        values_.download(lambda.data(), lambda.size());
        check_blas(cublasZgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, rows, count, nbase, &kOne, basis_.data(), rows,
                               subspace_.data(), nbase, &kZero, x, rows),
                   "cublasZgemm");
        check_blas(cublasZgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, rows, count, nbase, &kOne, hbasis_.data(), rows,
                               subspace_.data(), nbase, &kZero, hx, rows),
                   "cublasZgemm");
        if (general)
            check_blas(cublasZgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, rows, count, nbase, &kOne, sbasis_.data(),
                                   rows, subspace_.data(), nbase, &kZero, sx, rows),
                       "cublasZgemm");
        energies_.assign(lambda.data(), bands);
        residual_kernel<<<blocks(n * bands), kThreads>>>(n, count, sx, hx, energies_.data(), residual_.data());
        check_cuda(cudaGetLastError(), "the residuals");
        // |R|^2 of each band
        sums_.reserve(bands);
        column_norms<<<count, kThreads>>>(n, residual_.data(), sums_.data());
        check_cuda(cudaGetLastError(), "the residual norms");
        sums_.download(residuals.data(), bands);
        open.clear();
        result.residual = 0.0;
        result.converged = 0;
        for (int b = 0; b < count; ++b) {
            const double norm = std::sqrt(residuals[static_cast<std::size_t>(b)]);
            result.residual = std::max(result.residual, norm);
            if (norm < (b < tight ? tolerance : loose_tolerance))
                ++result.converged;
            else
                open.push_back(b);
        }
        int add = static_cast<int>(open.size());
        if (add > 0 && iteration < max_iterations && nbase == most) {
            // restart from the Ritz vectors
            copy(basis_.data(), x, n * bands);
            copy(hbasis_.data(), hx, n * bands);
            if (general) copy(sbasis_.data(), sx, n * bands);
            nbase = count;
            grow(0, nbase);
        }
        add = std::min(add, most - nbase);
        if (add == 0 || iteration >= max_iterations) {
            copy(psi, x, n * bands);
            eigenvalues.assign(lambda.begin(), lambda.begin() + count);
            return result;
        }
        // the preconditioned residuals of the open bands
        open.resize(static_cast<std::size_t>(add));
        index_.assign(open.data(), open.size());
        std::vector<double> band(static_cast<std::size_t>(add));
        for (int i = 0; i < add; ++i)
            band[static_cast<std::size_t>(i)] = lambda[static_cast<std::size_t>(open[static_cast<std::size_t>(i)])];
        scale_.assign(band.data(), band.size());
        Complex* fresh = basis_.data() + n * nbase;
        precondition_kernel<<<blocks(n * add), kThreads>>>(n, add, index_.data(), diagonal_.data(),
                                                          overlap_diagonal_.data(), scale_.data(),
                                                          residual_.data(), fresh);
        check_cuda(cudaGetLastError(), "the preconditioner");
        Complex* sfresh = general ? sbasis_.data() + n * nbase : nullptr;
        if (general) h.apply_overlap(fresh, sfresh, add);
        const int kept = orthonormalize(basis_.data(), general ? sbasis_.data() : nullptr, nbase, fresh, sfresh, add,
                                        rows);
        if (kept == 0) {  // nothing new: the basis cannot grow
            copy(psi, x, n * bands);
            eigenvalues.assign(lambda.begin(), lambda.begin() + count);
            return result;
        }
        h.apply(fresh, hbasis_.data() + n * nbase, kept);
        grow(nbase, kept);
        nbase += kept;
    }
}

}  // namespace mdapy::dft
