// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/solver/davidson.h"

#include "dft/core/profile.h"
#include "dft/hamiltonian/gamma.h"

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

void copy(Complex* to, const Complex* from, std::size_t count) {
    check_cuda(cudaMemcpy(to, from, count * sizeof(Complex), cudaMemcpyDeviceToDevice), "cudaMemcpy");
}

const Complex kOne = make_cuDoubleComplex(1.0, 0.0);
const Complex kZero = make_cuDoubleComplex(0.0, 0.0);
const Complex kMinus = make_cuDoubleComplex(-1.0, 0.0);

double* real(Complex* p) { return reinterpret_cast<double*>(p); }
const double* real(const Complex* p) { return reinterpret_cast<const double*>(p); }

void copy_bytes(void* to, const void* from, std::size_t bytes) {
    check_cuda(cudaMemcpy(to, from, bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy");
}

void dgemm(Blas& blas, Op ta, Op tb, int m, int n, int k, double alpha,
           const double* a, int lda, const double* b, int ldb, double beta, double* c, int ldc) {
    blas.gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// one block per column c: dots[c] = sum a b, real
__global__ void real_column_dots(std::size_t n, const double* a, const double* b, double* dots) {
    const double* x = a + static_cast<std::size_t>(blockIdx.x) * n;
    const double* y = b + static_cast<std::size_t>(blockIdx.x) * n;
    double s = 0.0;
    for (std::size_t g = threadIdx.x; g < n; g += blockDim.x) s += x[g] * y[g];
    __shared__ double partial[kThreads];
    partial[threadIdx.x] = s;
    __syncthreads();
    for (int k = kThreads / 2; k > 0; k >>= 1) {
        if (static_cast<int>(threadIdx.x) < k) partial[threadIdx.x] += partial[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) dots[blockIdx.x] = partial[0];
}

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

// column c of out (rows, count) is column index[c] of in (rows, any)
__global__ void gather_columns(std::size_t rows, int count, const int* index, const Complex* in, Complex* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= rows * count) return;
    out[i] = in[static_cast<std::size_t>(index[i / rows]) * rows + i % rows];
}

// column c of x times s[c]
__global__ void scale_columns(std::size_t n, int count, const double* s, Complex* x) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    const double f = s[i / n];
    x[i] = make_cuDoubleComplex(f * x[i].x, f * x[i].y);
}

__global__ void scale_real_columns(std::size_t n, int count, const double* s, double* x) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    x[i] *= s[i / n];
}

__global__ void gather_real_columns(std::size_t rows, int count, const int* index, const double* in, double* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= rows * count) return;
    out[i] = in[static_cast<std::size_t>(index[i / rows]) * rows + i % rows];
}

}  // namespace

Davidson::Davidson(Kernels kernels) : blas_(kernels), lapack_(kernels) {}

Davidson::~Davidson() = default;

void Davidson::eigh(Complex* a, int n, int count) { lapack_.eigh(a, n, eigenvalues_, count); }

void Davidson::eigh_real(double* a, int n, int count) { lapack_.eigh(a, n, eigenvalues_, count); }

Complex* Davidson::projection(Complex* p, std::size_t c) const {
    const std::size_t offset = static_cast<std::size_t>(projectors_) * c;
    return gamma_ ? reinterpret_cast<Complex*>(real(p) + offset) : p + offset;
}

void Davidson::gram(Hamiltonian* h, const Complex* a, const Complex* pa, int ka, const Complex* b,
                     const Complex* pb, int kb, int rows, Complex* out) {
    MDAPY_SPAN("gram");
    if (gamma_) {  // real, over the whole sphere
        double* o = real(out);
        gamma_inner(blas_, static_cast<std::size_t>(rows), a, ka, b, kb, o, ka);
        if (h == nullptr) return;
        const int np = h->projectors();
        qp_.reserve(static_cast<std::size_t>(np) * kb);
        h->overlap_projections(pb, kb, qp_.data());
        dgemm(blas_, Op::T, Op::N, ka, kb, np, 1.0, real(pa), np, real(qp_.data()), np, 1.0, o, ka);
        return;
    }
    // out (ka, kb) = a^H b + pa^H q pb
    blas_.gemm(Op::C, Op::N, ka, kb, rows, kOne, a, rows, b, rows, kZero, out, ka);
    if (h == nullptr) return;
    const int np = h->projectors();
    qp_.reserve(static_cast<std::size_t>(np) * kb);
    h->overlap_projections(pb, kb, qp_.data());
    blas_.gemm(Op::C, Op::N, ka, kb, np, kOne, pa, np, qp_.data(), np, kOne, out, ka);
}

int Davidson::orthonormalize(Hamiltonian* h, const Complex* v, const Complex* pv, int n, Complex* w, Complex* pw,
                             int m, int rows) {
    MDAPY_SPAN("orthonormalize");
    if (m == 0) return 0;
    const bool general = h != nullptr;
    const int np = general ? h->projectors() : 0;
    const std::size_t r = static_cast<std::size_t>(rows), p = static_cast<std::size_t>(np);
    overlap_.reserve(static_cast<std::size_t>(std::max(n, m)) * m);
    work_.reserve(std::max(r, p) * m);
    std::vector<double> host(static_cast<std::size_t>(m));
    // unit columns in the metric of S, so that the overlap eigenvalues measure
    // independence: |w|^2 + pw^H q pw
    sums_.reserve(2 * static_cast<std::size_t>(m));
    if (gamma_)
        gamma_dots(r, m, w, w, sums_.data());
    else
        column_dots<<<m, kThreads>>>(r, w, w, sums_.data());
    if (general) {
        qp_.reserve(p * m);
        h->overlap_projections(pw, m, qp_.data());
        if (gamma_)
            real_column_dots<<<m, kThreads>>>(p, real(pw), real(qp_.data()), sums_.data() + m);
        else
            column_dots<<<m, kThreads>>>(p, pw, qp_.data(), sums_.data() + m);
    }
    check_cuda(cudaGetLastError(), "the column norms");
    std::vector<double> both(2 * static_cast<std::size_t>(m), 0.0);
    sums_.download(both.data(), general ? both.size() : host.size());
    for (int c = 0; c < m; ++c) {
        const double x = both[static_cast<std::size_t>(c)] + (general ? both[static_cast<std::size_t>(m + c)] : 0.0);
        host[static_cast<std::size_t>(c)] = x > 0.0 ? 1.0 / std::sqrt(x) : 0.0;
    }
    scale_.assign(host.data(), host.size());
    scale_columns<<<blocks(r * m), kThreads>>>(r, m, scale_.data(), w);
    if (general && gamma_)
        scale_real_columns<<<blocks(p * m), kThreads>>>(p, m, scale_.data(), real(pw));
    else if (general)
        scale_columns<<<blocks(p * m), kThreads>>>(p, m, scale_.data(), pw);
    check_cuda(cudaGetLastError(), "the unit columns");
    // the projections follow w through the linear steps; recomputed when a
    // step is ill conditioned, as their error grows with it
    bool fresh = false;
    for (int pass = 0; pass < 2; ++pass) {
        if (n > 0 && gamma_) {  // w -= v (v^T S w), real
            gram(h, v, pv, n, w, pw, m, rows, overlap_.data());
            const double* o = real(overlap_.data());
            gamma_combine(blas_, r, v, n, o, n, m, -1.0, 1.0, w);
            if (general) dgemm(blas_, Op::N, Op::N, np, m, n, -1.0, real(pv), np, o, n, 1.0, real(pw), np);
        } else if (n > 0) {  // w -= v (v^H S w), pw likewise
            gram(h, v, pv, n, w, pw, m, rows, overlap_.data());
            blas_.gemm(Op::N, Op::N, rows, m, n, kMinus, v, rows, overlap_.data(), n, kOne, w, rows);
            if (general)
                blas_.gemm(Op::N, Op::N, np, m, n, kMinus, pv, np, overlap_.data(), n, kOne, pw, np);
        }
        gram(h, w, pw, m, w, pw, m, rows, overlap_.data());
        // Cholesky QR, w <- w R^-1 with w^H S w = R^H R, while R is well
        // conditioned (its diagonal within 1e-4 of its largest)
        const std::size_t mm = static_cast<std::size_t>(m) * m;
        chol_.reserve(mm);
        copy_bytes(chol_.data(), overlap_.data(), mm * (gamma_ ? sizeof(double) : sizeof(Complex)));
        const int info = gamma_ ? lapack_.potrf(real(chol_.data()), m) : lapack_.potrf(chol_.data(), m);
        if (info == 0) {
            diagonal_r_.reserve(static_cast<std::size_t>(m));
            std::vector<double> d(static_cast<std::size_t>(m));
            if (gamma_) {
                blas_.copy(m, real(chol_.data()), m + 1, real(diagonal_r_.data()), 1);
                check_cuda(cudaMemcpy(d.data(), diagonal_r_.data(), d.size() * sizeof(double), cudaMemcpyDeviceToHost),
                           "cudaMemcpy");
            } else {
                blas_.copy(m, chol_.data(), m + 1, diagonal_r_.data(), 1);
                std::vector<Complex> z(static_cast<std::size_t>(m));
                diagonal_r_.download(z.data(), z.size());
                for (std::size_t i = 0; i < z.size(); ++i) d[i] = z[i].x;
            }
            double low = std::abs(d[0]), high = low;
            for (const double x : d) {
                low = std::min(low, std::abs(x));
                high = std::max(high, std::abs(x));
            }
            if (low > 1e-4 * high) {
                if (low < 1e-2 * high) fresh = true;
                if (gamma_) {
                    gamma_solve(blas_, r, m, real(chol_.data()), w);
                    if (general) {
                        blas_.trsm(np, m, real(chol_.data()), m, real(pw), np);
                    }
                } else {
                    blas_.trsm(rows, m, chol_.data(), m, w, rows);
                    if (general)
                        blas_.trsm(np, m, chol_.data(), m, pw, np);
                }
                continue;
            }
        }
        // else w <- w Q Lambda^-1/2 over the eigenpairs of w^H S w above
        // kDependent of the largest
        fresh = true;
        if (gamma_)
            eigh_real(real(overlap_.data()), m);
        else
            eigh(overlap_.data(), m);
        host = eigenvalues_;
        const double largest = std::max(host.back(), 1e-300);
        int first = 0;
        while (first < m && !(host[static_cast<std::size_t>(first)] > kDependent * largest)) ++first;
        const int kept = m - first;
        if (kept == 0) return 0;
        std::vector<double> inverse(static_cast<std::size_t>(kept));
        for (int i = 0; i < kept; ++i)
            inverse[static_cast<std::size_t>(i)] = 1.0 / std::sqrt(host[static_cast<std::size_t>(first + i)]);
        scale_.assign(inverse.data(), inverse.size());
        for (int part = 0; gamma_ && part < (general ? 2 : 1); ++part) {
            const double* q = real(overlap_.data()) + static_cast<std::size_t>(first) * m;
            if (part == 0) {
                gamma_combine(blas_, r, w, m, q, m, kept, 1.0, 0.0, work_.data());
                scale_columns<<<blocks(r * kept), kThreads>>>(r, kept, scale_.data(), work_.data());
                check_cuda(cudaGetLastError(), "the orthonormal columns");
                copy(w, work_.data(), r * kept);
            } else {
                dgemm(blas_, Op::N, Op::N, np, kept, m, 1.0, real(pw), np, q, m, 0.0, real(work_.data()), np);
                scale_real_columns<<<blocks(p * kept), kThreads>>>(p, kept, scale_.data(), real(work_.data()));
                check_cuda(cudaGetLastError(), "the orthonormal columns");
                copy_bytes(pw, work_.data(), p * kept * sizeof(double));
            }
        }
        for (int part = 0; !gamma_ && part < (general ? 2 : 1); ++part) {
            Complex* target = part == 0 ? w : pw;
            const int length = part == 0 ? rows : np;
            const std::size_t l = static_cast<std::size_t>(length);
            blas_.gemm(Op::N, Op::N, length, kept, m, kOne, target, length, overlap_.data() + static_cast<std::size_t>(first) * m, m, kZero, work_.data(), length);
            scale_columns<<<blocks(l * kept), kThreads>>>(l, kept, scale_.data(), work_.data());
            check_cuda(cudaGetLastError(), "the orthonormal columns");
            copy(target, work_.data(), l * kept);
        }
        m = kept;
    }
    if (general && fresh) h->project(w, m, pw);
    return m;
}

DavidsonResult Davidson::solve(Hamiltonian& h, Complex* psi, int count, std::vector<double>& eigenvalues,
                               double tolerance, int tight, double loose_tolerance, int max_iterations,
                               int largest_basis) {
    std::vector<double> tolerances(static_cast<std::size_t>(std::max(count, 0)));
    for (int b = 0; b < count; ++b) tolerances[static_cast<std::size_t>(b)] = b < tight ? tolerance : loose_tolerance;
    return solve(h, psi, count, eigenvalues, tolerances, max_iterations, largest_basis);
}

DavidsonResult Davidson::solve(Hamiltonian& h, Complex* psi, int count, std::vector<double>& eigenvalues,
                               const std::vector<double>& tolerances, int max_iterations, int largest_basis) {
    MDAPY_SPAN("davidson");
    const int rows = static_cast<int>(h.plane_waves());
    if (count < 1 || count > rows) throw std::invalid_argument("from one band to as many as the plane waves");
    if (tolerances.size() != static_cast<std::size_t>(count)) throw std::invalid_argument("a tolerance for each band");
    if (max_iterations < 1) throw std::invalid_argument("max_iterations must be positive");
    // one Rayleigh-Ritz needs no room to grow
    const int most = std::min(max_iterations == 1 ? count : std::max(largest_basis, 2 * count), rows);
    const std::size_t n = static_cast<std::size_t>(rows), bands = static_cast<std::size_t>(count);
    // with an overlap, the projections <beta|basis> carry the metric of S and
    // the nonlocal parts of H and S; without, H is applied as it is
    const bool general = h.has_overlap() && h.projectors() > 0;
    Hamiltonian* metric = general ? &h : nullptr;
    const int np = general ? h.projectors() : 0;
    gamma_ = h.gamma();
    projectors_ = np;
    const std::size_t p = static_cast<std::size_t>(np);
    basis_.reserve(n * most);
    hbasis_.reserve(n * most);
    if (general) {
        pbasis_.reserve(p * most);
        pvectors_.reserve(p * bands);
    }
    vectors_.reserve((general ? 3 : 2) * n * bands);  // X, HX and SX
    residual_.reserve(n * bands);
    subspace_.reserve(static_cast<std::size_t>(most) * most);
    diagonal_.reserve(n);
    h.diagonal(diagonal_.data());
    overlap_diagonal_.reserve(n);
    h.overlap_diagonal(overlap_diagonal_.data());
    Complex* pb = general ? pbasis_.data() : nullptr;
    // H of columns first ... first + m of the basis, their projections at hand
    auto apply = [&](int first, int m) {
        MDAPY_SPAN("apply");
        Complex* w = basis_.data() + n * static_cast<std::size_t>(first);
        Complex* hw = hbasis_.data() + n * static_cast<std::size_t>(first);
        if (!general) {
            h.apply(w, hw, m);
            return;
        }
        h.apply_local(w, hw, m);
        h.add_nonlocal(projection(pb, static_cast<std::size_t>(first)), m, hw);
    };
    copy(basis_.data(), psi, n * bands);
    if (general) h.project(basis_.data(), count, pb);
    // the guess as it is when it is orthonormal already, as the bands of the
    // step before are
    {
        overlap_.reserve(bands * bands);
        if (gamma_) gamma_clean(n, count, basis_.data());
        gram(metric, basis_.data(), pb, count, basis_.data(), pb, count, rows, overlap_.data());
        double off = 0.0;
        if (gamma_) {
            std::vector<double> g(bands * bands);
            check_cuda(cudaMemcpy(g.data(), overlap_.data(), g.size() * sizeof(double), cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
            for (std::size_t a = 0; a < bands; ++a)
                for (std::size_t b = 0; b < bands; ++b)
                    off = std::max(off, std::abs(g[a * bands + b] - (a == b ? 1.0 : 0.0)));
        } else {
            std::vector<Complex> g(bands * bands);
            overlap_.download(g.data(), g.size());
            for (std::size_t a = 0; a < bands; ++a)
                for (std::size_t b = 0; b < bands; ++b) {
                    const Complex x = g[a * bands + b];
                    off = std::max(off, std::abs(x.x - (a == b ? 1.0 : 0.0)) + std::abs(x.y));
                }
        }
        if (off > 1e-12) {
            const int independent = orthonormalize(metric, nullptr, nullptr, 0, basis_.data(), pb, count, rows);
            if (independent < count) throw std::invalid_argument("the guess of the bands is linearly dependent");
        }
    }
    int nbase = count;
    apply(0, nbase);
    // basis^H H basis, kept between iterations and grown by the columns of
    // the new vectors (the upper triangle, which the eigensolver reads)
    const std::size_t ld = static_cast<std::size_t>(most);
    hsub_.reserve(ld * ld);
    auto grow = [&](int first, int columns) {
        MDAPY_SPAN("grow");
        if (gamma_) {
            gamma_inner(blas_, n, basis_.data(), first + columns, hbasis_.data() + n * static_cast<std::size_t>(first),
                        columns, real(hsub_.data()) + ld * static_cast<std::size_t>(first), most);
            return;
        }
        blas_.gemm(Op::C, Op::N, first + columns, columns, rows, kOne, basis_.data(), rows, hbasis_.data() + n * static_cast<std::size_t>(first), rows, kZero, hsub_.data() + ld * static_cast<std::size_t>(first), most);
    };
    grow(0, nbase);
    Complex* x = vectors_.data();
    Complex* hx = vectors_.data() + n * bands;
    Complex* sx = general ? vectors_.data() + 2 * n * bands : x;
    Complex* px = general ? pvectors_.data() : nullptr;
    picked_.reserve(ld * bands);
    DavidsonResult result;
    std::vector<double> lambda, residuals(bands);
    std::vector<char> done(bands, 0);  // converged, not checked again until the end
    std::vector<int> open, column;     // the bands still open; their columns in residual_
    // the residuals of the bands b of `which` into residual_ (column i for
    // which[i]) and their squared norms into residuals[b]; with all, X, HX,
    // SX and the projections of X of every band too
    auto residuals_of = [&](const std::vector<int>& which, bool all) {
        MDAPY_SPAN("residuals");
        const int m = static_cast<int>(which.size());
        if (m == 0) return;
        if (gamma_) {
            const double* yr = real(subspace_.data());
            if (!all) {
                index_.assign(which.data(), which.size());
                gather_real_columns<<<blocks(static_cast<std::size_t>(nbase) * m), kThreads>>>(
                    static_cast<std::size_t>(nbase), m, index_.data(), real(subspace_.data()), real(picked_.data()));
                check_cuda(cudaGetLastError(), "the open eigenvectors");
                yr = real(picked_.data());
            }
            gamma_combine(blas_, n, hbasis_.data(), nbase, yr, nbase, m, 1.0, 0.0, hx);
            gamma_combine(blas_, n, basis_.data(), nbase, yr, nbase, m, 1.0, 0.0, sx);
            if (general) {
                dgemm(blas_, Op::N, Op::N, np, m, nbase, 1.0, real(pb), np, yr, nbase, 0.0, real(px), np);
                if (all) copy(x, sx, n * static_cast<std::size_t>(m));
                h.add_overlap(px, m, sx);
            }
        } else {
            const Complex* y = subspace_.data();
            if (!all) {
                index_.assign(which.data(), which.size());
                gather_columns<<<blocks(static_cast<std::size_t>(nbase) * m), kThreads>>>(
                    static_cast<std::size_t>(nbase), m, index_.data(), subspace_.data(), picked_.data());
                check_cuda(cudaGetLastError(), "the open eigenvectors");
                y = picked_.data();
            }
            blas_.gemm(Op::N, Op::N, rows, m, nbase, kOne, hbasis_.data(), rows, y, nbase, kZero, hx, rows);
            // S X: X with the overlap of its projections
            blas_.gemm(Op::N, Op::N, rows, m, nbase, kOne, basis_.data(), rows, y, nbase, kZero, sx, rows);
            if (general) {
                blas_.gemm(Op::N, Op::N, np, m, nbase, kOne, pb, np, y, nbase, kZero, px, np);
                if (all) copy(x, sx, n * static_cast<std::size_t>(m));
                h.add_overlap(px, m, sx);
            }
        }
        std::vector<double> e(static_cast<std::size_t>(m));
        for (int i = 0; i < m; ++i) e[static_cast<std::size_t>(i)] = lambda[static_cast<std::size_t>(which[static_cast<std::size_t>(i)])];
        energies_.assign(e.data(), e.size());
        residual_kernel<<<blocks(n * m), kThreads>>>(n, m, sx, hx, energies_.data(), residual_.data());
        check_cuda(cudaGetLastError(), "the residuals");
        sums_.reserve(bands);
        if (gamma_)
            gamma_dots(n, m, residual_.data(), residual_.data(), sums_.data());
        else
            column_norms<<<m, kThreads>>>(n, residual_.data(), sums_.data());
        check_cuda(cudaGetLastError(), "the residual norms");
        std::vector<double> norms(static_cast<std::size_t>(m));
        sums_.download(norms.data(), norms.size());
        for (int i = 0; i < m; ++i) residuals[static_cast<std::size_t>(which[static_cast<std::size_t>(i)])] = norms[static_cast<std::size_t>(i)];
    };
    auto converged = [&](int b) {
        return std::sqrt(residuals[static_cast<std::size_t>(b)]) < tolerances[static_cast<std::size_t>(b)];
    };
    std::vector<int> every(bands);
    for (int b = 0; b < count; ++b) every[static_cast<std::size_t>(b)] = b;
    // the end, X of the last Rayleigh-Ritz as the bands
    auto finish = [&](bool checked) {
        if (!checked) residuals_of(every, true);
        result.converged = 0;
        result.residual = 0.0;
        for (int b = 0; b < count; ++b) {
            result.residual = std::max(result.residual, std::sqrt(residuals[static_cast<std::size_t>(b)]));
            if (converged(b)) ++result.converged;
        }
        copy(psi, x, n * bands);
        eigenvalues.assign(lambda.begin(), lambda.begin() + count);
        return result;
    };
    for (int iteration = 1;; ++iteration) {
        result.iterations = iteration;
        // Rayleigh-Ritz: the count lowest eigenpairs of basis^H H basis (of the
        // others the eigenvalues alone)
        const std::size_t element = gamma_ ? sizeof(double) : sizeof(Complex);
        check_cuda(cudaMemcpy2D(subspace_.data(), static_cast<std::size_t>(nbase) * element, hsub_.data(),
                                ld * element, static_cast<std::size_t>(nbase) * element,
                                static_cast<std::size_t>(nbase), cudaMemcpyDeviceToDevice),
                   "cudaMemcpy2D");
        if (gamma_)
            eigh_real(real(subspace_.data()), nbase, count);
        else
            eigh(subspace_.data(), nbase, count);
        lambda = eigenvalues_;
        open.clear();
        for (int b = 0; b < count; ++b)
            if (!done[static_cast<std::size_t>(b)]) open.push_back(b);
        column.clear();
        bool full = nbase == most || iteration >= max_iterations;
        if (!full) {
            // the residuals of the open bands; a band that converges is not
            // checked again until all have, when all are
            residuals_of(open, false);
            std::vector<int> still;
            for (std::size_t i = 0; i < open.size(); ++i) {
                if (converged(open[i])) {
                    done[static_cast<std::size_t>(open[i])] = 1;
                } else {
                    still.push_back(open[i]);
                    column.push_back(static_cast<int>(i));
                }
            }
            open = still;
            full = open.empty();
        }
        if (full) {
            // every band, before a restart or the end
            residuals_of(every, true);
            open.clear();
            column.clear();
            for (int b = 0; b < count; ++b) {
                done[static_cast<std::size_t>(b)] = converged(b) ? 1 : 0;
                if (!done[static_cast<std::size_t>(b)]) {
                    open.push_back(b);
                    column.push_back(b);
                }
            }
            if (open.empty() || iteration >= max_iterations) return finish(true);
            if (nbase == most) {
                // restart from the Ritz vectors, where H is diagonal
                copy(basis_.data(), x, n * bands);
                copy(hbasis_.data(), hx, n * bands);
                if (general) copy_bytes(pb, px, p * bands * (gamma_ ? sizeof(double) : sizeof(Complex)));
                nbase = count;
                if (gamma_) {
                    check_cuda(cudaMemset2D(hsub_.data(), ld * sizeof(double), 0, bands * sizeof(double), bands),
                               "cudaMemset2D");
                    energies_.assign(lambda.data(), bands);
                    blas_.copy(count, energies_.data(), 1, real(hsub_.data()), most + 1);
                } else {
                    check_cuda(cudaMemset2D(hsub_.data(), ld * sizeof(Complex), 0, bands * sizeof(Complex), bands),
                               "cudaMemset2D");
                    std::vector<Complex> diag(bands);
                    for (std::size_t b = 0; b < bands; ++b) diag[b] = make_cuDoubleComplex(lambda[b], 0.0);
                    picked_.assign(diag.data(), bands);
                    blas_.copy(count, picked_.data(), 1, hsub_.data(), most + 1);
                }
            }
        }
        const int add = std::min(static_cast<int>(open.size()), most - nbase);
        if (add == 0) return finish(full);
        // the preconditioned residuals of the open bands
        std::vector<double> band(static_cast<std::size_t>(add));
        for (int i = 0; i < add; ++i)
            band[static_cast<std::size_t>(i)] = lambda[static_cast<std::size_t>(open[static_cast<std::size_t>(i)])];
        scale_.assign(band.data(), band.size());
        index_.assign(column.data(), static_cast<std::size_t>(add));
        Complex* fresh = basis_.data() + n * nbase;
        precondition_kernel<<<blocks(n * add), kThreads>>>(n, add, index_.data(), diagonal_.data(),
                                                          overlap_diagonal_.data(), scale_.data(),
                                                          residual_.data(), fresh);
        check_cuda(cudaGetLastError(), "the preconditioner");
        if (gamma_) gamma_clean(n, add, fresh);
        Complex* pfresh = general ? projection(pb, static_cast<std::size_t>(nbase)) : nullptr;
        if (general) h.project(fresh, add, pfresh);
        const int kept = orthonormalize(metric, basis_.data(), pb, nbase, fresh, pfresh, add, rows);
        if (kept == 0) return finish(full);  // nothing new: the basis cannot grow
        apply(nbase, kept);
        grow(nbase, kept);
        nbase += kept;
    }
}

}  // namespace mdapy::dft
