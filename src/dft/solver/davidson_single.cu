// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/solver/davidson_single.h"

#include "dft/hamiltonian/gamma.h"

#include <algorithm>
#include <cmath>
#include <new>
#include <stdexcept>
#include <string>

namespace mdapy::dft {

namespace {

constexpr int kThreads = 256;
// overlap eigenvalue of a dropped direction, of the largest: single precision
// resolves its overlaps to about 1e-7
constexpr float kDependent = 1e-6f;

unsigned blocks(std::size_t work) { return static_cast<unsigned>((work + kThreads - 1) / kThreads); }

void copy(cuComplex* to, const cuComplex* from, std::size_t count) {
    check_cuda(cudaMemcpy(to, from, count * sizeof(cuComplex), cudaMemcpyDeviceToDevice), "cudaMemcpy");
}

const cuComplex kOne = make_cuComplex(1.0f, 0.0f);
const cuComplex kZero = make_cuComplex(0.0f, 0.0f);
const cuComplex kMinus = make_cuComplex(-1.0f, 0.0f);

float* real(cuComplex* p) { return reinterpret_cast<float*>(p); }
const float* real(const cuComplex* p) { return reinterpret_cast<const float*>(p); }

void copy_bytes(void* to, const void* from, std::size_t bytes) {
    check_cuda(cudaMemcpy(to, from, bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy");
}

void sgemm(Blas& blas, Op ta, Op tb, int m, int n, int k, float alpha,
           const float* a, int lda, const float* b, int ldb, float beta, float* c, int ldc) {
    blas.gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm(Blas& blas, Op ta, Op tb, int m, int n, int k, const cuComplex& alpha,
          const cuComplex* a, int lda, const cuComplex* b, int ldb, const cuComplex& beta, cuComplex* c, int ldc) {
    blas.gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

__global__ void to_single(std::size_t n, const Complex* in, cuComplex* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = make_cuComplex(static_cast<float>(in[i].x), static_cast<float>(in[i].y));
}

__global__ void to_double(std::size_t n, const cuComplex* in, Complex* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = make_cuDoubleComplex(in[i].x, in[i].y);
}

__global__ void real_to_single(std::size_t n, const double* in, float* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = static_cast<float>(in[i]);
}

// R = HX - X diag(e)
__global__ void residual_kernel(std::size_t n, int count, const cuComplex* x, const cuComplex* hx, const float* e,
                                cuComplex* r) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    const float ev = e[i / n];
    r[i] = make_cuComplex(hx[i].x - ev * x[i].x, hx[i].y - ev * x[i].y);
}

// one block per column c: dots[c] = Re sum conj(a) b, summed in double
__global__ void column_dots(std::size_t n, const cuComplex* a, const cuComplex* b, double* dots) {
    const cuComplex* x = a + static_cast<std::size_t>(blockIdx.x) * n;
    const cuComplex* y = b + static_cast<std::size_t>(blockIdx.x) * n;
    double s = 0.0;
    for (std::size_t g = threadIdx.x; g < n; g += blockDim.x)
        s += static_cast<double>(x[g].x) * y[g].x + static_cast<double>(x[g].y) * y[g].y;
    __shared__ double partial[kThreads];
    partial[threadIdx.x] = s;
    __syncthreads();
    for (int k = kThreads / 2; k > 0; k >>= 1) {
        if (static_cast<int>(threadIdx.x) < k) partial[threadIdx.x] += partial[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) dots[blockIdx.x] = partial[0];
}

// one block per column c: dots[c] = sum a b, real, summed in double
__global__ void real_column_dots(std::size_t n, const float* a, const float* b, double* dots) {
    const float* x = a + static_cast<std::size_t>(blockIdx.x) * n;
    const float* y = b + static_cast<std::size_t>(blockIdx.x) * n;
    double s = 0.0;
    for (std::size_t g = threadIdx.x; g < n; g += blockDim.x) s += static_cast<double>(x[g]) * y[g];
    __shared__ double partial[kThreads];
    partial[threadIdx.x] = s;
    __syncthreads();
    for (int k = kThreads / 2; k > 0; k >>= 1) {
        if (static_cast<int>(threadIdx.x) < k) partial[threadIdx.x] += partial[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) dots[blockIdx.x] = partial[0];
}

// as Davidson's preconditioner
__global__ void precondition_kernel(std::size_t n, int count, const int* index, const float* diagonal,
                                    const float* overlap, const float* e, const cuComplex* r, cuComplex* t) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    const std::size_t c = i / n, g = i % n;
    const float x = 2.0f * (diagonal[g] - e[c] * overlap[g]);
    const float k = 2.0f / (1.0f + x + sqrtf(1.0f + (x - 1.0f) * (x - 1.0f)));
    const cuComplex v = r[static_cast<std::size_t>(index[c]) * n + g];
    t[i] = make_cuComplex(k * v.x, k * v.y);
}

__global__ void gather_columns(std::size_t rows, int count, const int* index, const cuComplex* in, cuComplex* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= rows * count) return;
    out[i] = in[static_cast<std::size_t>(index[i / rows]) * rows + i % rows];
}

__global__ void scale_columns(std::size_t n, int count, const float* s, cuComplex* x) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    const float f = s[i / n];
    x[i] = make_cuComplex(f * x[i].x, f * x[i].y);
}

__global__ void scale_real_columns(std::size_t n, int count, const float* s, float* x) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    x[i] *= s[i / n];
}

__global__ void gather_real_columns(std::size_t rows, int count, const int* index, const float* in, float* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= rows * count) return;
    out[i] = in[static_cast<std::size_t>(index[i / rows]) * rows + i % rows];
}

}  // namespace

DavidsonSingle::DavidsonSingle(Kernels kernels) : blas_(kernels), lapack_(kernels) {}

DavidsonSingle::~DavidsonSingle() = default;

void DavidsonSingle::eigh(cuComplex* a, int n) { lapack_.eigh(a, n, eigenvalues_); }

void DavidsonSingle::eigh_real(float* a, int n) { lapack_.eigh(a, n, eigenvalues_); }

cuComplex* DavidsonSingle::projection(cuComplex* p, std::size_t c) const {
    const std::size_t offset = static_cast<std::size_t>(projectors_) * c;
    return gamma_ ? reinterpret_cast<cuComplex*>(real(p) + offset) : p + offset;
}

void DavidsonSingle::gram(Hamiltonian* h, const cuComplex* a, const cuComplex* pa, int ka, const cuComplex* b,
                          const cuComplex* pb, int kb, int rows, cuComplex* out) {
    if (gamma_) {  // real, over the whole sphere
        float* o = real(out);
        gamma_inner(blas_, static_cast<std::size_t>(rows), a, ka, b, kb, o, ka);
        if (h == nullptr) return;
        const int np = h->projectors();
        qp_.reserve(static_cast<std::size_t>(np) * kb);
        h->overlap_projections(pb, kb, qp_.data());
        sgemm(blas_, Op::T, Op::N, ka, kb, np, 1.0f, real(pa), np, real(qp_.data()), np, 1.0f, o, ka);
        return;
    }
    gemm(blas_, Op::C, Op::N, ka, kb, rows, kOne, a, rows, b, rows, kZero, out, ka);
    if (h == nullptr) return;
    const int np = h->projectors();
    qp_.reserve(static_cast<std::size_t>(np) * kb);
    h->overlap_projections(pb, kb, qp_.data());
    gemm(blas_, Op::C, Op::N, ka, kb, np, kOne, pa, np, qp_.data(), np, kOne, out, ka);
}

int DavidsonSingle::orthonormalize(Hamiltonian* h, const cuComplex* v, const cuComplex* pv, int n, cuComplex* w,
                                   cuComplex* pw, int m, int rows) {
    if (m == 0) return 0;
    const bool general = h != nullptr;
    const int np = general ? h->projectors() : 0;
    const std::size_t r = static_cast<std::size_t>(rows), p = static_cast<std::size_t>(np);
    overlap_.reserve(static_cast<std::size_t>(std::max(n, m)) * m);
    work_.reserve(std::max(r, p) * m);
    std::vector<float> host(static_cast<std::size_t>(m));
    // unit columns in the metric of S
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
    sums_.download(both.data(), general ? both.size() : static_cast<std::size_t>(m));
    for (int c = 0; c < m; ++c) {
        const double x = both[static_cast<std::size_t>(c)] + (general ? both[static_cast<std::size_t>(m + c)] : 0.0);
        host[static_cast<std::size_t>(c)] = x > 0.0 ? static_cast<float>(1.0 / std::sqrt(x)) : 0.0f;
    }
    scale_.assign(host.data(), host.size());
    scale_columns<<<blocks(r * m), kThreads>>>(r, m, scale_.data(), w);
    if (general && gamma_)
        scale_real_columns<<<blocks(p * m), kThreads>>>(p, m, scale_.data(), real(pw));
    else if (general)
        scale_columns<<<blocks(p * m), kThreads>>>(p, m, scale_.data(), pw);
    check_cuda(cudaGetLastError(), "the unit columns");
    bool fresh = false;
    for (int pass = 0; pass < 2; ++pass) {
        if (n > 0 && gamma_) {  // w -= v (v^T S w), real
            gram(h, v, pv, n, w, pw, m, rows, overlap_.data());
            const float* o = real(overlap_.data());
            gamma_combine(blas_, r, v, n, o, n, m, -1.0f, 1.0f, w);
            if (general)
                sgemm(blas_, Op::N, Op::N, np, m, n, -1.0f, real(pv), np, o, n, 1.0f, real(pw), np);
        } else if (n > 0) {  // w -= v (v^H S w), pw likewise
            gram(h, v, pv, n, w, pw, m, rows, overlap_.data());
            gemm(blas_, Op::N, Op::N, rows, m, n, kMinus, v, rows, overlap_.data(), n, kOne, w, rows);
            if (general) gemm(blas_, Op::N, Op::N, np, m, n, kMinus, pv, np, overlap_.data(), n, kOne, pw, np);
        }
        gram(h, w, pw, m, w, pw, m, rows, overlap_.data());
        // Cholesky QR while R is well conditioned
        const std::size_t mm = static_cast<std::size_t>(m) * m;
        chol_.reserve(mm);
        copy_bytes(chol_.data(), overlap_.data(), mm * (gamma_ ? sizeof(float) : sizeof(cuComplex)));
        const int info = gamma_ ? lapack_.potrf(real(chol_.data()), m) : lapack_.potrf(chol_.data(), m);
        if (info == 0) {
            diagonal_r_.reserve(static_cast<std::size_t>(m));
            std::vector<float> d(static_cast<std::size_t>(m));
            if (gamma_) {
                blas_.copy(m, real(chol_.data()), m + 1, real(diagonal_r_.data()), 1);
                check_cuda(cudaMemcpy(d.data(), diagonal_r_.data(), d.size() * sizeof(float), cudaMemcpyDeviceToHost),
                           "cudaMemcpy");
            } else {
                blas_.copy(m, chol_.data(), m + 1, diagonal_r_.data(), 1);
                std::vector<cuComplex> z(static_cast<std::size_t>(m));
                diagonal_r_.download(z.data(), z.size());
                for (std::size_t i = 0; i < z.size(); ++i) d[i] = z[i].x;
            }
            float low = std::abs(d[0]), high = low;
            for (const float x : d) {
                low = std::min(low, std::abs(x));
                high = std::max(high, std::abs(x));
            }
            // single precision: a factor of condition below 1e2 only
            if (low > 1e-2f * high) {
                if (low < 1e-1f * high) fresh = true;
                if (gamma_) {
                    gamma_solve(blas_, r, m, real(chol_.data()), w);
                    if (general) {
                        const float one = 1.0f;
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
        // else w <- w Q Lambda^-1/2 over the eigenpairs above kDependent
        fresh = true;
        if (gamma_)
            eigh_real(real(overlap_.data()), m);
        else
            eigh(overlap_.data(), m);
        host.assign(eigenvalues_.begin(), eigenvalues_.end());
        const float largest = std::max(host.back(), 1e-30f);
        int first = 0;
        while (first < m && !(host[static_cast<std::size_t>(first)] > kDependent * largest)) ++first;
        const int kept = m - first;
        if (kept == 0) return 0;
        std::vector<float> inverse(static_cast<std::size_t>(kept));
        for (int i = 0; i < kept; ++i)
            inverse[static_cast<std::size_t>(i)] = 1.0f / std::sqrt(host[static_cast<std::size_t>(first + i)]);
        scale_.assign(inverse.data(), inverse.size());
        for (int part = 0; gamma_ && part < (general ? 2 : 1); ++part) {
            const float* q = real(overlap_.data()) + static_cast<std::size_t>(first) * m;
            if (part == 0) {
                gamma_combine(blas_, r, w, m, q, m, kept, 1.0f, 0.0f, work_.data());
                scale_columns<<<blocks(r * kept), kThreads>>>(r, kept, scale_.data(), work_.data());
                check_cuda(cudaGetLastError(), "the orthonormal columns");
                copy(w, work_.data(), r * kept);
            } else {
                sgemm(blas_, Op::N, Op::N, np, kept, m, 1.0f, real(pw), np, q, m, 0.0f, real(work_.data()),
                      np);
                scale_real_columns<<<blocks(p * kept), kThreads>>>(p, kept, scale_.data(), real(work_.data()));
                check_cuda(cudaGetLastError(), "the orthonormal columns");
                copy_bytes(pw, work_.data(), p * kept * sizeof(float));
            }
        }
        for (int part = 0; !gamma_ && part < (general ? 2 : 1); ++part) {
            cuComplex* target = part == 0 ? w : pw;
            const int length = part == 0 ? rows : np;
            const std::size_t l = static_cast<std::size_t>(length);
            gemm(blas_, Op::N, Op::N, length, kept, m, kOne, target, length,
                 overlap_.data() + static_cast<std::size_t>(first) * m, m, kZero, work_.data(), length);
            scale_columns<<<blocks(l * kept), kThreads>>>(l, kept, scale_.data(), work_.data());
            check_cuda(cudaGetLastError(), "the orthonormal columns");
            copy(target, work_.data(), l * kept);
        }
        m = kept;
        host.resize(static_cast<std::size_t>(m));
    }
    if (general && fresh) h->project(w, m, pw);
    return m;
}

DavidsonResult DavidsonSingle::solve(Hamiltonian& h, Complex* psi, int count, std::vector<double>& eigenvalues,
                                     const std::vector<double>& tolerances, int max_iterations, int largest_basis) {
    const int rows = static_cast<int>(h.plane_waves());
    if (count < 1 || count > rows) throw std::invalid_argument("from one band to as many as the plane waves");
    if (tolerances.size() != static_cast<std::size_t>(count)) throw std::invalid_argument("a tolerance for each band");
    if (max_iterations < 1) throw std::invalid_argument("max_iterations must be positive");
    const int most = std::min(max_iterations == 1 ? count : std::max(largest_basis, 2 * count), rows);
    const std::size_t n = static_cast<std::size_t>(rows), bands = static_cast<std::size_t>(count);
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
    vectors_.reserve((general ? 3 : 2) * n * bands);
    residual_.reserve(n * bands);
    subspace_.reserve(static_cast<std::size_t>(most) * most);
    {  // the diagonals of H and S in single precision
        DeviceArray<double> d(n);
        h.diagonal(d.data());
        diagonal_.reserve(n);
        real_to_single<<<blocks(n), kThreads>>>(n, d.data(), diagonal_.data());
        h.overlap_diagonal(d.data());
        overlap_diagonal_.reserve(n);
        real_to_single<<<blocks(n), kThreads>>>(n, d.data(), overlap_diagonal_.data());
        check_cuda(cudaGetLastError(), "the diagonals");
    }
    cuComplex* pb = general ? pbasis_.data() : nullptr;
    auto apply = [&](int first, int m) {
        cuComplex* w = basis_.data() + n * static_cast<std::size_t>(first);
        cuComplex* hw = hbasis_.data() + n * static_cast<std::size_t>(first);
        if (!general) {
            h.apply(w, hw, m);
            return;
        }
        h.apply_local(w, hw, m);
        h.add_nonlocal(projection(pb, static_cast<std::size_t>(first)), m, hw);
    };
    to_single<<<blocks(n * bands), kThreads>>>(n * bands, psi, basis_.data());
    check_cuda(cudaGetLastError(), "the bands in single precision");
    if (gamma_) gamma_clean(n, count, basis_.data());
    if (general) h.project(basis_.data(), count, pb);
    {  // the bands, orthonormal in double precision, again in single
        const int independent = orthonormalize(metric, nullptr, nullptr, 0, basis_.data(), pb, count, rows);
        if (independent < count) throw std::invalid_argument("the guess of the bands is linearly dependent");
    }
    int nbase = count;
    apply(0, nbase);
    const std::size_t ld = static_cast<std::size_t>(most);
    hsub_.reserve(ld * ld);
    auto grow = [&](int first, int columns) {
        if (gamma_) {
            gamma_inner(blas_, n, basis_.data(), first + columns, hbasis_.data() + n * static_cast<std::size_t>(first),
                        columns, real(hsub_.data()) + ld * static_cast<std::size_t>(first), most);
            return;
        }
        gemm(blas_, Op::C, Op::N, first + columns, columns, rows, kOne, basis_.data(), rows,
             hbasis_.data() + n * static_cast<std::size_t>(first), rows, kZero,
             hsub_.data() + ld * static_cast<std::size_t>(first), most);
    };
    grow(0, nbase);
    cuComplex* x = vectors_.data();
    cuComplex* hx = vectors_.data() + n * bands;
    cuComplex* sx = general ? vectors_.data() + 2 * n * bands : x;
    cuComplex* px = general ? pvectors_.data() : nullptr;
    picked_.reserve(ld * bands);
    DavidsonResult result;
    std::vector<double> lambda, residuals(bands);
    std::vector<char> done(bands, 0);
    std::vector<int> open, column;
    auto residuals_of = [&](const std::vector<int>& which, bool all) {
        const int m = static_cast<int>(which.size());
        if (m == 0) return;
        if (gamma_) {
            const float* yr = real(subspace_.data());
            if (!all) {
                index_.assign(which.data(), which.size());
                gather_real_columns<<<blocks(static_cast<std::size_t>(nbase) * m), kThreads>>>(
                    static_cast<std::size_t>(nbase), m, index_.data(), real(subspace_.data()), real(picked_.data()));
                check_cuda(cudaGetLastError(), "the open eigenvectors");
                yr = real(picked_.data());
            }
            gamma_combine(blas_, n, hbasis_.data(), nbase, yr, nbase, m, 1.0f, 0.0f, hx);
            gamma_combine(blas_, n, basis_.data(), nbase, yr, nbase, m, 1.0f, 0.0f, sx);
            if (general) {
                sgemm(blas_, Op::N, Op::N, np, m, nbase, 1.0f, real(pb), np, yr, nbase, 0.0f, real(px), np);
                if (all) copy(x, sx, n * static_cast<std::size_t>(m));
                h.add_overlap(px, m, sx);
            }
        } else {
            const cuComplex* y = subspace_.data();
            if (!all) {
                index_.assign(which.data(), which.size());
                gather_columns<<<blocks(static_cast<std::size_t>(nbase) * m), kThreads>>>(
                    static_cast<std::size_t>(nbase), m, index_.data(), subspace_.data(), picked_.data());
                check_cuda(cudaGetLastError(), "the open eigenvectors");
                y = picked_.data();
            }
            gemm(blas_, Op::N, Op::N, rows, m, nbase, kOne, hbasis_.data(), rows, y, nbase, kZero, hx,
                 rows);
            gemm(blas_, Op::N, Op::N, rows, m, nbase, kOne, basis_.data(), rows, y, nbase, kZero, sx,
                 rows);
            if (general) {
                gemm(blas_, Op::N, Op::N, np, m, nbase, kOne, pb, np, y, nbase, kZero, px, np);
                if (all) copy(x, sx, n * static_cast<std::size_t>(m));
                h.add_overlap(px, m, sx);
            }
        }
        std::vector<float> e(static_cast<std::size_t>(m));
        for (int i = 0; i < m; ++i)
            e[static_cast<std::size_t>(i)] = static_cast<float>(lambda[static_cast<std::size_t>(which[static_cast<std::size_t>(i)])]);
        energies_.assign(e.data(), e.size());
        residual_kernel<<<blocks(n * m), kThreads>>>(n, m, sx, hx, energies_.data(), residual_.data());
        check_cuda(cudaGetLastError(), "the residuals");
        sums_.reserve(bands);
        if (gamma_)
            gamma_dots(n, m, residual_.data(), residual_.data(), sums_.data());
        else
            column_dots<<<m, kThreads>>>(n, residual_.data(), residual_.data(), sums_.data());
        check_cuda(cudaGetLastError(), "the residual norms");
        std::vector<double> norms(static_cast<std::size_t>(m));
        sums_.download(norms.data(), norms.size());
        for (int i = 0; i < m; ++i)
            residuals[static_cast<std::size_t>(which[static_cast<std::size_t>(i)])] = norms[static_cast<std::size_t>(i)];
    };
    auto converged = [&](int b) {
        return std::sqrt(residuals[static_cast<std::size_t>(b)]) < tolerances[static_cast<std::size_t>(b)];
    };
    std::vector<int> every(bands);
    for (int b = 0; b < count; ++b) every[static_cast<std::size_t>(b)] = b;
    auto finish = [&](bool checked) {
        if (!checked) residuals_of(every, true);
        result.converged = 0;
        result.residual = 0.0;
        for (int b = 0; b < count; ++b) {
            result.residual = std::max(result.residual, std::sqrt(residuals[static_cast<std::size_t>(b)]));
            if (converged(b)) ++result.converged;
        }
        to_double<<<blocks(n * bands), kThreads>>>(n * bands, x, psi);
        check_cuda(cudaGetLastError(), "the bands in double precision");
        eigenvalues.assign(lambda.begin(), lambda.begin() + count);
        return result;
    };
    std::vector<float> host;
    for (int iteration = 1;; ++iteration) {
        result.iterations = iteration;
        const std::size_t element = gamma_ ? sizeof(float) : sizeof(cuComplex);
        check_cuda(cudaMemcpy2D(subspace_.data(), static_cast<std::size_t>(nbase) * element, hsub_.data(),
                                ld * element, static_cast<std::size_t>(nbase) * element,
                                static_cast<std::size_t>(nbase), cudaMemcpyDeviceToDevice),
                   "cudaMemcpy2D");
        if (gamma_)
            eigh_real(real(subspace_.data()), nbase);
        else
            eigh(subspace_.data(), nbase);
        lambda = eigenvalues_;
        open.clear();
        for (int b = 0; b < count; ++b)
            if (!done[static_cast<std::size_t>(b)]) open.push_back(b);
        column.clear();
        bool full = nbase == most || iteration >= max_iterations;
        if (!full) {
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
            if (nbase == most) {  // restart from the Ritz vectors
                copy(basis_.data(), x, n * bands);
                copy(hbasis_.data(), hx, n * bands);
                if (general) copy_bytes(pb, px, p * bands * (gamma_ ? sizeof(float) : sizeof(cuComplex)));
                nbase = count;
                if (gamma_) {
                    check_cuda(cudaMemset2D(hsub_.data(), ld * sizeof(float), 0, bands * sizeof(float), bands),
                               "cudaMemset2D");
                    std::vector<float> diag(bands);
                    for (std::size_t b = 0; b < bands; ++b) diag[b] = static_cast<float>(lambda[b]);
                    energies_.assign(diag.data(), bands);
                    blas_.copy(count, energies_.data(), 1, real(hsub_.data()), most + 1);
                } else {
                    check_cuda(cudaMemset2D(hsub_.data(), ld * sizeof(cuComplex), 0, bands * sizeof(cuComplex), bands),
                               "cudaMemset2D");
                    std::vector<cuComplex> diag(bands);
                    for (std::size_t b = 0; b < bands; ++b)
                        diag[b] = make_cuComplex(static_cast<float>(lambda[b]), 0.0f);
                    picked_.assign(diag.data(), bands);
                    blas_.copy(count, picked_.data(), 1, hsub_.data(), most + 1);
                }
            }
        }
        const int add = std::min(static_cast<int>(open.size()), most - nbase);
        if (add == 0) return finish(full);
        std::vector<float> band(static_cast<std::size_t>(add));
        for (int i = 0; i < add; ++i)
            band[static_cast<std::size_t>(i)] = static_cast<float>(lambda[static_cast<std::size_t>(open[static_cast<std::size_t>(i)])]);
        scale_.assign(band.data(), band.size());
        index_.assign(column.data(), static_cast<std::size_t>(add));
        cuComplex* fresh = basis_.data() + n * nbase;
        precondition_kernel<<<blocks(n * add), kThreads>>>(n, add, index_.data(), diagonal_.data(),
                                                          overlap_diagonal_.data(), scale_.data(), residual_.data(),
                                                          fresh);
        check_cuda(cudaGetLastError(), "the preconditioner");
        if (gamma_) gamma_clean(n, add, fresh);
        cuComplex* pfresh = general ? projection(pb, static_cast<std::size_t>(nbase)) : nullptr;
        if (general) h.project(fresh, add, pfresh);
        const int kept = orthonormalize(metric, basis_.data(), pb, nbase, fresh, pfresh, add, rows);
        if (kept == 0) return finish(full);
        apply(nbase, kept);
        grow(nbase, kept);
        nbase += kept;
    }
}

}  // namespace mdapy::dft
