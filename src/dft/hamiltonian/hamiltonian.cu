// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/hamiltonian/hamiltonian.h"

#include "dft/core/profile.h"
#include "dft/hamiltonian/gamma.h"
#include "dft/xc/dual.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace mdapy::dft {

namespace {

constexpr int kThreads = 256;
constexpr double kPi = 3.14159265358979323846264338327950288;

unsigned blocks(std::size_t work) {
    return static_cast<unsigned>((work + kThreads - 1) / kThreads);
}

// c = alpha op(a) op(b) + beta c, real, column-major
void dgemm(Blas& blas, Op ta, Op tb, int m, int n, int k, double alpha,
           const double* a, int lda, const double* b, int ldb, double beta, double* c, int ldc) {
    blas.gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

std::vector<double> real_parts(const std::vector<Complex>& c) {
    std::vector<double> out(c.size());
    for (std::size_t i = 0; i < c.size(); ++i) out[i] = c[i].x;
    return out;
}

double* real(Complex* p) { return reinterpret_cast<double*>(p); }
const double* real(const Complex* p) { return reinterpret_cast<const double*>(p); }
float* real(cuComplex* p) { return reinterpret_cast<float*>(p); }
const float* real(const cuComplex* p) { return reinterpret_cast<const float*>(p); }

void sgemm(Blas& blas, Op ta, Op tb, int m, int n, int k, float alpha,
           const float* a, int lda, const float* b, int ldb, float beta, float* c, int ldc) {
    blas.gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// The real spherical harmonics Y_lm of the direction of (x, y, z), l up to 3
// and m from -l to l: the sines of m < 0 and the cosines of m > 0; at the
// origin Y_00 and zero for l > 0.
__device__ double ylm(int l, int m, double x, double y, double z) {
    const double r2 = x * x + y * y + z * z;
    if (l == 0) return 0.28209479177387814347;  // 1 / (2 sqrt(pi))
    if (r2 < 1e-24) return 0.0;
    const double r = ::sqrt(r2);
    x /= r;
    y /= r;
    z /= r;
    if (l == 1) {
        const double c = 0.48860251190291992159;  // sqrt(3 / 4 pi)
        return c * (m == -1 ? y : (m == 0 ? z : x));
    }
    if (l == 2) {
        switch (m) {
            case -2: return 1.0925484305920790705 * x * y;
            case -1: return 1.0925484305920790705 * y * z;
            case 0: return 0.31539156525252000603 * (3.0 * z * z - 1.0);
            case 1: return 1.0925484305920790705 * x * z;
            default: return 0.54627421529603953527 * (x * x - y * y);
        }
    }
    switch (m) {
        case -3: return 0.59004358992664351035 * y * (3.0 * x * x - y * y);
        case -2: return 2.8906114426405540554 * x * y * z;
        case -1: return 0.45704579946446573616 * y * (5.0 * z * z - 1.0);
        case 0: return 0.37317633259011539141 * z * (5.0 * z * z - 3.0);
        case 1: return 0.45704579946446573616 * x * (5.0 * z * z - 1.0);
        case 2: return 1.4453057213202770277 * z * (x * x - y * y);
        default: return 0.59004358992664351035 * x * (x * x - 3.0 * y * y);
    }
}

// ylm of any number type (the duals of the stress), as ylm
template <class T>
__device__ T ylm_of(int l, int m, T x, T y, T z) {
    if (l == 0) return T(0.28209479177387814347);
    const T r = sqrt(x * x + y * y + z * z);
    x = x / r;
    y = y / r;
    z = z / r;
    if (l == 1) return 0.48860251190291992159 * (m == -1 ? y : (m == 0 ? z : x));
    if (l == 2) {
        switch (m) {
            case -2: return 1.0925484305920790705 * x * y;
            case -1: return 1.0925484305920790705 * y * z;
            case 0: return 0.31539156525252000603 * (3.0 * z * z - 1.0);
            case 1: return 1.0925484305920790705 * x * z;
            default: return 0.54627421529603953527 * (x * x - y * y);
        }
    }
    switch (m) {
        case -3: return 0.59004358992664351035 * y * (3.0 * x * x - y * y);
        case -2: return 2.8906114426405540554 * x * y * z;
        case -1: return 0.45704579946446573616 * y * (5.0 * z * z - 1.0);
        case 0: return 0.37317633259011539141 * z * (5.0 * z * z - 3.0);
        case 1: return 0.45704579946446573616 * x * (5.0 * z * z - 1.0);
        case 2: return 1.4453057213202770277 * z * (x * x - y * y);
        default: return 0.59004358992664351035 * x * (x * x - 3.0 * y * y);
    }
}

using Column = detail::ProjectorColumn;
using Reciprocal = detail::ReciprocalCell;

__global__ void projector_kernel(std::size_t n, int columns, const int* miller, Reciprocal rc,
                                 const Column* cols, const double* tables, Complex* beta) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * columns) return;
    const std::size_t g = i % n;
    const int c = static_cast<int>(i / n);
    const Column col = cols[c];
    const double f0 = rc.k[0] + miller[3 * g], f1 = rc.k[1] + miller[3 * g + 1],
                 f2 = rc.k[2] + miller[3 * g + 2];
    const double x = f0 * rc.b[0] + f1 * rc.b[3] + f2 * rc.b[6];
    const double y = f0 * rc.b[1] + f1 * rc.b[4] + f2 * rc.b[7];
    const double z = f0 * rc.b[2] + f1 * rc.b[5] + f2 * rc.b[8];
    const double q = ::sqrt(x * x + y * y + z * z);
    // Lagrange polynomials of four points from the one at or below q
    const double s = q / rc.dq;
    int i0 = static_cast<int>(floor(s));
    if (i0 > rc.points - 4) i0 = rc.points - 4;
    const double p = s - i0, u = 1.0 - p, v = 2.0 - p, w = 3.0 - p;
    const double* t = tables + col.table + i0;
    const double radial = t[0] * u * v * w / 6.0 + t[1] * p * v * w / 2.0 -
                          t[2] * p * u * w / 2.0 + t[3] * p * u * v / 6.0;
    const double amplitude = radial * ylm(col.l, col.m, x, y, z);
    // exp(-i (k+G).tau) times (-i)^l
    const double phase = -2.0 * kPi * (f0 * col.tau[0] + f1 * col.tau[1] + f2 * col.tau[2]);
    double re = cos(phase), im = sin(phase);
    for (int k = 0; k < col.l; ++k) {  // times -i
        const double old = re;
        re = im;
        im = -old;
    }
    beta[static_cast<std::size_t>(c) * n + g] = make_cuDoubleComplex(amplitude * re, amplitude * im);
}

// d beta / d epsilon_ab at a fixed set of plane waves, symmetrized:
//   -1/2 delta_ab beta - 1/2 (q_b du/dq_a + q_a du/dq_b) (-i)^l exp(-i q.tau),
// u = t(|q|) Y_lm(q), the derivatives by duals
__global__ void projector_strain_kernel(std::size_t n, int columns, const int* miller, Reciprocal rc,
                                        const Column* cols, const double* tables, int a, int b,
                                        Complex* dbeta) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * columns) return;
    const std::size_t g = i % n;
    const int c = static_cast<int>(i / n);
    const Column col = cols[c];
    const double f0 = rc.k[0] + miller[3 * g], f1 = rc.k[1] + miller[3 * g + 1],
                 f2 = rc.k[2] + miller[3 * g + 2];
    const double q[3] = {f0 * rc.b[0] + f1 * rc.b[3] + f2 * rc.b[6], f0 * rc.b[1] + f1 * rc.b[4] + f2 * rc.b[7],
                         f0 * rc.b[2] + f1 * rc.b[5] + f2 * rc.b[8]};
    const double modulus = ::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
    using D = Dual<3>;
    double u = 0.0, du[3] = {0.0, 0.0, 0.0};
    const double* t = tables + col.table;
    if (modulus < 1e-12) {  // q = 0: only l = 0 holds a value, and q_b du/dq_a vanishes
        u = col.l == 0 ? t[0] * 0.28209479177387814347 : 0.0;
    } else {
        const D x = D::input(q[0], 0), y = D::input(q[1], 1), z = D::input(q[2], 2);
        const D s = sqrt(x * x + y * y + z * z) / rc.dq;
        int i0 = static_cast<int>(floor(s.v));
        if (i0 > rc.points - 4) i0 = rc.points - 4;
        const D p = s - static_cast<double>(i0), uu = 1.0 - p, v = 2.0 - p, w = 3.0 - p;
        const D radial = t[i0] * uu * v * w / 6.0 + t[i0 + 1] * p * v * w / 2.0 -
                         t[i0 + 2] * p * uu * w / 2.0 + t[i0 + 3] * p * uu * v / 6.0;
        const D value = radial * ylm_of(col.l, col.m, x, y, z);
        u = value.v;
        for (int d = 0; d < 3; ++d) du[d] = value.d[d];
    }
    const double strain = -(a == b ? 0.5 * u : 0.0) - 0.5 * (q[b] * du[a] + q[a] * du[b]);
    const double phase = -2.0 * kPi * (f0 * col.tau[0] + f1 * col.tau[1] + f2 * col.tau[2]);
    double re = cos(phase), im = sin(phase);
    for (int k = 0; k < col.l; ++k) {
        const double old = re;
        re = im;
        im = -old;
    }
    dbeta[static_cast<std::size_t>(c) * n + g] = make_cuDoubleComplex(strain * re, strain * im);
}

// per band b: sum_G |psi|^2 q_a q_b for the six pairs (xx yy zz yz xz xy)
__global__ void kinetic_stress_kernel(std::size_t n, const double* q, const Complex* psi, double* out) {
    const Complex* column = psi + static_cast<std::size_t>(blockIdx.x) * n;
    double sum[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (std::size_t g = threadIdx.x; g < n; g += blockDim.x) {
        const double m = column[g].x * column[g].x + column[g].y * column[g].y;
        const double* x = q + 3 * g;
        sum[0] += m * x[0] * x[0];
        sum[1] += m * x[1] * x[1];
        sum[2] += m * x[2] * x[2];
        sum[3] += m * x[1] * x[2];
        sum[4] += m * x[0] * x[2];
        sum[5] += m * x[0] * x[1];
    }
    __shared__ double partial[6][kThreads];
    for (int d = 0; d < 6; ++d) partial[d][threadIdx.x] = sum[d];
    __syncthreads();
    for (int k = kThreads / 2; k > 0; k >>= 1) {
        if (static_cast<int>(threadIdx.x) < k)
            for (int d = 0; d < 6; ++d) partial[d][threadIdx.x] += partial[d][threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int d = 0; d < 6; ++d) out[6 * blockIdx.x + d] = partial[d][0];
}

// s_j = sum_b w_b Re(conj(a_jb) c_jb)
__global__ void projector_real_sums(int projectors, int count, const double* w, const Complex* a, const Complex* c,
                                    double* s) {
    const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= projectors) return;
    double sum = 0.0;
    for (int b = 0; b < count; ++b) {
        const std::size_t i = static_cast<std::size_t>(b) * projectors + j;
        sum += w[b] * (a[i].x * c[i].x + a[i].y * c[i].y);
    }
    s[j] = sum;
}

__global__ void kinetic_kernel(std::size_t n, int count, const double* kinetic, const Complex* psi,
                               Complex* hpsi) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    const double t = kinetic[i % n];
    hpsi[i] = make_cuDoubleComplex(t * psi[i].x, t * psi[i].y);
}

__global__ void multiply_kernel(std::size_t points, int count, const double* potential,
                                Complex* boxes) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= points * count) return;
    const double v = potential[i % points];
    boxes[i] = make_cuDoubleComplex(v * boxes[i].x, v * boxes[i].y);
}

__global__ void kinetic_single_kernel(std::size_t n, int count, const float* kinetic, const cuComplex* psi,
                                      cuComplex* hpsi) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    const float t = kinetic[i % n];
    hpsi[i] = make_cuComplex(t * psi[i].x, t * psi[i].y);
}

__global__ void multiply_single_kernel(std::size_t points, int count, const float* potential, cuComplex* boxes) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= points * count) return;
    const float v = potential[i % points];
    boxes[i] = make_cuComplex(v * boxes[i].x, v * boxes[i].y);
}

__global__ void multiply_real_single_kernel(std::size_t points, int count, const float* potential, float* values) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= points * count) return;
    values[i] *= potential[i % points];
}

__global__ void round_real_kernel(std::size_t n, const double* in, float* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = static_cast<float>(in[i]);
}

__global__ void round_complex_kernel(std::size_t n, const Complex* in, cuComplex* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = make_cuComplex(static_cast<float>(in[i].x), static_cast<float>(in[i].y));
}

// the diagonal of H: kinetic + mean potential + sum over the blocks of D of
// beta_i(G) D_ij conj(beta_j(G)); blocks (first, width) of each atom
__global__ void diagonal_kernel(std::size_t n, int atoms, const int* blocks, int projectors,
                                const double* kinetic, double mean, const Complex* beta,
                                const Complex* d, double* out) {
    const std::size_t g = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (g >= n) return;
    double sum = (kinetic == nullptr ? 0.0 : kinetic[g]) + mean;
    for (int a = 0; a < atoms; ++a) {
        const int first = blocks[2 * a], width = blocks[2 * a + 1];
        for (int j = first; j < first + width; ++j) {
            const Complex bj = beta[static_cast<std::size_t>(j) * n + g];
            for (int i = first; i < first + width; ++i) {
                const double dij = d[static_cast<std::size_t>(j) * projectors + i].x;
                if (dij == 0.0) continue;
                const Complex bi = beta[static_cast<std::size_t>(i) * n + g];
                sum += dij * (bi.x * bj.x + bi.y * bj.y);  // Re(b_i conj(b_j))
            }
        }
    }
    out[g] = sum;
}

// rho += sum_b w_b |box_b|^2 / Omega
__global__ void density_kernel(std::size_t points, int count, const Complex* boxes, const double* w,
                               double inverse_volume, double* rho) {
    const std::size_t r = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (r >= points) return;
    double sum = 0.0;
    for (int b = 0; b < count; ++b) {
        const Complex v = boxes[static_cast<std::size_t>(b) * points + r];
        sum += w[b] * (v.x * v.x + v.y * v.y);
    }
    rho[r] += sum * inverse_volume;
}

// out = q_d * psi, q (n, 3) the Cartesian k+G
__global__ void times_q_kernel(std::size_t n, int count, int d, const double* q, const Complex* psi, Complex* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    const double x = q[3 * (i % n) + static_cast<std::size_t>(d)];
    out[i] = make_cuDoubleComplex(x * psi[i].x, x * psi[i].y);
}

// s_j = sum_b w_b Im(conj(a_jb) c_jb), a and c (projectors, count)
__global__ void projector_sums(int projectors, int count, const double* w, const Complex* a, const Complex* c,
                               double* s) {
    const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= projectors) return;
    double sum = 0.0;
    for (int b = 0; b < count; ++b) {
        const std::size_t i = static_cast<std::size_t>(b) * projectors + j;
        sum += w[b] * (a[i].x * c[i].y - a[i].y * c[i].x);
    }
    s[j] = sum;
}

// out[j] of the packed pairs a <= b of each atom: Re c_aa, 2 Re c_ab, of the
// dense (projectors, projectors) c, added
__global__ void becsum_kernel(int atoms, const int* blocks, const std::size_t* offsets, int projectors,
                              const Complex* c, double* out) {
    const int a = static_cast<int>(blockIdx.x);
    if (a >= atoms) return;
    const int first = blocks[2 * a], width = blocks[2 * a + 1];
    const std::size_t base = offsets[a];
    for (int k = static_cast<int>(threadIdx.x); k < width * width; k += static_cast<int>(blockDim.x)) {
        const int i = k / width, j = k % width;
        if (j < i) continue;
        const int packed = i * width - i * (i - 1) / 2 + (j - i);  // pairs (i, j >= i) in order
        const double value = c[static_cast<std::size_t>(first + j) * projectors + (first + i)].x;
        out[base + static_cast<std::size_t>(packed)] += (i == j ? 1.0 : 2.0) * value;
    }
}

// dp[:, b] -= e_b qp[:, b]
__global__ void shift_kernel(int projectors, int count, const double* e, const Complex* qp, Complex* dp) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= static_cast<std::size_t>(projectors) * count) return;
    const double x = e[i / static_cast<std::size_t>(projectors)];
    dp[i] = make_cuDoubleComplex(dp[i].x - x * qp[i].x, dp[i].y - x * qp[i].y);
}

__global__ void sum_kernel(std::size_t count, const double* x, double* total) {
    __shared__ double partial[kThreads];
    double s = 0.0;
    for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < count;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
        s += x[i];
    partial[threadIdx.x] = s;
    __syncthreads();
    for (int k = kThreads / 2; k > 0; k >>= 1) {
        if (static_cast<int>(threadIdx.x) < k) partial[threadIdx.x] += partial[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(total, partial[0]);
}

__global__ void multiply_real_kernel(std::size_t points, int count, const double* potential, double* values) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= points * count) return;
    values[i] *= potential[i % points];
}

// rho += sum_b w_b psi_b(r)^2 / Omega of real bands
__global__ void density_real_kernel(std::size_t points, int count, const double* values, const double* w,
                                    double inverse_volume, double* rho) {
    const std::size_t r = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (r >= points) return;
    double sum = 0.0;
    for (int b = 0; b < count; ++b) {
        const double v = values[static_cast<std::size_t>(b) * points + r];
        sum += w[b] * v * v;
    }
    rho[r] += sum * inverse_volume;
}

// out = i q_d psi, q (n, 3) the Cartesian G
__global__ void times_iq_kernel(std::size_t n, int count, int d, const double* q, const Complex* psi, Complex* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    const double x = q[3 * (i % n) + static_cast<std::size_t>(d)];
    out[i] = make_cuDoubleComplex(-x * psi[i].y, x * psi[i].x);
}

// s_j = sum_b w_b a_jb c_jb, real (projectors, count)
__global__ void real_sums_kernel(int projectors, int count, const double* w, const double* a, const double* c,
                                 double* s) {
    const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (j >= projectors) return;
    double sum = 0.0;
    for (int b = 0; b < count; ++b) {
        const std::size_t i = static_cast<std::size_t>(b) * projectors + j;
        sum += w[b] * a[i] * c[i];
    }
    s[j] = sum;
}

// dp[:, b] -= e_b qp[:, b], real
__global__ void shift_real_kernel(int projectors, int count, const double* e, const double* qp, double* dp) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= static_cast<std::size_t>(projectors) * count) return;
    dp[i] -= e[i / static_cast<std::size_t>(projectors)] * qp[i];
}

// out[:, b] = w_b p[:, b], (rows, count)
__global__ void weigh_columns(std::size_t rows, int count, const double* w, const Complex* p, Complex* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= rows * count) return;
    const double x = w[i / rows];
    out[i] = make_cuDoubleComplex(x * p[i].x, x * p[i].y);
}

__global__ void weigh_real_columns(std::size_t rows, int count, const double* w, const double* p, double* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= rows * count) return;
    out[i] = w[i / rows] * p[i];
}

// becsum_kernel of a real c
__global__ void becsum_real_kernel(int atoms, const int* blocks, const std::size_t* offsets, int projectors,
                                   const double* c, double* out) {
    const int a = static_cast<int>(blockIdx.x);
    if (a >= atoms) return;
    const int first = blocks[2 * a], width = blocks[2 * a + 1];
    const std::size_t base = offsets[a];
    for (int k = static_cast<int>(threadIdx.x); k < width * width; k += static_cast<int>(blockDim.x)) {
        const int i = k / width, j = k % width;
        if (j < i) continue;
        const int packed = i * width - i * (i - 1) / 2 + (j - i);
        const double value = c[static_cast<std::size_t>(first + j) * projectors + (first + i)];
        out[base + static_cast<std::size_t>(packed)] += (i == j ? 1.0 : 2.0) * value;
    }
}

}  // namespace

Hamiltonian::Hamiltonian(const Lattice& lattice, const Int3& grid, double cutoff,
                         std::vector<SpeciesProjectors> species, std::vector<Atom> atoms, const Lattice* basis,
                         Kernels kernels)
    : lattice_(lattice),
      has_basis_(basis != nullptr),
      basis_(basis != nullptr ? *basis : lattice),
      grid_(grid),
      cutoff_(cutoff),
      species_(std::move(species)),
      atoms_(std::move(atoms)),
      fft_(grid, kernels),
      blas_(kernels) {
    // the tables of every species, one after the other
    std::vector<double> all;
    for (const auto& s : species_) {
        if (s.angular.size() * static_cast<std::size_t>(s.points) != s.tables.size() ||
            s.dij.size() != s.angular.size() * s.angular.size())
            throw std::invalid_argument("the tables, angular momenta and D of a species disagree");
        for (const int l : s.angular)
            if (l < 0 || l > 3) throw std::invalid_argument("projectors of l from 0 to 3 only");
        if (!s.angular.empty() && s.points < 4)
            throw std::invalid_argument("a table needs four points or more");
        all.insert(all.end(), s.tables.begin(), s.tables.end());
    }
    tables_.upload(all.data(), all.size());
    // the pseudo-orbitals, likewise
    std::vector<double> orbitals;
    for (const auto& s : species_) {
        if (s.orbital_angular.size() * static_cast<std::size_t>(s.points) != s.orbital_tables.size())
            throw std::invalid_argument("the tables and angular momenta of the orbitals of a species disagree");
        for (const int l : s.orbital_angular)
            if (l < 0 || l > 3) throw std::invalid_argument("orbitals of l from 0 to 3 only");
        orbitals.insert(orbitals.end(), s.orbital_tables.begin(), s.orbital_tables.end());
    }
    orbital_tables_.upload(orbitals.data(), orbitals.size());
    for (const Atom& a : atoms_)
        if (a.species >= 0 && a.species < static_cast<int>(species_.size()))
            for (const int l : species_[static_cast<std::size_t>(a.species)].orbital_angular) atomic_ += 2 * l + 1;
    // the projectors of every atom and D, block diagonal
    for (const Atom& a : atoms_) {
        if (a.species < 0 || a.species >= static_cast<int>(species_.size()))
            throw std::invalid_argument("an atom of a species not given");
        for (const int l : species_[static_cast<std::size_t>(a.species)].angular) projectors_ += 2 * l + 1;
    }
    std::vector<Complex> d(static_cast<std::size_t>(projectors_) * projectors_,
                           make_cuDoubleComplex(0.0, 0.0));
    int start = 0;
    for (const Atom& a : atoms_) {
        const auto& s = species_[static_cast<std::size_t>(a.species)];
        const int count = static_cast<int>(s.angular.size());
        std::vector<int> first(static_cast<std::size_t>(count));
        int width = 0;
        for (int p = 0; p < count; ++p) {
            first[static_cast<std::size_t>(p)] = start + width;
            width += 2 * s.angular[static_cast<std::size_t>(p)] + 1;
        }
        for (int p = 0; p < count; ++p)
            for (int q = 0; q < count; ++q) {
                const int l = s.angular[static_cast<std::size_t>(p)];
                if (l != s.angular[static_cast<std::size_t>(q)]) continue;
                const double value = s.dij[static_cast<std::size_t>(p * count + q)];
                for (int m = 0; m < 2 * l + 1; ++m) {
                    const std::size_t row = static_cast<std::size_t>(first[static_cast<std::size_t>(p)] + m);
                    const std::size_t column = static_cast<std::size_t>(first[static_cast<std::size_t>(q)] + m);
                    d[column * static_cast<std::size_t>(projectors_) + row] = make_cuDoubleComplex(value, 0.0);
                }
            }
        blocks_.push_back(start);
        blocks_.push_back(width);
        start += width;
    }
    dmatrix_.upload(d.data(), d.size());
    const auto dr = real_parts(d);
    dreal_.upload(dr.data(), dr.size());
    blocks_device_.upload(blocks_.data(), blocks_.size());
}

Hamiltonian::~Hamiltonian() = default;

void Hamiltonian::set_k(const Vec3& k) {
    MDAPY_SPAN("set_k");
    if (gamma_mode_ && (k[0] != 0.0 || k[1] != 0.0 || k[2] != 0.0))
        throw std::invalid_argument("real bands are those of Gamma only");
    gamma_ = gamma_mode_;
    single_stale_ = true;
    sphere_ = make_sphere(lattice_, has_basis_ ? &basis_ : nullptr, k, cutoff_, grid_, gamma_);
    const std::size_t n = sphere_.size();
    kinetic_.assign(sphere_.kinetic.data(), n);
    box_.assign(sphere_.box.data(), n);
    if (gamma_) partner_.assign(sphere_.partner.data(), n);
    beta_.reserve(n * static_cast<std::size_t>(projectors_));
    // the Cartesian k+G of the plane waves
    std::vector<double> q(3 * n);
    for (std::size_t g = 0; g < n; ++g)
        for (int d = 0; d < 3; ++d) {
            double x = 0.0;
            for (int j = 0; j < 3; ++j)
                x += (k[static_cast<std::size_t>(j)] + sphere_.miller[g][static_cast<std::size_t>(j)]) *
                     lattice_.b[static_cast<std::size_t>(j)][static_cast<std::size_t>(d)];
            q[3 * g + static_cast<std::size_t>(d)] = x;
        }
    q_.assign(q.data(), q.size());
    std::vector<int> miller(3 * n);
    for (std::size_t g = 0; g < n; ++g)
        for (int i = 0; i < 3; ++i) miller[3 * g + i] = sphere_.miller[g][i];
    DeviceArray<int>& m = miller_;
    m.assign(miller.data(), miller.size());
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            reciprocal_.b[3 * i + j] = lattice_.b[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        reciprocal_.k[i] = k[static_cast<std::size_t>(i)];
    }
    if (!species_.empty()) {
        reciprocal_.dq = species_.front().dq;
        reciprocal_.points = species_.front().points;
    }
    if (projectors_ == 0 || n == 0) return;
    std::vector<Column> columns;
    std::vector<std::int64_t> starts;  // of the tables of each species
    std::int64_t offset = 0;
    for (const auto& s : species_) {
        starts.push_back(offset);
        offset += static_cast<std::int64_t>(s.tables.size());
    }
    for (const Atom& a : atoms_) {
        const auto& s = species_[static_cast<std::size_t>(a.species)];
        for (std::size_t p = 0; p < s.angular.size(); ++p) {
            const int l = s.angular[p];
            for (int mm = -l; mm <= l; ++mm)
                columns.push_back({starts[static_cast<std::size_t>(a.species)] +
                                       static_cast<std::int64_t>(p) * s.points,
                                   l, mm, {a.position[0], a.position[1], a.position[2]}});
        }
    }
    DeviceArray<Column>& cols = columns_;
    cols.assign(columns.data(), columns.size());
    Reciprocal rc{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) rc.b[3 * i + j] = lattice_.b[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        rc.k[i] = k[static_cast<std::size_t>(i)];
    }
    // every species shares dq and the number of points of its tables
    rc.dq = species_.front().dq;
    rc.points = species_.front().points;
    for (const auto& s : species_)
        if (s.dq != rc.dq || (!s.angular.empty() && s.points != rc.points))
            throw std::invalid_argument("every species needs tables of the same steps and points");
    // the tables reach every |k+G| of the sphere
    const double reach = (rc.points - 4) * rc.dq;
    if (!(std::sqrt(2.0 * *std::max_element(sphere_.kinetic.begin(), sphere_.kinetic.end())) <= reach))
        throw std::invalid_argument("the tables of the projectors stop short of the sphere");
    projector_kernel<<<blocks(n * columns.size()), kThreads>>>(
        n, static_cast<int>(columns.size()), m.data(), rc, cols.data(), tables_.data(), beta_.data());
    check_cuda(cudaGetLastError(), "the projectors");
    reciprocal_ = rc;
}

void Hamiltonian::atomic(Complex* out) const {
    MDAPY_SPAN("atomic");
    const std::size_t n = sphere_.size();
    if (atomic_ == 0 || n == 0) return;
    std::vector<Column> columns;
    std::vector<std::int64_t> starts;
    std::int64_t offset = 0;
    for (const auto& s : species_) {
        starts.push_back(offset);
        offset += static_cast<std::int64_t>(s.orbital_tables.size());
    }
    for (const Atom& a : atoms_) {
        const auto& s = species_[static_cast<std::size_t>(a.species)];
        for (std::size_t o = 0; o < s.orbital_angular.size(); ++o) {
            const int l = s.orbital_angular[o];
            for (int mm = -l; mm <= l; ++mm)
                columns.push_back({starts[static_cast<std::size_t>(a.species)] + static_cast<std::int64_t>(o) * s.points,
                                   l, mm, {a.position[0], a.position[1], a.position[2]}});
        }
    }
    const double reach = (reciprocal_.points - 4) * reciprocal_.dq;
    if (!(std::sqrt(2.0 * *std::max_element(sphere_.kinetic.begin(), sphere_.kinetic.end())) <= reach))
        throw std::invalid_argument("the tables of the orbitals stop short of the sphere");
    DeviceArray<Column> cols;
    cols.assign(columns.data(), columns.size());
    projector_kernel<<<blocks(n * columns.size()), kThreads>>>(n, static_cast<int>(columns.size()), miller_.data(),
                                                              reciprocal_, cols.data(), orbital_tables_.data(), out);
    check_cuda(cudaGetLastError(), "the atomic orbitals");
}

void Hamiltonian::set_potential(const double* potential) {
    MDAPY_SPAN("set_potential");
    single_stale_ = true;
    const std::size_t points = fft_.points();
    potential_.resize(points);
    check_cuda(cudaMemcpy(potential_.data(), potential, points * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "cudaMemcpy of the potential");
    DeviceArray<double>& total = total_;
    total.reserve(1);
    total.zero();
    sum_kernel<<<std::min<unsigned>(blocks(points), 1024u), kThreads>>>(points, potential_.data(), total.data());
    check_cuda(cudaGetLastError(), "the mean potential");
    total.download(&mean_potential_, 1);
    mean_potential_ /= static_cast<double>(points);
}

void Hamiltonian::add_density(const Complex* psi, int count, const double* weights, double* rho) {
    MDAPY_SPAN("add_density");
    const std::size_t n = sphere_.size(), points = fft_.points();
    if (n == 0 || count == 0) return;
    const int step = std::min(batch(), count);
    boxes_.reserve(points * static_cast<std::size_t>(step));
    weights_.assign(weights, static_cast<std::size_t>(count));
    if (gamma_) {  // real bands: rho += w psi(r)^2
        const std::size_t spectrum = fft_.spectrum();
        values_.reserve(points * static_cast<std::size_t>(step));
        for (int first = 0; first < count; first += step) {
            const int m = std::min(step, count - first);
            scatter(psi + n * static_cast<std::size_t>(first), n, box_.data(), partner_.data(), m, boxes_.data(),
                    spectrum);
            fft_.to_real(boxes_.data(), values_.data(), m);
            density_real_kernel<<<blocks(points), kThreads>>>(points, m, values_.data(), weights_.data() + first,
                                                              1.0 / lattice_.volume, rho);
            check_cuda(cudaGetLastError(), "the density of the bands");
        }
        return;
    }
    for (int first = 0; first < count; first += step) {
        const int m = std::min(step, count - first);
        scatter(psi + n * static_cast<std::size_t>(first), n, box_.data(), nullptr, m, boxes_.data(), points);
        fft_.to_real(boxes_.data(), m);
        density_kernel<<<blocks(points), kThreads>>>(points, m, boxes_.data(), weights_.data() + first,
                                                     1.0 / lattice_.volume, rho);
        check_cuda(cudaGetLastError(), "the density of the bands");
    }
}

void Hamiltonian::nonlocal_forces(const Complex* psi, int count, const double* weights, std::vector<double>& forces,
                                  const double* eigenvalues) {
    const std::size_t n = sphere_.size(), np = static_cast<std::size_t>(projectors_);
    if (n == 0 || count == 0 || projectors_ == 0) return;
    if (forces.size() != 3 * atoms_.size()) throw std::invalid_argument("forces are (atoms, 3)");
    const std::size_t m = static_cast<std::size_t>(count);
    if (gamma_) {
        // real projections; <beta|i q psi> is minus the imaginary part of
        // <beta|q psi> over the whole sphere, so that s_j = -sum_b w_b dp qp
        DeviceArray<double> p(np * m), dp(np * m), qp(np * m), w(m), s(np);
        DeviceArray<Complex> qpsi(n * m);
        w.upload(weights, m);
        gamma_inner(blas_, n, beta_.data(), projectors_, psi, count, p.data(), projectors_);
        dgemm(blas_, Op::N, Op::N, projectors_, count, projectors_, 1.0, dreal_.data(), projectors_,
              p.data(), projectors_, 0.0, dp.data(), projectors_);
        if (has_overlap_) {
            if (eigenvalues == nullptr)
                throw std::invalid_argument("the forces of ultrasoft bands need their eigenvalues");
            DeviceArray<double> qpp(np * m), e;
            dgemm(blas_, Op::N, Op::N, projectors_, count, projectors_, 1.0, qreal_.data(), projectors_,
                  p.data(), projectors_, 0.0, qpp.data(), projectors_);
            e.upload(eigenvalues, m);
            shift_real_kernel<<<blocks(np * m), kThreads>>>(projectors_, count, e.data(), qpp.data(), dp.data());
            check_cuda(cudaGetLastError(), "D - e q");
        }
        std::vector<double> sums(np);
        for (int d = 0; d < 3; ++d) {
            times_iq_kernel<<<blocks(n * m), kThreads>>>(n, count, d, q_.data(), psi, qpsi.data());
            check_cuda(cudaGetLastError(), "i G times the bands");
            gamma_inner(blas_, n, beta_.data(), projectors_, qpsi.data(), count, qp.data(), projectors_);
            real_sums_kernel<<<blocks(np), kThreads>>>(projectors_, count, w.data(), dp.data(), qp.data(), s.data());
            check_cuda(cudaGetLastError(), "the sums of the projectors");
            s.download(sums.data(), np);
            for (std::size_t a = 0; a < atoms_.size(); ++a) {
                double total = 0.0;
                for (int j = blocks_[2 * a]; j < blocks_[2 * a] + blocks_[2 * a + 1]; ++j)
                    total += sums[static_cast<std::size_t>(j)];
                forces[3 * a + static_cast<std::size_t>(d)] -= 2.0 * total;
            }
        }
        return;
    }
    DeviceArray<Complex> p(np * m), dp(np * m), qpsi(n * m), qp(np * m);
    DeviceArray<double> w, s(np);
    w.upload(weights, m);
    const Complex one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
    const int nn = static_cast<int>(n);
    blas_.gemm(Op::C, Op::N, projectors_, count, nn, one, beta_.data(), nn, psi, nn, zero, p.data(), projectors_);
    blas_.gemm(Op::N, Op::N, projectors_, count, projectors_, one, dmatrix_.data(), projectors_, p.data(), projectors_, zero, dp.data(), projectors_);
    if (has_overlap_) {
        // (D - e q) P
        if (eigenvalues == nullptr) throw std::invalid_argument("the forces of ultrasoft bands need their eigenvalues");
        DeviceArray<Complex> qpp(np * m);
        blas_.gemm(Op::N, Op::N, projectors_, count, projectors_, one, qmatrix_.data(), projectors_, p.data(), projectors_, zero, qpp.data(), projectors_);
        DeviceArray<double> e;
        e.upload(eigenvalues, m);
        shift_kernel<<<blocks(np * m), kThreads>>>(projectors_, count, e.data(), qpp.data(), dp.data());
        check_cuda(cudaGetLastError(), "D - e q");
    }
    std::vector<double> sums(np);
    for (int d = 0; d < 3; ++d) {
        times_q_kernel<<<blocks(n * m), kThreads>>>(n, count, d, q_.data(), psi, qpsi.data());
        check_cuda(cudaGetLastError(), "k+G times the bands");
        blas_.gemm(Op::C, Op::N, projectors_, count, nn, one, beta_.data(), nn, qpsi.data(), nn, zero, qp.data(), projectors_);
        projector_sums<<<blocks(np), kThreads>>>(projectors_, count, w.data(), dp.data(), qp.data(), s.data());
        check_cuda(cudaGetLastError(), "the sums of the projectors");
        s.download(sums.data(), np);
        // F = -dE/dtau = 2 sum_j of the atom s_j
        for (std::size_t a = 0; a < atoms_.size(); ++a) {
            double total = 0.0;
            for (int j = blocks_[2 * a]; j < blocks_[2 * a] + blocks_[2 * a + 1]; ++j)
                total += sums[static_cast<std::size_t>(j)];
            forces[3 * a + static_cast<std::size_t>(d)] += 2.0 * total;
        }
    }
}

void Hamiltonian::strain_derivative(const Complex* psi, int count, const double* weights,
                                    std::array<double, 6>& out, const double* eigenvalues) {
    const std::size_t n = sphere_.size(), np = static_cast<std::size_t>(projectors_), m = static_cast<std::size_t>(count);
    if (n == 0 || count == 0) return;
    // the kinetic energy: -sum_b w_b sum_G |psi|^2 q_a q_b
    DeviceArray<double> sums(6 * m);
    kinetic_stress_kernel<<<static_cast<unsigned>(count), kThreads>>>(n, q_.data(), psi, sums.data());
    check_cuda(cudaGetLastError(), "the kinetic stress");
    std::vector<double> host(6 * m);
    sums.download(host.data(), host.size());
    // the half sphere at Gamma counts twice, G = 0 holding nothing here
    const double half = gamma_ ? 2.0 : 1.0;
    for (std::size_t b = 0; b < m; ++b)
        for (int d = 0; d < 6; ++d)
            out[static_cast<std::size_t>(d)] -= half * weights[b] * host[6 * b + static_cast<std::size_t>(d)];
    if (projectors_ == 0) return;
    if (gamma_) {
        DeviceArray<double> p(np * m), dp(np * m), dpsi(np * m), w(m), s(np);
        DeviceArray<Complex> dbeta(n * np);
        w.upload(weights, m);
        gamma_inner(blas_, n, beta_.data(), projectors_, psi, count, p.data(), projectors_);
        dgemm(blas_, Op::N, Op::N, projectors_, count, projectors_, 1.0, dreal_.data(), projectors_,
              p.data(), projectors_, 0.0, dp.data(), projectors_);
        if (has_overlap_) {
            if (eigenvalues == nullptr)
                throw std::invalid_argument("the stress of ultrasoft bands needs their eigenvalues");
            DeviceArray<double> qpp(np * m), e;
            dgemm(blas_, Op::N, Op::N, projectors_, count, projectors_, 1.0, qreal_.data(), projectors_,
                  p.data(), projectors_, 0.0, qpp.data(), projectors_);
            e.upload(eigenvalues, m);
            shift_real_kernel<<<blocks(np * m), kThreads>>>(projectors_, count, e.data(), qpp.data(), dp.data());
            check_cuda(cudaGetLastError(), "D - e q");
        }
        const int pairs[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};
        std::vector<double> values(np);
        for (int d = 0; d < 6; ++d) {
            projector_strain_kernel<<<blocks(n * np), kThreads>>>(n, projectors_, miller_.data(), reciprocal_,
                                                                  columns_.data(), tables_.data(), pairs[d][0],
                                                                  pairs[d][1], dbeta.data());
            check_cuda(cudaGetLastError(), "the strain of the projectors");
            gamma_inner(blas_, n, dbeta.data(), projectors_, psi, count, dpsi.data(), projectors_);
            real_sums_kernel<<<blocks(np), kThreads>>>(projectors_, count, w.data(), dp.data(), dpsi.data(), s.data());
            check_cuda(cudaGetLastError(), "the sums of the projectors");
            s.download(values.data(), np);
            double total = 0.0;
            for (const double x : values) total += x;
            out[static_cast<std::size_t>(d)] += 2.0 * total;
        }
        return;
    }
    // the projectors: 2 sum_b w_b Re sum_j conj(D P)_jb dP_jb, dP = dbeta^H psi
    DeviceArray<Complex> p(np * m), dp(np * m), dbeta(n * np), dpsi(np * m);
    DeviceArray<double> w, s(np);
    w.upload(weights, m);
    const Complex one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
    const int nn = static_cast<int>(n);
    blas_.gemm(Op::C, Op::N, projectors_, count, nn, one, beta_.data(), nn, psi, nn, zero, p.data(), projectors_);
    blas_.gemm(Op::N, Op::N, projectors_, count, projectors_, one, dmatrix_.data(), projectors_, p.data(), projectors_, zero, dp.data(), projectors_);
    if (has_overlap_) {  // (D - e q) P
        if (eigenvalues == nullptr) throw std::invalid_argument("the stress of ultrasoft bands needs their eigenvalues");
        DeviceArray<Complex> qpp(np * m);
        blas_.gemm(Op::N, Op::N, projectors_, count, projectors_, one, qmatrix_.data(), projectors_, p.data(), projectors_, zero, qpp.data(), projectors_);
        DeviceArray<double> e;
        e.upload(eigenvalues, m);
        shift_kernel<<<blocks(np * m), kThreads>>>(projectors_, count, e.data(), qpp.data(), dp.data());
        check_cuda(cudaGetLastError(), "D - e q");
    }
    const Reciprocal rc = reciprocal_;
    const int pairs[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};
    std::vector<double> values(np);
    for (int d = 0; d < 6; ++d) {
        projector_strain_kernel<<<blocks(n * np), kThreads>>>(n, projectors_, miller_.data(), rc, columns_.data(),
                                                              tables_.data(), pairs[d][0], pairs[d][1], dbeta.data());
        check_cuda(cudaGetLastError(), "the strain of the projectors");
        blas_.gemm(Op::C, Op::N, projectors_, count, nn, one, dbeta.data(), nn, psi, nn, zero, dpsi.data(), projectors_);
        projector_real_sums<<<blocks(np), kThreads>>>(projectors_, count, w.data(), dp.data(), dpsi.data(), s.data());
        check_cuda(cudaGetLastError(), "the sums of the projectors");
        s.download(values.data(), np);
        double total = 0.0;
        for (const double x : values) total += x;
        out[static_cast<std::size_t>(d)] += 2.0 * total;
    }
}

void Hamiltonian::diagonal(double* out) const {
    MDAPY_SPAN("diagonal");
    const std::size_t n = sphere_.size();
    if (n == 0) return;
    if (potential_.size() != fft_.points()) throw std::logic_error("no potential set");
    diagonal_kernel<<<blocks(n), kThreads>>>(n, static_cast<int>(atoms_.size()), blocks_device_.data(), projectors_,
                                             kinetic_.data(), mean_potential_, beta_.data(), dmatrix_.data(), out);
    check_cuda(cudaGetLastError(), "the diagonal of H");
}

void Hamiltonian::overlap_diagonal(double* out) const {
    MDAPY_SPAN("diagonal");
    const std::size_t n = sphere_.size();
    if (n == 0) return;
    if (!has_overlap_) {
        std::vector<double> ones(n, 1.0);
        check_cuda(cudaMemcpy(out, ones.data(), n * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy");
        return;
    }
    diagonal_kernel<<<blocks(n), kThreads>>>(n, static_cast<int>(atoms_.size()), blocks_device_.data(), projectors_,
                                             nullptr, 1.0, beta_.data(), qmatrix_.data(), out);
    check_cuda(cudaGetLastError(), "the diagonal of S");
}

int Hamiltonian::batch() const noexcept {
    // bands per batch of FFTs: boxes of at most 256 MB
    const std::size_t points = fft_.points();
    const std::size_t most = (std::size_t{256} << 20) / (points * sizeof(Complex));
    return static_cast<int>(std::max<std::size_t>(1, std::min<std::size_t>(most, 1u << 20)));
}

void Hamiltonian::apply_overlap(const Complex* psi, Complex* spsi, int count) {
    const std::size_t n = sphere_.size();
    if (n == 0 || count == 0) return;
    check_cuda(cudaMemcpy(spsi, psi, n * static_cast<std::size_t>(count) * sizeof(Complex), cudaMemcpyDeviceToDevice),
               "cudaMemcpy");
    if (!has_overlap_ || projectors_ == 0) return;
    const std::size_t np = static_cast<std::size_t>(projectors_);
    projections_.reserve(np * static_cast<std::size_t>(count));
    if (gamma_) {
        project(psi, count, projections_.data());
        add_overlap(projections_.data(), count, spsi);
        return;
    }
    scaled_.reserve(np * static_cast<std::size_t>(count));
    const Complex one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
    const int nn = static_cast<int>(n);
    blas_.gemm(Op::C, Op::N, projectors_, count, nn, one, beta_.data(), nn, psi, nn, zero, projections_.data(), projectors_);
    blas_.gemm(Op::N, Op::N, projectors_, count, projectors_, one, qmatrix_.data(), projectors_, projections_.data(), projectors_, zero, scaled_.data(), projectors_);
    blas_.gemm(Op::N, Op::N, nn, count, projectors_, one, beta_.data(), nn, scaled_.data(), projectors_, one, spsi, nn);
}

namespace {
// the dense matrix of blocks per atom, (width, width) row by row each
std::vector<Complex> dense_of(const std::vector<int>& blocks, int projectors, const std::vector<double>& values) {
    std::vector<Complex> d(static_cast<std::size_t>(projectors) * projectors, make_cuDoubleComplex(0.0, 0.0));
    std::size_t k = 0;
    for (std::size_t a = 0; a < blocks.size() / 2; ++a) {
        const int first = blocks[2 * a], width = blocks[2 * a + 1];
        for (int i = 0; i < width; ++i)
            for (int j = 0; j < width; ++j, ++k) {
                if (k >= values.size()) throw std::invalid_argument("too few values for the blocks of the atoms");
                d[static_cast<std::size_t>(first + j) * projectors + (first + i)] = make_cuDoubleComplex(values[k], 0.0);
            }
    }
    if (k != values.size()) throw std::invalid_argument("the blocks of the atoms take other sizes");
    return d;
}
}  // namespace

void Hamiltonian::set_strengths(const std::vector<double>& blocks) {
    MDAPY_SPAN("set_strengths");
    single_stale_ = true;
    const auto d = dense_of(blocks_, projectors_, blocks);
    dmatrix_.assign(d.data(), d.size());
    const auto r = real_parts(d);
    dreal_.assign(r.data(), r.size());
}

void Hamiltonian::set_overlaps(const std::vector<double>& blocks) {
    single_stale_ = true;
    const auto q = dense_of(blocks_, projectors_, blocks);
    qmatrix_.assign(q.data(), q.size());
    const auto r = real_parts(q);
    qreal_.assign(r.data(), r.size());
    has_overlap_ = true;
}

std::size_t Hamiltonian::becsum_size() const {
    std::size_t total = 0;
    for (std::size_t a = 0; a < blocks_.size() / 2; ++a) {
        const std::size_t w = static_cast<std::size_t>(blocks_[2 * a + 1]);
        total += w * (w + 1) / 2;
    }
    return total;
}

void Hamiltonian::add_becsum(const Complex* psi, int count, const double* weights, std::vector<double>& becsum) {
    if (becsum.size() != becsum_size()) throw std::invalid_argument("becsum holds the pairs of each atom");
    DeviceArray<double> out;
    out.upload(becsum.data(), becsum.size());
    add_becsum_device(psi, count, weights, out.data());
    out.download(becsum.data(), becsum.size());
}

void Hamiltonian::add_becsum_device(const Complex* psi, int count, const double* weights, double* becsum) {
    MDAPY_SPAN("add_becsum");
    const std::size_t n = sphere_.size(), np = static_cast<std::size_t>(projectors_), m = static_cast<std::size_t>(count);
    if (n == 0 || count == 0 || projectors_ == 0) return;
    if (becsum_offsets_.size() != atoms_.size()) {  // where the pairs of each atom start
        std::vector<std::size_t> offsets;
        std::size_t total = 0;
        for (std::size_t a = 0; a < blocks_.size() / 2; ++a) {
            offsets.push_back(total);
            const std::size_t w = static_cast<std::size_t>(blocks_[2 * a + 1]);
            total += w * (w + 1) / 2;
        }
        becsum_offsets_.upload(offsets.data(), offsets.size());
    }
    weights_.assign(weights, m);
    const unsigned atoms = static_cast<unsigned>(atoms_.size());
    if (gamma_) {  // c = P W P^T, real
        DeviceArray<double> pr(np * m), pw(np * m), c(np * np);
        gamma_inner(blas_, n, beta_.data(), projectors_, psi, count, pr.data(), projectors_);
        weigh_real_columns<<<blocks(np * m), kThreads>>>(np, count, weights_.data(), pr.data(), pw.data());
        check_cuda(cudaGetLastError(), "the weighted projections");
        dgemm(blas_, Op::N, Op::T, projectors_, projectors_, count, 1.0, pw.data(), projectors_,
              pr.data(), projectors_, 0.0, c.data(), projectors_);
        becsum_real_kernel<<<atoms, kThreads>>>(static_cast<int>(atoms), blocks_device_.data(),
                                                becsum_offsets_.data(), projectors_, c.data(), becsum);
        check_cuda(cudaGetLastError(), "becsum");
        return;
    }
    const Complex one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
    const int nn = static_cast<int>(n);
    DeviceArray<Complex> pc(np * m), pw(np * m), c(np * np);
    blas_.gemm(Op::C, Op::N, projectors_, count, nn, one, beta_.data(), nn, psi, nn, zero, pc.data(), projectors_);
    // c = P W P^H, c_ab = sum_n w_n P_an conj(P_bn), of real part becsum_ab
    weigh_columns<<<blocks(np * m), kThreads>>>(np, count, weights_.data(), pc.data(), pw.data());
    check_cuda(cudaGetLastError(), "the weighted projections");
    blas_.gemm(Op::N, Op::C, projectors_, projectors_, count, one, pw.data(), projectors_, pc.data(), projectors_, zero, c.data(), projectors_);
    becsum_kernel<<<atoms, kThreads>>>(static_cast<int>(atoms), blocks_device_.data(), becsum_offsets_.data(),
                                       projectors_, c.data(), becsum);
    check_cuda(cudaGetLastError(), "becsum");
}

void Hamiltonian::apply_local(const Complex* psi, Complex* hpsi, int count) {
    MDAPY_SPAN("apply_local");
    const std::size_t n = sphere_.size(), points = fft_.points();
    if (n == 0 || count == 0) return;
    if (potential_.size() != points) throw std::logic_error("no potential set");
    kinetic_kernel<<<blocks(n * count), kThreads>>>(n, count, kinetic_.data(), psi, hpsi);
    check_cuda(cudaGetLastError(), "the kinetic energy");
    // the local potential on the grid, in batches of bands
    const int step = std::min(batch(), count);
    boxes_.reserve(points * static_cast<std::size_t>(step));
    if (gamma_) {  // real transforms of real bands
        const std::size_t spectrum = fft_.spectrum();
        values_.reserve(points * static_cast<std::size_t>(step));
        for (int first = 0; first < count; first += step) {
            const int m = std::min(step, count - first);
            const std::size_t offset = n * static_cast<std::size_t>(first);
            scatter(psi + offset, n, box_.data(), partner_.data(), m, boxes_.data(), spectrum);
            fft_.to_real(boxes_.data(), values_.data(), m);
            multiply_real_kernel<<<blocks(points * m), kThreads>>>(points, m, potential_.data(), values_.data());
            check_cuda(cudaGetLastError(), "the local potential");
            fft_.to_reciprocal(values_.data(), boxes_.data(), m);
            gather_add(boxes_.data(), spectrum, box_.data(), n, m, 1.0 / static_cast<double>(points), hpsi + offset);
        }
        gamma_clean(n, count, hpsi);
        return;
    }
    for (int first = 0; first < count; first += step) {
        const int m = std::min(step, count - first);
        const std::size_t offset = n * static_cast<std::size_t>(first);
        scatter(psi + offset, n, box_.data(), nullptr, m, boxes_.data(), points);
        fft_.to_real(boxes_.data(), m);
        multiply_kernel<<<blocks(points * m), kThreads>>>(points, m, potential_.data(), boxes_.data());
        check_cuda(cudaGetLastError(), "the local potential");
        fft_.to_reciprocal(boxes_.data(), m);
        gather_add(boxes_.data(), points, box_.data(), n, m, 1.0 / static_cast<double>(points), hpsi + offset);
    }
}

void Hamiltonian::project(const Complex* psi, int count, Complex* p) {
    MDAPY_SPAN("project");
    if (projectors_ == 0 || count == 0) return;
    const Complex one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
    const int nn = static_cast<int>(sphere_.size());
    if (gamma_) {
        gamma_inner(blas_, sphere_.size(), beta_.data(), projectors_, psi, count, real(p), projectors_);
        return;
    }
    blas_.gemm(Op::C, Op::N, projectors_, count, nn, one, beta_.data(), nn, psi, nn, zero, p, projectors_);
}

namespace {
// out += beta (m p), m (projectors, projectors) on the device
void add_projected(Blas& blas, const Complex* beta, int n, int projectors, const Complex* m,
                   const Complex* p, int count, DeviceArray<Complex>& scaled, Complex* out) {
    scaled.reserve(static_cast<std::size_t>(projectors) * static_cast<std::size_t>(count));
    const Complex one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
    blas.gemm(Op::N, Op::N, projectors, count, projectors, one, m, projectors, p, projectors, zero, scaled.data(), projectors);
    blas.gemm(Op::N, Op::N, n, count, projectors, one, beta, n, scaled.data(), projectors, one, out, n);
}

// the same of real m and p at Gamma
void add_projected_real(Blas& blas, const Complex* beta, std::size_t n, int projectors, const double* m,
                        const double* p, int count, DeviceArray<Complex>& scaled, Complex* out) {
    scaled.reserve(static_cast<std::size_t>(projectors) * static_cast<std::size_t>(count));
    double* s = real(scaled.data());
    dgemm(blas, Op::N, Op::N, projectors, count, projectors, 1.0, m, projectors, p, projectors, 0.0, s,
          projectors);
    gamma_combine(blas, n, beta, projectors, s, projectors, count, 1.0, 1.0, out);
}
}  // namespace

void Hamiltonian::add_nonlocal(const Complex* p, int count, Complex* hpsi) {
    MDAPY_SPAN("add_nonlocal");
    if (projectors_ == 0 || count == 0) return;
    if (gamma_) {
        add_projected_real(blas_, beta_.data(), sphere_.size(), projectors_, dreal_.data(), real(p), count, scaled_,
                           hpsi);
        return;
    }
    add_projected(blas_, beta_.data(), static_cast<int>(sphere_.size()), projectors_, dmatrix_.data(), p, count,
                  scaled_, hpsi);
}

void Hamiltonian::add_overlap(const Complex* p, int count, Complex* spsi) {
    MDAPY_SPAN("add_overlap");
    if (!has_overlap_ || projectors_ == 0 || count == 0) return;
    if (gamma_) {
        add_projected_real(blas_, beta_.data(), sphere_.size(), projectors_, qreal_.data(), real(p), count, scaled_,
                           spsi);
        return;
    }
    add_projected(blas_, beta_.data(), static_cast<int>(sphere_.size()), projectors_, qmatrix_.data(), p, count,
                  scaled_, spsi);
}

void Hamiltonian::overlap_projections(const Complex* p, int count, Complex* out) {
    MDAPY_SPAN("overlap_projections");
    if (projectors_ == 0 || count == 0) return;
    const Complex one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
    if (!has_overlap_) {
        check_cuda(cudaMemset(out, 0, static_cast<std::size_t>(projectors_) * count * (gamma_ ? sizeof(double) : sizeof(Complex))),
                   "cudaMemset");
        return;
    }
    if (gamma_) {
        dgemm(blas_, Op::N, Op::N, projectors_, count, projectors_, 1.0, qreal_.data(), projectors_,
              real(p), projectors_, 0.0, real(out), projectors_);
        return;
    }
    blas_.gemm(Op::N, Op::N, projectors_, count, projectors_, one, qmatrix_.data(), projectors_, p, projectors_, zero, out, projectors_);
}

void Hamiltonian::apply(const Complex* psi, Complex* hpsi, int count) {
    apply_local(psi, hpsi, count);
    if (projectors_ == 0 || sphere_.size() == 0 || count == 0) return;
    // the projectors: P = beta^H psi, H psi += beta D P
    projections_.reserve(static_cast<std::size_t>(projectors_) * static_cast<std::size_t>(count));
    project(psi, count, projections_.data());
    add_nonlocal(projections_.data(), count, hpsi);
}


void Hamiltonian::single() {
    if (!single_stale_) return;
    const std::size_t n = sphere_.size(), points = fft_.points(), np = static_cast<std::size_t>(projectors_);
    kinetic_f_.resize(n);
    if (n > 0) round_real_kernel<<<blocks(n), kThreads>>>(n, kinetic_.data(), kinetic_f_.data());
    potential_f_.resize(points);
    if (potential_.size() == points)
        round_real_kernel<<<blocks(points), kThreads>>>(points, potential_.data(), potential_f_.data());
    if (np > 0) {
        beta_f_.resize(n * np);
        if (n > 0) round_complex_kernel<<<blocks(n * np), kThreads>>>(n * np, beta_.data(), beta_f_.data());
        if (dmatrix_.size() == np * np) {
            dmatrix_f_.resize(np * np);
            round_complex_kernel<<<blocks(np * np), kThreads>>>(np * np, dmatrix_.data(), dmatrix_f_.data());
        }
        if (has_overlap_ && qmatrix_.size() == np * np) {
            qmatrix_f_.resize(np * np);
            round_complex_kernel<<<blocks(np * np), kThreads>>>(np * np, qmatrix_.data(), qmatrix_f_.data());
        }
        if (gamma_ && dreal_.size() == np * np) {
            dreal_f_.resize(np * np);
            round_real_kernel<<<blocks(np * np), kThreads>>>(np * np, dreal_.data(), dreal_f_.data());
        }
        if (gamma_ && has_overlap_ && qreal_.size() == np * np) {
            qreal_f_.resize(np * np);
            round_real_kernel<<<blocks(np * np), kThreads>>>(np * np, qreal_.data(), qreal_f_.data());
        }
    }
    check_cuda(cudaGetLastError(), "the single-precision Hamiltonian");
    single_stale_ = false;
}

void Hamiltonian::apply_local(const cuComplex* psi, cuComplex* hpsi, int count) {
    MDAPY_SPAN("apply_local");
    const std::size_t n = sphere_.size(), points = fft_.points();
    if (n == 0 || count == 0) return;
    if (potential_.size() != points) throw std::logic_error("no potential set");
    single();
    kinetic_single_kernel<<<blocks(n * count), kThreads>>>(n, count, kinetic_f_.data(), psi, hpsi);
    check_cuda(cudaGetLastError(), "the kinetic energy");
    // boxes of half the size of those of double precision, twice the bands
    const int step = std::min(2 * batch(), count);
    boxes_f_.reserve(points * static_cast<std::size_t>(step));
    if (gamma_) {  // real transforms of real bands
        const std::size_t spectrum = fft_.spectrum();
        values_f_.reserve(points * static_cast<std::size_t>(step));
        for (int first = 0; first < count; first += step) {
            const int m = std::min(step, count - first);
            const std::size_t offset = n * static_cast<std::size_t>(first);
            scatter(psi + offset, n, box_.data(), partner_.data(), m, boxes_f_.data(), spectrum);
            fft_.to_real(boxes_f_.data(), values_f_.data(), m);
            multiply_real_single_kernel<<<blocks(points * m), kThreads>>>(points, m, potential_f_.data(),
                                                                         values_f_.data());
            check_cuda(cudaGetLastError(), "the local potential");
            fft_.to_reciprocal(values_f_.data(), boxes_f_.data(), m);
            gather_add(boxes_f_.data(), spectrum, box_.data(), n, m, 1.0f / static_cast<float>(points), hpsi + offset);
        }
        gamma_clean(n, count, hpsi);
        return;
    }
    for (int first = 0; first < count; first += step) {
        const int m = std::min(step, count - first);
        const std::size_t offset = n * static_cast<std::size_t>(first);
        scatter(psi + offset, n, box_.data(), m, boxes_f_.data(), points);
        fft_.to_real(boxes_f_.data(), m);
        multiply_single_kernel<<<blocks(points * m), kThreads>>>(points, m, potential_f_.data(), boxes_f_.data());
        check_cuda(cudaGetLastError(), "the local potential");
        fft_.to_reciprocal(boxes_f_.data(), m);
        gather_add(boxes_f_.data(), points, box_.data(), n, m, 1.0f / static_cast<float>(points), hpsi + offset);
    }
}

void Hamiltonian::project(const cuComplex* psi, int count, cuComplex* p) {
    MDAPY_SPAN("project");
    if (projectors_ == 0 || count == 0) return;
    single();
    if (gamma_) {
        gamma_inner(blas_, sphere_.size(), beta_f_.data(), projectors_, psi, count, real(p), projectors_);
        return;
    }
    const cuComplex one = make_cuComplex(1.0f, 0.0f), zero = make_cuComplex(0.0f, 0.0f);
    const int nn = static_cast<int>(sphere_.size());
    blas_.gemm(Op::C, Op::N, projectors_, count, nn, one, beta_f_.data(), nn, psi, nn, zero, p, projectors_);
}

namespace {
// out += beta (m p) in single precision
void add_projected(Blas& blas, const cuComplex* beta, int n, int projectors, const cuComplex* m,
                   const cuComplex* p, int count, DeviceArray<cuComplex>& scaled, cuComplex* out) {
    scaled.reserve(static_cast<std::size_t>(projectors) * static_cast<std::size_t>(count));
    const cuComplex one = make_cuComplex(1.0f, 0.0f), zero = make_cuComplex(0.0f, 0.0f);
    blas.gemm(Op::N, Op::N, projectors, count, projectors, one, m, projectors, p, projectors, zero, scaled.data(), projectors);
    blas.gemm(Op::N, Op::N, n, count, projectors, one, beta, n, scaled.data(), projectors, one, out, n);
}

// the same of real m and p at Gamma
void add_projected_real(Blas& blas, const cuComplex* beta, std::size_t n, int projectors, const float* m,
                        const float* p, int count, DeviceArray<cuComplex>& scaled, cuComplex* out) {
    scaled.reserve(static_cast<std::size_t>(projectors) * static_cast<std::size_t>(count));
    float* s = real(scaled.data());
    sgemm(blas, Op::N, Op::N, projectors, count, projectors, 1.0f, m, projectors, p, projectors, 0.0f, s,
          projectors);
    gamma_combine(blas, n, beta, projectors, s, projectors, count, 1.0f, 1.0f, out);
}
}  // namespace

void Hamiltonian::add_nonlocal(const cuComplex* p, int count, cuComplex* hpsi) {
    MDAPY_SPAN("add_nonlocal");
    if (projectors_ == 0 || count == 0) return;
    single();
    if (gamma_) {
        add_projected_real(blas_, beta_f_.data(), sphere_.size(), projectors_, dreal_f_.data(), real(p), count,
                           scaled_f_, hpsi);
        return;
    }
    add_projected(blas_, beta_f_.data(), static_cast<int>(sphere_.size()), projectors_, dmatrix_f_.data(), p, count,
                  scaled_f_, hpsi);
}

void Hamiltonian::add_overlap(const cuComplex* p, int count, cuComplex* spsi) {
    MDAPY_SPAN("add_overlap");
    if (!has_overlap_ || projectors_ == 0 || count == 0) return;
    single();
    if (gamma_) {
        add_projected_real(blas_, beta_f_.data(), sphere_.size(), projectors_, qreal_f_.data(), real(p), count,
                           scaled_f_, spsi);
        return;
    }
    add_projected(blas_, beta_f_.data(), static_cast<int>(sphere_.size()), projectors_, qmatrix_f_.data(), p, count,
                  scaled_f_, spsi);
}

void Hamiltonian::overlap_projections(const cuComplex* p, int count, cuComplex* out) {
    MDAPY_SPAN("overlap_projections");
    if (projectors_ == 0 || count == 0) return;
    if (!has_overlap_) {
        check_cuda(cudaMemset(out, 0, static_cast<std::size_t>(projectors_) * count *
                                          (gamma_ ? sizeof(float) : sizeof(cuComplex))),
                   "cudaMemset");
        return;
    }
    single();
    if (gamma_) {
        sgemm(blas_, Op::N, Op::N, projectors_, count, projectors_, 1.0f, qreal_f_.data(), projectors_,
              real(p), projectors_, 0.0f, real(out), projectors_);
        return;
    }
    const cuComplex one = make_cuComplex(1.0f, 0.0f), zero = make_cuComplex(0.0f, 0.0f);
    blas_.gemm(Op::N, Op::N, projectors_, count, projectors_, one, qmatrix_f_.data(), projectors_, p, projectors_, zero, out, projectors_);
}

void Hamiltonian::apply(const cuComplex* psi, cuComplex* hpsi, int count) {
    apply_local(psi, hpsi, count);
    if (projectors_ == 0 || sphere_.size() == 0 || count == 0) return;
    projections_f_.reserve(static_cast<std::size_t>(projectors_) * static_cast<std::size_t>(count));
    project(psi, count, projections_f_.data());
    add_nonlocal(projections_f_.data(), count, hpsi);
}

}  // namespace mdapy::dft
