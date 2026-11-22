// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/hamiltonian/hamiltonian.h"

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

void check_blas(cublasStatus_t status, const char* call) {
    if (status == CUBLAS_STATUS_SUCCESS) return;
    if (status == CUBLAS_STATUS_ALLOC_FAILED) throw std::bad_alloc();
    throw std::runtime_error(std::string(call) + " failed with cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
}

// The real spherical harmonics Y_lm of the direction of (x, y, z), l up to 3
// and m from -l to l: the sines of m < 0 and the cosines of m > 0; at the
// origin Y_00 and zero for l > 0.
__device__ double ylm(int l, int m, double x, double y, double z) {
    const double r2 = x * x + y * y + z * z;
    if (l == 0) return 0.28209479177387814347;  // 1 / (2 sqrt(pi))
    if (r2 < 1e-24) return 0.0;
    const double r = sqrt(r2);
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

struct Column {
    std::int64_t table;  // where the table of its projector starts
    int l, m;
    double tau[3];  // the atom, fractions of the cell vectors
};

struct Reciprocal {
    double b[9];  // rows
    double k[3];
    double dq;
    int points;
};

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
    const double q = sqrt(x * x + y * y + z * z);
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

// the diagonal of H: kinetic + mean potential + sum over the blocks of D of
// beta_i(G) D_ij conj(beta_j(G)); blocks (first, width) of each atom
__global__ void diagonal_kernel(std::size_t n, int atoms, const int* blocks, int projectors,
                                const double* kinetic, double mean, const Complex* beta,
                                const Complex* d, double* out) {
    const std::size_t g = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (g >= n) return;
    double sum = kinetic[g] + mean;
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

}  // namespace

Hamiltonian::Hamiltonian(const Lattice& lattice, const Int3& grid, double cutoff,
                         std::vector<SpeciesProjectors> species, std::vector<Atom> atoms)
    : lattice_(lattice),
      grid_(grid),
      cutoff_(cutoff),
      species_(std::move(species)),
      atoms_(std::move(atoms)),
      fft_(grid) {
    check_blas(cublasCreate(&blas_), "cublasCreate");
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
    blocks_device_.upload(blocks_.data(), blocks_.size());
}

Hamiltonian::~Hamiltonian() {
    if (blas_ != nullptr) cublasDestroy(blas_);
}

void Hamiltonian::set_k(const Vec3& k) {
    sphere_ = make_sphere(lattice_, k, cutoff_, grid_, false);
    const std::size_t n = sphere_.size();
    kinetic_.upload(sphere_.kinetic.data(), n);
    box_.upload(sphere_.box.data(), n);
    beta_.resize(n * static_cast<std::size_t>(projectors_));
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
    q_.upload(q.data(), q.size());
    if (projectors_ == 0 || n == 0) return;
    std::vector<int> miller(3 * n);
    for (std::size_t g = 0; g < n; ++g)
        for (int i = 0; i < 3; ++i) miller[3 * g + i] = sphere_.miller[g][i];
    DeviceArray<int> m;
    m.upload(miller.data(), miller.size());
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
    DeviceArray<Column> cols;
    cols.upload(columns.data(), columns.size());
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
    if (!(std::sqrt(2.0 * sphere_.kinetic.back()) <= reach))
        throw std::invalid_argument("the tables of the projectors stop short of the sphere");
    projector_kernel<<<blocks(n * columns.size()), kThreads>>>(
        n, static_cast<int>(columns.size()), m.data(), rc, cols.data(), tables_.data(), beta_.data());
    check_cuda(cudaGetLastError(), "the projectors");
}

void Hamiltonian::set_potential(const double* potential) {
    const std::size_t points = fft_.points();
    potential_.resize(points);
    check_cuda(cudaMemcpy(potential_.data(), potential, points * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "cudaMemcpy of the potential");
    DeviceArray<double> total(1);
    total.zero();
    sum_kernel<<<std::min<unsigned>(blocks(points), 1024u), kThreads>>>(points, potential_.data(), total.data());
    check_cuda(cudaGetLastError(), "the mean potential");
    total.download(&mean_potential_, 1);
    mean_potential_ /= static_cast<double>(points);
}

void Hamiltonian::add_density(const Complex* psi, int count, const double* weights, double* rho) {
    const std::size_t n = sphere_.size(), points = fft_.points();
    if (n == 0 || count == 0) return;
    const int step = std::min(batch(), count);
    boxes_.reserve(points * static_cast<std::size_t>(step));
    weights_.assign(weights, static_cast<std::size_t>(count));
    for (int first = 0; first < count; first += step) {
        const int m = std::min(step, count - first);
        scatter(psi + n * static_cast<std::size_t>(first), n, box_.data(), nullptr, m, boxes_.data(), points);
        fft_.to_real(boxes_.data(), m);
        density_kernel<<<blocks(points), kThreads>>>(points, m, boxes_.data(), weights_.data() + first,
                                                     1.0 / lattice_.volume, rho);
        check_cuda(cudaGetLastError(), "the density of the bands");
    }
}

void Hamiltonian::nonlocal_forces(const Complex* psi, int count, const double* weights, std::vector<double>& forces) {
    const std::size_t n = sphere_.size(), np = static_cast<std::size_t>(projectors_);
    if (n == 0 || count == 0 || projectors_ == 0) return;
    if (forces.size() != 3 * atoms_.size()) throw std::invalid_argument("forces are (atoms, 3)");
    const std::size_t m = static_cast<std::size_t>(count);
    DeviceArray<Complex> p(np * m), dp(np * m), qpsi(n * m), qp(np * m);
    DeviceArray<double> w, s(np);
    w.upload(weights, m);
    const Complex one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
    const int nn = static_cast<int>(n);
    check_blas(cublasZgemm(blas_, CUBLAS_OP_C, CUBLAS_OP_N, projectors_, count, nn, &one, beta_.data(), nn, psi, nn,
                           &zero, p.data(), projectors_),
               "cublasZgemm");
    check_blas(cublasZgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, projectors_, count, projectors_, &one, dmatrix_.data(),
                           projectors_, p.data(), projectors_, &zero, dp.data(), projectors_),
               "cublasZgemm");
    std::vector<double> sums(np);
    for (int d = 0; d < 3; ++d) {
        times_q_kernel<<<blocks(n * m), kThreads>>>(n, count, d, q_.data(), psi, qpsi.data());
        check_cuda(cudaGetLastError(), "k+G times the bands");
        check_blas(cublasZgemm(blas_, CUBLAS_OP_C, CUBLAS_OP_N, projectors_, count, nn, &one, beta_.data(), nn,
                               qpsi.data(), nn, &zero, qp.data(), projectors_),
                   "cublasZgemm");
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

void Hamiltonian::diagonal(double* out) const {
    const std::size_t n = sphere_.size();
    if (n == 0) return;
    if (potential_.size() != fft_.points()) throw std::logic_error("no potential set");
    diagonal_kernel<<<blocks(n), kThreads>>>(n, static_cast<int>(atoms_.size()), blocks_device_.data(), projectors_,
                                             kinetic_.data(), mean_potential_, beta_.data(), dmatrix_.data(), out);
    check_cuda(cudaGetLastError(), "the diagonal of H");
}

int Hamiltonian::batch() const noexcept {
    // bands per batch of FFTs: boxes of at most 256 MB
    const std::size_t points = fft_.points();
    const std::size_t most = (std::size_t{256} << 20) / (points * sizeof(Complex));
    return static_cast<int>(std::max<std::size_t>(1, std::min<std::size_t>(most, 1u << 20)));
}

void Hamiltonian::apply(const Complex* psi, Complex* hpsi, int count) {
    const std::size_t n = sphere_.size(), points = fft_.points();
    if (n == 0 || count == 0) return;
    if (potential_.size() != points) throw std::logic_error("no potential set");
    kinetic_kernel<<<blocks(n * count), kThreads>>>(n, count, kinetic_.data(), psi, hpsi);
    check_cuda(cudaGetLastError(), "the kinetic energy");
    // the local potential on the grid, in batches of bands
    const int step = std::min(batch(), count);
    boxes_.reserve(points * static_cast<std::size_t>(step));
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
    if (projectors_ == 0) return;
    // the projectors: P = beta^H psi, D P, H psi += beta D P
    const std::size_t np = static_cast<std::size_t>(projectors_);
    projections_.resize(np * static_cast<std::size_t>(count));
    scaled_.resize(np * static_cast<std::size_t>(count));
    const Complex one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
    const int nn = static_cast<int>(n);
    check_blas(cublasZgemm(blas_, CUBLAS_OP_C, CUBLAS_OP_N, projectors_, count, nn, &one, beta_.data(),
                           nn, psi, nn, &zero, projections_.data(), projectors_),
               "cublasZgemm");
    check_blas(cublasZgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, projectors_, count, projectors_, &one,
                           dmatrix_.data(), projectors_, projections_.data(), projectors_, &zero,
                           scaled_.data(), projectors_),
               "cublasZgemm");
    check_blas(cublasZgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, nn, count, projectors_, &one, beta_.data(),
                           nn, scaled_.data(), projectors_, &one, hpsi, nn),
               "cublasZgemm");
}

}  // namespace mdapy::dft
