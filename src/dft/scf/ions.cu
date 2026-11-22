// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/scf/ions.h"

#include <stdexcept>

namespace mdapy::dft {

namespace {

constexpr int kThreads = 256;
constexpr double kTwoPi = 6.28318530717958647692528676655900577;

unsigned blocks(std::size_t work) { return static_cast<unsigned>((work + kThreads - 1) / kThreads); }

// out(G) = sum_I factor_{s(I)}(G) exp(-2 pi i m.tau_I)
__global__ void structure_kernel(std::size_t n, int atoms, const int* miller, const double* tau, const int* kinds,
                                 const double* factors, Complex* out) {
    const std::size_t g = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (g >= n) return;
    const double m0 = miller[3 * g], m1 = miller[3 * g + 1], m2 = miller[3 * g + 2];
    double re = 0.0, im = 0.0;
    for (int a = 0; a < atoms; ++a) {
        const double f = factors[static_cast<std::size_t>(kinds[a]) * n + g];
        const double phase = -kTwoPi * (m0 * tau[3 * a] + m1 * tau[3 * a + 1] + m2 * tau[3 * a + 2]);
        double s, c;
        sincos(phase, &s, &c);
        re += f * c;
        im += f * s;
    }
    out[g] = make_cuDoubleComplex(re, im);
}

// one block per atom: sum_G Re[i G f(G) exp(-i G.tau) conj(x(G))] for the
// three directions, of each (factor, field) pair given
__global__ void force_kernel(std::size_t n, const int* miller, const double* g, const double* tau, const int* kinds,
                             const double* local, const Complex* rho, const double* core, const Complex* vxc,
                             double* out) {
    const int a = static_cast<int>(blockIdx.x);
    const double t0 = tau[3 * a], t1 = tau[3 * a + 1], t2 = tau[3 * a + 2];
    const std::size_t offset = static_cast<std::size_t>(kinds[a]) * n;
    double sum[3] = {0.0, 0.0, 0.0};
    for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const double phase = -kTwoPi * (miller[3 * i] * t0 + miller[3 * i + 1] * t1 + miller[3 * i + 2] * t2);
        double s, c;
        sincos(phase, &s, &c);
        // Re[i f e conj(x)] = -f Im[e conj(x)], e = c + i s
        double value = local[offset + i] * (s * rho[i].x - c * rho[i].y);
        if (core != nullptr) value += core[offset + i] * (s * vxc[i].x - c * vxc[i].y);
        for (int d = 0; d < 3; ++d) sum[d] -= g[3 * i + d] * value;
    }
    __shared__ double partial[3][kThreads];
    for (int d = 0; d < 3; ++d) partial[d][threadIdx.x] = sum[d];
    __syncthreads();
    for (int k = kThreads / 2; k > 0; k >>= 1) {
        if (static_cast<int>(threadIdx.x) < k)
            for (int d = 0; d < 3; ++d) partial[d][threadIdx.x] += partial[d][threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        for (int d = 0; d < 3; ++d) out[3 * a + d] = partial[d][0];
}

}  // namespace

Ions::Ions(const Lattice& lattice, const Sphere& sphere, const std::vector<Atom>& atoms, int species,
           const std::vector<double>& local, const std::vector<double>& core)
    : lattice_(lattice),
      n_(sphere.size()),
      atoms_(static_cast<int>(atoms.size())),
      species_(species),
      has_core_(!core.empty()) {
    const std::size_t total = static_cast<std::size_t>(species) * n_;
    if (local.size() != total || (has_core_ && core.size() != total))
        throw std::invalid_argument("the form factors are (species, the sphere of the density)");
    std::vector<int> miller(3 * n_), kinds;
    std::vector<double> g(3 * n_), tau;
    for (std::size_t i = 0; i < n_; ++i)
        for (int d = 0; d < 3; ++d) {
            miller[3 * i + d] = sphere.miller[i][static_cast<std::size_t>(d)];
            double x = 0.0;
            for (int j = 0; j < 3; ++j)
                x += sphere.miller[i][static_cast<std::size_t>(j)] *
                     lattice.b[static_cast<std::size_t>(j)][static_cast<std::size_t>(d)];
            g[3 * i + d] = x;
        }
    for (const Atom& a : atoms) {
        if (a.species < 0 || a.species >= species) throw std::invalid_argument("an atom of a species not given");
        kinds.push_back(a.species);
        for (int d = 0; d < 3; ++d) tau.push_back(a.position[static_cast<std::size_t>(d)]);
    }
    miller_.upload(miller.data(), miller.size());
    g_.upload(g.data(), g.size());
    kinds_.upload(kinds.data(), kinds.size());
    tau_.upload(tau.data(), tau.size());
    local_.upload(local.data(), local.size());
    if (has_core_) core_.upload(core.data(), core.size());
}

void Ions::structure(const DeviceArray<double>& factors, Complex* out) const {
    structure_kernel<<<blocks(n_), kThreads>>>(n_, atoms_, miller_.data(), tau_.data(), kinds_.data(), factors.data(),
                                               out);
    check_cuda(cudaGetLastError(), "the structure factors");
}

void Ions::local_components(Complex* out) const { structure(local_, out); }

void Ions::core_components(Complex* out) const {
    if (!has_core_) throw std::logic_error("no core charges");
    structure(core_, out);
}

void Ions::set_slopes(const std::vector<double>& local, const std::vector<double>& core) {
    const std::size_t total = static_cast<std::size_t>(species_) * n_;
    if (local.size() != total || core.size() != (has_core_ ? total : 0))
        throw std::invalid_argument("the slopes of the form factors are (species, the sphere of the density)");
    local_slope_.upload(local.data(), local.size());
    if (has_core_) core_slope_.upload(core.data(), core.size());
}

void Ions::local_slope_components(Complex* out) const {
    if (!has_slopes()) throw std::logic_error("no slopes of the form factors");
    structure(local_slope_, out);
}

void Ions::core_slope_components(Complex* out) const {
    if (!has_slopes() || !has_core_) throw std::logic_error("no slopes of the core charges");
    structure(core_slope_, out);
}

std::vector<double> Ions::forces(const Complex* rho, const Complex* vxc) const {
    std::vector<double> out(3 * static_cast<std::size_t>(atoms_), 0.0);
    if (atoms_ == 0) return out;
    DeviceArray<double> f(out.size());
    force_kernel<<<static_cast<unsigned>(atoms_), kThreads>>>(n_, miller_.data(), g_.data(), tau_.data(),
                                                              kinds_.data(), local_.data(), rho,
                                                              has_core_ ? core_.data() : nullptr, vxc, f.data());
    check_cuda(cudaGetLastError(), "the forces of the ions");
    f.download(out.data(), out.size());
    for (double& x : out) x *= lattice_.volume;
    return out;
}

}  // namespace mdapy::dft
