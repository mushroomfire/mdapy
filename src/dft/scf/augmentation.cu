// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/scf/augmentation.h"

#include <algorithm>
#include <new>
#include <stdexcept>
#include <string>

#include "dft/hamiltonian/harmonics.h"
#include "dft/xc/dual.h"

namespace mdapy::dft {

namespace {

constexpr int kThreads = 256;
constexpr double kTwoPi = 6.28318530717958647692528676655900577;

unsigned blocks(std::size_t work) { return static_cast<unsigned>((work + kThreads - 1) / kThreads); }

struct Terms {
    const int *pair_start, *lm, *l, *table;
    const double *coef, *tables;
    int points, lmax, pairs;
    double dq;
};

// Q_ab(G) of the G first ... first + count, out[j pairs + pair]
__global__ void q_kernel(std::size_t first, std::size_t count, const double* g, Terms t, Complex* out) {
    const std::size_t j = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (j >= count) return;
    const double* v = g + 3 * (first + j);
    double y[(kHarmonicsMax + 1) * (kHarmonicsMax + 1)];
    real_harmonics(t.lmax, v[0], v[1], v[2], y);
    const double q = ::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    const double s = q / t.dq;
    int i0 = static_cast<int>(floor(s));
    if (i0 > t.points - 4) i0 = t.points - 4;
    const double p = s - i0, u = 1.0 - p, w2 = 2.0 - p, w3 = 3.0 - p;
    const double c0 = u * w2 * w3 / 6.0, c1 = p * w2 * w3 / 2.0, c2 = -p * u * w3 / 2.0, c3 = p * u * w2 / 6.0;
    for (int pair = 0; pair < t.pairs; ++pair) {
        double re = 0.0, im = 0.0;
        for (int k = t.pair_start[pair]; k < t.pair_start[pair + 1]; ++k) {
            const double* table = t.tables + static_cast<std::size_t>(t.table[k]) * t.points + i0;
            const double radial = c0 * table[0] + c1 * table[1] + c2 * table[2] + c3 * table[3];
            const double value = t.coef[k] * radial * y[t.lm[k]];
            switch (t.l[k] & 3) {  // times (-i)^L
                case 0: re += value; break;
                case 1: im -= value; break;
                case 2: re -= value; break;
                default: im += value; break;
            }
        }
        out[j * t.pairs + pair] = make_cuDoubleComplex(re, im);
    }
}

// w[j pairs + pair] = Omega dQ_ab(G)/d epsilon_k conj(v(G)) at a fixed volume,
// the derivative of the table and of the harmonics through duals, with
// dG_c / d epsilon_ab = -(delta_ca G_b + delta_cb G_a) / 2
__global__ void strain_weight_kernel(std::size_t first, std::size_t count, const double* g, Terms t, int a, int b,
                                     double volume, const Complex* v, Complex* w) {
    const std::size_t j = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (j >= count) return;
    using D = Dual<3>;
    const double* gv = g + 3 * (first + j);
    const double modulus = ::sqrt(gv[0] * gv[0] + gv[1] * gv[1] + gv[2] * gv[2]);
    const Complex vc = v[first + j];
    if (modulus < 1e-12) {  // G = 0 does not move
        for (int pair = 0; pair < t.pairs; ++pair) w[j * t.pairs + pair] = make_cuDoubleComplex(0.0, 0.0);
        return;
    }
    const D x = D::input(gv[0], 0), y = D::input(gv[1], 1), z = D::input(gv[2], 2);
    D harm[(kHarmonicsMax + 1) * (kHarmonicsMax + 1)];
    real_harmonics(t.lmax, x, y, z, harm);
    const D s = sqrt(x * x + y * y + z * z) / t.dq;
    int i0 = static_cast<int>(floor(s.v));
    if (i0 > t.points - 4) i0 = t.points - 4;
    const D p = s - static_cast<double>(i0), u = 1.0 - p, w2 = 2.0 - p, w3 = 3.0 - p;
    const D c0 = u * w2 * w3 / 6.0, c1 = p * w2 * w3 / 2.0, c2 = -1.0 * p * u * w3 / 2.0, c3 = p * u * w2 / 6.0;
    for (int pair = 0; pair < t.pairs; ++pair) {
        double re = 0.0, im = 0.0;
        for (int k = t.pair_start[pair]; k < t.pair_start[pair + 1]; ++k) {
            const double* table = t.tables + static_cast<std::size_t>(t.table[k]) * t.points + i0;
            const D radial = c0 * table[0] + c1 * table[1] + c2 * table[2] + c3 * table[3];
            const D value = t.coef[k] * radial * harm[t.lm[k]];
            const double dvalue = -0.5 * (gv[b] * value.d[a] + gv[a] * value.d[b]);
            switch (t.l[k] & 3) {
                case 0: re += dvalue; break;
                case 1: im -= dvalue; break;
                case 2: re -= dvalue; break;
                default: im += dvalue; break;
            }
        }
        // (re + i im) conj(v) Omega
        w[j * t.pairs + pair] =
            make_cuDoubleComplex(volume * (re * vc.x + im * vc.y), volume * (im * vc.x - re * vc.y));
    }
}

// E[j atoms + i] = exp(-2 pi i m_j . tau_i)
__global__ void phase_kernel(std::size_t first, std::size_t count, int atoms, const int* miller, const double* tau,
                             Complex* out) {
    const std::size_t idx = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (idx >= count * atoms) return;
    const std::size_t j = idx / atoms;
    const int i = static_cast<int>(idx % atoms);
    const int* m = miller + 3 * (first + j);
    const double phase = -kTwoPi * (m[0] * tau[3 * i] + m[1] * tau[3 * i + 1] + m[2] * tau[3 * i + 2]);
    double s, c;
    sincos(phase, &s, &c);
    out[idx] = make_cuDoubleComplex(c, s);
}

// rho[first + j] += sum_pair q[j pairs + pair] t[j pairs + pair]
__global__ void contract_kernel(std::size_t first, std::size_t count, int pairs, const Complex* q, const Complex* t,
                                Complex* rho) {
    const std::size_t j = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (j >= count) return;
    double re = 0.0, im = 0.0;
    for (int p = 0; p < pairs; ++p) {
        const Complex a = q[j * pairs + p], b = t[j * pairs + p];
        re += a.x * b.x - a.y * b.y;
        im += a.x * b.y + a.y * b.x;
    }
    rho[first + j].x += re;
    rho[first + j].y += im;
}

// w[j pairs + pair] = q[j pairs + pair] conj(v[first + j])
__global__ void weight_kernel(std::size_t first, std::size_t count, int pairs, const Complex* q, const Complex* v,
                              Complex* w) {
    const std::size_t idx = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (idx >= count * pairs) return;
    const Complex a = q[idx], b = v[first + idx / pairs];
    w[idx] = make_cuDoubleComplex(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}

// w[j pairs + pair] = q[j pairs + pair] conj(v[first + j]) i G_d
__global__ void force_weight_kernel(std::size_t first, std::size_t count, int pairs, int d, const double* g,
                                    const Complex* q, const Complex* v, Complex* w) {
    const std::size_t idx = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (idx >= count * pairs) return;
    const std::size_t j = first + idx / pairs;
    const Complex a = q[idx], b = v[j];
    const double re = a.x * b.x + a.y * b.y, im = a.y * b.x - a.x * b.y, gd = g[3 * j + d];
    w[idx] = make_cuDoubleComplex(-gd * im, gd * re);  // times i G_d
}

}  // namespace

Augmentation::Augmentation(const Lattice& lattice, const Sphere& sphere, const std::vector<Atom>& atoms,
                           std::vector<SpeciesAugmentation> species, Kernels kernels)
    : lattice_(lattice), n_(sphere.size()), atoms_(atoms), species_(std::move(species)), blas_(kernels) {
    members_.resize(species_.size());
    offsets_.resize(species_.size());
    std::size_t offset = 0;
    for (std::size_t i = 0; i < atoms_.size(); ++i) {
        const auto s = static_cast<std::size_t>(atoms_[i].species);
        if (s >= species_.size()) throw std::invalid_argument("an atom of a species not given");
        members_[s].push_back(static_cast<int>(i));
        offsets_[s].push_back(offset);
        offset += static_cast<std::size_t>(std::max(species_[s].pairs(), 0));
    }
    std::vector<int> miller(3 * n_);
    std::vector<double> g(3 * n_);
    for (std::size_t i = 0; i < n_; ++i)
        for (int d = 0; d < 3; ++d) {
            miller[3 * i + d] = sphere.miller[i][static_cast<std::size_t>(d)];
            double x = 0.0;
            for (int j = 0; j < 3; ++j)
                x += sphere.miller[i][static_cast<std::size_t>(j)] *
                     lattice.b[static_cast<std::size_t>(j)][static_cast<std::size_t>(d)];
            g[3 * i + d] = x;
        }
    miller_.upload(miller.data(), miller.size());
    g_.upload(g.data(), g.size());
    device_.resize(species_.size());
    for (std::size_t s = 0; s < species_.size(); ++s) {
        const auto& sp = species_[s];
        if (sp.pairs() <= 0) continue;
        if (sp.lmax > kHarmonicsMax || sp.points < 4) throw std::invalid_argument("augmentation of L up to 6");
        const std::size_t terms = sp.term_lm.size();
        if (sp.term_l.size() != terms || sp.term_table.size() != terms || sp.term_coef.size() != terms ||
            sp.pair_start.back() != static_cast<int>(terms))
            throw std::invalid_argument("the terms of the augmentation disagree");
        any_ = true;
        auto& d = device_[s];
        d.pair_start.upload(sp.pair_start.data(), sp.pair_start.size());
        d.term_lm.upload(sp.term_lm.data(), terms);
        d.term_l.upload(sp.term_l.data(), terms);
        d.term_table.upload(sp.term_table.data(), terms);
        d.term_coef.upload(sp.term_coef.data(), terms);
        d.tables.upload(sp.tables.data(), sp.tables.size());
        std::vector<double> tau;
        for (const int a : members_[s])
            for (int k = 0; k < 3; ++k) tau.push_back(atoms_[static_cast<std::size_t>(a)].position[static_cast<std::size_t>(k)]);
        d.tau.upload(tau.data(), tau.size());
    }
}

Augmentation::~Augmentation() = default;

void Augmentation::block_q(int species, std::size_t first, std::size_t count, Complex* out) {
    const auto& sp = species_[static_cast<std::size_t>(species)];
    const auto& d = device_[static_cast<std::size_t>(species)];
    const Terms t{d.pair_start.data(), d.term_lm.data(), d.term_l.data(), d.term_table.data(),
                  d.term_coef.data(), d.tables.data(), sp.points, sp.lmax, sp.pairs(), sp.dq};
    q_kernel<<<blocks(count), kThreads>>>(first, count, g_.data(), t, out);
    check_cuda(cudaGetLastError(), "the augmentation charges");
}

void Augmentation::block_phases(int species, std::size_t first, std::size_t count, Complex* out) {
    const int na = static_cast<int>(members_[static_cast<std::size_t>(species)].size());
    phase_kernel<<<blocks(count * na), kThreads>>>(first, count, na, miller_.data(),
                                                   device_[static_cast<std::size_t>(species)].tau.data(), out);
    check_cuda(cudaGetLastError(), "the phases of the atoms");
}

std::vector<double> Augmentation::overlaps(int species) const {
    const auto& sp = species_[static_cast<std::size_t>(species)];
    std::vector<double> out(static_cast<std::size_t>(std::max(sp.pairs(), 0)), 0.0);
    if (sp.pairs() <= 0 || n_ == 0) return out;
    DeviceArray<Complex> q(out.size());
    const_cast<Augmentation*>(this)->block_q(species, 0, 1, q.data());  // G = 0 comes first
    std::vector<Complex> host(out.size());
    q.download(host.data(), host.size());
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = lattice_.volume * host[i].x;
    return out;
}

std::vector<Complex> Augmentation::components(int species) {
    const auto& sp = species_[static_cast<std::size_t>(species)];
    const std::size_t pairs = static_cast<std::size_t>(std::max(sp.pairs(), 0));
    std::vector<Complex> out(pairs * n_);
    if (pairs == 0) return out;
    DeviceArray<Complex> q(pairs * n_);
    block_q(species, 0, n_, q.data());
    q.download(out.data(), out.size());
    return out;
}

void Augmentation::add_density(const std::vector<double>& becsum, Complex* rho) {
    const Complex one = make_cuDoubleComplex(1.0, 0.0), zero = make_cuDoubleComplex(0.0, 0.0);
    for (std::size_t s = 0; s < species_.size(); ++s) {
        const int pairs = species_[s].pairs();
        const int na = static_cast<int>(members_[s].size());
        if (pairs <= 0 || na == 0) continue;
        // becsum of the species, (pairs, atoms), as complex numbers
        std::vector<Complex> b(static_cast<std::size_t>(pairs) * na);
        for (int i = 0; i < na; ++i)
            for (int p = 0; p < pairs; ++p)
                b[static_cast<std::size_t>(i) * pairs + p] =
                    make_cuDoubleComplex(becsum[offsets_[s][static_cast<std::size_t>(i)] + static_cast<std::size_t>(p)], 0.0);
        DeviceArray<Complex> bd;
        bd.upload(b.data(), b.size());
        const std::size_t step = std::max<std::size_t>(1, (std::size_t{256} << 20) / (48 * static_cast<std::size_t>(pairs + na)));
        for (std::size_t first = 0; first < n_; first += step) {
            const std::size_t count = std::min(step, n_ - first);
            q_.reserve(count * pairs);
            phases_.reserve(count * na);
            work_.reserve(count * pairs);
            block_q(static_cast<int>(s), first, count, q_.data());
            block_phases(static_cast<int>(s), first, count, phases_.data());
            blas_.gemm(Op::N, Op::N, pairs, static_cast<int>(count), na, one, bd.data(), pairs, phases_.data(), na, zero, work_.data(), pairs);
            contract_kernel<<<blocks(count), kThreads>>>(first, count, pairs, q_.data(), work_.data(), rho);
            check_cuda(cudaGetLastError(), "the augmented density");
        }
    }
}

std::vector<double> Augmentation::strengths(const Complex* v) {
    std::size_t total = 0;
    for (std::size_t i = 0; i < atoms_.size(); ++i)
        total += static_cast<std::size_t>(std::max(species_[static_cast<std::size_t>(atoms_[i].species)].pairs(), 0));
    std::vector<double> out(total, 0.0);
    const Complex one = make_cuDoubleComplex(1.0, 0.0);
    for (std::size_t s = 0; s < species_.size(); ++s) {
        const int pairs = species_[s].pairs();
        const int na = static_cast<int>(members_[s].size());
        if (pairs <= 0 || na == 0) continue;
        values_.reserve(static_cast<std::size_t>(pairs) * na);
        values_.zero();
        const std::size_t step = std::max<std::size_t>(1, (std::size_t{256} << 20) / (48 * static_cast<std::size_t>(pairs + na)));
        for (std::size_t first = 0; first < n_; first += step) {
            const std::size_t count = std::min(step, n_ - first);
            q_.reserve(count * pairs);
            phases_.reserve(count * na);
            work_.reserve(count * pairs);
            block_q(static_cast<int>(s), first, count, q_.data());
            block_phases(static_cast<int>(s), first, count, phases_.data());
            weight_kernel<<<blocks(count * pairs), kThreads>>>(first, count, pairs, q_.data(), v, work_.data());
            check_cuda(cudaGetLastError(), "the weights of the augmentation");
            // values (pairs, atoms) += work (pairs, count) E^T (count, atoms)
            blas_.gemm(Op::N, Op::T, pairs, na, static_cast<int>(count), one, work_.data(), pairs, phases_.data(), na, one, values_.data(), pairs);
        }
        std::vector<Complex> host(static_cast<std::size_t>(pairs) * na);
        values_.download(host.data(), host.size());
        for (int i = 0; i < na; ++i)
            for (int p = 0; p < pairs; ++p)
                out[offsets_[s][static_cast<std::size_t>(i)] + static_cast<std::size_t>(p)] =
                    lattice_.volume * host[static_cast<std::size_t>(i) * pairs + p].x;
    }
    return out;
}

std::vector<double> Augmentation::forces(const Complex* v, const std::vector<double>& becsum) {
    std::vector<double> out(3 * atoms_.size(), 0.0);
    const Complex one = make_cuDoubleComplex(1.0, 0.0);
    for (std::size_t s = 0; s < species_.size(); ++s) {
        const int pairs = species_[s].pairs();
        const int na = static_cast<int>(members_[s].size());
        if (pairs <= 0 || na == 0) continue;
        const std::size_t step = std::max<std::size_t>(1, (std::size_t{256} << 20) / (48 * static_cast<std::size_t>(pairs + na)));
        for (int d = 0; d < 3; ++d) {
            values_.reserve(static_cast<std::size_t>(pairs) * na);
            values_.zero();
            for (std::size_t first = 0; first < n_; first += step) {
                const std::size_t count = std::min(step, n_ - first);
                q_.reserve(count * pairs);
                phases_.reserve(count * na);
                work_.reserve(count * pairs);
                block_q(static_cast<int>(s), first, count, q_.data());
                block_phases(static_cast<int>(s), first, count, phases_.data());
                force_weight_kernel<<<blocks(count * pairs), kThreads>>>(first, count, pairs, d, g_.data(), q_.data(),
                                                                       v, work_.data());
                check_cuda(cudaGetLastError(), "the weights of the forces of the augmentation");
                blas_.gemm(Op::N, Op::T, pairs, na, static_cast<int>(count), one, work_.data(), pairs, phases_.data(), na, one, values_.data(), pairs);
            }
            std::vector<Complex> host(static_cast<std::size_t>(pairs) * na);
            values_.download(host.data(), host.size());
            for (int i = 0; i < na; ++i) {
                double f = 0.0;
                for (int p = 0; p < pairs; ++p)
                    f += becsum[offsets_[s][static_cast<std::size_t>(i)] + static_cast<std::size_t>(p)] *
                         host[static_cast<std::size_t>(i) * pairs + p].x;
                out[3 * static_cast<std::size_t>(members_[s][static_cast<std::size_t>(i)]) + static_cast<std::size_t>(d)] +=
                    lattice_.volume * f;
            }
        }
    }
    return out;
}

std::array<double, 6> Augmentation::strain(const Complex* v, const std::vector<double>& becsum) {
    std::array<double, 6> out{};
    const Complex one = make_cuDoubleComplex(1.0, 0.0);
    const int pairs6[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};
    for (std::size_t s = 0; s < species_.size(); ++s) {
        const auto& sp = species_[s];
        const int pairs = sp.pairs();
        const int na = static_cast<int>(members_[s].size());
        if (pairs <= 0 || na == 0) continue;
        const auto& d = device_[s];
        const Terms t{d.pair_start.data(), d.term_lm.data(), d.term_l.data(), d.term_table.data(),
                      d.term_coef.data(), d.tables.data(), sp.points, sp.lmax, pairs, sp.dq};
        const std::size_t step = std::max<std::size_t>(1, (std::size_t{256} << 20) / (48 * static_cast<std::size_t>(pairs + na)));
        for (int k = 0; k < 6; ++k) {
            values_.reserve(static_cast<std::size_t>(pairs) * na);
            values_.zero();
            for (std::size_t first = 0; first < n_; first += step) {
                const std::size_t count = std::min(step, n_ - first);
                phases_.reserve(count * na);
                work_.reserve(count * pairs);
                block_phases(static_cast<int>(s), first, count, phases_.data());
                strain_weight_kernel<<<blocks(count), kThreads>>>(first, count, g_.data(), t, pairs6[k][0], pairs6[k][1],
                                                                  lattice_.volume, v, work_.data());
                check_cuda(cudaGetLastError(), "the strain of the augmentation");
                blas_.gemm(Op::N, Op::T, pairs, na, static_cast<int>(count), one, work_.data(), pairs, phases_.data(), na, one, values_.data(), pairs);
            }
            std::vector<Complex> host(static_cast<std::size_t>(pairs) * na);
            values_.download(host.data(), host.size());
            for (int i = 0; i < na; ++i)
                for (int p = 0; p < pairs; ++p)
                    out[static_cast<std::size_t>(k)] +=
                        becsum[offsets_[s][static_cast<std::size_t>(i)] + static_cast<std::size_t>(p)] *
                        host[static_cast<std::size_t>(i) * pairs + p].x;
        }
    }
    return out;
}

}  // namespace mdapy::dft
