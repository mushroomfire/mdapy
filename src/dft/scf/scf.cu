// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/scf/scf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <atomic>
#include <exception>
#include <random>
#include <thread>
#include <unordered_map>
#include <stdexcept>

#include "dft/core/device.h"
#include "dft/scf/augmentation.h"
#include "dft/scf/ions.h"
#include "dft/scf/potentials.h"
#include "dft/solver/davidson.h"

namespace mdapy::dft {

namespace {

constexpr int kThreads = 256;

unsigned blocks(std::size_t work) { return static_cast<unsigned>((work + kThreads - 1) / kThreads); }

__global__ void add_kernel(std::size_t n, const double* a, const double* b, double* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void difference_kernel(std::size_t n, const double* a, const double* b, double* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

// out = d (v - c), d clamped at zero
__global__ void screened_kernel(std::size_t n, const double* d, const double* v, double c, double* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = fmax(d[i], 0.0) * (v[i] - c);
}

__global__ void clamp_kernel(std::size_t n, double* x) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n && x[i] < 0.0) x[i] = 0.0;
}

void copy_device(DeviceArray<double>& to, const DeviceArray<double>& from, std::size_t n) {
    to.resize(n);
    check_cuda(cudaMemcpy(to.data(), from.data(), n * sizeof(double), cudaMemcpyDeviceToDevice), "cudaMemcpy");
}

// seconds between laps, the GPU finished
struct Clock {
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
    double lap() {
        check_cuda(cudaStreamSynchronize(cudaStreamPerThread), "cudaStreamSynchronize");
        const auto now = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(now - last).count();
        last = now;
        return seconds;
    }
};

// random bands decaying as 1 / (1 + |k+G|^2 / 2); real at Gamma, Im c_0 = 0
std::vector<Complex> random_bands(const std::vector<double>& kinetic, int bands, unsigned long long seed,
                                  bool gamma) {
    std::mt19937_64 engine(seed);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    std::vector<Complex> psi(kinetic.size() * static_cast<std::size_t>(bands));
    for (int b = 0; b < bands; ++b)
        for (std::size_t g = 0; g < kinetic.size(); ++g) {
            const double scale = 1.0 / (1.0 + kinetic[g]);
            const double re = uniform(engine), im = uniform(engine);
            psi[static_cast<std::size_t>(b) * kinetic.size() + g] =
                make_cuDoubleComplex(scale * re, gamma && g == 0 ? 0.0 : scale * im);
        }
    return psi;
}

// (up, down) components on the sphere to (total, magnetization) and back
void to_total(std::vector<Complex>& c, std::size_t n) {
    for (std::size_t g = 0; g < n; ++g) {
        const Complex up = c[g], down = c[n + g];
        c[g] = make_cuDoubleComplex(up.x + down.x, up.y + down.y);
        c[n + g] = make_cuDoubleComplex(up.x - down.x, up.y - down.y);
    }
}

void to_spins(std::vector<Complex>& c, std::size_t n) {
    for (std::size_t g = 0; g < n; ++g) {
        const Complex total = c[g], m = c[n + g];
        c[g] = make_cuDoubleComplex(0.5 * (total.x + m.x), 0.5 * (total.y + m.y));
        c[n + g] = make_cuDoubleComplex(0.5 * (total.x - m.x), 0.5 * (total.y - m.y));
    }
}

// the index of each Miller index of `to` in `from`, -1 where it is missing
std::vector<std::int64_t> match(const std::vector<Int3>& from, const std::vector<Int3>& to) {
    auto key = [](const Int3& m) {
        return (static_cast<std::int64_t>(m[0] + 4096) << 26) | (static_cast<std::int64_t>(m[1] + 4096) << 13) |
               static_cast<std::int64_t>(m[2] + 4096);
    };
    std::unordered_map<std::int64_t, std::int64_t> where;
    where.reserve(from.size());
    for (std::size_t i = 0; i < from.size(); ++i) where.emplace(key(from[i]), static_cast<std::int64_t>(i));
    std::vector<std::int64_t> out(to.size(), -1);
    for (std::size_t i = 0; i < to.size(); ++i) {
        const auto found = where.find(key(to[i]));
        if (found != where.end()) out[i] = found->second;
    }
    return out;
}

// column b of out (n, count) is column b of in (m, count) at the rows index,
// zero where the index is -1
__global__ void match_kernel(std::size_t n, std::size_t m, int count, const std::int64_t* index, const Complex* in,
                             Complex* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n * count) return;
    const std::size_t b = i / n, g = i % n;
    const std::int64_t from = index[g];
    out[i] = from < 0 ? make_cuDoubleComplex(0.0, 0.0) : in[b * m + static_cast<std::size_t>(from)];
}

// the width of the projector functions of a species
int width_of(const SpeciesProjectors& s) {
    int w = 0;
    for (const int l : s.angular) w += 2 * l + 1;
    return w;
}

// the strengths of a species as a block (width, width), D_pq between the
// functions of one l and m
std::vector<double> species_block(const SpeciesProjectors& s) {
    const int w = width_of(s);
    std::vector<double> out(static_cast<std::size_t>(w) * w, 0.0);
    const int count = static_cast<int>(s.angular.size());
    std::vector<int> first(static_cast<std::size_t>(count));
    for (int p = 0, f = 0; p < count; f += 2 * s.angular[static_cast<std::size_t>(p)] + 1, ++p)
        first[static_cast<std::size_t>(p)] = f;
    for (int p = 0; p < count; ++p)
        for (int q = 0; q < count; ++q) {
            const int l = s.angular[static_cast<std::size_t>(p)];
            if (l != s.angular[static_cast<std::size_t>(q)]) continue;
            for (int m = 0; m < 2 * l + 1; ++m)
                out[static_cast<std::size_t>(first[static_cast<std::size_t>(p)] + m) * w +
                    (first[static_cast<std::size_t>(q)] + m)] = s.dij[static_cast<std::size_t>(p * count + q)];
        }
    return out;
}

// the packed pairs a <= b of a block of width w added to it, symmetric
void add_packed(const double* packed, int w, double* block) {
    int k = 0;
    for (int a = 0; a < w; ++a)
        for (int b = a; b < w; ++b, ++k) {
            block[a * w + b] += packed[k];
            if (b != a) block[b * w + a] += packed[k];
        }
}

bool fits(const ScfState* state, const ScfInput& input, int spins, int bands) {
    if (state == nullptr || !state->valid || state->atoms != input.atoms.size() || state->spins != spins ||
        state->bands != bands || state->gamma != input.gamma || state->rank != input.rank ||
        state->ranks != input.ranks || state->kpoints.size() != input.kpoints.size())
        return false;
    for (std::size_t k = 0; k < input.kpoints.size(); ++k)
        for (std::size_t d = 0; d < 3; ++d)
            if (std::abs(state->kpoints[k][d] - input.kpoints[k][d]) > 1e-12) return false;
    return true;
}

}  // namespace

std::array<double, 2> extrapolation_coefficients(const std::vector<Vec3>& before_last, const std::vector<Vec3>& last,
                                                 const std::vector<Vec3>& now, const std::vector<Vec3>& next,
                                                 const Lattice& lattice) {
    const std::size_t n = now.size();
    if (before_last.size() != n || last.size() != n || next.size() != n)
        throw std::invalid_argument("the positions of the same atoms");
    // the displacement from p to q in bohr, of the nearest images
    auto move = [&](const Vec3& q, const Vec3& p) {
        Vec3 r{};
        for (std::size_t d = 0; d < 3; ++d) {
            const double f = q[d] - p[d] - std::round(q[d] - p[d]);
            for (std::size_t c = 0; c < 3; ++c) r[c] += f * lattice.a[d][c];
        }
        return r;
    };
    double aa = 0.0, ab = 0.0, bb = 0.0, ac = 0.0, bc = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const Vec3 a = move(now[i], last[i]), b = move(last[i], before_last[i]), c = move(next[i], now[i]);
        for (std::size_t d = 0; d < 3; ++d) {
            aa += a[d] * a[d];
            ab += a[d] * b[d];
            bb += b[d] * b[d];
            ac += a[d] * c[d];
            bc += b[d] * c[d];
        }
    }
    const double det = aa * bb - ab * ab;
    if (!(det > 1e-8 * aa * bb)) return {1.0, 0.0};
    return {(ac * bb - bc * ab) / det, (aa * bc - ab * ac) / det};
}

ScfResult run_scf(const ScfInput& input, const ScfOptions& options, ScfState* state) {
    const std::size_t points =
        static_cast<std::size_t>(input.grid[0]) * input.grid[1] * input.grid[2];
    const std::size_t nk = input.kpoints.size();
    const int nb = options.bands, ns = options.spins;
    if (ns != 1 && ns != 2) throw std::invalid_argument("one or two spins");
    const std::size_t spins = static_cast<std::size_t>(ns), bands = static_cast<std::size_t>(nb);
    if (nk == 0 || input.weights.size() != nk) throw std::invalid_argument("k-points and their weights");
    if (input.density.size() != spins * points) throw std::invalid_argument("rho_in of each spin on the grid");
    if (nb < 1) throw std::invalid_argument("at least one band");
    if (input.ranks < 1 || input.rank < 0 || input.rank >= input.ranks || (input.ranks > 1 && !input.allreduce))
        throw std::invalid_argument("a pool of the pools, and their sum");
    // the pairs (s, k) of this pool, and the k-points with any of them
    auto mine = [&](std::size_t s, std::size_t k) {
        return static_cast<int>((s * nk + k) % static_cast<std::size_t>(input.ranks)) == input.rank;
    };
    auto any_mine = [&](std::size_t k) { return mine(0, k) || (ns == 2 && mine(1, k)); };
    auto sum = [&](double* x, std::size_t count) {
        if (input.ranks > 1 && count > 0) input.allreduce(x, count);
    };
    if (options.max_iterations < 1) throw std::invalid_argument("max_iterations must be positive");
    const double f_max = ns == 2 ? 1.0 : 2.0;
    const Lattice* basis = input.has_basis ? &input.basis : nullptr;
    Clock clock;
    // the bands on the smooth grid, the density and the potentials on the
    // dense one; the smooth sphere is the start of the dense one
    const Int3 smooth = input.smooth_grid[0] > 0 ? input.smooth_grid : input.grid;
    const bool two_grids = smooth != input.grid;
    const std::size_t points_s = static_cast<std::size_t>(smooth[0]) * smooth[1] * smooth[2];
    Hamiltonian h(input.lattice, smooth, input.cutoff, input.species, input.atoms, basis);
    if (input.gamma) {
        const Vec3& k = input.kpoints.front();
        if (nk != 1 || k[0] != 0.0 || k[1] != 0.0 || k[2] != 0.0)
            throw std::invalid_argument("real bands take the one k-point Gamma");
        h.set_gamma(true);
    }
    Potentials pot(input.lattice, input.grid, input.density_cutoff, options.functional, ns, options.xc_threshold,
                   basis);
    std::unique_ptr<Potentials> pot_s;
    std::size_t ng_s = 0;
    if (two_grids) {
        pot_s = std::make_unique<Potentials>(input.lattice, smooth, 4.0 * input.cutoff, options.functional, ns,
                                             options.xc_threshold, basis);
        ng_s = pot_s->sphere().size();
        if (ng_s > pot.sphere().size()) throw std::invalid_argument("the dense grid is smaller than the smooth one");
        for (std::size_t g = 0; g < ng_s; ++g)
            if (pot_s->sphere().miller[g] != pot.sphere().miller[g])
                throw std::logic_error("the smooth sphere is not the start of the dense one");
    }
    DeviceArray<Complex> smooth_g(two_grids ? spins * ng_s : 0);
    DeviceArray<double> veff_s(two_grids ? points_s : 0), rho_s(two_grids ? spins * points_s : 0);
    DeviceArray<Complex> veff_dense_g(two_grids ? pot.sphere().size() : 0);
    // V on the dense grid to the Hamiltonian, through the smooth sphere with two
    auto set_potential = [&](const DeviceArray<double>& v) {
        if (!two_grids) {
            h.set_potential(v.data());
            return;
        }
        pot.to_sphere(v.data(), veff_dense_g.data(), 1);
        pot_s->from_sphere(veff_dense_g.data(), veff_s.data(), 1);
        h.set_potential(veff_s.data());
    };
    // V_loc and rho_core of the atoms on the grid
    const std::size_t ng = pot.sphere().size();
    Ions ions(input.lattice, pot.sphere(), input.atoms, static_cast<int>(input.species.size()),
              input.local_factors, input.core_factors);
    DeviceArray<Complex> components(spins * ng);
    DeviceArray<double> local(points);
    ions.local_components(components.data());
    pot.from_sphere(components.data(), local.data(), 1);
    if (!input.local_slopes.empty()) ions.set_slopes(input.local_slopes, input.core_slopes);
    if (!input.atom_factors.empty()) ions.set_atom_factors(input.atom_factors);
    if (ions.has_core()) {
        DeviceArray<double> core(points);
        ions.core_components(components.data());
        pot.from_sphere(components.data(), core.data(), 1);
        std::vector<double> host(points);
        core.download(host.data(), points);
        pot.set_core(host.data());
    }
    // ultrasoft species: their augmentation, the overlaps and the strengths of
    // the bare D for each atom
    std::vector<SpeciesAugmentation> augmented = input.augmentation;
    augmented.resize(input.species.size());
    Augmentation aug(input.lattice, pot.sphere(), input.atoms, augmented);
    const bool ultrasoft = aug.any();
    // DFT+U: the block of the pseudo-orbital of each Hubbard atom, where it
    // starts among the projector functions of the atom, its l and U
    struct HubbardBlock {
        std::size_t atom;
        int first, l;
        double u;
    };
    std::vector<HubbardBlock> hubbard_blocks;
    if (!input.hubbard_function.empty()) {
        if (input.hubbard_function.size() != input.species.size() || input.hubbard_u.size() != input.species.size())
            throw std::invalid_argument("a Hubbard function and U of every species");
        for (std::size_t i = 0; i < input.atoms.size(); ++i) {
            const auto sp = static_cast<std::size_t>(input.atoms[i].species);
            const int f = input.hubbard_function[sp];
            if (f < 0) continue;
            const auto& s = input.species[sp];
            if (f >= static_cast<int>(s.angular.size())) throw std::invalid_argument("the Hubbard function of a species");
            if (augmented[sp].pairs() > 0)
                throw std::invalid_argument("DFT+U takes norm-conserving pseudopotentials only, for now");
            int first = 0;
            for (int q = 0; q < f; ++q) first += 2 * s.angular[static_cast<std::size_t>(q)] + 1;
            hubbard_blocks.push_back({i, first, s.angular[static_cast<std::size_t>(f)], input.hubbard_u[sp]});
        }
    }
    const bool hubbard = !hubbard_blocks.empty();
    std::vector<double> bare;  // the blocks of D0 of the atoms
    std::vector<std::size_t> pair_offsets, block_offsets;  // of the packed pairs and of the blocks of each atom
    std::vector<double> overlaps;                          // the blocks of q of the atoms, ultrasoft
    if (ultrasoft || hubbard) {
        std::size_t offset = 0;
        for (const Atom& atom : input.atoms) {
            const auto& sp = input.species[static_cast<std::size_t>(atom.species)];
            const int w = width_of(sp);
            const auto block = species_block(sp);
            bare.insert(bare.end(), block.begin(), block.end());
            std::vector<double> q(static_cast<std::size_t>(w) * w, 0.0);
            const auto& sa = augmented[static_cast<std::size_t>(atom.species)];
            if (sa.pairs() > 0) {
                if (sa.functions != w) throw std::invalid_argument("the augmentation and the projectors of a species disagree");
                const auto packed = aug.overlaps(atom.species);
                add_packed(packed.data(), w, q.data());
            }
            overlaps.insert(overlaps.end(), q.begin(), q.end());
            block_offsets.push_back(bare.size() - block.size());
            pair_offsets.push_back(offset);
            offset += static_cast<std::size_t>(w) * (w + 1) / 2;
        }
        pair_offsets.push_back(offset);
        if (ultrasoft) h.set_overlaps(overlaps);
    }
    // the strengths of each spin: D0 + Omega sum_G conj(V_s(G)) Q(G) of a potential
    std::vector<std::vector<double>> strengths(spins);
    DeviceArray<Complex> veff_g(ng);
    DeviceArray<double> veff_r(ultrasoft ? points : 0);
    auto potential_of = [&](std::size_t s, const DeviceArray<double>& v) {
        add_kernel<<<blocks(points), kThreads>>>(points, local.data(), v.data() + s * points, veff_r.data());
        check_cuda(cudaGetLastError(), "the effective potential");
        pot.to_sphere(veff_r.data(), veff_g.data(), 1);
    };
    auto update_strengths = [&](const DeviceArray<double>& v) {
        if (!ultrasoft) {
            if (hubbard) std::fill(strengths.begin(), strengths.end(), bare);
            return;
        }
        for (std::size_t s = 0; s < spins; ++s) {
            potential_of(s, v);
            const auto packed = aug.strengths(veff_g.data());
            strengths[s] = bare;
            std::size_t at = 0;
            for (std::size_t i = 0; i < input.atoms.size(); ++i) {
                const int w = width_of(input.species[static_cast<std::size_t>(input.atoms[i].species)]);
                const std::size_t first = pair_offsets[i];
                if (pair_offsets[i + 1] > first && first < packed.size())
                    add_packed(packed.data() + first, w, strengths[s].data() + at);
                at += static_cast<std::size_t>(w) * w;
            }
        }
    };
    std::vector<std::vector<double>> becsum(spins);
    std::vector<DeviceArray<double>> becsum_d(spins);  // the same on the device, during the density
    // DFT+U from occupations (spins, pairs) as becsum holds them, n of a spin
    // channel their half with one channel: E_U = sum (U/2) Tr[n (1 - n)]; with
    // add, v = U (1/2 - n) into the strengths of each spin; vn = sum Tr[v n]
    const double channels = ns == 2 ? 1.0 : 2.0;
    auto hubbard_occupation = [&](const std::vector<double>& occ, const HubbardBlock& b, int x, int y) {
        const int w = width_of(input.species[static_cast<std::size_t>(input.atoms[b.atom].species)]);
        const int i = b.first + std::min(x, y), j = b.first + std::max(x, y);
        const std::size_t k = static_cast<std::size_t>(i) * static_cast<std::size_t>(w) -
                              static_cast<std::size_t>(i) * static_cast<std::size_t>(i - 1) / 2 +
                              static_cast<std::size_t>(j - i);
        return occ[pair_offsets[b.atom] + k] * (x == y ? 1.0 : 0.5) / channels;
    };
    auto hubbard_of = [&](const std::vector<std::vector<double>>& occ, bool add, double* vn) {
        double energy = 0.0, sum_vn = 0.0;
        for (const auto& b : hubbard_blocks) {
            const int w = width_of(input.species[static_cast<std::size_t>(input.atoms[b.atom].species)]);
            const int d = 2 * b.l + 1;
            for (std::size_t s = 0; s < spins; ++s)
                for (int x = 0; x < d; ++x)
                    for (int y = 0; y < d; ++y) {
                        const double n = hubbard_occupation(occ[s], b, x, y);
                        const double v = b.u * ((x == y ? 0.5 : 0.0) - n);
                        energy += channels * 0.5 * b.u * ((x == y ? n : 0.0) - n * n);
                        sum_vn += channels * v * n;
                        if (add)
                            strengths[s][block_offsets[b.atom] + static_cast<std::size_t>((b.first + x) * w + b.first + y)] += v;
                    }
        }
        if (vn != nullptr) *vn = sum_vn;
        return energy;
    };
    // sum Tr[v(a) n(b)], v of the occupations a on the occupations b
    auto hubbard_cross = [&](const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b_) {
        double r = 0.0;
        for (const auto& b : hubbard_blocks)
            for (std::size_t s = 0; s < spins; ++s)
                for (int x = 0; x < 2 * b.l + 1; ++x)
                    for (int y = 0; y < 2 * b.l + 1; ++y)
                        r += channels * b.u * ((x == y ? 0.5 : 0.0) - hubbard_occupation(a[s], b, x, y)) *
                             hubbard_occupation(b_[s], b, x, y);
        return r;
    };
    // sum (U/2) |n_out - n_in|^2, the part of the occupations in the accuracy
    auto hubbard_residual = [&](const std::vector<std::vector<double>>& in, const std::vector<std::vector<double>>& out) {
        double r = 0.0;
        for (const auto& b : hubbard_blocks)
            for (std::size_t s = 0; s < spins; ++s)
                for (int x = 0; x < 2 * b.l + 1; ++x)
                    for (int y = 0; y < 2 * b.l + 1; ++y) {
                        const double dn = hubbard_occupation(out[s], b, x, y) - hubbard_occupation(in[s], b, x, y);
                        r += channels * 0.5 * b.u * dn * dn;
                    }
        return r;
    };
    // the symmetry of the crystal, on the density and the occupations
    Symmetrizer symmetrizer(input.symmetry, pot.sphere().miller);
    std::vector<std::vector<int>> atom_angular;
    for (const Atom& atom : input.atoms) atom_angular.push_back(input.species[static_cast<std::size_t>(atom.species)].angular);
    // PAW: the input occupations, mixed, and the one-center strengths of them,
    // of the callable given or of the GPU
    std::unique_ptr<OneCenter> centers;
    std::function<double(const std::vector<std::vector<double>>&, std::vector<std::vector<double>>&)> one =
        input.onecenter;
    if (!one && std::any_of(input.paw.begin(), input.paw.end(), [](const SpeciesOneCenter& x) { return x.mesh > 0; })) {
        std::vector<int> kinds, widths;
        for (const Atom& atom : input.atoms) {
            kinds.push_back(atom.species);
            widths.push_back(width_of(input.species[static_cast<std::size_t>(atom.species)]));
        }
        std::vector<SpeciesOneCenter> data = input.paw;
        data.resize(input.species.size());
        centers = std::make_unique<OneCenter>(std::move(data), kinds, widths, ns, options.functional,
                                              options.xc_threshold);
        one = [&centers](const std::vector<std::vector<double>>& b, std::vector<std::vector<double>>& d) {
            return centers->compute(b, d);
        };
    }
    const bool paw = static_cast<bool>(one);
    if (paw && !ultrasoft) throw std::invalid_argument("PAW without augmentation");
    std::vector<std::vector<double>> becsum_in, onecenter(spins);
    double e1 = 0.0;
    auto add_onecenter = [&](const std::vector<std::vector<double>>& d) {
        for (std::size_t s = 0; s < spins; ++s) {
            if (d[s].size() != pair_offsets.back()) throw std::invalid_argument("the one-center strengths of each spin");
            std::size_t at = 0;
            for (std::size_t i = 0; i < input.atoms.size(); ++i) {
                const int w = width_of(input.species[static_cast<std::size_t>(input.atoms[i].species)]);
                add_packed(d[s].data() + pair_offsets[i], w, strengths[s].data() + at);
                at += static_cast<std::size_t>(w) * w;
            }
        }
    };
    Davidson davidson;
    // the bands of each spin and k-point, (s, k) at s nk + k, random at the
    // start and the same for both spins, so that a density without
    // magnetization keeps none
    std::vector<DeviceArray<Complex>> psi(spins * nk);
    const bool restart = fits(state, input, ns, nb);
    std::vector<std::vector<Int3>> millers(nk);
    for (std::size_t k = 0; k < nk; ++k) {
        if (!any_mine(k)) continue;
        h.set_k(input.kpoints[k]);
        if (bands > h.plane_waves()) throw std::invalid_argument("more bands than plane waves at a k-point");
        millers[k] = h.sphere().miller;
        if (restart) {
            // the bands of the last run on these plane waves
            const auto index = match(state->millers[k], millers[k]);
            DeviceArray<std::int64_t> where;
            where.upload(index.data(), index.size());
            const std::size_t n = h.plane_waves(), m = state->millers[k].size();
            for (std::size_t s = 0; s < spins; ++s) {
                if (!mine(s, k)) continue;
                psi[s * nk + k].resize(n * bands);
                match_kernel<<<blocks(n * bands), kThreads>>>(n, m, nb, where.data(), state->psi[s * nk + k].data(),
                                                              psi[s * nk + k].data());
                check_cuda(cudaGetLastError(), "the bands of the last run");
            }
            continue;
        }
        const auto start = random_bands(h.sphere().kinetic, nb, options.seed * 1000003ULL + k, input.gamma);
        for (std::size_t s = 0; s < spins; ++s)
            if (mine(s, k)) psi[s * nk + k].upload(start.data(), start.size());
    }
    // the mixer of (total, magnetization) on the sphere of the density
    std::vector<double> g2(ng);
    for (std::size_t g = 0; g < ng; ++g) g2[g] = 2.0 * pot.sphere().kinetic[g];
    Mixer mixer(g2, input.lattice.volume, ns, options.mixing);
    DeviceArray<double> rho_in, rho_out, hxc(spins * points), veff(points), hxc_output;
    // adaptive damping: the step taken last (its input on the sphere, output,
    // potentials of both, output energy and residual), the step of the mixing
    // from it, and the factor on it tried
    const bool adaptive = options.adaptive_damping;
    if (adaptive && !hubbard_blocks.empty()) throw std::invalid_argument("adaptive damping without DFT+U");
    const bool outputs = options.output_energies || adaptive;
    DeviceArray<double> taken_out, taken_hxc, taken_hxc_out, d_out, d_v, d_vo, d_r;
    // the local density of states of the preconditioner (on the grid of the
    // density, and on the smooth one), and the work of its operator
    DeviceArray<double> ldos_r, ldos_s, ldos_v, ldos_out;
    DeviceArray<Complex> ldos_g;
    std::vector<Complex> taken_in, delta;
    std::vector<std::vector<double>> taken_becsum, delta_becsum;
    double taken_energy = 0.0, taken_accuracy = 0.0, trial = 1.0, scale = 1.0;
    bool any_taken = false;
    int trials = 0;
    constexpr int kTrials = 5;           // trials of one step, the last taken whatever it gives
    constexpr double kLeastScale = 0.25;  // of the shares of the mixing, Herbst and Levitt's 0.2 of 0.8
    constexpr double kLeastTrial = 0.05;  // the smallest factor on a step
    rho_in.upload(input.density.data(), spins * points);
    rho_out.resize(spins * points);
    std::vector<Complex> in_g(spins * ng), out_g(spins * ng), last_in;
    pot.to_sphere(rho_in.data(), components.data(), ns);
    components.download(in_g.data(), in_g.size());
    if (ns == 2) to_total(in_g, ng);
    const std::vector<Complex> atoms_g = in_g;  // of the free atoms where they are
    if (restart) {
        // the last output density less its free atoms plus these; with more
        // runs kept, their differences extrapolated to the new positions
        const std::size_t used = std::min<std::size_t>(state->past.size(), static_cast<std::size_t>(state->steps));
        double alpha = 0.0, beta = 0.0;
        if (used >= 2) {
            alpha = 1.0;
            if (used >= 3) {
                std::vector<Vec3> next(input.atoms.size());
                for (std::size_t i = 0; i < next.size(); ++i) next[i] = input.atoms[i].position;
                const auto ab = extrapolation_coefficients(state->past[2].positions, state->past[1].positions,
                                                           state->past[0].positions, next, input.lattice);
                alpha = ab[0];
                beta = ab[1];
            }
        }
        // d(t+1) = (1 + alpha) d(t) + (beta - alpha) d(t-1) - beta d(t-2)
        const double weight[3] = {1.0 + alpha, beta - alpha, -beta};
        for (std::size_t j = 0; j < used; ++j) {
            const auto& p = state->past[j];
            const auto index = match(p.miller, pot.sphere().miller);
            const std::size_t m = p.miller.size();
            for (std::size_t c = 0; c < spins; ++c)
                for (std::size_t g = 0; g < ng; ++g) {
                    const std::int64_t from = index[g];
                    if (from < 0) continue;
                    const Complex d = p.difference[c * m + static_cast<std::size_t>(from)];
                    in_g[c * ng + g] = make_cuDoubleComplex(in_g[c * ng + g].x + weight[j] * d.x,
                                                            in_g[c * ng + g].y + weight[j] * d.y);
                }
        }
        std::vector<Complex> start = in_g;
        if (ns == 2) to_spins(start, ng);
        components.upload(start.data(), start.size());
        pot.from_sphere(components.data(), rho_in.data(), ns);
    }
    const bool mixes_occupations = paw || hubbard;  // the occupations of the projectors mixed with the density
    if (mixes_occupations) {
        becsum_in = restart && state->becsum.size() == spins ? state->becsum : input.becsum;
        if (becsum_in.size() != spins) throw std::invalid_argument("the first occupations of each spin");
        for (const auto& b : becsum_in)
            if (b.size() != pair_offsets.back()) throw std::invalid_argument("the first occupations of each atom");
    }
    ScfResult result;
    result.restarted = restart;
    result.ewald = input.ewald;
    // the k-points of each spin, their weights and eigenvalues
    std::vector<double> weights(spins * nk), eigenvalues(spins * nk * bands), values, w(bands);
    for (std::size_t s = 0; s < spins; ++s)
        std::copy(input.weights.begin(), input.weights.end(), weights.begin() + static_cast<std::ptrdiff_t>(s * nk));
    // the first bands of a cold start: the lowest Ritz vectors of the
    // pseudo-orbitals of the free atoms in the first potential, with random
    // bands after them when they are fewer than the bands; the random bands as
    // they are when the orbitals are more than twice the bands (memory) or
    // linearly dependent
    auto start_atomic = [&](std::size_t row) {
        const std::size_t n = h.plane_waves();
        const int m = h.atomic_functions(), total = std::max(m, nb);
        if (m > 2 * nb || static_cast<std::size_t>(total) > n) return;
        DeviceArray<Complex> start(n * static_cast<std::size_t>(total));
        h.atomic(start.data());
        if (m < nb)
            check_cuda(cudaMemcpy(start.data() + n * static_cast<std::size_t>(m),
                                  psi[row].data() + n * static_cast<std::size_t>(m),
                                  n * static_cast<std::size_t>(nb - m) * sizeof(Complex), cudaMemcpyDeviceToDevice),
                       "cudaMemcpy");
        try {
            std::vector<double> e, loose(static_cast<std::size_t>(total), 1e-2);
            davidson.solve(h, start.data(), total, e, loose, 1, total);
        } catch (const std::invalid_argument&) {
            return;
        }
        check_cuda(cudaMemcpy(psi[row].data(), start.data(), n * bands * sizeof(Complex), cudaMemcpyDeviceToDevice),
                   "cudaMemcpy");
    };
    // the residual norm of the bands of the first step: loose from random or
    // atomic bands; from the bands of the last run, tight, as Quantum ESPRESSO
    // starts the steps of dynamics (1e-6 Ry on the eigenvalues), since loose
    // bands there would give an output density worse than the input
    double tolerance = options.first_tolerance > 0.0 ? options.first_tolerance : (restart ? 1e-4 : 1e-2);
    double previous = std::numeric_limits<double>::infinity();
    // the occupations of the step before, which weight the tolerance of each
    // band: tolerance / sqrt(|f|), never above the loose one of empty bands
    std::vector<double> held;
    // before them, the bands that may hold electrons (all by default)
    const std::size_t first_held = static_cast<std::size_t>(std::max(options.first_held, 0));
    std::vector<double> limits(bands);
    // the pairs (s, k) of this pool, and the host threads that take them side
    // by side: small cells of many k-points, whose bands alone leave the GPU
    // idle; each thread its own Hamiltonian, solver, density and occupations
    std::vector<std::pair<std::size_t, std::size_t>> tasks;
    for (std::size_t k = 0; k < nk; ++k)
        for (std::size_t s = 0; s < spins; ++s)
            if (mine(s, k)) tasks.emplace_back(s, k);
    std::size_t largest = 0;
    for (const auto& m : millers) largest = std::max(largest, m.size());
    int threads = options.kpoint_threads;
    if (threads == 0) threads = (input.gamma || tasks.size() < 4 || largest * bands > 500000) ? 1 : 8;
    threads = static_cast<int>(std::min<std::size_t>(static_cast<std::size_t>(std::max(threads, 1)), tasks.size()));
    if (input.gamma) threads = 1;
    struct Worker {
        Hamiltonian h;
        Davidson davidson;
        DeviceArray<double> rho;
        std::vector<DeviceArray<double>> becsum;
        std::vector<double> values, limits, w;
        int iterations = 0;
        double band = 0.0;
        std::exception_ptr error;
        Worker(const Lattice& lattice, const Int3& grid, double cutoff, const std::vector<SpeciesProjectors>& species,
               const std::vector<Atom>& atoms, const Lattice* basis_)
            : h(lattice, grid, cutoff, species, atoms, basis_) {}
    };
    std::vector<std::unique_ptr<Worker>> workers;
    std::vector<DeviceArray<double>> veff_spin;  // the effective potential of each spin on the grid of the bands
    if (threads > 1) {
        for (int i = 0; i < threads; ++i) {
            workers.push_back(std::make_unique<Worker>(input.lattice, smooth, input.cutoff, input.species,
                                                       input.atoms, basis));
            if (ultrasoft) workers.back()->h.set_overlaps(overlaps);
            workers.back()->becsum.resize(spins);
            workers.back()->limits.resize(bands);
            workers.back()->w.resize(bands);
        }
        veff_spin.resize(spins);
        for (auto& v : veff_spin) v.resize(two_grids ? points_s : points);
        check_cuda(cudaStreamSynchronize(cudaStreamPerThread), "cudaStreamSynchronize");
    }
    // each task of tasks by one of the workers, f(worker, s, k); the work of
    // this thread done first, the errors of the workers raised here
    auto side_by_side = [&](const std::function<void(Worker&, std::size_t, std::size_t)>& f) {
        check_cuda(cudaStreamSynchronize(cudaStreamPerThread), "cudaStreamSynchronize");
        std::atomic<std::size_t> next{0};
        std::vector<std::thread> pool;
        for (auto& worker : workers) {
            Worker* wk = worker.get();
            wk->error = nullptr;
            pool.emplace_back([&, wk] {
                try {
                    for (std::size_t i = next++; i < tasks.size(); i = next++) f(*wk, tasks[i].first, tasks[i].second);
                    check_cuda(cudaStreamSynchronize(cudaStreamPerThread), "cudaStreamSynchronize");
                } catch (...) {
                    wk->error = std::current_exception();
                }
            });
        }
        for (auto& t : pool) t.join();
        for (auto& worker : workers)
            if (worker->error) std::rethrow_exception(worker->error);
    };
    for (int iteration = 1; iteration <= options.max_iterations; ++iteration) {
        // the potential of rho_in
        if (iteration == 1) result.times.setup += clock.lap();
        const PotentialEnergies e = pot.compute(rho_in.data(), hxc.data());
        update_strengths(hxc);
        double paw_energy = 0.0, hubbard_energy = 0.0, hubbard_vn = 0.0;
        if (paw) {
            e1 = one(becsum_in, onecenter);
            // the one-center strengths made symmetric too, as QE does: its
            // quadrature of directions is not
            for (std::size_t s = 0; s < spins; ++s)
                symmetrizer.occupations(onecenter[s], pair_offsets, atom_angular, false);
            add_onecenter(onecenter);
            paw_energy = e1;
            for (std::size_t s = 0; s < spins; ++s)
                for (std::size_t i = 0; i < becsum_in[s].size(); ++i) paw_energy -= onecenter[s][i] * becsum_in[s][i];
        }
        if (hubbard) result.hubbard = hubbard_of(becsum_in, true, &hubbard_vn);
        result.times.potentials += clock.lap();
        last_in = in_g;
        // the bands
        ScfStep step;
        step.tolerance = tolerance;
        if (threads > 1) {
            // the potential of each spin once, then the k-points on the workers
            for (std::size_t s = 0; s < spins; ++s) {
                add_kernel<<<blocks(points), kThreads>>>(points, local.data(), hxc.data() + s * points, veff.data());
                check_cuda(cudaGetLastError(), "the effective potential");
                if (two_grids) {
                    pot.to_sphere(veff.data(), veff_dense_g.data(), 1);
                    pot_s->from_sphere(veff_dense_g.data(), veff_spin[s].data(), 1);
                } else {
                    check_cuda(cudaMemcpy(veff_spin[s].data(), veff.data(), points * sizeof(double),
                                          cudaMemcpyDeviceToDevice),
                               "cudaMemcpy");
                }
            }
            for (auto& worker : workers) worker->iterations = 0;
            side_by_side([&](Worker& wk, std::size_t s, std::size_t k) {
                wk.h.set_k(input.kpoints[k]);
                wk.h.set_potential(veff_spin[s].data());
                if (ultrasoft || hubbard) wk.h.set_strengths(strengths[s]);
                const double loose = std::max(tolerance, 1e-2);
                for (std::size_t b = 0; b < bands; ++b) {
                    const double f = held.empty() ? (first_held > 0 && b >= first_held ? 0.0 : 1.0)
                                                   : std::abs(held[(s * nk + k) * bands + b]);
                    wk.limits[b] = f > 1e-10 ? std::min(loose, tolerance / std::sqrt(std::min(f, 1.0))) : loose;
                }
                const std::size_t row = s * nk + k;
                if (iteration == 1 && !restart && wk.h.atomic_functions() > 0) {
                    // the lowest Ritz vectors of the pseudo-orbitals, as start_atomic
                    const std::size_t n = wk.h.plane_waves();
                    const int m = wk.h.atomic_functions(), total = std::max(m, nb);
                    if (m <= 2 * nb && static_cast<std::size_t>(total) <= n) {
                        DeviceArray<Complex> start(n * static_cast<std::size_t>(total));
                        wk.h.atomic(start.data());
                        if (m < nb)
                            check_cuda(cudaMemcpy(start.data() + n * static_cast<std::size_t>(m),
                                                  psi[row].data() + n * static_cast<std::size_t>(m),
                                                  n * static_cast<std::size_t>(nb - m) * sizeof(Complex),
                                                  cudaMemcpyDeviceToDevice),
                                       "cudaMemcpy");
                        bool solved = true;
                        try {
                            std::vector<double> ev, first(static_cast<std::size_t>(total), 1e-2);
                            wk.davidson.solve(wk.h, start.data(), total, ev, first, 1, total);
                        } catch (const std::invalid_argument&) {
                            solved = false;
                        }
                        if (solved)
                            check_cuda(cudaMemcpy(psi[row].data(), start.data(), n * bands * sizeof(Complex),
                                                  cudaMemcpyDeviceToDevice),
                                       "cudaMemcpy");
                    }
                }
                const auto r = wk.davidson.solve(wk.h, psi[row].data(), nb, wk.values, wk.limits,
                                                 options.davidson_iterations, options.davidson_basis * nb);
                wk.iterations += r.iterations;
                std::copy(wk.values.begin(), wk.values.end(),
                          eigenvalues.begin() + static_cast<std::ptrdiff_t>(row * bands));
            });
            for (auto& worker : workers) step.davidson += worker->iterations;
        }
        for (std::size_t k = 0; k < nk && threads == 1; ++k) {
            if (!any_mine(k)) continue;
            h.set_k(input.kpoints[k]);
            for (std::size_t s = 0; s < spins; ++s) {
                if (!mine(s, k)) continue;
                add_kernel<<<blocks(points), kThreads>>>(points, local.data(), hxc.data() + s * points, veff.data());
                check_cuda(cudaGetLastError(), "the effective potential");
                set_potential(veff);
                if (ultrasoft || hubbard) h.set_strengths(strengths[s]);
                const double loose = std::max(tolerance, 1e-2);
                for (std::size_t b = 0; b < bands; ++b) {
                    const double f = held.empty() ? (first_held > 0 && b >= first_held ? 0.0 : 1.0)
                                                   : std::abs(held[(s * nk + k) * bands + b]);
                    limits[b] = f > 1e-10 ? std::min(loose, tolerance / std::sqrt(std::min(f, 1.0))) : loose;
                }
                if (iteration == 1 && !restart && h.atomic_functions() > 0) start_atomic(s * nk + k);
                const auto r = davidson.solve(h, psi[s * nk + k].data(), nb, values, limits,
                                              options.davidson_iterations, options.davidson_basis * nb);
                step.davidson += r.iterations;
                std::copy(values.begin(), values.end(),
                          eigenvalues.begin() + static_cast<std::ptrdiff_t>((s * nk + k) * bands));
            }
        }
        if (input.ranks > 1) {  // the eigenvalues of every pool
            for (std::size_t s = 0; s < spins; ++s)
                for (std::size_t k = 0; k < nk; ++k)
                    if (!mine(s, k))
                        std::fill_n(eigenvalues.begin() + static_cast<std::ptrdiff_t>((s * nk + k) * bands), bands, 0.0);
            sum(eigenvalues.data(), eigenvalues.size());
            double count = step.davidson;
            sum(&count, 1);
            step.davidson = static_cast<int>(std::lround(count));
        }
        result.times.bands += clock.lap();
        const Occupation occ =
            occupy(eigenvalues, weights, nb, options.electrons, f_max, options.smearing, options.sigma);
        // the output density and the energy
        rho_out.zero();
        if (two_grids) rho_s.zero();
        if (ultrasoft || hubbard)
            for (std::size_t s = 0; s < spins; ++s) {
                becsum[s].assign(pair_offsets.back(), 0.0);
                becsum_d[s].resize(pair_offsets.back());
            }
        double band = 0.0;
        if (ultrasoft || hubbard)  // the occupations of the projectors gathered on the device
            for (auto& b : becsum_d) b.zero();
        if (threads > 1) {
            // each worker its share of the density and of the occupations,
            // added here
            const std::size_t per = two_grids ? points_s : points;
            for (auto& worker : workers) {
                worker->band = 0.0;
                worker->rho.resize(spins * per);
                worker->rho.zero();
                if (ultrasoft || hubbard)
                    for (auto& b : worker->becsum) {
                        b.resize(pair_offsets.back());
                        b.zero();
                    }
            }
            side_by_side([&](Worker& wk, std::size_t s, std::size_t k) {
                wk.h.set_k(input.kpoints[k]);
                const std::size_t row = (s * nk + k) * bands;
                for (std::size_t b = 0; b < bands; ++b) {
                    wk.w[b] = weights[s * nk + k] * f_max * occ.f[row + b];
                    wk.band += wk.w[b] * eigenvalues[row + b];
                }
                wk.h.add_density(psi[s * nk + k].data(), nb, wk.w.data(), wk.rho.data() + s * per);
                if (ultrasoft || hubbard)
                    wk.h.add_becsum_device(psi[s * nk + k].data(), nb, wk.w.data(), wk.becsum[s].data());
            });
            DeviceArray<double>& target = two_grids ? rho_s : rho_out;
            for (auto& worker : workers) {
                band += worker->band;
                add_kernel<<<blocks(spins * per), kThreads>>>(spins * per, target.data(), worker->rho.data(),
                                                             target.data());
                check_cuda(cudaGetLastError(), "the density of the workers");
                if (ultrasoft || hubbard)
                    for (std::size_t s = 0; s < spins; ++s) {
                        add_kernel<<<blocks(pair_offsets.back()), kThreads>>>(
                            pair_offsets.back(), becsum_d[s].data(), worker->becsum[s].data(), becsum_d[s].data());
                        check_cuda(cudaGetLastError(), "the occupations of the workers");
                    }
            }
        }
        for (std::size_t k = 0; k < nk && threads == 1; ++k) {
            if (!any_mine(k)) continue;
            h.set_k(input.kpoints[k]);
            for (std::size_t s = 0; s < spins; ++s) {
                if (!mine(s, k)) continue;
                const std::size_t row = (s * nk + k) * bands;
                for (std::size_t b = 0; b < bands; ++b) {
                    w[b] = weights[s * nk + k] * f_max * occ.f[row + b];
                    band += w[b] * eigenvalues[row + b];
                }
                h.add_density(psi[s * nk + k].data(), nb, w.data(),
                              two_grids ? rho_s.data() + s * points_s : rho_out.data() + s * points);
                if (ultrasoft || hubbard) h.add_becsum_device(psi[s * nk + k].data(), nb, w.data(), becsum_d[s].data());
            }
        }
        if (ultrasoft || hubbard)
            for (std::size_t s = 0; s < spins; ++s) becsum_d[s].download(becsum[s].data(), becsum[s].size());
        if (input.ranks > 1) {  // the densities and occupations of every pool
            sum(&band, 1);
            DeviceArray<double>& partial = two_grids ? rho_s : rho_out;
            std::vector<double> host(spins * (two_grids ? points_s : points));
            partial.download(host.data(), host.size());
            sum(host.data(), host.size());
            partial.upload(host.data(), host.size());
            if (ultrasoft || hubbard)
                for (auto& b : becsum) sum(b.data(), b.size());
        }
        if (hubbard) {
            // the occupations of the orbitals, symmetric; on a cold start the
            // first output ones are taken as input, as Quantum ESPRESSO does
            for (std::size_t s = 0; s < spins; ++s)
                if (!ultrasoft) symmetrizer.occupations(becsum[s], pair_offsets, atom_angular);
            if (iteration == 1 && !restart) {
                hubbard_vn = hubbard_cross(becsum_in, becsum);  // the strengths of the bands on what comes in
                for (const auto& b : hubbard_blocks) {
                    const int w = width_of(input.species[static_cast<std::size_t>(input.atoms[b.atom].species)]);
                    for (int x = b.first; x < b.first + 2 * b.l + 1; ++x)
                        for (int y = x; y < b.first + 2 * b.l + 1; ++y) {
                            const std::size_t k = pair_offsets[b.atom] + static_cast<std::size_t>(x) * w -
                                                  static_cast<std::size_t>(x) * static_cast<std::size_t>(x - 1) / 2 +
                                                  static_cast<std::size_t>(y - x);
                            for (std::size_t s = 0; s < spins; ++s) becsum_in[s][k] = becsum[s][k];
                        }
                }
                result.hubbard = hubbard_of(becsum_in, false, nullptr);
            }
            hubbard_energy = result.hubbard - hubbard_vn;
        }
        const double energy =
            band - e.hxc_density + e.hartree + e.xc + input.ewald + occ.entropy + paw_energy + hubbard_energy;
        if (two_grids) {
            // the smooth density on the start of the dense sphere
            pot_s->to_sphere(rho_s.data(), smooth_g.data(), ns);
            components.zero();
            for (std::size_t s = 0; s < spins; ++s)
                check_cuda(cudaMemcpy(components.data() + s * ng, smooth_g.data() + s * ng_s, ng_s * sizeof(Complex),
                                      cudaMemcpyDeviceToDevice),
                           "cudaMemcpy");
        } else {
            pot.to_sphere(rho_out.data(), components.data(), ns);
        }
        if (ultrasoft)  // the occupations made symmetric, the augmentation of each spin
            for (std::size_t s = 0; s < spins; ++s) {
                symmetrizer.occupations(becsum[s], pair_offsets, atom_angular);
                aug.add_density(becsum[s], components.data() + s * ng);
            }
        symmetrizer.density(components.data(), ns);
        if (ultrasoft || two_grids || symmetrizer.active()) pot.from_sphere(components.data(), rho_out.data(), ns);
        components.download(out_g.data(), out_g.size());
        if (ns == 2) to_total(out_g, ng);
        result.times.density += clock.lap();
        const double accuracy = mixer.residual_energy(in_g, out_g) + (hubbard ? hubbard_residual(becsum_in, becsum) : 0.0);
        if (!std::isfinite(energy) || !std::isfinite(accuracy))
            throw std::runtime_error("the energy or the density is not finite");
        step.energy = energy;
        step.accuracy = accuracy;
        if (outputs) {
            // band - int V_Hxc[rho_in] rho_out + E_Hxc[rho_out], with PAW
            // E^1 of the output occupations less the strengths of the input on them
            step.output_energy = std::numeric_limits<double>::quiet_NaN();
            if (!hubbard) {
                hxc_output.resize(spins * points);
                const PotentialEnergies e_o = pot.compute(rho_out.data(), hxc_output.data());
                double paw_o = 0.0;
                if (paw) {
                    std::vector<std::vector<double>> strengths_o(spins);
                    paw_o = one(becsum, strengths_o);
                    for (std::size_t s = 0; s < spins; ++s)
                        for (std::size_t i = 0; i < becsum[s].size(); ++i) paw_o -= onecenter[s][i] * becsum[s][i];
                }
                step.output_energy = band - pot.integral(hxc.data(), rho_out.data(), static_cast<int>(spins)) +
                                     e_o.hartree + e_o.xc + input.ewald + occ.entropy + paw_o;
            }
        }
        step.fermi = occ.fermi;
        step.magnetization = ns == 2 ? out_g[ng].x * input.lattice.volume : 0.0;
        result.steps.push_back(step);
        result.iterations = iteration;
        result.energy = energy;
        result.band = band;
        result.one_electron = band - e.hxc_density;
        result.hartree = e.hartree;
        result.xc = e.xc;
        result.entropy = occ.entropy;
        result.fermi = occ.fermi;
        result.lowest_empty = occ.lowest_empty;
        result.eigenvalues = eigenvalues;
        result.occupations = occ.f;
        if (adaptive && any_taken) {
            // Herbst and Levitt, J. Comput. Phys. 459, 111127 (2022): this step
            // tried the input taken + trial delta; along delta, in u = 1 at this
            // trial, the energy phi(u) = E + u a - u^2 b / 2 with
            // a = <d rho_out, V_out - V> of the step taken (its residual in
            // potential) and b = <d rho_out, d V_in - d V_out>, d of this trial
            // from the step taken; a good model (its prediction within a tenth
            // of the change, convex) places the next trial at its minimum
            const std::size_t n = spins * points;
            d_out.resize(n);
            d_v.resize(n);
            d_vo.resize(n);
            d_r.resize(n);
            difference_kernel<<<blocks(n), kThreads>>>(n, rho_out.data(), taken_out.data(), d_out.data());
            difference_kernel<<<blocks(n), kThreads>>>(n, hxc.data(), taken_hxc.data(), d_v.data());
            difference_kernel<<<blocks(n), kThreads>>>(n, hxc_output.data(), taken_hxc_out.data(), d_vo.data());
            difference_kernel<<<blocks(n), kThreads>>>(n, taken_hxc_out.data(), taken_hxc.data(), d_r.data());
            check_cuda(cudaGetLastError(), "the differences of a step");
            const int c = static_cast<int>(spins);
            const double a = pot.integral(d_out.data(), d_r.data(), c);
            const double b = pot.integral(d_out.data(), d_v.data(), c) - pot.integral(d_out.data(), d_vo.data(), c);
            const double change = step.output_energy - taken_energy;
            const double ratio = std::abs(change - (a - 0.5 * b)) / std::max(std::abs(change), 1e-300);
            const bool good = ratio < 0.1 && b < 0.0;
            const double best = good ? a / b : 0.0;  // u of the minimum
            const bool take = step.output_energy < taken_energy || accuracy < taken_accuracy || trials >= kTrials;
            result.steps.back().damping = trial * scale;
            if (!take) {
                double u = 0.5;
                if (good && (best > 0.0 || ratio < 0.01)) u = std::copysign(std::min(std::abs(best), 0.95), best);
                if (std::abs(u * trial) < kLeastTrial) u = std::copysign(kLeastTrial / std::abs(trial), u);
                trial *= u;
                ++trials;
                result.steps.back().accepted = false;
                for (std::size_t i = 0; i < in_g.size(); ++i)
                    in_g[i] = make_cuDoubleComplex(taken_in[i].x + trial * delta[i].x, taken_in[i].y + trial * delta[i].y);
                if (mixes_occupations)
                    for (std::size_t s = 0; s < spins; ++s)
                        for (std::size_t i = 0; i < becsum_in[s].size(); ++i)
                            becsum_in[s][i] = taken_becsum[s][i] + trial * delta_becsum[s][i];
                std::vector<Complex> next = in_g;
                if (ns == 2) to_spins(next, ng);
                components.upload(next.data(), next.size());
                pot.from_sphere(components.data(), rho_in.data(), ns);
                result.times.mixing += clock.lap();
                continue;
            }
            // the shares of the mixing: those of the trial taken after trials
            // back, or up to 1.1 times the minimum of a good model after none
            if (trials > 0)
                scale = std::max(kLeastScale, scale * std::abs(trial));
            else if (good && best > 0.0)
                scale = std::min(1.0, std::max(scale, 1.1 * best * scale));
            mixer.set_scale(scale);
        }
        bool done = accuracy < options.density_tolerance && std::abs(energy - previous) < options.energy_tolerance;
        if (input.ranks > 1) {  // all pools stop together, when every one has converged
            double count = done ? 1.0 : 0.0;
            sum(&count, 1);
            done = count > input.ranks - 0.5;
        }
        previous = energy;
        if (done) {
            result.converged = true;
            break;
        }
        if (options.mixing.ldos) {
            // the local density of states at the Fermi level, both spins, by a
            // Gaussian of the width of the smearing, on the grid of the density
            // and symmetric; P = 1 - chi0 v_c with chi0 = -D(r) + D(r) D(r') / D
            ldos_r.resize(points);
            DeviceArray<double>& acc = two_grids ? ldos_s : ldos_r;
            acc.resize(two_grids ? points_s : points);
            acc.zero();
            const double width = options.sigma;
            std::vector<double> wd(bands);
            for (std::size_t k = 0; k < nk && width > 0.0; ++k) {
                if (!any_mine(k)) continue;
                h.set_k(input.kpoints[k]);
                for (std::size_t s = 0; s < spins; ++s) {
                    if (!mine(s, k)) continue;
                    const std::size_t row = (s * nk + k) * bands;
                    for (std::size_t b = 0; b < bands; ++b) {
                        const double x = (eigenvalues[row + b] - occ.fermi) / width;
                        wd[b] = weights[s * nk + k] * f_max * std::exp(-x * x) / (std::sqrt(3.14159265358979323846) * width);
                    }
                    h.add_density(psi[s * nk + k].data(), nb, wd.data(), acc.data());
                }
            }
            if (input.ranks > 1) {
                std::vector<double> host(acc.size());
                acc.download(host.data(), host.size());
                sum(host.data(), host.size());
                acc.upload(host.data(), host.size());
            }
            if (two_grids) {
                pot_s->to_sphere(ldos_s.data(), smooth_g.data(), 1);
                components.zero();
                check_cuda(cudaMemcpy(components.data(), smooth_g.data(), ng_s * sizeof(Complex), cudaMemcpyDeviceToDevice),
                           "cudaMemcpy");
            } else {
                pot.to_sphere(ldos_r.data(), components.data(), 1);
            }
            symmetrizer.density(components.data(), 1);
            pot.from_sphere(components.data(), ldos_r.data(), 1);
            clamp_kernel<<<blocks(points), kThreads>>>(points, ldos_r.data());
            check_cuda(cudaGetLastError(), "the local density of states");
            const double d_total = pot.integral(ldos_r.data(), nullptr, 1);
            if (d_total > 1e-8) {
                mixer.set_preconditioner([&, d_total](std::vector<Complex>& r) {
                    const auto apply = [&, d_total](const std::vector<Complex>& x) {
                        // x + D v_c x - D int D v_c x / D, on the sphere
                        std::vector<Complex> y(ng);
                        for (std::size_t g = 0; g < ng; ++g) {
                            const double g2_ = 2.0 * pot.sphere().kinetic[g];
                            const double f = g2_ > 0.0 ? 4.0 * 3.14159265358979323846 / g2_ : 0.0;
                            y[g] = make_cuDoubleComplex(f * x[g].x, f * x[g].y);
                        }
                        ldos_g.resize(ng);
                        ldos_v.resize(points);
                        ldos_out.resize(points);
                        ldos_g.upload(y.data(), ng);
                        pot.from_sphere(ldos_g.data(), ldos_v.data(), 1);
                        const double c = pot.integral(ldos_r.data(), ldos_v.data(), 1) / d_total;
                        screened_kernel<<<blocks(points), kThreads>>>(points, ldos_r.data(), ldos_v.data(), c,
                                                                       ldos_out.data());
                        check_cuda(cudaGetLastError(), "the screening of the local density of states");
                        pot.to_sphere(ldos_out.data(), ldos_g.data(), 1);
                        std::vector<Complex> z(ng);
                        ldos_g.download(z.data(), ng);
                        for (std::size_t g = 0; g < ng; ++g) z[g] = make_cuDoubleComplex(x[g].x + z[g].x, x[g].y + z[g].y);
                        z[0] = make_cuDoubleComplex(0.0, 0.0);
                        return z;
                    };
                    r = gmres(apply, r, 30, 1e-4);
                });
            } else {
                mixer.set_preconditioner([](std::vector<Complex>&) {});  // no states at the Fermi level: P = 1
            }
        }
        if (adaptive) {  // this step taken: what the next trials start from
            const std::size_t n = spins * points;
            copy_device(taken_out, rho_out, n);
            copy_device(taken_hxc, hxc, n);
            copy_device(taken_hxc_out, hxc_output, n);
            taken_in = in_g;
            taken_becsum = becsum_in;
            taken_energy = step.output_energy;
            taken_accuracy = accuracy;
            any_taken = true;
            trials = 0;
            trial = 1.0;
        }
        // the next input density, and with PAW the occupations of (total,
        // magnetization) along
        if (mixes_occupations) {
            const std::size_t n = pair_offsets.back();
            std::vector<double> x(spins * n), x_out(spins * n);
            for (std::size_t i = 0; i < n; ++i) {
                if (ns == 2) {
                    x[i] = becsum_in[0][i] + becsum_in[1][i];
                    x[n + i] = becsum_in[0][i] - becsum_in[1][i];
                    x_out[i] = becsum[0][i] + becsum[1][i];
                    x_out[n + i] = becsum[0][i] - becsum[1][i];
                } else {
                    x[i] = becsum_in[0][i];
                    x_out[i] = becsum[0][i];
                }
            }
            in_g = mixer.mix(in_g, out_g, &x, &x_out);
            for (std::size_t i = 0; i < n; ++i) {
                if (ns == 2) {
                    becsum_in[0][i] = 0.5 * (x[i] + x[n + i]);
                    becsum_in[1][i] = 0.5 * (x[i] - x[n + i]);
                } else {
                    becsum_in[0][i] = x[i];
                }
            }
        } else {
            in_g = mixer.mix(in_g, out_g);
        }
        if (adaptive) {  // the step of the mixing, of which the trials take a share
            delta.resize(in_g.size());
            for (std::size_t i = 0; i < in_g.size(); ++i)
                delta[i] = make_cuDoubleComplex(in_g[i].x - taken_in[i].x, in_g[i].y - taken_in[i].y);
            delta_becsum = becsum_in;
            for (std::size_t s = 0; s < delta_becsum.size(); ++s)
                for (std::size_t i = 0; i < delta_becsum[s].size(); ++i) delta_becsum[s][i] -= taken_becsum[s][i];
        }
        result.times.mixing += clock.lap();
        std::vector<Complex> next = in_g;
        if (ns == 2) to_spins(next, ng);
        components.upload(next.data(), next.size());
        pot.from_sphere(components.data(), rho_in.data(), ns);
        // the occupations and the tolerance of the next step
        held = occ.f;
        tolerance = std::clamp(0.01 * std::sqrt(accuracy), 1e-9, tolerance);
    }
    clock.lap();
    // the forces: the local term of rho_out, the core charge in V_xc of the
    // last rho_in (V_Hxc of each spin, averaged, less V_H), the projectors
    {
        DeviceArray<Complex> rho_g, vxc_g;
        rho_g.upload(out_g.data(), ng);
        pot.to_sphere(hxc.data(), components.data(), ns);
        std::vector<Complex> v(spins * ng);
        components.download(v.data(), v.size());
        for (std::size_t g = 0; g < ng; ++g) {
            Complex x = v[g];
            if (ns == 2) x = make_cuDoubleComplex(0.5 * (x.x + v[ng + g].x), 0.5 * (x.y + v[ng + g].y));
            const double g2_ = 2.0 * pot.sphere().kinetic[g];
            if (g2_ > 0.0) {
                const double f = 4.0 * 3.14159265358979323846 / g2_;
                x = make_cuDoubleComplex(x.x - f * last_in[g].x, x.y - f * last_in[g].y);
            }
            v[g] = x;
        }
        vxc_g.upload(v.data(), ng);
        result.forces = ions.forces(rho_g.data(), vxc_g.data());
        if (ultrasoft)  // the augmentation in the potential of the bands, V of rho_in
            for (std::size_t s = 0; s < spins; ++s) {
                potential_of(s, hxc);
                const auto f = aug.forces(veff_g.data(), becsum[s]);
                for (std::size_t i = 0; i < f.size(); ++i) result.forces[i] += f[i];
            }
        std::vector<double> nonlocal(result.forces.size(), 0.0);
        for (std::size_t k = 0; k < nk; ++k) {
            if (!any_mine(k)) continue;
            h.set_k(input.kpoints[k]);
            for (std::size_t s = 0; s < spins; ++s) {
                if (!mine(s, k)) continue;
                const std::size_t row = (s * nk + k) * bands;
                for (std::size_t b = 0; b < bands; ++b) w[b] = weights[s * nk + k] * f_max * result.occupations[row + b];
                if (ultrasoft || hubbard) h.set_strengths(strengths[s]);
                h.nonlocal_forces(psi[s * nk + k].data(), nb, w.data(), nonlocal, eigenvalues.data() + row);
            }
        }
        sum(nonlocal.data(), nonlocal.size());
        for (std::size_t i = 0; i < nonlocal.size(); ++i) result.forces[i] += nonlocal[i];
    }
    result.times.forces += clock.lap();
    // V_Hxc of rho_out, for the correction of the forces and the stress (the
    // last potentials computed, whose gradients the stress of GGA takes)
    DeviceArray<double> hxc_out(spins * points);
    const PotentialEnergies e_out = pot.compute(rho_out.data(), hxc_out.data());
    if (ions.has_atom_factors()) {
        // dV = V_Hxc[rho_out] - V_Hxc[rho_in], averaged over the spins
        std::vector<double> a(spins * points), b(spins * points);
        hxc_out.download(a.data(), a.size());
        hxc.download(b.data(), b.size());
        std::vector<double> dv(points);
        for (std::size_t i = 0; i < points; ++i) {
            double x = 0.0;
            for (std::size_t s = 0; s < spins; ++s) x += a[s * points + i] - b[s * points + i];
            dv[i] = x / static_cast<double>(spins);
        }
        DeviceArray<double> dv_r;
        dv_r.upload(dv.data(), points);
        DeviceArray<Complex> dv_g(ng);
        pot.to_sphere(dv_r.data(), dv_g.data(), 1);
        const auto correction = ions.correction_forces(dv_g.data());
        for (std::size_t i = 0; i < correction.size(); ++i) result.forces[i] += correction[i];
    }
    result.times.forces += clock.lap();
    // the strain derivative of the energy of Kohn and Sham of the last bands
    if (ions.has_slopes()) {
        const auto& sphere = pot.sphere();
        const double omega = input.lattice.volume, pi = 3.14159265358979323846;
        const int pairs[6][2] = {{0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}};
        std::vector<double> gv(3 * ng);
        for (std::size_t g = 0; g < ng; ++g)
            for (int d = 0; d < 3; ++d) {
                double x = 0.0;
                for (int j = 0; j < 3; ++j)
                    x += sphere.miller[g][static_cast<std::size_t>(j)] *
                         input.lattice.b[static_cast<std::size_t>(j)][static_cast<std::size_t>(d)];
                gv[3 * g + static_cast<std::size_t>(d)] = x;
            }
        auto download = [&](void (Ions::*f)(Complex*) const) {
            (ions.*f)(components.data());
            std::vector<Complex> out(ng);
            components.download(out.data(), ng);
            return out;
        };
        std::array<double, 6> st{};
        auto add_diagonal = [&](double x) {
            for (int d = 0; d < 3; ++d) st[static_cast<std::size_t>(d)] += x;
        };
        // sum_G w(G) G_a G_b, G = 0 left out
        auto add_shells = [&](const std::function<double(std::size_t)>& weight) {
            for (std::size_t g = 1; g < ng; ++g) {
                const double w_ = weight(g);
                for (int d = 0; d < 6; ++d)
                    st[static_cast<std::size_t>(d)] += w_ * gv[3 * g + static_cast<std::size_t>(pairs[d][0])] *
                                                       gv[3 * g + static_cast<std::size_t>(pairs[d][1])];
            }
        };
        const std::vector<Complex>& rho = out_g;  // the total density first
        auto g2_of = [&](std::size_t g) { return 2.0 * sphere.kinetic[g]; };
        // Hartree: -delta E_H + 4 pi Omega sum |rho|^2 G_a G_b / G^4
        double hartree = 0.0;
        for (std::size_t g = 1; g < ng; ++g)
            hartree += (rho[g].x * rho[g].x + rho[g].y * rho[g].y) / g2_of(g);
        hartree *= 2.0 * pi * omega;
        add_diagonal(-hartree);
        add_shells([&](std::size_t g) {
            const double x = g2_of(g);
            return 4.0 * pi * omega * (rho[g].x * rho[g].x + rho[g].y * rho[g].y) / (x * x);
        });
        // local: -delta E_loc - Omega sum Re[conj(rho) S V'] G_a G_b / |G|
        const auto vloc = download(&Ions::local_components);
        const auto vslope = download(&Ions::local_slope_components);
        double eloc = 0.0;
        for (std::size_t g = 0; g < ng; ++g) eloc += rho[g].x * vloc[g].x + rho[g].y * vloc[g].y;
        add_diagonal(-omega * eloc);
        add_shells([&](std::size_t g) {
            return -omega * (rho[g].x * vslope[g].x + rho[g].y * vslope[g].y) / std::sqrt(g2_of(g));
        });
        // exchange and correlation of rho_out: delta [E_xc - int V_xc (rho + rho_core)]
        // - Omega sum Re[conj(V_xc) S c'] G_a G_b / |G| + the gradients of GGA
        const PotentialEnergies& e = e_out;
        pot.to_sphere(hxc_out.data(), components.data(), ns);
        std::vector<Complex> v(spins * ng);
        components.download(v.data(), v.size());
        for (std::size_t g = 0; g < ng; ++g) {
            Complex x = v[g];
            if (ns == 2) x = make_cuDoubleComplex(0.5 * (x.x + v[ng + g].x), 0.5 * (x.y + v[ng + g].y));
            if (g > 0) {
                const double f = 4.0 * pi / g2_of(g);
                x = make_cuDoubleComplex(x.x - f * rho[g].x, x.y - f * rho[g].y);
            }
            v[g] = x;
        }
        double vxc_core = 0.0;
        if (ions.has_core()) {
            const auto core = download(&Ions::core_components);
            const auto cslope = download(&Ions::core_slope_components);
            for (std::size_t g = 0; g < ng; ++g) vxc_core += v[g].x * core[g].x + v[g].y * core[g].y;
            vxc_core *= omega;
            add_shells([&](std::size_t g) {
                return -omega * (v[g].x * cslope[g].x + v[g].y * cslope[g].y) / std::sqrt(g2_of(g));
            });
        }
        add_diagonal(e.xc - (e.hxc_density - 2.0 * e.hartree) - vxc_core);
        const auto gga = pot.gga_strain();
        for (int d = 0; d < 6; ++d) st[static_cast<std::size_t>(d)] += gga[static_cast<std::size_t>(d)];
        if (ultrasoft || hubbard) {
            // the strengths of the energy of Kohn and Sham of rho_out, and the
            // augmentation moving with the strain in V of rho_out
            update_strengths(hxc_out);
            if (hubbard) hubbard_of(becsum, true, nullptr);  // v of the output occupations
            if (paw) {
                // the one-center strengths of the output occupations
                std::vector<std::vector<double>> d(spins);
                one(becsum, d);
                for (std::size_t s = 0; s < spins; ++s) symmetrizer.occupations(d[s], pair_offsets, atom_angular, false);
                add_onecenter(d);
            }
            if (ultrasoft)
                for (std::size_t s = 0; s < spins; ++s) {
                    potential_of(s, hxc_out);
                    const auto a = aug.strain(veff_g.data(), becsum[s]);
                    for (int d = 0; d < 6; ++d) st[static_cast<std::size_t>(d)] += a[static_cast<std::size_t>(d)];
                }
        }
        // the kinetic energy and the projectors of the bands
        std::array<double, 6> of_bands{};
        for (std::size_t k = 0; k < nk; ++k) {
            if (!any_mine(k)) continue;
            h.set_k(input.kpoints[k]);
            for (std::size_t s = 0; s < spins; ++s) {
                if (!mine(s, k)) continue;
                const std::size_t row = (s * nk + k) * bands;
                for (std::size_t b = 0; b < bands; ++b) w[b] = weights[s * nk + k] * f_max * result.occupations[row + b];
                if (ultrasoft || hubbard) h.set_strengths(strengths[s]);
                h.strain_derivative(psi[s * nk + k].data(), nb, w.data(), of_bands, eigenvalues.data() + row);
            }
        }
        sum(of_bands.data(), of_bands.size());
        for (int d = 0; d < 6; ++d) st[static_cast<std::size_t>(d)] += of_bands[static_cast<std::size_t>(d)];
        result.has_strain = true;
        result.strain = st;
    }
    result.times.stress += clock.lap();
    if (state != nullptr) {
        state->valid = result.converged;
        state->atoms = input.atoms.size();
        state->spins = ns;
        state->bands = nb;
        state->gamma = input.gamma;
        state->rank = input.rank;
        state->ranks = input.ranks;
        state->kpoints = input.kpoints;
        state->millers = std::move(millers);
        state->psi = std::move(psi);
        // this run's difference first, then those kept from the runs before
        // it when this one started from them
        ScfState::Past latest;
        latest.miller = pot.sphere().miller;
        latest.difference.resize(out_g.size());
        for (std::size_t i = 0; i < out_g.size(); ++i)
            latest.difference[i] = make_cuDoubleComplex(out_g[i].x - atoms_g[i].x, out_g[i].y - atoms_g[i].y);
        for (const auto& atom : input.atoms) latest.positions.push_back(atom.position);
        std::vector<ScfState::Past> past;
        past.push_back(std::move(latest));
        if (restart)
            for (auto& earlier : state->past)
                if (past.size() < static_cast<std::size_t>(state->steps)) past.push_back(std::move(earlier));
        state->past = std::move(past);
        state->becsum = mixes_occupations ? becsum : std::vector<std::vector<double>>{};
    }
    result.onecenter = e1;
    if (mixes_occupations) result.becsum = becsum;
    result.density.resize(spins * points);
    rho_out.download(result.density.data(), result.density.size());
    if (ns == 2) {
        const double dv = input.lattice.volume / static_cast<double>(points);
        for (std::size_t i = 0; i < points; ++i) {
            const double m = result.density[i] - result.density[points + i];
            result.magnetization += m * dv;
            result.absolute_magnetization += std::abs(m) * dv;
        }
    }
    return result;
}

}  // namespace mdapy::dft
