// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// NEP on the device, the terms of calculators/nep.cpp in four kernels:
//   descriptors: each atom its radial descriptors, the list of its angular
//     pairs, then for each angular function n the sums of harmonics s_n of
//     its pairs, kept, and the angular descriptors of n;
//   network: each atom on a warp, its energy and the derivatives of the
//     energy by its descriptors, in place of the descriptors;
//   angular: for each n the derivatives of the energy by s_n, then by the
//     vector of each angular pair, kept per pair;
//   forces: each atom the radial forces of its pairs from both ends, the ZBL
//     forces, and the angular derivatives of its pairs and of their
//     reverses; its virial from the derivatives of its neighbors' energies.
// The sums of each atom are held in registers: unrolled to at most 8 or 20
// functions of each kind and to harmonics up to l = 4 or l = 8, the basis
// functions and the harmonics computed again where they are needed.

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "calculators/nep.h"
#include "cuda/atoms.cuh"
#include "cuda/neighbors.cuh"

namespace mdapy::cuda {

namespace {

// radial and angular functions, and Chebyshev polynomials, of each kind
constexpr int kMaxFunctions = 20;
constexpr int kWarp = kLanes;  // an atom of the network on a group of lanes

template <class Real>
struct NepTables {
    int types, nr, na, kr, ka, l_max, harmonics, dim, neurons;
    bool four_body, five_body, zbl;
    const Real* rc_radial;   // (types)
    const Real* rc_angular;  // (types)
    const Real* c_radial;    // (types, types, kr, nr)
    const Real* c_angular;   // (types, types, ka, na)
    const Real* w0;          // (types, neurons, dim)
    const Real* w0_swapped;  // (types, dim, neurons)
    const Real* b0;          // (types, neurons)
    const Real* w1;          // (types, neurons)
    Real b1;
    const Real* q_scaler;      // (dim)
    const Real* zbl_rows;      // (types, types, 12)
    const Real* coefficients;  // of the harmonics, NepModel::harmonics
};

// The buffers of a computation: the descriptors and then the derivatives
// of the energy by them, (n, dim); the sums of harmonics, (na, harmonics,
// n); the angular pairs of each atom as the neighbor list packs them,
// (most, n), and their counts; the place of each pair of the list among
// the angular pairs of its atom, or -1, (most, n); the derivatives by the
// vectors of the angular pairs, 3 planes of (most, n).
template <class Real>
struct NepWork {
    Real* q;
    Real* s;
    std::int32_t* angular;
    std::int32_t* angular_counts;
    std::int32_t* angular_of;
    Real* derivative;
};

// use(k, f_k, f'_k) for k = 0..k_max: f_k(r) = (T_k(x) + 1) fc(r) / 2 with
// x = 2 (r / rc - 1)^2 - 1 and the cosine cutoff fc, and its derivative by r.
template <class Real, class Use>
__device__ inline void for_chebyshev(Real r, Real rc, int k_max, Use&& use) {
    const Real pi = Real(3.14159265358979323846);
    const Real u = r / rc - Real(1);
    const Real x = Real(2) * u * u - Real(1);
    const Real dx = Real(4) * u / rc;
    Real sine, cosine;
    if constexpr (sizeof(Real) == 4) {
        sincosf(pi * r / rc, &sine, &cosine);
    } else {
        sincos(pi * r / rc, &sine, &cosine);
    }
    const Real fc = Real(0.5) * cosine + Real(0.5);
    const Real fcp = Real(-0.5) * pi / rc * sine;
    use(0, fc, fcp);
    Real t0 = 1, t1 = x, d0 = 0, d1 = 1;
    for (int k = 1; k <= k_max; ++k) {
        if (k >= 2) {
            const Real t2 = Real(2) * x * t1 - t0;
            const Real d2 = Real(2) * t1 + Real(2) * x * d1 - d0;
            t0 = t1;
            t1 = t2;
            d0 = d1;
            d1 = d2;
        }
        use(k, Real(0.5) * (t1 + Real(1)) * fc,
            Real(0.5) * (d1 * dx * fc + (t1 + Real(1)) * fcp));
    }
}

// use(h, y_h, g_h) for the real spherical harmonics of the unit vector u up
// to l_max <= L, as calculators/nep.cpp orders them, with g_h their
// gradients by u when Gradient (zero otherwise); the gradient by the vector
// u r is the part of g_h across u, over r. Unrolled, so that h is known
// when compiled.
template <class Real, int L, bool Gradient, class Use>
__device__ inline void for_harmonics(int l_max, const Real* __restrict__ c, const Real* u,
                                     Use&& use) {
    Real re[L + 1], im[L + 1], zp[L + 1];
    re[0] = 1;
    im[0] = 0;
    zp[0] = 1;
#pragma unroll
    for (int m = 1; m <= L; ++m) {
        re[m] = re[m - 1] * u[0] - im[m - 1] * u[1];
        im[m] = re[m - 1] * u[1] + im[m - 1] * u[0];
        zp[m] = zp[m - 1] * u[2];
    }
    int offset = 0, h = 0;
#pragma unroll
    for (int l = 1; l <= L; ++l) {
        if (l > l_max) break;
#pragma unroll
        for (int m = 0; m <= l; ++m) {
            Real p = 0, dp = 0;
#pragma unroll
            for (int k = 0; k <= l - m; ++k) {
                const Real coefficient = c[offset + k];
                p += coefficient * zp[k];
                if (Gradient && k > 0) dp += Real(k) * coefficient * zp[k - 1];
            }
            offset += l - m + 1;
            if (m == 0) {
                const Real g[3] = {0, 0, dp};
                use(h, p, g);
                h += 1;
            } else {
                const Real pm = p * Real(m);
                const Real g0[3] = {pm * re[m - 1], -pm * im[m - 1], dp * re[m]};
                const Real g1[3] = {pm * im[m - 1], pm * re[m - 1], dp * im[m]};
                use(h, p * re[m], g0);
                use(h + 1, p * im[m], g1);
                h += 2;
            }
        }
    }
}

template <class Real>
__device__ inline void zbl_pair(const Real* row, Real r, Real& energy, Real& derivative) {
    const Real pi = Real(3.14159265358979323846);
    const Real inner = row[0], outer = row[1];
    Real s = 1, sp = 0;
    if (r >= outer) {
        energy = derivative = 0;
        return;
    }
    if (r >= inner) {
        const Real w = pi / (outer - inner);
        s = Real(0.5) * cos(w * (r - inner)) + Real(0.5);
        sp = Real(-0.5) * w * sin(w * (r - inner));
    }
    const Real x = r * row[3];
    Real phi = 0, phip = 0;
    for (int k = 0; k < 4; ++k) {
        const Real term = row[4 + 2 * k] * exp(-row[5 + 2 * k] * x);
        phi += term;
        phip -= row[5 + 2 * k] * term;
    }
    const Real v = row[2] * phi / r;
    const Real vp = row[2] * (phip * row[3] / r - phi / (r * r));
    energy = v * s;
    derivative = vp * s + v * sp;
}

// The vector from atom i to the neighbor of a slot, its length, the
// neighbor's place and species.
template <class Real>
struct Pair {
    Real v[3];
    Real r;
    std::int64_t j;
    int species;
};

template <class Real>
__device__ inline Pair<Real> pair_of(std::int32_t entry, const Real4<Real>* __restrict__ x,
                                     const Real4<Real>& xi, const Real* steps) {
    Pair<Real> p;
    const auto packed = static_cast<std::uint32_t>(entry);
    p.j = packed & kIndexMask;
    const Real4<Real> xj = x[p.j];
    p.v[0] = xj.x - xi.x;
    p.v[1] = xj.y - xi.y;
    p.v[2] = xj.z - xi.z;
    add_step(steps, packed >> kIndexBits, p.v[0], p.v[1], p.v[2]);
    p.species = static_cast<int>(xj.w);
    p.r = sqrt(p.v[0] * p.v[0] + p.v[1] * p.v[1] + p.v[2] * p.v[2]);
    return p;
}

template <class Real>
__device__ inline Real mean(const Real* a, int ti, int tj) {
    return Real(0.5) * (a[ti] + a[tj]);
}

template <class Real, int N, int L>
__global__ void nep_descriptors(std::int64_t n, std::int64_t centers, const Real4<Real>* x,
                                Images<Real> images, const std::int32_t* indices,
                                const std::int32_t* counts, NepTables<Real> t, NepWork<Real> w,
                                double* energies) {
    constexpr int kH = (L + 1) * (L + 1) - 1;
    const Real alpha = Real(0.11952286093343936);  // 1 / sqrt(70)
    const Real beta = Real(0.6210590034081188);    // 3 sqrt(210) / 70
    const Real five = Real(7.0 / 15.0);
    __shared__ Real steps[81];
    fill_steps(images, steps);
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (i >= centers) {
        // a ghost under MPI, the centre of no energy here: no angular pairs,
        // its slopes zero, so that the forces take only the energies of the
        // atoms of the process
        for (int k = 0; k < counts[i]; ++k) w.angular_of[static_cast<std::int64_t>(k) * n + i] = -1;
        w.angular_counts[i] = 0;
        for (int d = 0; d < t.dim; ++d) w.q[i * t.dim + d] = Real(0);
        energies[i] = 0.0;
        return;
    }
    const Real4<Real> xi = x[i];
    const int ti = static_cast<int>(xi.w);
    const int nr = t.nr, na = t.na, l_max = t.l_max, H = t.harmonics;
    Real q[N];
#pragma unroll
    for (int m = 0; m < N; ++m) q[m] = 0;
    double repulsion = 0.0;
    int angular = 0;
    const int count = counts[i];
    for (int k = 0; k < count; ++k) {
        const std::int64_t slot = static_cast<std::int64_t>(k) * n + i;
        const std::int32_t entry = indices[slot];
        const Pair<Real> p = pair_of(entry, x, xi, steps);
        const Real rc_r = mean(t.rc_radial, ti, p.species);
        if (p.r < rc_r) {
            const Real* c = t.c_radial + (ti * t.types + p.species) * nr * t.kr;
            for_chebyshev(p.r, rc_r, t.kr - 1, [&](int b, Real f, Real) {
#pragma unroll
                for (int m = 0; m < N; ++m) {
                    if (m < nr) q[m] += c[b * nr + m] * f;
                }
            });
        }
        if (p.r < mean(t.rc_angular, ti, p.species)) {
            w.angular[static_cast<std::int64_t>(angular) * n + i] = entry;
            w.angular_of[slot] = angular;
            ++angular;
            if (t.zbl) {
                Real e, de;
                zbl_pair(t.zbl_rows + (ti * t.types + p.species) * 12, p.r, e, de);
                repulsion += 0.5 * static_cast<double>(e);
            }
        } else {
            w.angular_of[slot] = -1;
        }
    }
    w.angular_counts[i] = angular;
    Real* qi = w.q + i * t.dim;
#pragma unroll
    for (int m = 0; m < N; ++m) {
        if (m < nr) qi[m] = q[m] * t.q_scaler[m];
    }
    energies[i] = repulsion;
    for (int m = 0; m < na; ++m) {
        Real s[kH];
#pragma unroll
        for (int h = 0; h < kH; ++h) s[h] = 0;
        for (int a = 0; a < angular; ++a) {
            const Pair<Real> p = pair_of(w.angular[static_cast<std::int64_t>(a) * n + i], x, xi, steps);
            const Real* c = t.c_angular + (ti * t.types + p.species) * na * t.ka + m;
            Real g = 0;
            for_chebyshev(p.r, mean(t.rc_angular, ti, p.species), t.ka - 1,
                          [&](int b, Real f, Real) { g += c[b * na] * f; });
            const Real inverse = Real(1) / p.r;
            const Real u[3] = {p.v[0] * inverse, p.v[1] * inverse, p.v[2] * inverse};
            for_harmonics<Real, L, false>(l_max, t.coefficients, u,
                                          [&](int h, Real y, const Real*) { s[h] += g * y; });
        }
#pragma unroll
        for (int h = 0; h < kH; ++h) {
            if (h < H) w.s[(static_cast<std::int64_t>(m) * H + h) * n + i] = s[h];
        }
        int first = 0;
#pragma unroll
        for (int l = 1; l <= L; ++l) {
            if (l <= l_max) {
                Real sum = 0;
#pragma unroll
                for (int h = 0; h < 2 * l + 1; ++h) sum += s[first + h] * s[first + h];
                const int d = nr + (l - 1) * na + m;
                qi[d] = sum * t.q_scaler[d];
            }
            first += 2 * l + 1;
        }
        if (t.four_body) {
            const Real* b = s + 3;
            const int d = nr + l_max * na + m;
            qi[d] = (alpha * b[0] *
                         (Real(-2) * b[0] * b[0] - Real(3) * (b[1] * b[1] + b[2] * b[2]) +
                          Real(6) * (b[3] * b[3] + b[4] * b[4])) +
                     beta * (b[3] * (b[2] * b[2] - b[1] * b[1]) - Real(2) * b[1] * b[2] * b[4])) *
                    t.q_scaler[d];
        }
        if (t.five_body) {
            const Real sum = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
            const int d = nr + (l_max + 1) * na + m;
            qi[d] = five * sum * sum * t.q_scaler[d];
        }
    }
}

// An atom on a warp: the lanes over the neurons, then over the descriptors.
template <class Real>
__global__ void nep_network(std::int64_t n, std::int64_t centers, const Real4<Real>* x,
                            NepTables<Real> t, Real* q, double* energies) {
    extern __shared__ __align__(16) unsigned char raw[];
    const int lane = static_cast<int>(threadIdx.x) % kWarp;
    const int warp = static_cast<int>(threadIdx.x) / kWarp;
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * (blockDim.x / kWarp) + warp;
    if (i >= centers) return;  // the whole warp
    const int dim = t.dim, neurons = t.neurons;
    Real* descriptors = reinterpret_cast<Real*>(raw) + warp * (dim + neurons);
    Real* factor = descriptors + dim;
    const int ti = static_cast<int>(x[i].w);
    Real* row = q + i * dim;
    for (int d = lane; d < dim; d += kWarp) descriptors[d] = row[d];
    MDAPY_SYNC_LANES();
    double energy = 0.0;
    for (int k = lane; k < neurons; k += kWarp) {
        const Real* w = t.w0_swapped + static_cast<std::int64_t>(ti) * dim * neurons + k;
        Real input = -t.b0[ti * neurons + k];
        for (int d = 0; d < dim; ++d) input += w[d * neurons] * descriptors[d];
        const Real th = tanh(input);
        const Real w1 = t.w1[ti * neurons + k];
        energy += static_cast<double>(w1 * th);
        factor[k] = w1 * (Real(1) - th * th);
    }
    MDAPY_SYNC_LANES();
    for (int d = lane; d < dim; d += kWarp) {
        const Real* w = t.w0 + static_cast<std::int64_t>(ti) * neurons * dim + d;
        Real slope = 0;
        for (int k = 0; k < neurons; ++k) slope += factor[k] * w[k * dim];
        row[d] = slope * t.q_scaler[d];
    }
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        energy += MDAPY_SHUFFLE_DOWN(energy, offset);
    }
    if (lane == 0) energies[i] += energy - static_cast<double>(t.b1);
}

template <class Real, int N, int L>
__global__ void nep_angular(std::int64_t n, std::int64_t centers, std::int64_t plane,
                            const Real4<Real>* x, Images<Real> images, NepTables<Real> t,
                            NepWork<Real> w) {
    constexpr int kH = (L + 1) * (L + 1) - 1;
    const Real alpha = Real(0.11952286093343936);
    const Real beta = Real(0.6210590034081188);
    const Real five = Real(7.0 / 15.0);
    __shared__ Real steps[81];
    fill_steps(images, steps);
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= centers) return;
    const Real4<Real> xi = x[i];
    const int ti = static_cast<int>(xi.w);
    const int nr = t.nr, na = t.na, l_max = t.l_max, H = t.harmonics;
    const Real* slope = w.q + i * t.dim;
    const int angular = w.angular_counts[i];
    for (int m = 0; m < na; ++m) {
        // dE / ds of the sums of harmonics of n = m
        Real s[kH], g[kH];
#pragma unroll
        for (int h = 0; h < kH; ++h) {
            s[h] = h < H ? w.s[(static_cast<std::int64_t>(m) * H + h) * n + i] : Real(0);
        }
        int first = 0;
#pragma unroll
        for (int l = 1; l <= L; ++l) {
            const Real e = l <= l_max ? slope[nr + (l - 1) * na + m] : Real(0);
#pragma unroll
            for (int h = 0; h < 2 * l + 1; ++h) g[first + h] = Real(2) * e * s[first + h];
            first += 2 * l + 1;
        }
        if (t.four_body) {
            const Real e = slope[nr + l_max * na + m];
            const Real* b = s + 3;
            g[3] += e * alpha *
                    (Real(-6) * b[0] * b[0] - Real(3) * (b[1] * b[1] + b[2] * b[2]) +
                     Real(6) * (b[3] * b[3] + b[4] * b[4]));
            g[4] += e * (Real(-6) * alpha * b[0] * b[1] -
                         Real(2) * beta * (b[3] * b[1] + b[2] * b[4]));
            g[5] += e * (Real(-6) * alpha * b[0] * b[2] +
                         Real(2) * beta * (b[3] * b[2] - b[1] * b[4]));
            g[6] += e * (Real(12) * alpha * b[0] * b[3] + beta * (b[2] * b[2] - b[1] * b[1]));
            g[7] += e * (Real(12) * alpha * b[0] * b[4] - Real(2) * beta * b[1] * b[2]);
        }
        if (t.five_body) {
            const Real e = slope[nr + (l_max + 1) * na + m];
            const Real sum = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
#pragma unroll
            for (int h = 0; h < 3; ++h) g[h] += e * Real(4) * five * sum * s[h];
        }
        // the derivatives by the vectors of the angular pairs
        for (int a = 0; a < angular; ++a) {
            const std::int64_t place = static_cast<std::int64_t>(a) * n + i;
            const Pair<Real> p = pair_of(w.angular[place], x, xi, steps);
            const Real* c = t.c_angular + (ti * t.types + p.species) * na * t.ka + m;
            Real gn = 0, gp = 0;
            for_chebyshev(p.r, mean(t.rc_angular, ti, p.species), t.ka - 1,
                          [&](int b, Real f, Real fp) {
                              gn += c[b * na] * f;
                              gp += c[b * na] * fp;
                          });
            const Real inverse = Real(1) / p.r;
            const Real u[3] = {p.v[0] * inverse, p.v[1] * inverse, p.v[2] * inverse};
            Real along = 0, side[3] = {0, 0, 0};
            for_harmonics<Real, L, true>(l_max, t.coefficients, u,
                                         [&](int h, Real y, const Real* grad) {
                                             along += g[h] * y;
                                             side[0] += g[h] * grad[0];
                                             side[1] += g[h] * grad[1];
                                             side[2] += g[h] * grad[2];
                                         });
            // the part across u, over r
            const Real radial = side[0] * u[0] + side[1] * u[1] + side[2] * u[2];
            for (int a3 = 0; a3 < 3; ++a3) {
                const Real d = gp * along * u[a3] + gn * (side[a3] - radial * u[a3]) * inverse;
                if (m == 0) {
                    w.derivative[a3 * plane + place] = d;
                } else {
                    w.derivative[a3 * plane + place] += d;
                }
            }
        }
    }
}

// The force on each atom: the radial terms of its pairs from both ends, the
// ZBL ones, and the angular derivatives of its pairs and of their reverses;
// its virial from the derivatives of its neighbors' energies.
// Under MPI only the first centers atoms have energies here: the forces of
// the energies of the ghosts come back from their processes.
template <class Real, int N, bool Virial>
__global__ void nep_forces(std::int64_t n, std::int64_t centers, std::int64_t plane,
                           const Real4<Real>* x,
                           Images<Real> images, const std::int32_t* indices,
                           const std::int32_t* counts, const std::int32_t* reverse,
                           NepTables<Real> t, NepWork<Real> w, double* forces, double* virials) {
    __shared__ Real steps[81];
    fill_steps(images, steps);
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const Real4<Real> xi = x[i];
    const int ti = static_cast<int>(xi.w);
    const int nr = t.nr;
    Real own[N];
#pragma unroll
    for (int m = 0; m < N; ++m) own[m] = m < nr ? w.q[i * t.dim + m] : Real(0);
    double force[3] = {0.0, 0.0, 0.0};
    double virial[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const int count = counts[i];
    for (int k = 0; k < count; ++k) {
        const std::int64_t slot = static_cast<std::int64_t>(k) * n + i;
        const Pair<Real> p = pair_of(indices[slot], x, xi, steps);
        const int tj = p.species;
        const Real rc_r = mean(t.rc_radial, ti, tj);
        const bool angular = p.r < mean(t.rc_angular, ti, tj);
        if (p.r >= rc_r && !angular) continue;
        const Real inverse = Real(1) / p.r;
        const Real u[3] = {p.v[0] * inverse, p.v[1] * inverse, p.v[2] * inverse};
        // the derivatives of the energy of i by r_ij along u, and of that of
        // j by r_ji along -u
        Real along_ij = 0, along_ji = 0;
        if (p.r < rc_r) {
            Real other[N];
#pragma unroll
            for (int m = 0; m < N; ++m) other[m] = m < nr ? w.q[p.j * t.dim + m] : Real(0);
            const Real* cij = t.c_radial + (ti * t.types + tj) * nr * t.kr;
            const Real* cji = t.c_radial + (tj * t.types + ti) * nr * t.kr;
            for_chebyshev(p.r, rc_r, t.kr - 1, [&](int b, Real, Real fp) {
                Real sij = 0, sji = 0;
#pragma unroll
                for (int m = 0; m < N; ++m) {
                    if (m < nr) {
                        sij += cij[b * nr + m] * own[m];
                        sji += cji[b * nr + m] * other[m];
                    }
                }
                along_ij += fp * sij;
                along_ji += fp * sji;
            });
        }
        if (angular && t.zbl) {
            Real e, de;  // the rows of i j and of j i are the same
            zbl_pair(t.zbl_rows + (ti * t.types + tj) * 12, p.r, e, de);
            if (i < centers) along_ij += Real(0.5) * de;
            if (p.j < centers) along_ji += Real(0.5) * de;
        }
        Real tij[3], tji[3];
        for (int a = 0; a < 3; ++a) {
            tij[a] = along_ij * u[a];
            tji[a] = -along_ji * u[a];
        }
        if (angular) {
            // the reverse is angular too, unless rounding put its length
            // on the other side of the cutoff
            const int mine = w.angular_of[slot];  // -1 for a ghost
            const int back = w.angular_of[static_cast<std::int64_t>(reverse[slot]) * n + p.j];
            if (mine >= 0) {
                const std::int64_t own = static_cast<std::int64_t>(mine) * n + i;
                for (int a = 0; a < 3; ++a) tij[a] += w.derivative[a * plane + own];
            }
            if (back >= 0) {
                const std::int64_t place = static_cast<std::int64_t>(back) * n + p.j;
                for (int a = 0; a < 3; ++a) tji[a] += w.derivative[a * plane + place];
            }
        }
        for (int a = 0; a < 3; ++a) {
            force[a] += static_cast<double>(tij[a]) - static_cast<double>(tji[a]);
        }
        if (Virial) {
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    virial[3 * a + b] += static_cast<double>(p.v[a]) * static_cast<double>(tji[b]);
                }
            }
        }
    }
    for (int a = 0; a < 3; ++a) forces[3 * i + a] = force[a];
    if (Virial) {
        for (int a = 0; a < 9; ++a) virials[static_cast<std::int64_t>(a) * n + i] = virial[a];
    }
}

template <class Real>
class Nep final : public Potential {
public:
    explicit Nep(const calculators::NepModel& m) : model_(m) {
        if (m.n_radial + 1 > kMaxFunctions || m.n_angular + 1 > kMaxFunctions ||
            m.k_radial + 1 > kMaxFunctions || m.k_angular + 1 > kMaxFunctions) {
            throw std::invalid_argument("on the GPU, NEP takes at most 20 functions of each kind");
        }
        if (network_shared(m) > 48 * 1024) {
            throw std::invalid_argument("on the GPU, the network of NEP is too large");
        }
        cutoff_ = 0.0;
        for (const double r : m.rc_radial) cutoff_ = std::max(cutoff_, r);
        for (const double r : m.rc_angular) cutoff_ = std::max(cutoff_, r);
        const auto up = [](Buffer<Real>& b, const std::vector<double>& v) {
            std::vector<Real> r(v.begin(), v.end());
            b.upload(r.data(), r.size());
        };
        up(rc_radial_, m.rc_radial);
        up(rc_angular_, m.rc_angular);
        up(c_radial_, m.c_radial_swapped);
        up(c_angular_, m.c_angular_swapped);
        up(w0_, m.w0);
        up(w0_swapped_, m.w0_swapped);
        up(b0_, m.b0);
        up(w1_, m.w1);
        up(q_scaler_, m.q_scaler);
        up(zbl_, m.zbl.empty() ? std::vector<double>(12, 0.0) : m.zbl);
        up(coefficients_, m.harmonics);
    }

    double cutoff() const override { return cutoff_; }
    int species() const override { return model_.types; }
    // Under MPI the atoms of the process alone have energies, from ghosts
    // within the cutoff; the forces on the ghosts, and the virials, which each
    // atom gathers from the energies of its neighbors, go back to their atoms.
    bool thin() const override { return true; }
    bool reverse() const override { return true; }
    bool gathered() const override { return true; }
    void compute_stage(int stage, Atoms& atoms, Neighbors& neighbors, const Box& box,
                       bool energy, bool virial, bool heat) override {
        (void)energy;
        if (stage == 0) evaluate(atoms, neighbors, box, virial || heat, true);
    }

    void compute(Atoms& atoms, Neighbors& neighbors, const Box& box, bool energy, bool virial,
                 bool atomic) override {
        (void)energy;  // the energies come with the derivatives
        (void)atomic;  // the virial of each atom is that of the processor
        evaluate(atoms, neighbors, box, virial, false);
    }

private:
    // centred: the first owned atoms alone have energies
    void evaluate(Atoms& atoms, Neighbors& neighbors, const Box& box, bool virial,
                  bool centred) {
        Atoms::Impl& a = atoms.impl();
        Neighbors::Impl& l = neighbors.impl();
        if (l.n != a.n || l.builds == 0 || l.cutoff + 1e-12 < cutoff_) {
            throw std::logic_error("the neighbor list does not fit the atoms or the cutoff");
        }
        const std::int64_t n = a.n;
        const std::int64_t plane = static_cast<std::int64_t>(l.most) * n;
        const std::int32_t* reverse = l.reversed();
        const int na = model_.n_angular + 1, harmonics = model_.harmonic_count();
        q_.resize(static_cast<std::size_t>(n * model_.dim));
        s_.resize(static_cast<std::size_t>(n * na * harmonics));
        angular_.resize(static_cast<std::size_t>(plane));
        angular_counts_.resize(static_cast<std::size_t>(n));
        angular_of_.resize(static_cast<std::size_t>(plane));
        derivative_.resize(static_cast<std::size_t>(3 * plane));
        if (virial) a.virials.resize(static_cast<std::size_t>(9 * n));
        const Real4<Real>* x = l.local<Real>(a, box);
        const Images<Real> b = l.steps<Real>(box);
        const NepTables<Real> t{model_.types,
                                model_.n_radial + 1,
                                na,
                                model_.k_radial + 1,
                                model_.k_angular + 1,
                                model_.l_max,
                                harmonics,
                                model_.dim,
                                model_.neurons,
                                model_.four_body,
                                model_.five_body,
                                !model_.zbl.empty(),
                                rc_radial_.data(),
                                rc_angular_.data(),
                                c_radial_.data(),
                                c_angular_.data(),
                                w0_.data(),
                                w0_swapped_.data(),
                                b0_.data(),
                                w1_.data(),
                                static_cast<Real>(model_.b1),
                                q_scaler_.data(),
                                zbl_.data(),
                                coefficients_.data()};
        const NepWork<Real> w{q_.data(),           s_.data(),           angular_.data(),
                              angular_counts_.data(), angular_of_.data(), derivative_.data()};
        const Launch launch{n, centred ? a.active() : n, plane, x, b, l, reverse, t, w, a,
                            virial};
        const bool few = std::max(model_.n_radial, model_.n_angular) < 8;
        const bool low = model_.l_max <= 4;
        if (few && low) {
            run<8, 4>(launch);
        } else if (few) {
            run<8, 8>(launch);
        } else if (low) {
            run<kMaxFunctions, 4>(launch);
        } else {
            run<kMaxFunctions, 8>(launch);
        }
        a.has_energies = true;
        a.has_virials = virial;
    }

    static constexpr int kPerBlock = kBlock / kWarp;  // atoms of a block of the network

    static std::size_t network_shared(const calculators::NepModel& m) {
        return static_cast<std::size_t>(kPerBlock) * (m.dim + m.neurons) * sizeof(Real);
    }

    struct Launch {
        std::int64_t n, centers, plane;
        const Real4<Real>* x;
        Images<Real> b;
        Neighbors::Impl& l;
        const std::int32_t* reverse;
        NepTables<Real> t;
        NepWork<Real> w;
        Atoms::Impl& a;
        bool virial;
    };

    template <int N, int L>
    void run(const Launch& r) const {
        const unsigned g = blocks(r.n);
        const auto* idx = r.l.indices.data();
        const auto* cnt = r.l.counts.data();
        nep_descriptors<Real, N, L>
            <<<g, kBlock>>>(r.n, r.centers, r.x, r.b, idx, cnt, r.t, r.w, r.a.energies.data());
        MDAPY_LAUNCHED();
        const auto network_blocks = static_cast<unsigned>((r.n + kPerBlock - 1) / kPerBlock);
        nep_network<Real><<<network_blocks, kBlock, network_shared(model_)>>>(
            r.n, r.centers, r.x, r.t, r.w.q, r.a.energies.data());
        MDAPY_LAUNCHED();
        nep_angular<Real, N, L><<<g, kBlock>>>(r.n, r.centers, r.plane, r.x, r.b, r.t, r.w);
        MDAPY_LAUNCHED();
        if (r.virial) {
            nep_forces<Real, N, true><<<g, kBlock>>>(r.n, r.centers, r.plane, r.x, r.b, idx, cnt,
                                                     r.reverse, r.t, r.w, r.a.forces.data(),
                                                     r.a.virials.data());
        } else {
            nep_forces<Real, N, false><<<g, kBlock>>>(r.n, r.centers, r.plane, r.x, r.b, idx,
                                                      cnt, r.reverse, r.t, r.w,
                                                      r.a.forces.data(), nullptr);
        }
        MDAPY_LAUNCHED();
    }

    calculators::NepModel model_;
    double cutoff_;
    Buffer<Real> rc_radial_, rc_angular_, c_radial_, c_angular_, w0_, w0_swapped_, b0_, w1_,
        q_scaler_, zbl_, coefficients_;
    Buffer<Real> q_, s_, derivative_;
    Buffer<std::int32_t> angular_, angular_counts_, angular_of_;
};

}  // namespace

std::unique_ptr<Potential> make_nep(const calculators::NepModel& model, Precision precision) {
    if (precision == Precision::Mixed) {
        return std::make_unique<Nep<float>>(model);
    }
    return std::make_unique<Nep<double>>(model);
}

}  // namespace mdapy::cuda
