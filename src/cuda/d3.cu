// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// DFT-D3 on the device, as calculators/d3.cpp computes it. Its pairs reach
// tens of Å, beyond what the list of the device holds in memory: D3 finds them
// itself, from the atoms sorted into cells of a third of its reach, each atom
// on a warp whose lanes go over the atoms of the cells within reach, through
// as many images of the box as the reach crosses. The passes of the
// processor follow: the coordination numbers, the weights of the
// references, the pairs, the triangles, the coordination numbers through the
// C6; each atom gathers its own terms.

#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "calculators/d3.h"
#include "cuda/atoms.cuh"
#include "cuda/neighbors.cuh"

namespace mdapy::cuda {

namespace {

using calculators::D3Damping;
using calculators::kD3References;

constexpr int kWarp = 32;
constexpr unsigned kAll = 0xffffffffu;
constexpr double kCountSteepness = 16.0;
constexpr double kWeightWidth = 4.0;
constexpr double kClosest2 = 1e-12;
// The corners of the triangles held at once, a batch of atoms at a time.
constexpr std::int64_t kCornersAtOnce = std::int64_t{1} << 20;

// A buffer of at least size elements, which keeps its memory when smaller
// ones are asked for.
template <class T>
void room(Buffer<T>& buffer, std::size_t size) {
    if (buffer.size() < size) buffer.resize(size + size / 8);
}

// Three numbers of a precision, passed by value to the kernels.
template <class Real>
struct Real3w {
    Real v[3];
};

__device__ inline double warp_sum(double v) {
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        v += __shfl_down_sync(kAll, v, offset);
    }
    return v;
}

// The cells: the atoms sorted by them, the position of each from the corner
// of its cell, its fractional coordinates and its cell.
template <class Real>
struct Grid {
    int n[3];
    int pbc[3];
    // the thickness of the box along each vector, and whether the box is
    // orthogonal: the nearest a cell can be to an atom, to pass over it
    Real thick[3];
    int orthogonal;
    Real step[9];  // h_k / n_k, rows
    const int* start;
    const int* end;
    const Real4<Real>* local;  // x, y, z from the corner of the cell, species as w
    const Real4<Real>* frac;   // fractional coordinates in the box
    const int4* cell;          // the cell, its key as w
};

// The tables of the potential in a precision.
template <class Real>
struct Tables {
    int species;
    int rational;
    int half_alpha;
    Real s6, s8, s9, alpha, hartree;
    // the contracted C6 of the atoms (see calculators/d3.cpp): width rows of
    // atoms numbers, those of species t from row slots[t], so that the lanes
    // of a warp read those of neighboring atoms together
    std::int64_t width;
    std::int64_t atoms;
    const std::int64_t* slots;
    const int* references;
    const Real* reference_cn;
    const Real* rcov;
    const Real* ratio;
    const Real* radius6;
    const Real* radius8;
    const Real* atm_radius;
    const std::int64_t* blocks;
    const Real* c6;
};

// A neighbor of an atom within the cutoff of the triangles.
template <class Real>
struct Corner {
    Real v[3];
    Real r2;
    Real c6, dc6, radius;
    int q;  // its place among the sorted atoms
};

// The cell, the fractional coordinates and the position from the corner of
// the cell of a position: wrapped into the box along the periodic
// directions; along the others an atom outside the box is in the cell at its
// face, its fractional coordinate held there, its position kept.
__device__ inline void place(const double4a& x, const DeviceBox<double>& box, const int* n,
                             int c[3], double s[3], double l[3]) {
    const double r[3] = {x.x - box.lo[0], x.y - box.lo[1], x.z - box.lo[2]};
    double f[3];
    for (int k = 0; k < 3; ++k) {
        double v = r[0] * box.inverse[k] + r[1] * box.inverse[3 + k] + r[2] * box.inverse[6 + k];
        if (box.pbc[k]) v -= floor(v);
        const double held = box.pbc[k] ? v : fmin(fmax(v, 0.0), 1.0);
        int ck = static_cast<int>(floor(held * n[k]));
        ck = min(max(ck, 0), n[k] - 1);
        c[k] = ck;
        s[k] = held;
        f[k] = v - static_cast<double>(ck) / n[k];
    }
    for (int a = 0; a < 3; ++a) {
        l[a] = f[0] * box.h[a] + f[1] * box.h[3 + a] + f[2] * box.h[6 + a];
    }
}

__global__ void bin(std::int64_t n, const double4a* x, DeviceBox<double> box, int3 cells,
                    int* keys, int* order) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int count[3] = {cells.x, cells.y, cells.z};
    int c[3];
    double s[3], l[3];
    place(x[i], box, count, c, s, l);
    keys[i] = c[0] + cells.x * (c[1] + cells.y * c[2]);
    order[i] = static_cast<int>(i);
}

template <class Real>
__global__ void arrange_atoms(std::int64_t n, const double4a* x, DeviceBox<double> box, int3 cells,
                        const int* order, Real4<Real>* local, Real4<Real>* frac, int4* cell) {
    const std::int64_t p = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const int count[3] = {cells.x, cells.y, cells.z};
    const double4a xi = x[order[p]];
    int c[3];
    double s[3], l[3];
    place(xi, box, count, c, s, l);
    local[p] = Real4<Real>{static_cast<Real>(l[0]), static_cast<Real>(l[1]),
                           static_cast<Real>(l[2]), static_cast<Real>(xi.w)};
    frac[p] = Real4<Real>{static_cast<Real>(s[0]), static_cast<Real>(s[1]),
                          static_cast<Real>(s[2]), Real(0)};
    cell[p] = make_int4(c[0], c[1], c[2], c[0] + cells.x * (c[1] + cells.y * c[2]));
}

// The counts as 64-bit numbers, a zero after them, for their sums.
__global__ void widen(std::int64_t n, const int* counts, std::int64_t* out) {
    const std::int64_t p = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p > n) return;
    out[p] = p < n ? counts[p] : 0;
}

__global__ void bounds(std::int64_t n, const int* keys, int* start, int* end) {
    const std::int64_t p = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const int key = keys[p];
    if (p == 0 || keys[p - 1] != key) start[key] = static_cast<int>(p);
    if (p == n - 1 || keys[p + 1] != key) end[key] = static_cast<int>(p + 1);
}

// Calls visit(hit, q, v, r2, species) on every lane for the atoms of the
// cells around atom p within the fractional half-widths reach, the lanes over
// the atoms of each cell; hit when the image is within cut2 and not p itself;
// plane() after each plane of cells. Every lane of the warp makes the same
// calls.
template <class Real, class Visit, class Plane>
__device__ void each_pair(const Grid<Real>& g, std::int64_t p, const Real* reach, Real cut2,
                          int lane, Visit&& visit, Plane&& plane) {
    const Real4<Real> lp = g.local[p];
    const Real4<Real> sp = g.frac[p];
    const int4 cp = g.cell[p];
    const Real s[3] = {sp.x, sp.y, sp.z};
    const int c[3] = {cp.x, cp.y, cp.z};
    int low[3], high[3];
    for (int k = 0; k < 3; ++k) {
        low[k] = static_cast<int>(floor((s[k] - reach[k]) * g.n[k]));
        high[k] = static_cast<int>(floor((s[k] + reach[k]) * g.n[k]));
        if (!g.pbc[k]) {
            low[k] = max(low[k], 0);
            high[k] = min(high[k], g.n[k] - 1);
        }
    }
    // how far along vector k the cell u is from the atom, at least
    auto gap = [&](int k, int u) {
        const Real lo = static_cast<Real>(u) / g.n[k], hi = static_cast<Real>(u + 1) / g.n[k];
        const Real f = s[k] < lo ? lo - s[k] : (s[k] > hi ? s[k] - hi : Real(0));
        return f * g.thick[k];
    };
    for (int u2 = low[2]; u2 <= high[2]; ++u2) {
        const int w2 = ((u2 % g.n[2]) + g.n[2]) % g.n[2];
        const int d2 = u2 - c[2];
        const Real g2 = gap(2, u2);
        for (int u1 = low[1]; u1 <= high[1]; ++u1) {
            const int w1 = ((u1 % g.n[1]) + g.n[1]) % g.n[1];
            const int d1 = u1 - c[1];
            const Real g1 = gap(1, u1);
            for (int u0 = low[0]; u0 <= high[0]; ++u0) {
                const int w0 = ((u0 % g.n[0]) + g.n[0]) % g.n[0];
                const int d0 = u0 - c[0];
                // the cells beyond the cutoff, in an orthogonal box by their
                // distance, in another by the planes of the box between them,
                // a little closer for the rounding of the fractional coordinates
                const Real g0 = gap(0, u0);
                const Real nearest =
                    g.orthogonal ? g0 * g0 + g1 * g1 + g2 * g2
                                 : fmax(g0 * g0, fmax(g1 * g1, g2 * g2));
                if (!(nearest < cut2 * Real(1.0001))) continue;
                const int key = w0 + g.n[0] * (w1 + g.n[1] * w2);
                const Real shift[3] = {
                    d0 * g.step[0] + d1 * g.step[3] + d2 * g.step[6],
                    d0 * g.step[1] + d1 * g.step[4] + d2 * g.step[7],
                    d0 * g.step[2] + d1 * g.step[5] + d2 * g.step[8],
                };
                const bool home = d0 == 0 && d1 == 0 && d2 == 0;
                const int begin = g.start[key], stop = g.end[key];
                for (int first = begin; first < stop; first += kWarp) {
                    const int q = first + lane;
                    bool hit = q < stop && !(home && q == p);
                    Real v[3] = {Real(0), Real(0), Real(0)};
                    Real r2 = Real(0), species = Real(0);
                    if (hit) {
                        const Real4<Real> lq = g.local[q];
                        v[0] = lq.x - lp.x + shift[0];
                        v[1] = lq.y - lp.y + shift[1];
                        v[2] = lq.z - lp.z + shift[2];
                        r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                        species = lq.w;
                        hit = r2 < cut2 && r2 >= Real(kClosest2);
                    }
                    visit(hit, q, v, r2, static_cast<int>(species));
                }
            }
        }
        plane();
    }
}

template <class Real, class Visit>
__device__ void each_pair(const Grid<Real>& g, std::int64_t p, const Real* reach, Real cut2,
                          int lane, Visit&& visit) {
    each_pair(g, p, reach, cut2, lane, visit, [] {});
}

// The sums of a lane, 3 of a force, 9 of a virial, an energy and its
// derivative with the CN: in the precision of the terms over a plane of
// cells, some 30 pairs a lane, then added to those in double.
template <class Real>
struct Sums {
    Real part[14] = {};
    double whole[14] = {};
    __device__ void flush() {
        for (int k = 0; k < 14; ++k) {
            whole[k] += static_cast<double>(part[k]);
            part[k] = Real(0);
        }
    }
};

template <class Real>
__device__ inline void count(Real rc, Real r2, Real& value, Real& slope) {
    const Real r = sqrt(r2);
    value = Real(1) / (Real(1) + exp(-Real(kCountSteepness) * (rc / r - Real(1))));
    slope = -value * (Real(1) - value) * Real(kCountSteepness) * rc / r2;
}

template <class Real>
__global__ void coordination(std::int64_t n, Grid<Real> g, Tables<Real> t, Real3w<Real> reach,
                             Real cut2, double* cn) {
    const int lane = static_cast<int>(threadIdx.x) % kWarp;
    const std::int64_t p =
        static_cast<std::int64_t>(blockIdx.x) * (blockDim.x / kWarp) + threadIdx.x / kWarp;
    if (p >= n) return;
    const int si = static_cast<int>(g.local[p].w);
    double sum = 0.0;
    each_pair(g, p, reach.v, cut2, lane, [&](bool hit, int, const Real*, Real r2, int sj) {
        if (!hit) return;
        Real value, slope;
        count(t.rcov[si] + t.rcov[sj], r2, value, slope);
        sum += static_cast<double>(value);
    });
    sum = warp_sum(sum);
    if (lane == 0) cn[p] = sum;
}

template <class Real>
__global__ void weights(std::int64_t n, Grid<Real> g, Tables<Real> t, const double* cn, Real* w,
                        Real* dw, Real* u) {
    const std::int64_t p = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const int si = static_cast<int>(g.local[p].w);
    const int m = t.references[si];
    const Real* ref = t.reference_cn + si * kD3References;
    double d[kD3References], e[kD3References];
    double least = 1e300;
    for (int a = 0; a < m; ++a) {
        d[a] = static_cast<double>(ref[a]) - cn[p];
        least = fmin(least, d[a] * d[a]);
    }
    double norm = 0.0;
    for (int a = 0; a < m; ++a) {
        e[a] = exp(-kWeightWidth * (d[a] * d[a] - least));
        norm += e[a];
    }
    double mean = 0.0;
    for (int a = 0; a < m; ++a) {
        e[a] /= norm;
        mean += e[a] * 2.0 * kWeightWidth * d[a];
    }
    for (int a = 0; a < m; ++a) {
        w[p * kD3References + a] = static_cast<Real>(e[a]);
        dw[p * kD3References + a] = static_cast<Real>(e[a] * (2.0 * kWeightWidth * d[a] - mean));
    }
    for (int s = 0; s < t.species; ++s) {
        const Real* block = t.c6 + t.blocks[s * t.species + si];
        const int ns = t.references[s];
        for (int a = 0; a < ns; ++a) {
            double sum = 0.0;
            for (int b = 0; b < m; ++b) sum += static_cast<double>(block[a * m + b]) * e[b];
            u[(t.slots[s] + a) * t.atoms + p] = static_cast<Real>(sum);
        }
    }
}

// The C6 of a pair (i, j) from the weights of i and the contracted C6 of j,
// and its derivative with the CN of i when dwi is given.
template <class Real>
__device__ inline void pair_c6(const Tables<Real>& t, int si, const Real* wi, const Real* dwi,
                               const Real* u, int q, Real& c6, Real& dc6) {
    const Real* uq = u + t.slots[si] * t.atoms + q;
    const int ni = t.references[si];
    c6 = Real(0);
    dc6 = Real(0);
    for (int a = 0; a < ni; ++a) {
        const Real ua = uq[a * t.atoms];
        c6 += wi[a] * ua;
        if (dwi != nullptr) dc6 += dwi[a] * ua;
    }
}

template <class Real>
__device__ inline Real power(Real x, int k) {
    Real out = Real(1);
    for (int i = 0; i < k; ++i) out *= x;
    return out;
}

// The damping of a pair: e, with the energy -C6 e, and g = (de/dr) / r.
template <class Real>
__device__ inline void damping(const Tables<Real>& t, int k, Real r2, Real& e, Real& g) {
    const Real ratio = t.ratio[k];
    if (t.rational) {
        const Real r4 = r2 * r2;
        const Real r6 = r4 * r2;
        const Real t6 = Real(1) / (r6 + t.radius6[k]);
        const Real t8 = Real(1) / (r4 * r4 + t.radius8[k]);
        e = t.s6 * t6 + t.s8 * ratio * t8;
        g = t.s6 * (Real(-6) * r4 * t6 * t6) + t.s8 * ratio * (Real(-8) * r6 * t8 * t8);
        return;
    }
    const Real r6 = r2 * r2 * r2;
    const Real r8 = r6 * r2;
    const Real t6 = power(t.radius6[k] * t.radius6[k] / r2, t.half_alpha);
    const Real t8 = power(t.radius8[k] * t.radius8[k] / r2, t.half_alpha + 1);
    const Real f6 = Real(1) / (Real(1) + Real(6) * t6);
    const Real f8 = Real(1) / (Real(1) + Real(6) * t8);
    const Real d6 = (Real(-6) * f6 + Real(6) * t.alpha * t6 * f6 * f6) / r2;
    const Real d8 = (Real(-8) * f8 + Real(6) * (t.alpha + Real(2)) * t8 * f8 * f8) / r2;
    e = t.s6 * f6 / r6 + t.s8 * ratio * f8 / r8;
    g = t.s6 * d6 / r6 + t.s8 * ratio * d8 / r8;
}

template <class Real>
struct Outputs {
    const int* order;
    std::int64_t n;
    double* forces;    // (n, 3) of the atoms in their places
    double* energies;  // (n,), or null
    double* virials;   // 9 columns of n, or null
    double* dedcn;     // (n,) sorted
};

template <class Real>
__device__ inline void store(const Outputs<Real>& o, std::int64_t p, int lane, double* f,
                             double energy, double* w, bool add) {
    for (int c = 0; c < 3; ++c) f[c] = warp_sum(f[c]);
    energy = warp_sum(energy);
    if (o.virials != nullptr) {
        for (int c = 0; c < 9; ++c) w[c] = warp_sum(w[c]);
    }
    if (lane != 0) return;
    const std::int64_t i = o.order[p];
    for (int c = 0; c < 3; ++c) {
        o.forces[3 * i + c] = (add ? o.forces[3 * i + c] : 0.0) + f[c];
    }
    if (o.energies != nullptr) o.energies[i] = (add ? o.energies[i] : 0.0) + energy;
    if (o.virials != nullptr) {
        for (int c = 0; c < 9; ++c) {
            double& slot = o.virials[static_cast<std::int64_t>(c) * o.n + i];
            slot = (add ? slot : 0.0) + w[c];
        }
    }
}

template <class Real>
__global__ void pairs(std::int64_t n, Grid<Real> g, Tables<Real> t, Real3w<Real> reach,
                      Real cut2, const Real* w, const Real* dw, const Real* u, Outputs<Real> o) {
    const int lane = static_cast<int>(threadIdx.x) % kWarp;
    const std::int64_t p =
        static_cast<std::int64_t>(blockIdx.x) * (blockDim.x / kWarp) + threadIdx.x / kWarp;
    if (p >= n) return;
    const int si = static_cast<int>(g.local[p].w);
    const Real* wi = w + p * kD3References;
    const Real* dwi = dw + p * kD3References;
    const bool virial = o.virials != nullptr;
    Sums<Real> sums;
    each_pair(
        g, p, reach.v, cut2, lane,
        [&](bool hit, int q, const Real* v, Real r2, int sj) {
            if (!hit) return;
            Real c6, dc6;
            pair_c6(t, si, wi, dwi, u, q, c6, dc6);
            Real e, gr;
            damping(t, si * t.species + sj, r2, e, gr);
            sums.part[12] -= Real(0.5) * c6 * e;
            sums.part[13] -= dc6 * e;
            const Real fpair = -c6 * gr;
            for (int a = 0; a < 3; ++a) sums.part[a] += v[a] * fpair;
            if (virial) {
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        sums.part[3 + 3 * a + b] += Real(0.5) * v[a] * v[b] * c6 * gr;
                    }
                }
            }
        },
        [&] { sums.flush(); });
    double* whole = sums.whole;
    const double de = warp_sum(whole[13]);
    if (lane == 0) o.dedcn[p] = de;
    store(o, p, lane, whole, whole[12], whole + 3, false);
}

template <class Real>
__global__ void corner_counts(std::int64_t n, Grid<Real> g, Real3w<Real> reach, Real cut2,
                              int* counts) {
    const int lane = static_cast<int>(threadIdx.x) % kWarp;
    const std::int64_t p =
        static_cast<std::int64_t>(blockIdx.x) * (blockDim.x / kWarp) + threadIdx.x / kWarp;
    if (p >= n) return;
    int found = 0;
    each_pair(g, p, reach.v, cut2, lane, [&](bool hit, int, const Real*, Real, int) {
        found += __popc(__ballot_sync(kAll, hit));
    });
    if (lane == 0) counts[p] = found;
}

// The corners of the atoms from first, in the order the lanes find them.
template <class Real>
__global__ void corners_of(std::int64_t first, std::int64_t last, Grid<Real> g, Tables<Real> t,
                           Real3w<Real> reach, Real cut2, const Real* w, const Real* dw,
                           const Real* u, const std::int64_t* offsets, Corner<Real>* corners) {
    const int lane = static_cast<int>(threadIdx.x) % kWarp;
    const std::int64_t p = first + static_cast<std::int64_t>(blockIdx.x) * (blockDim.x / kWarp) +
                           threadIdx.x / kWarp;
    if (p >= last) return;
    const int si = static_cast<int>(g.local[p].w);
    const Real* wi = w + p * kD3References;
    const Real* dwi = dw + p * kD3References;
    Corner<Real>* out = corners + (offsets[p] - offsets[first]);
    int written = 0;
    each_pair(g, p, reach.v, cut2, lane, [&](bool hit, int q, const Real* v, Real r2, int sj) {
        const unsigned found = __ballot_sync(kAll, hit);
        if (hit) {
            Corner<Real>& c = out[written + __popc(found & ((1u << lane) - 1u))];
            pair_c6(t, si, wi, dwi, u, q, c.c6, c.dc6);
            c.v[0] = v[0];
            c.v[1] = v[1];
            c.v[2] = v[2];
            c.r2 = r2;
            c.radius = t.atm_radius[si * t.species + sj];
            c.q = q;
        }
        written += __popc(found);
    });
}

template <class Real>
__device__ inline Real dang(Real x, Real y, Real z, Real r5) {
    return Real(-0.375) *
           (x * x * x + x * x * (y + z) + x * (Real(3) * y * y + Real(2) * y * z + Real(3) * z * z) -
            Real(5) * (y - z) * (y - z) * (y + z)) /
           r5;
}

// The triangles of each atom with two of its corners, a warp an atom.
template <class Real>
__global__ void triangles(std::int64_t first, std::int64_t last, Grid<Real> g, Tables<Real> t,
                          Real cut2, const Real* w, const Real* u, const std::int64_t* offsets,
                          const int* counts, const Corner<Real>* corners, Outputs<Real> o) {
    const int lane = static_cast<int>(threadIdx.x) % kWarp;
    const std::int64_t p = first + static_cast<std::int64_t>(blockIdx.x) * (blockDim.x / kWarp) +
                           threadIdx.x / kWarp;
    if (p >= last) return;
    const Corner<Real>* own = corners + (offsets[p] - offsets[first]);
    const int m = counts[p];
    const bool virial = o.virials != nullptr;
    const Real exponent = (t.alpha + Real(2)) / Real(3);
    double f[3] = {0.0, 0.0, 0.0}, vir[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    double energy = 0.0, de = 0.0;
    for (int a = 0; a < m; ++a) {
        const Corner<Real> cj = own[a];
        const int sj = static_cast<int>(g.local[cj.q].w);
        const Real* wj = w + static_cast<std::int64_t>(cj.q) * kD3References;
        for (int b = a + 1 + lane; b < m; b += kWarp) {
            const Corner<Real> ck = own[b];
            const Real vjk[3] = {ck.v[0] - cj.v[0], ck.v[1] - cj.v[1], ck.v[2] - cj.v[2]};
            const Real r2jk = vjk[0] * vjk[0] + vjk[1] * vjk[1] + vjk[2] * vjk[2];
            if (!(r2jk < cut2) || r2jk < Real(kClosest2)) continue;
            const int sk = static_cast<int>(g.local[ck.q].w);
            Real c6jk, unused;
            pair_c6(t, sj, wj, static_cast<const Real*>(nullptr), u, ck.q, c6jk, unused);
            const Real r2ij = cj.r2, r2ik = ck.r2;
            const Real c9 = -t.s9 * sqrt(fabs(cj.c6 * ck.c6 * c6jk) / t.hartree);
            const Real r0 = cj.radius * ck.radius * t.atm_radius[sj * t.species + sk];
            const Real rr2 = r2ij * r2ik * r2jk;
            const Real r1 = sqrt(rr2);
            const Real r3 = rr2 * r1;
            const Real r5 = r3 * rr2;
            const Real x = pow(r0 / r1, exponent);
            const Real fdmp = Real(1) / (Real(1) + Real(6) * x);
            const Real ang = Real(0.375) * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) *
                                 (-r2ij + r2jk + r2ik) / r5 +
                             Real(1) / r3;
            const Real dfdmp = Real(-2) * (t.alpha + Real(2)) * x * fdmp * fdmp;
            const Real gij = c9 * (-dang(r2ij, r2jk, r2ik, r5) * fdmp + ang * dfdmp) / r2ij;
            const Real gik = c9 * (-dang(r2ik, r2jk, r2ij, r5) * fdmp + ang * dfdmp) / r2ik;
            const Real e = -c9 * ang * fdmp;
            energy += static_cast<double>(e) / 3.0;
            de += 0.5 * static_cast<double>(e * (cj.dc6 / cj.c6 + ck.dc6 / ck.c6));
            for (int c = 0; c < 3; ++c) f[c] += static_cast<double>(gij * cj.v[c] + gik * ck.v[c]);
            if (virial) {
                const Real gjk = c9 * (-dang(r2jk, r2ik, r2ij, r5) * fdmp + ang * dfdmp) / r2jk;
                for (int c = 0; c < 3; ++c) {
                    for (int d = 0; d < 3; ++d) {
                        vir[3 * c + d] -= static_cast<double>(
                            (gij * cj.v[c] * cj.v[d] + gik * ck.v[c] * ck.v[d] +
                             gjk * vjk[c] * vjk[d]) /
                            Real(3));
                    }
                }
            }
        }
    }
    de = warp_sum(de);
    if (lane == 0) o.dedcn[p] += de;
    store(o, p, lane, f, energy, vir, true);
}

template <class Real>
__global__ void chain(std::int64_t n, Grid<Real> g, Tables<Real> t, Real3w<Real> reach,
                      Real cut2, Outputs<Real> o) {
    const int lane = static_cast<int>(threadIdx.x) % kWarp;
    const std::int64_t p =
        static_cast<std::int64_t>(blockIdx.x) * (blockDim.x / kWarp) + threadIdx.x / kWarp;
    if (p >= n) return;
    const int si = static_cast<int>(g.local[p].w);
    const double dei = o.dedcn[p];
    const bool virial = o.virials != nullptr;
    Sums<Real> sums;
    each_pair(
        g, p, reach.v, cut2, lane,
        [&](bool hit, int q, const Real* v, Real r2, int sj) {
            if (!hit) return;
            Real value, slope;
            count(t.rcov[si] + t.rcov[sj], r2, value, slope);
            const Real fpair = static_cast<Real>(dei + o.dedcn[q]) * slope / sqrt(r2);
            for (int a = 0; a < 3; ++a) sums.part[a] += v[a] * fpair;
            if (virial) {
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        sums.part[3 + 3 * a + b] -= Real(0.5) * v[a] * v[b] * fpair;
                    }
                }
            }
        },
        [&] { sums.flush(); });
    Outputs<Real> forces_only = o;
    forces_only.energies = nullptr;
    store(forces_only, p, lane, sums.whole, 0.0, sums.whole + 3, true);
}

// The thickness of the box along each vector: its volume over the area of
// the face of the other two.
void thickness(const Box& box, double out[3], double& volume) {
    const double* h = box.h;
    auto cross = [&](int a, int b, double c[3]) {
        c[0] = h[3 * a + 1] * h[3 * b + 2] - h[3 * a + 2] * h[3 * b + 1];
        c[1] = h[3 * a + 2] * h[3 * b] - h[3 * a] * h[3 * b + 2];
        c[2] = h[3 * a] * h[3 * b + 1] - h[3 * a + 1] * h[3 * b];
    };
    double c[3];
    cross(1, 2, c);
    volume = std::fabs(h[0] * c[0] + h[1] * c[1] + h[2] * c[2]);
    for (int k = 0; k < 3; ++k) {
        cross((k + 1) % 3, (k + 2) % 3, c);
        out[k] = volume / std::sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
    }
}

template <class Real>
class D3 final : public Potential {
public:
    explicit D3(const calculators::D3Potential& p) : p_(p) {
        const auto s = static_cast<std::size_t>(p.species);
        auto real = [](const std::vector<double>& v) {
            return std::vector<Real>(v.begin(), v.end());
        };
        references_.upload(p.references.data(), s);
        reference_cn_.upload(real(p.reference_cn).data(), p.reference_cn.size());
        rcov_.upload(real(p.rcov).data(), s);
        ratio_.upload(real(p.ratio).data(), s * s);
        radius6_.upload(real(p.radius6).data(), s * s);
        radius8_.upload(real(p.radius8).data(), s * s);
        atm_radius_.upload(real(p.atm_radius).data(), s * s);
        blocks_.upload(p.blocks.data(), s * s);
        slots_.upload(p.slots.data(), s + 1);
        c6_.upload(real(p.c6).data(), p.c6.size());
    }

    // It finds its pairs itself: the list of the device holds none of them.
    double cutoff() const override { return 0.0; }
    int species() const override { return p_.species; }

    void compute(Atoms& atoms, Neighbors&, const Box& box, bool energy, bool virial,
                 bool atomic) override {
        Atoms::Impl& a = atoms.impl();
        const std::int64_t n = a.n;
        virial = virial || atomic;
        if (virial) a.virials.resize(static_cast<std::size_t>(9 * n));
        Grid<Real> g = sort_atoms(a, box);
        const std::int64_t width = p_.slots.back();
        const Tables<Real> t{p_.species,
                             p_.damping == D3Damping::Rational ? 1 : 0,
                             p_.half_alpha,
                             static_cast<Real>(p_.s6),
                             static_cast<Real>(p_.s8),
                             static_cast<Real>(p_.s9),
                             static_cast<Real>(p_.alpha),
                             static_cast<Real>(p_.hartree),
                             width,
                             n,
                             slots_.data(),
                             references_.data(),
                             reference_cn_.data(),
                             rcov_.data(),
                             ratio_.data(),
                             radius6_.data(),
                             radius8_.data(),
                             atm_radius_.data(),
                             blocks_.data(),
                             c6_.data()};
        room(cn_, static_cast<std::size_t>(n));
        room(dedcn_, static_cast<std::size_t>(n));
        room(w_, static_cast<std::size_t>(n * kD3References));
        room(dw_, static_cast<std::size_t>(n * kD3References));
        room(u_, static_cast<std::size_t>(n * width));
        const Outputs<Real> o{order_sorted_.data(), n, a.forces.data(),
                              energy ? a.energies.data() : nullptr,
                              virial ? a.virials.data() : nullptr, dedcn_.data()};
        const unsigned warps = static_cast<unsigned>((n + kBlock / kWarp - 1) / (kBlock / kWarp));
        const Real cn2 = static_cast<Real>(p_.cn_cutoff * p_.cn_cutoff);
        const Real cut2 = static_cast<Real>(p_.cutoff * p_.cutoff);
        coordination<Real><<<warps, kBlock>>>(n, g, t, reach(p_.cn_cutoff), cn2, cn_.data());
        MDAPY_LAUNCHED();
        weights<Real><<<blocks(n), kBlock>>>(n, g, t, cn_.data(), w_.data(), dw_.data(),
                                             u_.data());
        MDAPY_LAUNCHED();
        pairs<Real><<<warps, kBlock>>>(n, g, t, reach(p_.cutoff), cut2, w_.data(), dw_.data(),
                                       u_.data(), o);
        MDAPY_LAUNCHED();
        if (p_.s9 != 0.0) three_body(n, g, t, o);
        chain<Real><<<warps, kBlock>>>(n, g, t, reach(p_.cn_cutoff), cn2, o);
        MDAPY_LAUNCHED();
        a.has_energies = energy;
        a.has_virials = virial;
    }

private:
    // The fractional half-widths of a cutoff along each vector, a little more
    // for the rounding of the fractional coordinates.
    Real3w<Real> reach(double cutoff) const {
        Real3w<Real> r;
        for (int k = 0; k < 3; ++k) {
            r.v[k] = static_cast<Real>(cutoff / thick_[k] * (1.0 + 1e-5) + 1e-6);
        }
        return r;
    }

    // The atoms sorted into cells of about a third of the reach, at least two
    // atoms to a cell on average.
    Grid<Real> sort_atoms(Atoms::Impl& a, const Box& box) {
        const std::int64_t n = a.n;
        double volume;
        thickness(box, thick_, volume);
        const double reach_most = p_.reach;
        const double width = std::max(reach_most / 3.0, std::cbrt(2.0 * volume / n));
        int cells[3];
        std::int64_t total = 1;
        for (int k = 0; k < 3; ++k) {
            cells[k] = std::max(1, static_cast<int>(std::floor(thick_[k] / width)));
            total *= cells[k];
        }
        const DeviceBox<double> dbox = device_box<double>(box);
        const int3 grid = make_int3(cells[0], cells[1], cells[2]);
        const auto size = static_cast<std::size_t>(n);
        keys_.resize(size);
        keys_sorted_.resize(size);
        order_.resize(size);
        order_sorted_.resize(size);
        local_.resize(size);
        frac_.resize(size);
        cell_.resize(size);
        start_.resize(static_cast<std::size_t>(total));
        end_.resize(static_cast<std::size_t>(total));
        const double4a* x = a.positions<double>();
        bin<<<blocks(n), kBlock>>>(n, x, dbox, grid, keys_.data(), order_.data());
        MDAPY_LAUNCHED();
        int bits = 1;
        while ((std::int64_t{1} << bits) < total) ++bits;
        std::size_t scratch = 0;
        MDAPY_CUDA(cub::DeviceRadixSort::SortPairs(nullptr, scratch, keys_.data(),
                                                   keys_sorted_.data(), order_.data(),
                                                   order_sorted_.data(), static_cast<int>(n), 0,
                                                   bits));
        if (sort_scratch_.size() < scratch) sort_scratch_.resize(scratch);
        MDAPY_CUDA(cub::DeviceRadixSort::SortPairs(sort_scratch_.data(), scratch, keys_.data(),
                                                   keys_sorted_.data(), order_.data(),
                                                   order_sorted_.data(), static_cast<int>(n), 0,
                                                   bits));
        start_.zero();
        end_.zero();
        bounds<<<blocks(n), kBlock>>>(n, keys_sorted_.data(), start_.data(), end_.data());
        MDAPY_LAUNCHED();
        arrange_atoms<Real><<<blocks(n), kBlock>>>(n, x, dbox, grid, order_sorted_.data(),
                                                   local_.data(), frac_.data(), cell_.data());
        MDAPY_LAUNCHED();
        Grid<Real> g;
        for (int k = 0; k < 3; ++k) {
            g.n[k] = cells[k];
            g.pbc[k] = box.pbc[k] ? 1 : 0;
            for (int c = 0; c < 3; ++c) {
                g.step[3 * k + c] = static_cast<Real>(box.h[3 * k + c] / cells[k]);
            }
        }
        g.start = start_.data();
        g.end = end_.data();
        g.local = local_.data();
        g.frac = frac_.data();
        g.cell = cell_.data();
        for (int k = 0; k < 3; ++k) g.thick[k] = static_cast<Real>(thick_[k]);
        g.orthogonal = dbox.orthogonal;
        return g;
    }

    // The triangles, a batch of atoms at a time so that their corners fit.
    void three_body(std::int64_t n, const Grid<Real>& g, const Tables<Real>& t,
                    const Outputs<Real>& o) {
        const unsigned warps = static_cast<unsigned>((n + kBlock / kWarp - 1) / (kBlock / kWarp));
        const Real atm2 = static_cast<Real>(p_.three_body_cutoff * p_.three_body_cutoff);
        const Real3w<Real> r = reach(p_.three_body_cutoff);
        room(counts_, static_cast<std::size_t>(n));
        room(offsets_, static_cast<std::size_t>(n + 1));
        corner_counts<Real><<<warps, kBlock>>>(n, g, r, atm2, counts_.data());
        MDAPY_LAUNCHED();
        // offsets: exclusive sums of the counts, n + 1 of them
        Buffer<std::int64_t>& wide = wide_counts_;
        room(wide, static_cast<std::size_t>(n + 1));
        widen<<<blocks(n + 1), kBlock>>>(n, counts_.data(), wide.data());
        MDAPY_LAUNCHED();
        std::size_t scratch = 0;
        MDAPY_CUDA(cub::DeviceScan::ExclusiveSum(nullptr, scratch, wide.data(), offsets_.data(),
                                                 static_cast<int>(n + 1)));
        if (scan_scratch_.size() < scratch) scan_scratch_.resize(scratch);
        MDAPY_CUDA(cub::DeviceScan::ExclusiveSum(scan_scratch_.data(), scratch, wide.data(),
                                                 offsets_.data(), static_cast<int>(n + 1)));
        std::vector<std::int64_t> host(static_cast<std::size_t>(n + 1));
        offsets_.download(host.data(), host.size());
        std::int64_t first = 0;
        while (first < n) {
            std::int64_t last = first + 1;
            while (last < n && host[static_cast<std::size_t>(last + 1)] -
                                       host[static_cast<std::size_t>(first)] <=
                                   kCornersAtOnce) {
                ++last;
            }
            const std::int64_t held = host[static_cast<std::size_t>(last)] -
                                      host[static_cast<std::size_t>(first)];
            room(corners_, static_cast<std::size_t>(std::max<std::int64_t>(held, 1)));
            const auto batch = static_cast<unsigned>(
                (last - first + kBlock / kWarp - 1) / (kBlock / kWarp));
            corners_of<Real><<<batch, kBlock>>>(first, last, g, t, r, atm2, w_.data(), dw_.data(),
                                                u_.data(), offsets_.data(), corners_.data());
            MDAPY_LAUNCHED();
            triangles<Real><<<batch, kBlock>>>(first, last, g, t, atm2, w_.data(), u_.data(),
                                               offsets_.data(), counts_.data(), corners_.data(),
                                               o);
            MDAPY_LAUNCHED();
            first = last;
        }
    }

    calculators::D3Potential p_;
    double thick_[3] = {1.0, 1.0, 1.0};
    Buffer<int> references_;
    Buffer<Real> reference_cn_, rcov_, ratio_, radius6_, radius8_, atm_radius_, c6_;
    Buffer<std::int64_t> blocks_, slots_;
    Buffer<int> keys_, keys_sorted_, order_, order_sorted_, start_, end_, counts_;
    Buffer<unsigned char> sort_scratch_, scan_scratch_;
    Buffer<Real4<Real>> local_, frac_;
    Buffer<int4> cell_;
    Buffer<double> cn_, dedcn_;
    Buffer<Real> w_, dw_, u_;
    Buffer<std::int64_t> offsets_, wide_counts_;
    Buffer<Corner<Real>> corners_;
};

}  // namespace

std::unique_ptr<Potential> make_d3(const calculators::D3Potential& potential,
                                   Precision precision) {
    if (precision == Precision::Mixed) {
        return std::make_unique<D3<float>>(potential);
    }
    return std::make_unique<D3<double>>(potential);
}

}  // namespace mdapy::cuda
