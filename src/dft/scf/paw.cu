// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/scf/paw.h"

#include <algorithm>
#include <cmath>
#include <new>
#include <stdexcept>
#include <string>

namespace mdapy::dft {

namespace {

constexpr int kThreads = 256;
constexpr double kFourPi = 12.56637061435917295384870733753984;
constexpr std::size_t kBudget = std::size_t(1) << 26;  // doubles of work, half a gigabyte

unsigned blocks(std::size_t work) { return static_cast<unsigned>((work + kThreads - 1) / kThreads); }

void launched(const char* what) { check_cuda(cudaGetLastError(), what); }

// df/dr at point i of f on the mesh r: the parabola through i - 1, i and
// i + 1, zero at the last point and extrapolated linearly to the first
__device__ double slope3(const double* f, const double* r, int i) {
    const double a = r[i + 1] - r[i], b = r[i - 1] - r[i];
    return (a * a * (f[i - 1] - f[i]) - b * b * (f[i + 1] - f[i])) / (a * b * (r[i + 1] - r[i - 1]));
}

__device__ double radial_slope(const double* f, const double* r, int i, int mesh) {
    if (i == mesh - 1) return 0.0;
    if (i == 0) {
        const double g1 = slope3(f, r, 1), g2 = slope3(f, r, 2);
        return g1 + (g2 - g1) * (r[0] - r[1]) / (r[2] - r[1]);
    }
    return slope3(f, r, i);
}

// cq[(as, LM, q)] = sum_p bec[as, p] G(LM, p) over the pairs p of radial pair q
__global__ void coef_kernel(std::size_t total, int pairs, int l2, int radial, const double* bec, const double* coef,
                            const int* pair, double* cq) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= total) return;
    const int q = static_cast<int>(i % radial), lm = static_cast<int>((i / radial) % l2);
    const double* b = bec + (i / (static_cast<std::size_t>(radial) * l2)) * pairs;
    double x = 0.0;
    for (int p = 0; p < pairs; ++p)
        if (pair[p] == q) x += b[p] * coef[static_cast<std::size_t>(p) * l2 + lm];
    cq[i] = x;
}

// rho[(as, LM, r)] = sum_q cq[(as, LM, q)] F[q, L(LM), r], F of one L for AE
__global__ void density_kernel(std::size_t total, int mesh, int l2, int radial, int lcount, const int* ell,
                               const double* cq, const double* f, double* rho) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= total) return;
    const int r = static_cast<int>(i % mesh), lm = static_cast<int>((i / mesh) % l2);
    const int L = lcount == 1 ? 0 : ell[lm];
    const double* c = cq + (i / mesh) * radial;
    double x = 0.0;
    for (int q = 0; q < radial; ++q) x += c[q] * f[(static_cast<std::size_t>(q) * lcount + L) * mesh + r];
    rho[i] = x;
}

// the Hartree potential of each (atom, LM) of the densities summed over the
// spins, by Numerov's method in x = ln r on y = sqrt(r) V,
//   y'' - (l + 1/2)^2 y = -(2l + 1) f sqrt(r),  f = 4 pi n_LM / (2l + 1),
// y ~ r^(l + 1/2) at the first point (with the series c2 r^2 + c3 r^3 for l <
// 2) and ~ r^-(l + 1/2) at the last; the tridiagonal system by elimination.
// eh[i] = 1/2 int V n_LM dr over the mesh.
__global__ void hartree_kernel(int count, int spins, int l2, int mesh, const int* ell, const double* rho,
                               const double* r, const double* sqr, const double* whole, double dx, double* vh,
                               double* scratch, double* eh) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count) return;
    const int a = i / l2, lm = i % l2, L = ell[lm];
    const std::size_t m = static_cast<std::size_t>(mesh);
    auto total = [&](int k) {
        double x = 0.0;
        for (int s = 0; s < spins; ++s) x += rho[((static_cast<std::size_t>(a) * spins + s) * l2 + lm) * m + k];
        return x;
    };
    const double pref = kFourPi / (2 * L + 1);
    const double k21 = 2.0 * L + 1.0;
    const int nst = 2 * L + 2;
    double c2 = 0.0, c3 = 0.0;
    if (nst - L - 1 < 3) {
        // the cubic through the first four points of -(2l+1) f / r^(2l+2), by
        // divided differences, gives c2 and c3 of the series
        double d[4], x[4];
        for (int j = 0; j < 4; ++j) {
            x[j] = r[j];
            d[j] = -k21 * pref * total(j) / ::pow(r[j], static_cast<double>(nst));
        }
        double d1[3], d2[2];
        for (int j = 0; j < 3; ++j) d1[j] = (d[j + 1] - d[j]) / (x[j + 1] - x[j]);
        for (int j = 0; j < 2; ++j) d2[j] = (d1[j + 1] - d1[j]) / (x[j + 2] - x[j]);
        const double b3 = (d2[1] - d2[0]) / (x[3] - x[0]);
        const double b2 = d2[0] - b3 * (x[0] + x[1] + x[2]);
        const double b1 = d1[0] - b2 * (x[1] + x[0]) - b3 * (x[0] * x[0] + x[1] * x[1] + x[0] * x[1]);
        const double b0 = d[0] - x[0] * (b1 + x[0] * (b2 + x[0] * b3));
        const double b[4] = {b0, b1, b2, b3};
        double e[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int j = 0; j < 4; ++j) e[nst - L - 2 + j] = b[j];
        c2 = e[0] / (4.0 * L + 6.0);
        c3 = e[1] / (6.0 * L + 12.0);
    }
    const double ch = dx * dx / 12.0, xkh2 = ch * (L + 0.5) * (L + 0.5);
    const double ei = 1.0 - xkh2, di = -(2.0 + 10.0 * xkh2), off = -ei;
    double* v = vh + static_cast<std::size_t>(i) * m;
    double* c = scratch + static_cast<std::size_t>(i) * m;
    const double f1 = ::pow(sqr[0] / sqr[1], k21), fn = ::pow(sqr[mesh - 2] / sqr[mesh - 1], k21);
    const double s0k = ::pow(sqr[0], k21);
    auto source = [&](int k) { return k21 * ch * sqr[k] * pref * total(k); };
    double sm = source(0), s0 = source(1);
    for (int k = 1; k <= mesh - 2; ++k) {
        const double sp = source(k + 1);
        double rhs = sm + 10.0 * s0 + sp, diag = -di;
        if (k == 1) {
            diag -= ei * f1;
            rhs -= ei * s0k * (c2 * (r[1] * r[1] - r[0] * r[0]) + c3 * (r[1] * r[1] * r[1] - r[0] * r[0] * r[0]));
        }
        if (k == mesh - 2) diag -= ei * fn;
        const double w = k == 1 ? diag : diag - off * c[k - 1];
        c[k] = off / w;
        v[k] = (k == 1 ? rhs : rhs - off * v[k - 1]) / w;
        sm = s0;
        s0 = sp;
    }
    for (int k = mesh - 3; k >= 1; --k) v[k] -= c[k] * v[k + 1];
    const double c0 = v[1] / ::pow(sqr[1], k21) - c2 * r[1] * r[1] - c3 * r[1] * r[1] * r[1];
    v[0] = s0k * (c0 + c2 * r[0] * r[0] + c3 * r[0] * r[0] * r[0]);
    v[mesh - 1] = v[mesh - 2] * fn;
    double e = 0.0;
    for (int k = 0; k < mesh; ++k) {
        v[k] /= sqr[k];
        e += whole[k] * v[k] * total(k);
    }
    eh[i] = 0.5 * e;
}

// n[(as, x, r)] = raw / r^2 + core / spins
__global__ void finish_density_kernel(std::size_t total, int mesh, int spins, const double* r, const double* core,
                                      double* n) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= total) return;
    const int k = static_cast<int>(i % mesh);
    n[i] = n[i] / (r[k] * r[k]) + core[k] / spins;
}

// the radial derivative of each (as, x) row
__global__ void radial_kernel(std::size_t total, int mesh, const double* r, const double* f, double* g) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= total) return;
    const int k = static_cast<int>(i % mesh);
    g[i] = radial_slope(f + (i - k), r, k, mesh);
}

struct Gradient {
    const double *gr, *gp, *gt;  // the radial derivative, the raw phi and theta parts (times r^3)
};

// the gradient of spin s at (a, x, k), with the phi and theta parts over r^3
__device__ void gradient_at(const Gradient& g, std::size_t row, int k, double rk, double out[3]) {
    const std::size_t at = row + k;
    const double r3 = rk * rk * rk;
    out[0] = g.gr[at];
    out[1] = g.gp[at] / r3;
    out[2] = g.gt[at] / r3;
}

// the points (a, x, r) for the XC kernel: the spin densities and sigma, both
// halves of n without spin
__global__ void assemble_kernel(std::size_t total, int mesh, int nx, int spins, bool gga, const double* r,
                                const double* n, Gradient g, double* rho, double* sigma) {
    const std::size_t p = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (p >= total) return;
    const int k = static_cast<int>(p % mesh);
    const std::size_t ax = p / mesh, a = ax / nx, x = ax % nx;
    const std::size_t m = static_cast<std::size_t>(mesh);
    if (spins == 1) {
        const std::size_t row = (a * nx + x) * m;
        rho[2 * p] = rho[2 * p + 1] = 0.5 * n[row + k];
        double s2 = 0.0;
        if (gga) {
            double v[3];
            gradient_at(g, row, k, r[k], v);
            s2 = 0.25 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        }
        sigma[3 * p] = sigma[3 * p + 1] = sigma[3 * p + 2] = s2;
        return;
    }
    const std::size_t up = ((a * 2) * nx + x) * m, down = ((a * 2 + 1) * nx + x) * m;
    rho[2 * p] = n[up + k];
    rho[2 * p + 1] = n[down + k];
    double u[3] = {0.0, 0.0, 0.0}, d[3] = {0.0, 0.0, 0.0};
    if (gga) {
        gradient_at(g, up, k, r[k], u);
        gradient_at(g, down, k, r[k], d);
    }
    sigma[3 * p] = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    sigma[3 * p + 1] = u[0] * d[0] + u[1] * d[1] + u[2] * d[2];
    sigma[3 * p + 2] = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
}

// exc[(a, x)] = ww[x] sum_r whole r^2 f
__global__ void energy_kernel(int count, int mesh, int nx, const double* ww, const double* whole, const double* r,
                              const double* f, double* exc) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count) return;
    const double* row = f + static_cast<std::size_t>(i) * mesh;
    double e = 0.0;
    for (int k = 0; k < mesh; ++k) e += whole[k] * r[k] * r[k] * row[k];
    exc[i] = ww[i % nx] * e;
}

// v_rad[(as, x, r)] from the potentials of the points
__global__ void scatter_kernel(std::size_t total, int mesh, int nx, int spins, const double* v, double* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= total) return;
    const std::size_t k = i % mesh, asx = i / mesh, x = asx % nx, as = asx / nx;
    const std::size_t a = as / spins, s = as % spins;
    const std::size_t p = (a * nx + x) * mesh + k;
    out[i] = v[2 * p + s];
}

// the flux of GGA h[(as, c, x, r)] times r^2, its theta part over sin(theta)
__global__ void flux_kernel(std::size_t total, int mesh, int nx, int spins, const double* r, const double* sin_theta,
                            const double* vs, Gradient g, double* h) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= total) return;
    const std::size_t m = static_cast<std::size_t>(mesh);
    const int k = static_cast<int>(i % m);
    const std::size_t asx = i / m, x = asx % nx, as = asx / nx;
    const std::size_t a = as / spins, s = as % spins;
    const std::size_t p = (a * nx + x) * m + k;
    const double uu = vs[3 * p], ud = vs[3 * p + 1], dd = vs[3 * p + 2];
    double out[3];
    if (spins == 1) {
        double v[3];
        gradient_at(g, (a * nx + x) * m, k, r[k], v);
        for (int c = 0; c < 3; ++c) out[c] = (2.0 * uu + ud) * 0.5 * v[c];
    } else {
        double u[3], d[3];
        gradient_at(g, ((a * 2) * nx + x) * m, k, r[k], u);
        gradient_at(g, ((a * 2 + 1) * nx + x) * m, k, r[k], d);
        for (int c = 0; c < 3; ++c) out[c] = s == 0 ? 2.0 * uu * u[c] + ud * d[c] : 2.0 * dd * d[c] + ud * u[c];
    }
    const double r2 = r[k] * r[k];
    for (int c = 0; c < 3; ++c) {
        double value = out[c] * r2;
        if (c == 2) value /= sin_theta[x];
        h[((as * 3 + c) * nx + x) * m + k] = value;
    }
}

// v_LM -= div_LM / r^3 + d(H_r,LM)/dr / r^2
__global__ void divergence_kernel(std::size_t total, int mesh, int l2, int lh, const double* r, const double* hlm,
                                  const double* div_lm, double* v_lm) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= total) return;
    const std::size_t m = static_cast<std::size_t>(mesh);
    const int k = static_cast<int>(i % m);
    const std::size_t aslm = i / m, lm = aslm % l2, as = aslm / l2;
    const double* hr = hlm + ((as * 3) * lh + lm) * m;
    const double rk = r[k];
    v_lm[i] -= div_lm[i] / (rk * rk * rk) + radial_slope(hr, r, k, mesh) / (rk * rk);
}

// I[(as, LM, q)] = sum_r inner (v_LM + v_H,LM) F[q, L(LM), r]
__global__ void integral_kernel(std::size_t total, int mesh, int spins, int l2, int radial, int lcount,
                                const int* ell, const double* v_lm, const double* vh, const double* f,
                                const double* inner, double* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= total) return;
    const std::size_t m = static_cast<std::size_t>(mesh);
    const int q = static_cast<int>(i % radial);
    const std::size_t aslm = i / radial, lm = aslm % l2, as = aslm / l2, a = as / spins;
    const int L = lcount == 1 ? 0 : ell[lm];
    const double* v = v_lm + aslm * m;
    const double* w = vh + (a * l2 + lm) * m;
    const double* fq = f + (static_cast<std::size_t>(q) * lcount + L) * m;
    double x = 0.0;
    for (std::size_t k = 0; k < m; ++k) x += inner[k] * (v[k] + w[k]) * fq[k];
    out[i] = x;
}

// D[(as, p)] += sign sum_LM G(LM, p) I[(as, LM, pair(p))]
__global__ void strengths_kernel(std::size_t total, int pairs, int l2, int radial, double sign, const double* coef,
                                 const int* pair, const double* integral, double* d) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= total) return;
    const int p = static_cast<int>(i % pairs);
    const std::size_t as = i / pairs;
    double x = 0.0;
    for (int lm = 0; lm < l2; ++lm)
        x += coef[static_cast<std::size_t>(p) * l2 + lm] *
             integral[(as * l2 + lm) * radial + static_cast<std::size_t>(pair[p])];
    d[i] += sign * x;
}

}  // namespace

OneCenter::OneCenter(std::vector<SpeciesOneCenter> species, std::vector<int> atom_species, std::vector<int> widths,
                     int spins, Functional functional, double threshold, Kernels kernels)
    : species_(std::move(species)),
      atom_species_(std::move(atom_species)),
      spins_(spins),
      functional_(functional),
      threshold_(threshold),
      gga_(xc::gradient(functional)),
      blas_(kernels) {
    if (spins_ != 1 && spins_ != 2) throw std::invalid_argument("one or two spins");
    if (widths.size() != atom_species_.size()) throw std::invalid_argument("the width of each atom");
    offsets_.push_back(0);
    for (const int w : widths) offsets_.push_back(offsets_.back() + static_cast<std::size_t>(w) * (w + 1) / 2);
    device_.resize(species_.size());
    for (std::size_t t = 0; t < species_.size(); ++t) {
        const auto& sp = species_[t];
        if (sp.mesh == 0) continue;
        const std::size_t m = static_cast<std::size_t>(sp.mesh), nx = static_cast<std::size_t>(sp.nx),
                          lh = static_cast<std::size_t>(sp.lh);
        if (sp.mesh < 8 || sp.r.size() != m || sp.whole.size() != m || sp.inner.size() != m || sp.ww.size() != nx ||
            sp.ylm.size() != nx * lh || sp.dylmt.size() != nx * lh || sp.dylmp.size() != nx * lh ||
            sp.sin_theta.size() != nx || sp.cos_theta.size() != nx || sp.l2 > sp.lh ||
            sp.ae.size() != static_cast<std::size_t>(sp.radial) * m ||
            sp.ps.size() != static_cast<std::size_t>(sp.radial) * (sp.lmax_rho + 1) * m ||
            sp.coef.size() != static_cast<std::size_t>(sp.pairs()) * sp.l2 ||
            sp.pair.size() != static_cast<std::size_t>(sp.pairs()) || sp.ae_core.size() != m ||
            sp.ps_core.size() != m || sp.l2 != (sp.lmax_rho + 1) * (sp.lmax_rho + 1))
            throw std::invalid_argument("the one-center data of a PAW species");
        for (std::size_t a = 0; a < atom_species_.size(); ++a)
            if (atom_species_[a] == static_cast<int>(t) && widths[a] != sp.functions)
                throw std::invalid_argument("the projector functions of a PAW atom");
        auto& d = device_[t];
        std::vector<double> sqr(m), wwylm(nx * lh), angular(nx * lh);
        for (std::size_t k = 0; k < m; ++k) sqr[k] = std::sqrt(sp.r[k]);
        for (std::size_t x = 0; x < nx; ++x)
            for (std::size_t l = 0; l < lh; ++l) {
                wwylm[x * lh + l] = sp.ww[x] * sp.ylm[x * lh + l];
                angular[x * lh + l] = sp.dylmt[x * lh + l] * sp.sin_theta[x] + 2.0 * sp.ylm[x * lh + l] * sp.cos_theta[x];
            }
        std::vector<int> ell(static_cast<std::size_t>(sp.l2));
        for (int lm = 0; lm < sp.l2; ++lm) ell[static_cast<std::size_t>(lm)] = static_cast<int>(std::sqrt(lm + 0.5));
        d.r.assign(sp.r.data(), m);
        d.sqr.assign(sqr.data(), m);
        d.whole.assign(sp.whole.data(), m);
        d.inner.assign(sp.inner.data(), m);
        d.ww.assign(sp.ww.data(), nx);
        d.sin_theta.assign(sp.sin_theta.data(), nx);
        d.ylm.assign(sp.ylm.data(), nx * lh);
        d.wwylm.assign(wwylm.data(), nx * lh);
        d.dylmt.assign(sp.dylmt.data(), nx * lh);
        d.dylmp.assign(sp.dylmp.data(), nx * lh);
        d.angular.assign(angular.data(), nx * lh);
        d.ae.assign(sp.ae.data(), sp.ae.size());
        d.ps.assign(sp.ps.data(), sp.ps.size());
        d.coef.assign(sp.coef.data(), sp.coef.size());
        d.pair.assign(sp.pair.data(), sp.pair.size());
        d.ell.assign(ell.data(), ell.size());
        d.ae_core.assign(sp.ae_core.data(), m);
        d.ps_core.assign(sp.ps_core.data(), m);
    }
}

OneCenter::~OneCenter() = default;

void OneCenter::gemm(bool trans_b, int m, int n, int k, const double* a, int lda, long long stride_a,
                     const double* b, int ldb, double beta, double* c, int ldc, long long stride_c, int batch) {
    const double one = 1.0;
    blas_.gemm_batched(trans_b ? Op::T : Op::N, m, n, k, a, lda, stride_a, b, ldb, beta, c, ldc, stride_c, batch);
}

double OneCenter::compute(const std::vector<std::vector<double>>& becsum, std::vector<std::vector<double>>& d,
                          OneCenterEnergies* parts) {
    if (becsum.size() != static_cast<std::size_t>(spins_)) throw std::invalid_argument("becsum of each spin");
    for (const auto& b : becsum)
        if (b.size() != offsets_.back()) throw std::invalid_argument("becsum of the pairs of all atoms");
    d.assign(static_cast<std::size_t>(spins_), std::vector<double>(offsets_.back(), 0.0));
    OneCenterEnergies energies;
    for (std::size_t t = 0; t < species_.size(); ++t) {
        const auto& sp = species_[t];
        if (sp.mesh == 0) continue;
        std::vector<int> atoms;
        for (std::size_t a = 0; a < atom_species_.size(); ++a)
            if (atom_species_[a] == static_cast<int>(t)) atoms.push_back(static_cast<int>(a));
        const std::size_t m = static_cast<std::size_t>(sp.mesh), nx = static_cast<std::size_t>(sp.nx),
                          s = static_cast<std::size_t>(spins_);
        const std::size_t per_atom = m * (static_cast<std::size_t>(sp.l2) * (3 * s + 2) + nx * (8 * s + 11) +
                                          3 * s * static_cast<std::size_t>(sp.lh));
        const std::size_t size = std::max<std::size_t>(1, kBudget / per_atom);
        for (std::size_t first = 0; first < atoms.size(); first += size) {
            const std::vector<int> part(atoms.begin() + static_cast<std::ptrdiff_t>(first),
                                        atoms.begin() + static_cast<std::ptrdiff_t>(std::min(atoms.size(), first + size)));
            chunk(static_cast<int>(t), part, becsum, d, energies);
        }
    }
    if (parts != nullptr) *parts = energies;
    return energies.total();
}

void OneCenter::chunk(int species, const std::vector<int>& atoms, const std::vector<std::vector<double>>& becsum,
                      std::vector<std::vector<double>>& d, OneCenterEnergies& energies) {
    const auto& sp = species_[static_cast<std::size_t>(species)];
    const auto& dev = device_[static_cast<std::size_t>(species)];
    const int S = spins_, P = sp.pairs(), Q = sp.radial, l2 = sp.l2, lh = sp.lh, nx = sp.nx, M = sp.mesh;
    const std::size_t C = atoms.size(), s = static_cast<std::size_t>(S), m = static_cast<std::size_t>(M),
                      cs = C * s;
    const int batch = static_cast<int>(cs);
    // the occupations of the chunk, (a, s, p)
    std::vector<double> bec(cs * static_cast<std::size_t>(P));
    for (std::size_t a = 0; a < C; ++a)
        for (std::size_t j = 0; j < s; ++j)
            std::copy_n(becsum[j].begin() + static_cast<std::ptrdiff_t>(offsets_[static_cast<std::size_t>(atoms[a])]), P,
                        bec.begin() + static_cast<std::ptrdiff_t>((a * s + j) * static_cast<std::size_t>(P)));
    bec_.assign(bec.data(), bec.size());
    const std::size_t lm_size = cs * static_cast<std::size_t>(l2) * m, rad_size = cs * static_cast<std::size_t>(nx) * m;
    const std::size_t points = C * static_cast<std::size_t>(nx) * m;
    cq_.reserve(cs * static_cast<std::size_t>(l2 * Q));
    rho_lm_.reserve(lm_size);
    vh_.reserve(C * static_cast<std::size_t>(l2) * m);
    scratch_.reserve(C * static_cast<std::size_t>(l2) * m);
    eh_.reserve(C * static_cast<std::size_t>(l2));
    n_rad_.reserve(rad_size);
    rho_xc_.reserve(2 * points);
    sigma_.reserve(3 * points);
    f_.reserve(points);
    v_.reserve(2 * points);
    vs_.reserve(3 * points);
    exc_.reserve(C * static_cast<std::size_t>(nx));
    v_rad_.reserve(rad_size);
    v_lm_.reserve(lm_size);
    integral_.reserve(cs * static_cast<std::size_t>(l2 * Q));
    d_.reserve(cs * static_cast<std::size_t>(P));
    if (gga_) {
        gr_.reserve(rad_size);
        gp_.reserve(rad_size);
        gt_.reserve(rad_size);
        h_.reserve(3 * rad_size);
        hlm_.reserve(3 * cs * static_cast<std::size_t>(lh) * m);
        div_.reserve(rad_size);
        div_lm_.reserve(lm_size);
    }
    check_cuda(cudaMemset(d_.data(), 0, cs * static_cast<std::size_t>(P) * sizeof(double)), "cudaMemset");
    std::vector<double> eh(C * static_cast<std::size_t>(l2)), exc(C * static_cast<std::size_t>(nx));
    const long long lm_stride = static_cast<long long>(l2) * M, rad_stride = static_cast<long long>(nx) * M;
    for (int kind = 0; kind < 2; ++kind) {
        const bool ae = kind == 0;
        const double* f = ae ? dev.ae.data() : dev.ps.data();
        const int lcount = ae ? 1 : sp.lmax_rho + 1;
        // the densities n_LM of each atom and spin
        std::size_t work = cs * static_cast<std::size_t>(l2 * Q);
        coef_kernel<<<blocks(work), kThreads>>>(work, P, l2, Q, bec_.data(), dev.coef.data(), dev.pair.data(),
                                                cq_.data());
        launched("the coefficients of the one-center densities");
        density_kernel<<<blocks(lm_size), kThreads>>>(lm_size, M, l2, Q, lcount, dev.ell.data(), cq_.data(), f,
                                                     rho_lm_.data());
        launched("the one-center densities");
        // Hartree
        const int count = static_cast<int>(C) * l2;
        hartree_kernel<<<blocks(static_cast<std::size_t>(count)), kThreads>>>(
            count, S, l2, M, dev.ell.data(), rho_lm_.data(), dev.r.data(), dev.sqr.data(), dev.whole.data(), sp.dx,
            vh_.data(), scratch_.data(), eh_.data());
        launched("the one-center Hartree potential");
        // the densities on the directions, with the core
        gemm(false, M, nx, l2, rho_lm_.data(), M, lm_stride, dev.ylm.data(), lh, 0.0, n_rad_.data(), M, rad_stride,
             batch);
        finish_density_kernel<<<blocks(rad_size), kThreads>>>(rad_size, M, S, dev.r.data(),
                                                             ae ? dev.ae_core.data() : dev.ps_core.data(),
                                                             n_rad_.data());
        launched("the one-center densities on the directions");
        Gradient g{gr_.data(), gp_.data(), gt_.data()};
        if (gga_) {
            radial_kernel<<<blocks(rad_size), kThreads>>>(rad_size, M, dev.r.data(), n_rad_.data(), gr_.data());
            launched("the radial gradients");
            gemm(false, M, nx, l2, rho_lm_.data(), M, lm_stride, dev.dylmp.data(), lh, 0.0, gp_.data(), M, rad_stride,
                 batch);
            gemm(false, M, nx, l2, rho_lm_.data(), M, lm_stride, dev.dylmt.data(), lh, 0.0, gt_.data(), M, rad_stride,
                 batch);
        }
        assemble_kernel<<<blocks(points), kThreads>>>(points, M, nx, S, gga_, dev.r.data(), n_rad_.data(), g,
                                                     rho_xc_.data(), sigma_.data());
        launched("the one-center points of XC");
        evaluate_xc(functional_, points, rho_xc_.data(), sigma_.data(), threshold_, f_.data(), v_.data(), vs_.data(),
                    S == 2, S == 2);
        const int rows = static_cast<int>(C) * nx;
        energy_kernel<<<blocks(static_cast<std::size_t>(rows)), kThreads>>>(rows, M, nx, dev.ww.data(),
                                                                           dev.whole.data(), dev.r.data(), f_.data(),
                                                                           exc_.data());
        launched("the one-center XC energy");
        // v_LM of XC
        scatter_kernel<<<blocks(rad_size), kThreads>>>(rad_size, M, nx, S, v_.data(), v_rad_.data());
        launched("the one-center XC potential");
        gemm(true, M, l2, nx, v_rad_.data(), M, rad_stride, dev.wwylm.data(), lh, 0.0, v_lm_.data(), M, lm_stride,
             batch);
        if (gga_) {
            // less the divergence of the flux, expanded to lh
            flux_kernel<<<blocks(rad_size), kThreads>>>(rad_size, M, nx, S, dev.r.data(), dev.sin_theta.data(),
                                                       vs_.data(), g, h_.data());
            launched("the one-center flux of GGA");
            const long long hl_stride = static_cast<long long>(lh) * M;
            gemm(true, M, lh, nx, h_.data(), M, rad_stride, dev.wwylm.data(), lh, 0.0, hlm_.data(), M, hl_stride,
                 3 * batch);
            gemm(false, M, nx, lh, hlm_.data() + hl_stride, M, 3 * hl_stride, dev.dylmp.data(), lh, 0.0, div_.data(),
                 M, rad_stride, batch);
            gemm(false, M, nx, lh, hlm_.data() + 2 * hl_stride, M, 3 * hl_stride, dev.angular.data(), lh, 1.0,
                 div_.data(), M, rad_stride, batch);
            gemm(true, M, l2, nx, div_.data(), M, rad_stride, dev.wwylm.data(), lh, 0.0, div_lm_.data(), M, lm_stride,
                 batch);
            divergence_kernel<<<blocks(lm_size), kThreads>>>(lm_size, M, l2, lh, dev.r.data(), hlm_.data(),
                                                            div_lm_.data(), v_lm_.data());
            launched("the one-center divergence of GGA");
        }
        // D of this kind
        work = cs * static_cast<std::size_t>(l2 * Q);
        integral_kernel<<<blocks(work), kThreads>>>(work, M, S, l2, Q, lcount, dev.ell.data(), v_lm_.data(),
                                                   vh_.data(), f, dev.inner.data(), integral_.data());
        launched("the one-center integrals");
        work = cs * static_cast<std::size_t>(P);
        strengths_kernel<<<blocks(work), kThreads>>>(work, P, l2, Q, ae ? 1.0 : -1.0, dev.coef.data(),
                                                    dev.pair.data(), integral_.data(), d_.data());
        launched("the one-center strengths");
        eh_.download(eh.data(), eh.size());
        exc_.download(exc.data(), exc.size());
        double e_h = 0.0, e_xc = 0.0;
        for (const double x : eh) e_h += x;
        for (const double x : exc) e_xc += x;
        (ae ? energies.hartree_ae : energies.hartree_ps) += e_h;
        (ae ? energies.xc_ae : energies.xc_ps) += e_xc;
    }
    std::vector<double> out(cs * static_cast<std::size_t>(P));
    d_.download(out.data(), out.size());
    for (std::size_t a = 0; a < C; ++a)
        for (std::size_t j = 0; j < s; ++j)
            std::copy_n(out.begin() + static_cast<std::ptrdiff_t>((a * s + j) * static_cast<std::size_t>(P)), P,
                        d[j].begin() + static_cast<std::ptrdiff_t>(offsets_[static_cast<std::size_t>(atoms[a])]));
}

}  // namespace mdapy::dft
