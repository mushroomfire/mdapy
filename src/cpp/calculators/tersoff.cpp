// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "calculators/tersoff.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "calculators/many_body.h"
#include "neighbor/frame.h"

namespace mdapy::calculators {

namespace {

using P = TersoffParameters;

constexpr double kHalfPi = 1.57079632679489661923;
constexpr double kQuarterPi = 0.78539816339744830962;
constexpr double kThird = 1.0 / 3.0;
// beyond this exponent the factor of the bond order is taken as 1e30 or 0
constexpr double kLargestExponent = 69.0776;

double cutoff_function(double r, const P& p) {
    if (r < p.r - p.big_d) {
        return 1.0;
    }
    if (r > p.r + p.big_d) {
        return 0.0;
    }
    return 0.5 * (1.0 - std::sin(kHalfPi * (r - p.r) / p.big_d));
}

double cutoff_slope(double r, const P& p) {
    if (r < p.r - p.big_d || r > p.r + p.big_d) {
        return 0.0;
    }
    return -(kQuarterPi / p.big_d) * std::cos(kHalfPi * (r - p.r) / p.big_d);
}

// f_A(r) = -B exp(-lambda2 r) f_c(r) and its derivative
double attraction(double r, const P& p) {
    if (r > p.r + p.big_d) {
        return 0.0;
    }
    return -p.b * std::exp(-p.lam2 * r) * cutoff_function(r, p);
}

double attraction_slope(double r, const P& p) {
    if (r > p.r + p.big_d) {
        return 0.0;
    }
    return p.b * std::exp(-p.lam2 * r) * (p.lam2 * cutoff_function(r, p) - cutoff_slope(r, p));
}

// b_ij = (1 + (beta zeta)^n)^(-1/2n), by its limits far from beta zeta = 1
double bond_order(double zeta, const P& p) {
    const double x = p.beta * zeta;
    if (x > p.c1) {
        return 1.0 / std::sqrt(x);
    }
    if (x > p.c2) {
        return (1.0 - std::pow(x, -p.n) / (2.0 * p.n)) / std::sqrt(x);
    }
    if (x < p.c4) {
        return 1.0;
    }
    if (x < p.c3) {
        return 1.0 - std::pow(x, p.n) / (2.0 * p.n);
    }
    return std::pow(1.0 + std::pow(x, p.n), -1.0 / (2.0 * p.n));
}

double bond_order_slope(double zeta, const P& p) {
    const double x = p.beta * zeta;
    if (x > p.c1) {
        return p.beta * -0.5 * std::pow(x, -1.5);
    }
    if (x > p.c2) {
        return p.beta *
               (-0.5 * std::pow(x, -1.5) * (1.0 - (1.0 + 1.0 / (2.0 * p.n)) * std::pow(x, -p.n)));
    }
    if (x < p.c4) {
        return 0.0;
    }
    if (x < p.c3) {
        return -0.5 * p.beta * std::pow(x, p.n - 1.0);
    }
    const double xn = std::pow(x, p.n);
    return -0.5 * std::pow(1.0 + xn, -1.0 - (1.0 / (2.0 * p.n))) * xn / zeta;
}

// g(theta) = gamma (1 + c^2/d^2 - c^2 / (d^2 + (h - cos theta)^2))
double angular(double cosine, const P& p) {
    const double c2 = p.c * p.c, d2 = p.d * p.d, s = p.h - cosine;
    return p.gamma * (1.0 + c2 / d2 - c2 / (d2 + s * s));
}

double angular_slope(double cosine, const P& p) {
    const double c2 = p.c * p.c, d2 = p.d * p.d, s = p.h - cosine;
    const double inverse = 1.0 / (d2 + s * s);
    return p.gamma * (-2.0 * c2 * s) * inverse * inverse;
}

// exp(lambda3^m (r_ij - r_ik)^m), m 3 or 1, and its derivative by r_ij
double exponential(double rij, double rik, const P& p) {
    const double s = p.lam3 * (rij - rik);
    const double x = p.m == 3 ? s * s * s : s;
    if (x > kLargestExponent) {
        return 1.0e30;
    }
    if (x < -kLargestExponent) {
        return 0.0;
    }
    return std::exp(x);
}

double exponential_slope(double rij, double rik, double value, const P& p) {
    if (p.m == 3) {
        const double d = rij - rik;
        return 3.0 * p.lam3 * p.lam3 * p.lam3 * d * d * value;
    }
    return p.lam3 * value;
}

// The forces on j and k of the term of k in zeta_ij, times prefactor =
// -dE/dzeta_ij: u and v the unit vectors to j and k at rij and rik.
void zeta_forces(double prefactor, const double* u, double rij, const double* v, double rik,
                 const P& p, double* fj, double* fk) {
    const double fc = cutoff_function(rik, p), dfc = cutoff_slope(rik, p);
    const double ex = exponential(rij, rik, p), dex = exponential_slope(rij, rik, ex, p);
    const double cosine = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    const double g = angular(cosine, p), dg = angular_slope(cosine, p);
    for (int a = 0; a < 3; ++a) {
        const double dcos_j = (v[a] - cosine * u[a]) / rij;
        const double dcos_k = (u[a] - cosine * v[a]) / rik;
        fj[a] = prefactor * (fc * dg * ex * dcos_j + fc * g * dex * u[a]);
        fk[a] = prefactor * (dfc * g * ex * v[a] + fc * dg * ex * dcos_k - fc * g * dex * v[a]);
    }
}

struct Near {
    std::int64_t pair, atom;
    int species;
    double r[3];  // x_j - x_i
    double rsq;
};

Near near_of(std::int64_t i, std::int64_t p, const std::int32_t* species,
             const double* positions, const double* matrix, const index_t* neighbors,
             const std::int8_t* shifts) {
    Near q;
    q.pair = p;
    q.atom = static_cast<std::int64_t>(neighbors[p]);
    q.species = species[q.atom];
    q.rsq = neighbor::pair_vector(positions + 3 * i, positions + 3 * q.atom, shifts + 3 * p,
                                  matrix, q.r);
    return q;
}

// The repulsion of species si and sj at r^2 by the entries i j j and j i i,
// which LAMMPS takes one of by the numbering of the atoms, here their mean,
// one when they agree: its energy and fpair; false where it is zero.
bool repulsion(const TersoffPotential& potential, int si, int sj, double rsq, double& energy,
               double& fpair) {
    const P& pij = potential.of(si, sj, sj);
    const P& pji = potential.of(sj, si, si);
    const bool same =
        pij.a == pji.a && pij.lam1 == pji.lam1 && pij.r == pji.r && pij.big_d == pji.big_d;
    energy = fpair = 0.0;
    for (const P* p : {&pij, &pji}) {
        const double weight = same ? 1.0 : 0.5;
        if (rsq < p->cutsq) {
            const double r = std::sqrt(rsq);
            const double fc = cutoff_function(r, *p), dfc = cutoff_slope(r, *p);
            const double ex = std::exp(-p->lam1 * r);
            fpair += weight * (-p->a * ex * (dfc - fc * p->lam1) / r);
            energy += weight * (fc * p->a * ex);
        }
        if (same) {
            break;
        }
    }
    return !(energy == 0.0 && fpair == 0.0);
}

// The attraction of each bond of centre i with its neighbors near, weakened
// by the bond order, into sink.
template <class Sink>
void bonds(const TersoffPotential& potential, std::int64_t i, int si,
           const std::vector<Near>& near, bool heat, Sink& sink) {
    for (std::size_t a = 0; a < near.size(); ++a) {
        const Near& j = near[a];
        const P& pij = potential.of(si, j.species, j.species);
        if (!(j.rsq < pij.cutsq)) {
            continue;
        }
        const double rij = std::sqrt(j.rsq), rij_inverse = 1.0 / rij;
        const double u[3] = {j.r[0] * rij_inverse, j.r[1] * rij_inverse, j.r[2] * rij_inverse};
        double zeta = 0.0;
        for (std::size_t b = 0; b < near.size(); ++b) {
            const Near& k = near[b];
            const P& pijk = potential.of(si, j.species, k.species);
            if (b == a || !(k.rsq < pijk.cutsq)) {
                continue;
            }
            const double rik = std::sqrt(k.rsq), rik_inverse = 1.0 / rik;
            const double v[3] = {k.r[0] * rik_inverse, k.r[1] * rik_inverse, k.r[2] * rik_inverse};
            const double cosine = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
            zeta += cutoff_function(rik, pijk) * angular(cosine, pijk) * exponential(rij, rik, pijk);
        }
        const double fa = attraction(rij, pij), dfa = attraction_slope(rij, pij);
        const double bij = bond_order(zeta, pij);
        const double energy = 0.5 * bij * fa;
        const double fpair = 0.5 * bij * dfa * rij_inverse;
        const double prefactor = -0.5 * fa * bond_order_slope(zeta, pij);
        // the bond at fixed zeta: half to each atom, or for the heat flux
        // whole to its centre
        if (heat) {
            sink.own_energy(i, energy);
        } else {
            sink.own_energy(i, 0.5 * energy);
            sink.pair_energy(j.pair, j.atom, 0.5 * energy);
        }
        const double f[3] = {-fpair * j.r[0], -fpair * j.r[1], -fpair * j.r[2]};
        sink.pair_force(i, j.pair, j.atom, j.r, f, 0.5);
        // zeta through each k: its virial a third to each atom
        for (std::size_t b = 0; b < near.size(); ++b) {
            const Near& k = near[b];
            const P& pijk = potential.of(si, j.species, k.species);
            if (b == a || !(k.rsq < pijk.cutsq)) {
                continue;
            }
            const double rik = std::sqrt(k.rsq), rik_inverse = 1.0 / rik;
            const double v[3] = {k.r[0] * rik_inverse, k.r[1] * rik_inverse, k.r[2] * rik_inverse};
            double fj[3], fk[3];
            zeta_forces(prefactor, u, rij, v, rik, pijk, fj, fk);
            sink.derivative(i, j.pair, j.atom, fj);
            sink.derivative(i, k.pair, k.atom, fk);
            double w[9];
            for (int x = 0; x < 3; ++x) {
                for (int y = 0; y < 3; ++y) {
                    w[3 * x + y] = kThird * (j.r[x] * fj[y] + k.r[x] * fk[y]);
                }
            }
            sink.virial3(i, j.pair, j.atom, k.pair, k.atom, w);
        }
    }
}

}  // namespace

TersoffPotential make_tersoff(int elements, const double* rows) {
    if (elements < 1) {
        throw std::invalid_argument("a Tersoff potential needs at least one element");
    }
    TersoffPotential potential;
    potential.elements = elements;
    const auto count = static_cast<std::size_t>(elements) * static_cast<std::size_t>(elements) *
                       static_cast<std::size_t>(elements);
    potential.parameters.resize(count);
    for (std::size_t t = 0; t < count; ++t) {
        const double* row = rows + 14 * t;
        P& p = potential.parameters[t];
        p.m = static_cast<int>(row[0]);
        p.gamma = row[1];
        p.lam3 = row[2];
        p.c = row[3];
        p.d = row[4];
        p.h = row[5];
        p.n = row[6];
        p.beta = row[7];
        p.lam2 = row[8];
        p.b = row[9];
        p.r = row[10];
        p.big_d = row[11];
        p.lam1 = row[12];
        p.a = row[13];
        p.cut = p.r + p.big_d;
        p.cutsq = p.cut * p.cut;
        if (p.n > 0.0) {
            p.c1 = std::pow(2.0 * p.n * 1.0e-16, -1.0 / p.n);
            p.c2 = std::pow(2.0 * p.n * 1.0e-8, -1.0 / p.n);
            p.c3 = 1.0 / p.c2;
            p.c4 = 1.0 / p.c1;
        }
        potential.cutoff = std::max(potential.cutoff, p.cut);
    }
    return potential;
}

void compute_tersoff(const TersoffPotential& potential, std::int64_t n,
                     const std::int32_t* species, const double* positions, const double* matrix,
                     const std::int64_t* offsets, const index_t* neighbors,
                     const std::int8_t* shifts, double* energies, double* forces,
                     double* virials, const Control& control, bool heat, std::int64_t centers,
                     const std::int64_t* reverse) {
    // the terms of the first centers atoms alone, all atoms gathering them
    const std::int64_t c = centers < 0 ? n : centers;
    Shares shares(n, offsets[n]);
    const double cutmax2 = potential.cutoff * potential.cutoff;
    parallel_for(
        c, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<Near> near;
            ToShares sink{shares};
            for (std::int64_t i = begin; i < end; ++i) {
                const int si = species[i];
                near.clear();
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const Near q = near_of(i, p, species, positions, matrix, neighbors, shifts);
                    if (q.rsq < cutmax2) {
                        near.push_back(q);
                    }
                }
                // the repulsion of each pair, half of it from each atom
                for (const Near& j : near) {
                    double energy, fpair;
                    if (!repulsion(potential, si, j.species, j.rsq, energy, fpair)) {
                        continue;
                    }
                    sink.own_energy(i, 0.5 * energy);
                    const double f[3] = {0.5 * fpair * j.r[0], 0.5 * fpair * j.r[1],
                                         0.5 * fpair * j.r[2]};
                    sink.pair_force(i, j.pair, j.atom, j.r, f, 0.0);
                }
                bonds(potential, i, si, near, heat, sink);
            }
        },
        stage(control, 0, 2));
    gather(shares, n, offsets, neighbors, shifts, energies, forces, virials, stage(control, 1, 2),
           heat, positions, matrix, reverse);
}

void compute_tersoff_once(const TersoffPotential& potential, std::int64_t n,
                          const std::int32_t* species, const double* positions,
                          const double* matrix, const std::int64_t* offsets,
                          const index_t* neighbors, const std::int8_t* shifts,
                          const std::int64_t* once_offsets, const index_t* once_neighbors,
                          const std::int8_t* once_shifts, std::int64_t centers, double* energies,
                          double* forces, double* virials) {
    std::fill(energies, energies + n, 0.0);
    std::fill(forces, forces + 3 * n, 0.0);
    if (virials != nullptr) std::fill(virials, virials + 9 * n, 0.0);
    const double cutmax2 = potential.cutoff * potential.cutoff;
    ToAtoms sink{energies, forces, virials};
    std::vector<Near> near;
    for (std::int64_t i = 0; i < centers; ++i) {
        const int si = species[i];
        // the repulsion of each pair once, half of its energy and virial to each atom
        for (std::int64_t p = once_offsets[i]; p < once_offsets[i + 1]; ++p) {
            const Near q = near_of(i, p, species, positions, matrix, once_neighbors, once_shifts);
            double energy, fpair;
            if (!(q.rsq < cutmax2) || !repulsion(potential, si, q.species, q.rsq, energy, fpair)) {
                continue;
            }
            sink.own_energy(i, 0.5 * energy);
            sink.pair_energy(p, q.atom, 0.5 * energy);
            const double f[3] = {fpair * q.r[0], fpair * q.r[1], fpair * q.r[2]};
            sink.pair_force(i, p, q.atom, q.r, f, 0.5);
        }
        near.clear();
        for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
            const Near q = near_of(i, p, species, positions, matrix, neighbors, shifts);
            if (q.rsq < cutmax2) {
                near.push_back(q);
            }
        }
        bonds(potential, i, si, near, false, sink);
    }
}

}  // namespace mdapy::calculators
