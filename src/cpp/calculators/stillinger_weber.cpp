// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "calculators/stillinger_weber.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "calculators/many_body.h"
#include "neighbor/frame.h"

namespace mdapy::calculators {

namespace {

using P = StillingerWeberParameters;
constexpr double kThird = 1.0 / 3.0;

struct Near {
    std::int64_t pair, atom;
    int species;
    double r[3];  // x_j - x_i
    double rsq;
    // what the three-body terms of the centre take of this neighbor alone
    double length, rinvsq, gsrainvsq, expgsrainv;
};

// The parts of the three-body terms of a neighbor of a centre of species si
// that depend on it alone, by the entry i j j.
void prepare(const StillingerWeberPotential& potential, int si, Near& q) {
    const P& pij = potential.of(si, q.species, q.species);
    q.length = std::sqrt(q.rsq);
    q.rinvsq = 1.0 / q.rsq;
    const double rainv = 1.0 / (q.length - pij.cut);
    const double gsrainv = pij.sigma_gamma * rainv;
    q.gsrainvsq = gsrainv * rainv / q.length;
    q.expgsrainv = std::exp(gsrainv);
}

// The two-body term of species si and sj at r^2 by the entries i j j and
// j i i, which LAMMPS takes one of by the numbering of the atoms, here their
// mean, one when they agree: its energy and fpair; false where neither
// reaches.
bool two_body(const StillingerWeberPotential& potential, int si, int sj, double rsq,
              double& energy, double& fpair) {
    const P& pij = potential.of(si, sj, sj);
    const P& pji = potential.of(sj, si, si);
    const bool same = pij.cutsq == pji.cutsq && pij.cut == pji.cut && pij.sigma == pji.sigma &&
                      pij.p == pji.p && pij.q == pji.q && pij.c1 == pji.c1 && pij.c2 == pji.c2 &&
                      pij.c3 == pji.c3 && pij.c4 == pji.c4 && pij.c5 == pji.c5 &&
                      pij.c6 == pji.c6;
    energy = fpair = 0.0;
    bool any = false;
    for (const P* e : {&pij, &pji}) {
        const double weight = same ? 1.0 : 0.5;
        if (rsq < e->cutsq) {
            any = true;
            const double r = std::sqrt(rsq), rinvsq = 1.0 / rsq;
            const double rp = std::pow(r, -e->p), rq = std::pow(r, -e->q);
            const double rainv = 1.0 / (r - e->cut);
            const double rainvsq = rainv * rainv * r;
            const double expsrainv = std::exp(e->sigma * rainv);
            fpair += weight * ((e->c1 * rp - e->c2 * rq + (e->c3 * rp - e->c4 * rq) * rainvsq) *
                               expsrainv * rinvsq);
            energy += weight * ((e->c5 * rp - e->c6 * rq) * expsrainv);
        }
        if (same) {
            break;
        }
    }
    return any;
}

// The three-body terms of centre i of each pair of its neighbors near, a
// third to each atom, or with heat whole with the centre, into sink.
template <class Sink>
void three_body(const StillingerWeberPotential& potential, std::int64_t i, int si,
                const std::vector<Near>& near, bool heat, Sink& sink) {
    for (std::size_t a = 0; a + 1 < near.size(); ++a) {
        const Near& j = near[a];
        const double r1 = j.length, rinvsq1 = j.rinvsq;
        const double gsrainvsq1 = j.gsrainvsq, expgsrainv1 = j.expgsrainv;
        for (std::size_t b = a + 1; b < near.size(); ++b) {
            const Near& k = near[b];
            const double r2 = k.length, rinvsq2 = k.rinvsq;
            const double gsrainvsq2 = k.gsrainvsq, expgsrainv2 = k.expgsrainv;
            const double rinv12 = 1.0 / (r1 * r2);
            const double cs = (j.r[0] * k.r[0] + j.r[1] * k.r[1] + j.r[2] * k.r[2]) * rinv12;
            const double facexp = expgsrainv1 * expgsrainv2;
            // the strength and the angle of the entries i j k and i k j:
            // LAMMPS takes one by the order of the neighbors, here their
            // mean, the same when they agree
            const P& pjk = potential.of(si, j.species, k.species);
            const P& pkj = potential.of(si, k.species, j.species);
            const bool same = pjk.lambda_epsilon == pkj.lambda_epsilon &&
                              pjk.lambda_epsilon2 == pkj.lambda_epsilon2 &&
                              pjk.costheta == pkj.costheta;
            double facrad = 0.0, facang = 0.0;
            for (const P* pijk : {&pjk, &pkj}) {
                const double weight = same ? 1.0 : 0.5;
                const double delcs = cs - pijk->costheta;
                facrad += weight * (pijk->lambda_epsilon * facexp * delcs * delcs);
                facang += weight * (pijk->lambda_epsilon2 * facexp * delcs);
                if (same) {
                    break;
                }
            }
            const double frad1 = facrad * gsrainvsq1, frad2 = facrad * gsrainvsq2;
            const double facang12 = rinv12 * facang;
            const double csfacang = cs * facang;
            const double csfac1 = rinvsq1 * csfacang, csfac2 = rinvsq2 * csfacang;
            double fj[3], fk[3];
            for (int x = 0; x < 3; ++x) {
                fj[x] = j.r[x] * (frad1 + csfac1) - k.r[x] * facang12;
                fk[x] = k.r[x] * (frad2 + csfac2) - j.r[x] * facang12;
            }
            if (heat) {  // the term whole with its centre
                sink.own_energy(i, facrad);
            } else {
                sink.own_energy(i, kThird * facrad);
                sink.pair_energy(j.pair, j.atom, kThird * facrad);
                sink.pair_energy(k.pair, k.atom, kThird * facrad);
            }
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

// The pair p of atom i as a neighbor, with its vector and r^2.
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

}  // namespace

StillingerWeberPotential make_stillinger_weber(int elements, const double* rows) {
    if (elements < 1) {
        throw std::invalid_argument("a Stillinger-Weber potential needs at least one element");
    }
    StillingerWeberPotential potential;
    potential.elements = elements;
    const auto count = static_cast<std::size_t>(elements) * static_cast<std::size_t>(elements) *
                       static_cast<std::size_t>(elements);
    potential.parameters.resize(count);
    for (std::size_t t = 0; t < count; ++t) {
        const double* row = rows + 11 * t;
        P& p = potential.parameters[t];
        p.epsilon = row[0];
        p.sigma = row[1];
        p.a = row[2];
        p.lambda = row[3];
        p.gamma = row[4];
        p.costheta = row[5];
        p.big_a = row[6];
        p.big_b = row[7];
        p.p = row[8];
        p.q = row[9];
        p.tol = row[10];
        p.cut = p.sigma * p.a;
        // a shorter radius, where the energy falls below tol
        double radius = p.cut;
        if (p.tol > 0.0) {
            p.tol = std::min(p.tol, 0.01);
            radius += (p.gamma < 1.0 ? p.gamma * p.sigma : p.sigma) / std::log(p.tol);
        }
        p.cutsq = radius * radius;
        p.sigma_gamma = p.sigma * p.gamma;
        p.lambda_epsilon = p.lambda * p.epsilon;
        p.lambda_epsilon2 = 2.0 * p.lambda * p.epsilon;
        p.c1 = p.big_a * p.epsilon * p.p * p.big_b * std::pow(p.sigma, p.p);
        p.c2 = p.big_a * p.epsilon * p.q * std::pow(p.sigma, p.q);
        p.c3 = p.big_a * p.epsilon * p.big_b * std::pow(p.sigma, p.p + 1.0);
        p.c4 = p.big_a * p.epsilon * std::pow(p.sigma, p.q + 1.0);
        p.c5 = p.big_a * p.epsilon * p.big_b * std::pow(p.sigma, p.p);
        p.c6 = p.big_a * p.epsilon * std::pow(p.sigma, p.q);
        potential.cutoff = std::max(potential.cutoff, std::sqrt(p.cutsq));
    }
    return potential;
}

void compute_stillinger_weber(const StillingerWeberPotential& potential, std::int64_t n,
                              const std::int32_t* species, const double* positions,
                              const double* matrix, const std::int64_t* offsets,
                              const index_t* neighbors, const std::int8_t* shifts,
                              double* energies, double* forces, double* virials,
                              const Control& control, bool heat, std::int64_t centers,
                              const std::int64_t* reverse) {
    // the terms of the first centers atoms alone, all atoms gathering them
    const std::int64_t c = centers < 0 ? n : centers;
    Shares shares(n, offsets[n]);
    parallel_for(
        c, 0,
        [&](std::int64_t begin, std::int64_t end) {
            std::vector<Near> near;
            ToShares sink{shares};
            for (std::int64_t i = begin; i < end; ++i) {
                const int si = species[i];
                // the neighbors within the radius of their pair, each pair half from each atom
                near.clear();
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    Near q = near_of(i, p, species, positions, matrix, neighbors, shifts);
                    if (q.rsq < potential.of(si, q.species, q.species).cutsq) {
                        prepare(potential, si, q);
                        near.push_back(q);
                    }
                    double energy, fpair;
                    if (!two_body(potential, si, q.species, q.rsq, energy, fpair)) {
                        continue;
                    }
                    sink.own_energy(i, 0.5 * energy);
                    const double f[3] = {0.5 * fpair * q.r[0], 0.5 * fpair * q.r[1],
                                         0.5 * fpair * q.r[2]};
                    sink.pair_force(i, p, q.atom, q.r, f, 0.0);
                }
                three_body(potential, i, si, near, heat, sink);
            }
        },
        stage(control, 0, 2));
    gather(shares, n, offsets, neighbors, shifts, energies, forces, virials, stage(control, 1, 2),
           heat, positions, matrix, reverse);
}

void compute_stillinger_weber_once(const StillingerWeberPotential& potential, std::int64_t n,
                                   const std::int32_t* species, const double* positions,
                                   const double* matrix, const std::int64_t* offsets,
                                   const index_t* neighbors, const std::int8_t* shifts,
                                   const std::int64_t* once_offsets, const index_t* once_neighbors,
                                   const std::int8_t* once_shifts, std::int64_t centers,
                                   double* energies, double* forces, double* virials) {
    std::fill(energies, energies + n, 0.0);
    std::fill(forces, forces + 3 * n, 0.0);
    if (virials != nullptr) std::fill(virials, virials + 9 * n, 0.0);
    ToAtoms sink{energies, forces, virials};
    const double cutmax2 = potential.cutoff * potential.cutoff;
    std::vector<Near> near;
    for (std::int64_t i = 0; i < centers; ++i) {
        const int si = species[i];
        // each pair once, half of its energy and its virial to each atom
        for (std::int64_t p = once_offsets[i]; p < once_offsets[i + 1]; ++p) {
            const Near q = near_of(i, p, species, positions, matrix, once_neighbors, once_shifts);
            double energy, fpair;
            if (!(q.rsq < cutmax2) || !two_body(potential, si, q.species, q.rsq, energy, fpair)) {
                continue;
            }
            sink.own_energy(i, 0.5 * energy);
            sink.pair_energy(p, q.atom, 0.5 * energy);
            const double f[3] = {fpair * q.r[0], fpair * q.r[1], fpair * q.r[2]};
            sink.pair_force(i, p, q.atom, q.r, f, 0.5);
        }
        near.clear();
        for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
            Near q = near_of(i, p, species, positions, matrix, neighbors, shifts);
            if (q.rsq < potential.of(si, q.species, q.species).cutsq) {
                prepare(potential, si, q);
                near.push_back(q);
            }
        }
        three_body(potential, i, si, near, false, sink);
    }
}

}  // namespace mdapy::calculators
