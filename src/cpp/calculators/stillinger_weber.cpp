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
    std::int64_t pair;
    int species;
    double r[3];  // x_j - x_i
    double rsq;
};

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
            for (std::int64_t i = begin; i < end; ++i) {
                const int si = species[i];
                double& own_energy = shares.own_energy[static_cast<std::size_t>(i)];
                double* own_virial = &shares.own_virial[9 * static_cast<std::size_t>(i)];
                // the neighbors within the radius of their pair, each pair half from each atom
                near.clear();
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    Near q;
                    q.pair = p;
                    q.species = species[neighbors[p]];
                    q.rsq = neighbor::pair_vector(
                        positions + 3 * i, positions + 3 * static_cast<std::int64_t>(neighbors[p]),
                        shifts + 3 * p, matrix, q.r);
                    const P& pij = potential.of(si, q.species, q.species);
                    const P& pji = potential.of(q.species, si, si);
                    if (q.rsq < pij.cutsq) {
                        near.push_back(q);
                    }
                    // the two-body term by the entries i j j and j i i, which
                    // LAMMPS takes one of by the numbering of the atoms, here
                    // their mean, one when they agree
                    const bool same = pij.cutsq == pji.cutsq && pij.cut == pji.cut &&
                                      pij.sigma == pji.sigma && pij.p == pji.p &&
                                      pij.q == pji.q && pij.c1 == pji.c1 && pij.c2 == pji.c2 &&
                                      pij.c3 == pji.c3 && pij.c4 == pji.c4 &&
                                      pij.c5 == pji.c5 && pij.c6 == pji.c6;
                    double energy = 0.0, fpair = 0.0;
                    bool any = false;
                    for (const P* e : {&pij, &pji}) {
                        const double weight = same ? 1.0 : 0.5;
                        if (q.rsq < e->cutsq) {
                            any = true;
                            const double r = std::sqrt(q.rsq), rinvsq = 1.0 / q.rsq;
                            const double rp = std::pow(r, -e->p), rq = std::pow(r, -e->q);
                            const double rainv = 1.0 / (r - e->cut);
                            const double rainvsq = rainv * rainv * r;
                            const double expsrainv = std::exp(e->sigma * rainv);
                            fpair += weight * ((e->c1 * rp - e->c2 * rq +
                                                (e->c3 * rp - e->c4 * rq) * rainvsq) *
                                               expsrainv * rinvsq);
                            energy += weight * ((e->c5 * rp - e->c6 * rq) * expsrainv);
                        }
                        if (same) {
                            break;
                        }
                    }
                    if (!any) {
                        continue;
                    }
                    own_energy += 0.5 * energy;
                    const double f[3] = {0.5 * fpair * q.r[0], 0.5 * fpair * q.r[1],
                                         0.5 * fpair * q.r[2]};
                    shares.pair_force(i, p, q.r, f, 0.0);
                }
                // the three-body terms of each pair of neighbors, a third to each atom
                for (std::size_t a = 0; a + 1 < near.size(); ++a) {
                    const Near& j = near[a];
                    const P& pij = potential.of(si, j.species, j.species);
                    const double r1 = std::sqrt(j.rsq), rinvsq1 = 1.0 / j.rsq;
                    const double rainv1 = 1.0 / (r1 - pij.cut);
                    const double gsrainv1 = pij.sigma_gamma * rainv1;
                    const double gsrainvsq1 = gsrainv1 * rainv1 / r1;
                    const double expgsrainv1 = std::exp(gsrainv1);
                    double* tj = &shares.derivative[3 * static_cast<std::size_t>(j.pair)];
                    double* wj = &shares.virial[9 * static_cast<std::size_t>(j.pair)];
                    for (std::size_t b = a + 1; b < near.size(); ++b) {
                        const Near& k = near[b];
                        const P& pik = potential.of(si, k.species, k.species);
                        const double r2 = std::sqrt(k.rsq), rinvsq2 = 1.0 / k.rsq;
                        const double rainv2 = 1.0 / (r2 - pik.cut);
                        const double gsrainv2 = pik.sigma_gamma * rainv2;
                        const double gsrainvsq2 = gsrainv2 * rainv2 / r2;
                        const double expgsrainv2 = std::exp(gsrainv2);
                        const double rinv12 = 1.0 / (r1 * r2);
                        const double cs =
                            (j.r[0] * k.r[0] + j.r[1] * k.r[1] + j.r[2] * k.r[2]) * rinv12;
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
                            own_energy += facrad;
                        } else {
                            own_energy += kThird * facrad;
                            shares.energy[static_cast<std::size_t>(j.pair)] += kThird * facrad;
                            shares.energy[static_cast<std::size_t>(k.pair)] += kThird * facrad;
                        }
                        double* tk = &shares.derivative[3 * static_cast<std::size_t>(k.pair)];
                        double* wk = &shares.virial[9 * static_cast<std::size_t>(k.pair)];
                        for (int x = 0; x < 3; ++x) {
                            tj[x] -= fj[x];
                            tk[x] -= fk[x];
                            for (int y = 0; y < 3; ++y) {
                                const double w = kThird * (j.r[x] * fj[y] + k.r[x] * fk[y]);
                                own_virial[3 * x + y] += w;
                                wj[3 * x + y] += w;
                                wk[3 * x + y] += w;
                            }
                        }
                    }
                }
            }
        },
        stage(control, 0, 2));
    gather(shares, n, offsets, neighbors, shifts, energies, forces, virials, stage(control, 1, 2),
           heat, positions, matrix, reverse);
}

}  // namespace mdapy::calculators
