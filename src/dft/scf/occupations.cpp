// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/scf/occupations.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mdapy::dft {

namespace {

constexpr double kSqrtPi = 1.77245385090551602730;
constexpr double kSqrt2 = 1.41421356237309504880;

}  // namespace

Smearing smearing_of(const std::string& name) {
    if (name == "fixed") return Smearing::Fixed;
    if (name == "gaussian") return Smearing::Gaussian;
    if (name == "fermi-dirac") return Smearing::FermiDirac;
    if (name == "methfessel-paxton") return Smearing::MethfesselPaxton;
    if (name == "marzari-vanderbilt") return Smearing::MarzariVanderbilt;
    throw std::invalid_argument("no smearing named " + name +
                                "; fixed, gaussian, fermi-dirac, methfessel-paxton or marzari-vanderbilt");
}

double smeared_occupation(Smearing smearing, double u) {
    switch (smearing) {
        case Smearing::Gaussian: return 0.5 * std::erfc(u);
        case Smearing::FermiDirac:
            return u > 0.0 ? std::exp(-u) / (1.0 + std::exp(-u)) : 1.0 / (1.0 + std::exp(u));
        case Smearing::MethfesselPaxton: return 0.5 * std::erfc(u) - u * std::exp(-u * u) / (2.0 * kSqrtPi);
        case Smearing::MarzariVanderbilt: {
            const double s = u + 1.0 / kSqrt2;
            return 0.5 * std::erfc(s) + std::exp(-s * s) / (kSqrt2 * kSqrtPi);
        }
        case Smearing::Fixed: break;
    }
    return u < 0.0 ? 1.0 : 0.0;
}

double smeared_entropy(Smearing smearing, double u) {
    switch (smearing) {
        case Smearing::Gaussian: return -std::exp(-u * u) / (2.0 * kSqrtPi);
        case Smearing::FermiDirac: {
            const double f = smeared_occupation(smearing, u);
            double e = 0.0;
            if (f > 0.0) e += f * std::log(f);
            if (f < 1.0) e += (1.0 - f) * std::log1p(-f);
            return e;
        }
        case Smearing::MethfesselPaxton: return std::exp(-u * u) * (2.0 * u * u - 1.0) / (4.0 * kSqrtPi);
        case Smearing::MarzariVanderbilt: {
            const double s = u + 1.0 / kSqrt2;
            return -s * std::exp(-s * s) / (kSqrt2 * kSqrtPi);
        }
        case Smearing::Fixed: break;
    }
    return 0.0;
}

Occupation occupy(const std::vector<double>& eigenvalues, const std::vector<double>& weights, int bands,
                  double electrons, double f_max, Smearing smearing, double sigma) {
    const std::size_t nk = weights.size(), nb = static_cast<std::size_t>(bands);
    if (bands < 1 || eigenvalues.size() != nk * nb)
        throw std::invalid_argument("the eigenvalues are (k-points, bands)");
    if (!(electrons > 0.0) || electrons > f_max * bands + 1e-9)
        throw std::invalid_argument("the bands cannot hold the electrons");
    Occupation out;
    out.f.assign(nk * nb, 0.0);
    if (smearing == Smearing::Fixed) {
        const double filled = electrons / f_max;
        const int occupied = static_cast<int>(std::lround(filled));
        if (std::abs(filled - occupied) > 1e-9)
            throw std::invalid_argument("fixed occupations need whole bands of electrons; use a smearing");
        out.fermi = -std::numeric_limits<double>::infinity();
        out.lowest_empty = std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < nk; ++k)
            for (std::size_t b = 0; b < nb; ++b) {
                const double e = eigenvalues[k * nb + b];
                if (static_cast<int>(b) < occupied) {
                    out.f[k * nb + b] = 1.0;
                    out.fermi = std::max(out.fermi, e);
                } else {
                    out.lowest_empty = std::min(out.lowest_empty, e);
                }
            }
        if (occupied == bands) out.lowest_empty = out.fermi;
        return out;
    }
    if (!(sigma > 0.0)) throw std::invalid_argument("a smearing needs a positive width");
    auto count = [&](double mu) {
        double n = 0.0;
        for (std::size_t k = 0; k < nk; ++k)
            for (std::size_t b = 0; b < nb; ++b)
                n += weights[k] * f_max * smeared_occupation(smearing, (eigenvalues[k * nb + b] - mu) / sigma);
        return n;
    };
    const auto [lo_it, hi_it] = std::minmax_element(eigenvalues.begin(), eigenvalues.end());
    double lo = *lo_it - 20.0 * sigma, hi = *hi_it + 20.0 * sigma;
    if (!(count(lo) < electrons) || !(count(hi) > electrons))
        throw std::runtime_error("no Fermi level holds the electrons; more bands are needed");
    for (int i = 0; i < 200 && hi - lo > 1e-15 * std::max(1.0, std::abs(lo)); ++i) {
        const double mid = 0.5 * (lo + hi);
        (count(mid) < electrons ? lo : hi) = mid;
    }
    const double mu = 0.5 * (lo + hi);
    out.fermi = out.lowest_empty = mu;
    for (std::size_t k = 0; k < nk; ++k)
        for (std::size_t b = 0; b < nb; ++b) {
            const double u = (eigenvalues[k * nb + b] - mu) / sigma;
            out.f[k * nb + b] = smeared_occupation(smearing, u);
            out.entropy += weights[k] * f_max * sigma * smeared_entropy(smearing, u);
        }
    return out;
}

}  // namespace mdapy::dft
