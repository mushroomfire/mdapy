// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "dft/scf/symmetry.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mdapy::dft {

namespace {

constexpr int kThreads = 256;
constexpr double kTwoPi = 6.28318530717958647692528676655900577;

unsigned blocks(std::size_t work) { return static_cast<unsigned>((work + kThreads - 1) / kThreads); }

struct Box {
    int low[3], size[3];
};

__device__ int lookup(const int* table, Box box, const int m[3]) {
    int at = 0;
    for (int d = 0; d < 3; ++d) {
        const int i = m[d] - box.low[d];
        if (i < 0 || i >= box.size[d]) return -1;
        at = at * box.size[d] + i;
    }
    return table[at];
}

// out[c n + g'] = 1/N sum_ops in[c n + index(R^-T m')] exp(2 pi i (R^-T m') . t)
__global__ void symmetrize_kernel(std::size_t n, int channels, int ops, const int* miller, const int* matrices,
                                  const double* shifts, const int* table, Box box, const Complex* in, Complex* out) {
    const std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const int* mp = miller + 3 * i;
    for (int c = 0; c < channels; ++c) {
        double re = 0.0, im = 0.0;
        for (int k = 0; k < ops; ++k) {
            const int* w = matrices + 9 * k;
            int m[3];
            for (int d = 0; d < 3; ++d) m[d] = w[3 * d] * mp[0] + w[3 * d + 1] * mp[1] + w[3 * d + 2] * mp[2];
            const int j = lookup(table, box, m);
            if (j < 0) continue;  // not met: the sphere is checked to hold every image
            const double phase = kTwoPi * (m[0] * shifts[3 * k] + m[1] * shifts[3 * k + 1] + m[2] * shifts[3 * k + 2]);
            const double cs = cos(phase), sn = sin(phase);
            const Complex v = in[static_cast<std::size_t>(c) * n + static_cast<std::size_t>(j)];
            re += v.x * cs - v.y * sn;
            im += v.x * sn + v.y * cs;
        }
        out[static_cast<std::size_t>(c) * n + i] = make_cuDoubleComplex(re / ops, im / ops);
    }
}

}  // namespace

Symmetrizer::Symmetrizer(const SymmetryInput& input, const std::vector<Int3>& miller)
    : operations_(static_cast<int>(std::max<std::size_t>(1, input.inverse_transposes.size()))),
      n_(miller.size()),
      input_(input) {
    if (!input.any()) {
        operations_ = 1;
        return;
    }
    const std::size_t ops = input.inverse_transposes.size();
    if (input.translations.size() != ops || input.atoms.size() != ops || input.harmonics.size() != ops)
        throw std::invalid_argument("each operation of the symmetry with its translation, atoms and harmonics");
    std::array<int, 3> high{};
    for (int d = 0; d < 3; ++d) low_[d] = high[d] = 0;
    for (const Int3& m : miller)
        for (int d = 0; d < 3; ++d) {
            low_[d] = std::min(low_[d], m[static_cast<std::size_t>(d)]);
            high[d] = std::max(high[d], m[static_cast<std::size_t>(d)]);
        }
    for (int d = 0; d < 3; ++d) size_[d] = high[d] - low_[d] + 1;
    const std::size_t cells = static_cast<std::size_t>(size_[0]) * size_[1] * size_[2];
    std::vector<int> table(cells, -1), flat(3 * n_);
    auto at = [&](const int m[3]) -> long long {
        long long k = 0;
        for (int d = 0; d < 3; ++d) {
            const int i = m[d] - low_[d];
            if (i < 0 || i >= size_[d]) return -1;
            k = k * size_[d] + i;
        }
        return k;
    };
    for (std::size_t g = 0; g < n_; ++g) {
        int m[3] = {miller[g][0], miller[g][1], miller[g][2]};
        table[static_cast<std::size_t>(at(m))] = static_cast<int>(g);
        for (int d = 0; d < 3; ++d) flat[3 * g + static_cast<std::size_t>(d)] = m[d];
    }
    // the sphere holds the image of every Miller index under every operation
    std::vector<int> matrices;
    std::vector<double> shifts;
    for (std::size_t k = 0; k < ops; ++k) {
        const auto& w = input.inverse_transposes[k];
        matrices.insert(matrices.end(), w.begin(), w.end());
        for (int d = 0; d < 3; ++d) shifts.push_back(input.translations[k][static_cast<std::size_t>(d)]);
        for (std::size_t g = 0; g < n_; ++g) {
            int m[3];
            for (int d = 0; d < 3; ++d)
                m[d] = w[static_cast<std::size_t>(3 * d)] * miller[g][0] +
                       w[static_cast<std::size_t>(3 * d + 1)] * miller[g][1] +
                       w[static_cast<std::size_t>(3 * d + 2)] * miller[g][2];
            const long long j = at(m);
            if (j < 0 || table[static_cast<std::size_t>(j)] < 0)
                throw std::invalid_argument("the sphere of the density is not closed under the symmetry of the crystal");
        }
    }
    table_.upload(table.data(), table.size());
    miller_.upload(flat.data(), flat.size());
    matrices_.upload(matrices.data(), matrices.size());
    shifts_.upload(shifts.data(), shifts.size());
}

void Symmetrizer::density(Complex* components, int channels) {
    if (!active() || n_ == 0) return;
    const std::size_t total = n_ * static_cast<std::size_t>(channels);
    work_.reserve(total);
    check_cuda(cudaMemcpy(work_.data(), components, total * sizeof(Complex), cudaMemcpyDeviceToDevice), "cudaMemcpy");
    const Box box{{low_[0], low_[1], low_[2]}, {size_[0], size_[1], size_[2]}};
    symmetrize_kernel<<<blocks(n_), kThreads>>>(n_, channels, operations_, miller_.data(), matrices_.data(),
                                               shifts_.data(), table_.data(), box, work_.data(), components);
    check_cuda(cudaGetLastError(), "the symmetric density");
}

void Symmetrizer::occupations(std::vector<double>& becsum, const std::vector<std::size_t>& offsets,
                              const std::vector<std::vector<int>>& angular, bool doubled) const {
    if (!active()) return;
    const std::size_t atoms = angular.size();
    if (offsets.size() != atoms + 1 || becsum.size() != offsets.back())
        throw std::invalid_argument("the occupations of each atom");
    // the full matrices of each atom
    std::vector<std::vector<double>> full(atoms);
    std::vector<int> width(atoms, 0);
    for (std::size_t a = 0; a < atoms; ++a) {
        for (const int l : angular[a]) width[a] += 2 * l + 1;
        const int w = width[a];
        if (offsets[a + 1] - offsets[a] != static_cast<std::size_t>(w) * (w + 1) / 2)
            throw std::invalid_argument("the pairs of an atom");
        full[a].assign(static_cast<std::size_t>(w) * w, 0.0);
        std::size_t k = offsets[a];
        for (int i = 0; i < w; ++i)
            for (int j = i; j < w; ++j, ++k) {
                const double x = i == j || !doubled ? becsum[k] : 0.5 * becsum[k];
                full[a][static_cast<std::size_t>(i) * w + j] = x;
                full[a][static_cast<std::size_t>(j) * w + i] = x;
            }
    }
    // where the harmonics of each l start in the flat block of an operation
    std::vector<std::size_t> start(static_cast<std::size_t>(input_.lmax) + 2, 0);
    for (int l = 0; l <= input_.lmax; ++l)
        start[static_cast<std::size_t>(l) + 1] = start[static_cast<std::size_t>(l)] +
                                                 static_cast<std::size_t>((2 * l + 1) * (2 * l + 1));
    std::vector<std::vector<double>> sum(atoms);
    for (std::size_t a = 0; a < atoms; ++a) sum[a].assign(full[a].size(), 0.0);
    std::vector<double> m, t;
    for (std::size_t k = 0; k < static_cast<std::size_t>(operations_); ++k) {
        const auto& h = input_.harmonics[k];
        for (std::size_t i = 0; i < atoms; ++i) {
            const std::size_t j = static_cast<std::size_t>(input_.atoms[k][i]);
            const int w = width[i];
            if (width[j] != w) throw std::invalid_argument("an operation takes an atom to one of another kind");
            // M, block diagonal over the projectors of the atom
            m.assign(static_cast<std::size_t>(w) * w, 0.0);
            int first = 0;
            for (const int l : angular[i]) {
                if (l > input_.lmax) throw std::invalid_argument("projectors above the l of the symmetry");
                const int d = 2 * l + 1;
                const double* block = h.data() + start[static_cast<std::size_t>(l)];
                for (int p = 0; p < d; ++p)
                    for (int q = 0; q < d; ++q)
                        m[static_cast<std::size_t>(first + p) * w + (first + q)] = block[p * d + q];
                first += d;
            }
            // sum_J += M B_I M^T
            t.assign(static_cast<std::size_t>(w) * w, 0.0);
            for (int p = 0; p < w; ++p)
                for (int q = 0; q < w; ++q) {
                    double x = 0.0;
                    for (int r = 0; r < w; ++r)
                        x += m[static_cast<std::size_t>(p) * w + r] * full[i][static_cast<std::size_t>(r) * w + q];
                    t[static_cast<std::size_t>(p) * w + q] = x;
                }
            for (int p = 0; p < w; ++p)
                for (int q = 0; q < w; ++q) {
                    double x = 0.0;
                    for (int r = 0; r < w; ++r)
                        x += t[static_cast<std::size_t>(p) * w + r] * m[static_cast<std::size_t>(q) * w + r];
                    sum[j][static_cast<std::size_t>(p) * w + q] += x;
                }
        }
    }
    for (std::size_t a = 0; a < atoms; ++a) {
        const int w = width[a];
        std::size_t k = offsets[a];
        for (int i = 0; i < w; ++i)
            for (int j = i; j < w; ++j, ++k) {
                const double x = sum[a][static_cast<std::size_t>(i) * w + j] / operations_;
                becsum[k] = i == j || !doubled ? x : 2.0 * x;
            }
    }
}

}  // namespace mdapy::dft
