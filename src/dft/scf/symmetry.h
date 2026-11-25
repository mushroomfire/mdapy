// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The symmetry of a crystal in the self-consistent field: the output density
// and the occupations of the projectors made the averages of their images
// under the operations x -> R x + t of the crystal (fractional coordinates).
// On the sphere of the density,
//   rho(m') = 1/N sum_ops rho(R^-T m') exp(2 pi i (R^-T m') . t),
// and for the occupations B_J (w, w) of atom J
//   B_J = 1/N sum_ops M B_I M^T,  I the atom the operation takes to J,
// M block diagonal, the rotations of the real harmonics of each projector.

#pragma once

#include <array>
#include <vector>

#include "dft/core/device.h"
#include "dft/lattice/lattice.h"

namespace mdapy::dft {

struct SymmetryInput {
    std::vector<std::array<int, 9>> inverse_transposes;  // R^-T of each operation, row by row
    std::vector<Vec3> translations;                      // t of each
    std::vector<std::vector<int>> atoms;                 // the atom J each takes atom I to
    // the rotations M of the real harmonics of l = 0 ... lmax of each
    // operation, (2l+1, 2l+1) row by row one l after the other
    std::vector<std::vector<double>> harmonics;
    int lmax = 3;
    bool any() const noexcept { return inverse_transposes.size() > 1; }
};

class Symmetrizer {
public:
    // the operations and the Miller indices of the sphere of the density
    Symmetrizer(const SymmetryInput& input, const std::vector<Int3>& miller);
    bool active() const noexcept { return operations_ > 1; }
    // components (channels, n) of the sphere on the device, in place
    void density(Complex* components, int channels);
    // becsum of one spin, the packed pairs of each atom from offsets[i] to
    // offsets[i + 1], the angular momenta of the projectors of each atom;
    // doubled when the pairs off the diagonal hold twice their element, as
    // becsum does and the strengths D do not
    void occupations(std::vector<double>& becsum, const std::vector<std::size_t>& offsets,
                     const std::vector<std::vector<int>>& angular, bool doubled = true) const;

private:
    int operations_ = 1;
    std::size_t n_ = 0;
    SymmetryInput input_;
    DeviceArray<int> table_;       // the index of each Miller index of the box, -1 off the sphere
    DeviceArray<int> miller_;      // (n, 3)
    DeviceArray<int> matrices_;    // (ops, 9) R^-T
    DeviceArray<double> shifts_;   // (ops, 3) t
    DeviceArray<Complex> work_;
    std::array<int, 3> low_{}, size_{};
};

}  // namespace mdapy::dft
