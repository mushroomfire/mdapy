// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Grids from mdapy_cuda12._dft.fft_grid at 4 E_cut, bands from the rule of
// the SCF for metals, batches from Hamiltonian::batch.

namespace {

const std::vector<Shape>& dft_shapes() {
    static const std::vector<Shape> shapes = {
        {"Al4 fcc, 15 Ha: bands", {27, 27, 27}, 10},
        {"Al4 fcc, 26 Ha: bands", {36, 36, 36}, 10},
        {"Al32 fcc, 15 Ha: bands", {54, 54, 54}, 58},
        {"Al32 fcc 600 eV, Fe54 20 Ha, Fe16 45 Ha: bands", {72, 72, 72}, 44},
        {"Al12 hcp 3x2x2, 600 eV: bands", {72, 45, 75}, 22},
        {"Al108 15 Ha, Al54 bcc 600 eV, Si64 18 Ha: bands", {80, 80, 80}, 32},
        {"Fe54 PAW 35.5 Ha: bands", {90, 90, 90}, 23},
        {"Fe54 45 Ha, Al108 600 eV: bands", {100, 100, 100}, 16},
        {"Si216 18 Ha: bands", {120, 120, 120}, 9},
        {"density and potentials, one box", {80, 80, 80}, 1},
        {"density and potentials, two spins", {100, 100, 100}, 2},
        {"Fe54 PAW density (248 Ha), two spins", {120, 120, 120}, 2},
        {"a large box", {200, 200, 200}, 2},
    };
    return shapes;
}

}  // namespace
