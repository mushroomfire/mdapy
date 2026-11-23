// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The potentials of a density on the dense grid, in hartree atomic units: the
// Hartree potential and energy from the Fourier components of the density on
// the sphere of the density cutoff, as QE keeps them; the exchange-correlation
// energy and potential of the spin densities with the core charge of the
// nonlinear core correction, half on each spin, the gradients and the
// divergence of GGA taken on the same sphere so that the potential is the
// exact derivative of the energy on the grid (the form of White and Bird).
// Densities are electrons per bohr^3 on the n1 n2 n3 points, one array per
// spin (one or two).

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "dft/core/device.h"
#include "dft/fft/fft.h"
#include "dft/lattice/lattice.h"
#include "dft/xc/xc.h"

namespace mdapy::dft {

struct PotentialEnergies {
    double hartree = 0.0;     // E_H
    double xc = 0.0;          // E_xc of the density with the core charge
    double hxc_density = 0.0;  // sum over spins of int (V_H + V_xc,s) rho_s
};

class Potentials {
public:
    // basis: the cell that chooses the sphere, for a fixed set of G as the cell
    // strains; the lattice itself by default
    Potentials(const Lattice& lattice, const Int3& grid, double cutoff, Functional functional,
               int spins, double threshold = 1e-10, const Lattice* basis = nullptr);

    // The core charge on the grid, n1 n2 n3 values on the host; none by default.
    void set_core(const double* core);

    // The Hartree and exchange-correlation potential of each spin, V_H + V_xc,s,
    // from the densities rho (spins x points on the device) into hxc (the same
    // shape on the device), and the energies.
    PotentialEnergies compute(const double* rho, double* hxc);

    // sum over count functions of int a b, on the device, count x points each
    double integral(const double* a, const double* b, int count);

    // The part of GGA in dE_xc/d epsilon of the last density computed,
    // -int sum_s h_s,a grad n_s,b with h_s the flux of the spin (xx yy zz yz xz
    // xy), hartree; zero for LDA.
    std::array<double, 6> gga_strain() const;

    // A density on the grid restricted to the sphere: its components beyond
    // the cutoff removed, in place, count functions.
    void restrict(double* values, int count);

    // Fourier components on the sphere (coefficients of exp(iGr)) of count
    // functions on the grid, and back; on the device.
    void to_sphere(const double* values, Complex* components, int count);
    void from_sphere(const Complex* components, double* values, int count);

    const Sphere& sphere() const noexcept { return sphere_; }
    std::size_t points() const noexcept { return fft_.points(); }
    int spins() const noexcept { return spins_; }
    double volume() const noexcept { return lattice_.volume; }
    Fft& fft() noexcept { return fft_; }

private:
    void gradient(const Complex* components, double* out);  // the 3 components of a gradient

    Lattice lattice_;
    Int3 grid_;
    Functional functional_;
    int spins_;
    double threshold_;
    Fft fft_;
    Sphere sphere_;
    DeviceArray<std::int64_t> box_;
    DeviceArray<double> g_;   // (n, 3) the Cartesian G of the sphere
    DeviceArray<double> core_;
    DeviceArray<Complex> boxes_, components_;
    DeviceArray<double> work_, rho_, sigma_, f_, v_, vs_, grads_, apart_;
    DeviceArray<Complex> spare_;  // components of the spins and of the gradients
    DeviceArray<double> sum_, flux_;
    DeviceArray<Complex> vh_, flux_g_, div_;
};

}  // namespace mdapy::dft
