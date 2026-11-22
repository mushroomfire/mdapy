// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The ions on the sphere of the density, on the GPU: the local potential and
// the core charge of the atoms,
//   V(G) = sum_s V_s(|G|) sum_{I of s} exp(-i G.tau_I),  rho_core(G) likewise,
// from the form factors V_s and c_s of each species (per unit volume of the
// cell, radial.local_potential and radial.radial_density), and the forces of
// a density and a potential on them:
//   F_I = Omega sum_G Re[i G V_s(G) exp(-i G.tau_I) conj(rho(G))]
//       + Omega sum_G Re[i G c_s(G) exp(-i G.tau_I) conj(V_xc(G))],
// the local term and that of the core charge, hartree / bohr.

#pragma once

#include <vector>

#include "dft/core/device.h"
#include "dft/hamiltonian/hamiltonian.h"
#include "dft/lattice/lattice.h"

namespace mdapy::dft {

class Ions {
public:
    // local and core: (species, n) form factors on the sphere, core empty
    // without core charges
    Ions(const Lattice& lattice, const Sphere& sphere, const std::vector<Atom>& atoms, int species,
         const std::vector<double>& local, const std::vector<double>& core);

    bool has_core() const noexcept { return has_core_; }
    // V(G) and rho_core(G) into n components on the device
    void local_components(Complex* out) const;
    void core_components(Complex* out) const;
    // the slopes dV_s/d|G| and dc_s/d|G| of the form factors (species, n), for
    // the stress; core empty without core charges
    void set_slopes(const std::vector<double>& local, const std::vector<double>& core);
    bool has_slopes() const noexcept { return local_slope_.size() > 0; }
    // sum_s S_s(G) dV_s/d|G| and sum_s S_s(G) dc_s/d|G| on the device
    void local_slope_components(Complex* out) const;
    void core_slope_components(Complex* out) const;
    // the forces (atoms, 3) on the host of rho(G) through V_loc and, with core
    // charges, of V_xc(G) through rho_core; both on the device
    std::vector<double> forces(const Complex* rho, const Complex* vxc) const;

private:
    void structure(const DeviceArray<double>& factors, Complex* out) const;

    Lattice lattice_;
    std::size_t n_;
    int atoms_, species_;
    bool has_core_;
    DeviceArray<int> miller_, kinds_;
    DeviceArray<double> g_, tau_, local_, core_, local_slope_, core_slope_;
};

}  // namespace mdapy::dft
