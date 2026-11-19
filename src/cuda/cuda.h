// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// MDAPY on an NVIDIA GPU with CUDA: atoms, a neighbor list and potentials
// held on the device. This header has no CUDA syntax, so that the bindings
// compile with the host compiler; the kernels are in the .cu files.

#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "calculators/adp.h"
#include "calculators/eam.h"
#include "calculators/meam.h"
#include "calculators/pair.h"
#include "calculators/stillinger_weber.h"
#include "calculators/tersoff.h"

namespace mdapy::cuda {

// Thrown when positions, energies or forces stop being finite.
class NotFinite : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// The name of the device in use; throws std::runtime_error without one.
std::string device_name();

// Mixed: single precision for the distance and the terms of each pair, as
// the GPU package of LAMMPS in mixed precision, double precision for the
// positions and the sums. Double: double precision throughout.
enum class Precision { Mixed, Double };

// A box: the rows of h are its vectors, lo its origin, pbc the directions it
// is periodic along.
struct Box {
    double h[9];
    double lo[3];
    bool pbc[3];
};

// The atoms on the device: positions, species, forces, and the energies and
// virials of the last computation.
class Atoms {
public:
    explicit Atoms(std::int64_t n);
    ~Atoms();
    Atoms(const Atoms&) = delete;
    Atoms& operator=(const Atoms&) = delete;

    std::int64_t size() const;
    void set_species(const std::int32_t* species);  // (n,)
    void set_positions(const double* positions);    // (n, 3)
    // In the box along its periodic directions after a neighbor list was
    // built, which wraps them; images counts the boxes they were moved by.
    void get_positions(double* positions) const;
    void get_images(std::int32_t* images) const;    // (n, 3)
    void get_forces(double* forces) const;          // (n, 3)
    void get_energies(double* energies) const;      // (n,), after an energy
    void get_virials(double* virials) const;        // (n, 3, 3), after atomic virials
    double total_energy() const;                     // after an energy
    void total_virial(double* virial) const;         // (3, 3), after a virial

    // Molecular dynamics: masses in amu and velocities in Å/ps, and the steps
    // of MDAPY's integrator (simulation/_integrator.py) on every atom.
    void set_masses(const double* masses);          // (n,)
    void set_velocities(const double* velocities);  // (n, 3)
    void get_velocities(double* velocities) const;
    void kick(double c);                            // v += c f / m
    void drift(double dt);                          // x += dt v
    void scale(double fx, double fy, double fz);    // v *= (fx, fy, fz)
    // v_x -= half (v_y xy + v_z xz), then v_y -= half v_z yz
    void shear(double half, double yz, double xz, double xy);
    // x = lo + (x - old_lo) old_inverse h: the atoms carried with the box
    void remap(const double* old_lo, const double* old_inverse, const double* lo,
               const double* h);
    // v = c1 v + c2 sqrt(kt / m) g, g Gaussian of a seed and a counter
    void langevin(double c1, double c2, double kt, std::uint64_t seed, std::uint64_t counter);
    void kinetic(double* tensor) const;  // (3, 3), the sum of m v v

    struct Impl;
    Impl& impl() { return *impl_; }
    const Impl& impl() const { return *impl_; }

private:
    std::unique_ptr<Impl> impl_;
};

// A full neighbor list on the device, of the pairs closer than the cutoff
// and a skin, by the minimum image along the periodic directions.
class Neighbors {
public:
    Neighbors(double cutoff, double skin);
    ~Neighbors();
    Neighbors(const Neighbors&) = delete;
    Neighbors& operator=(const Neighbors&) = delete;

    // Builds the list if it was never built, the number of atoms changed, or
    // the atoms and the box moved more than half the skin allows since it
    // was; true if it built. Throws std::invalid_argument for a box that is
    // not thicker than twice the cutoff and the skin along a periodic
    // direction, where the minimum image would miss pairs.
    // Building wraps the atoms into the box along its periodic directions.
    bool update(Atoms& atoms, const Box& box);
    double cutoff() const;
    double skin() const;
    std::int64_t builds() const;
    // The neighbors of each atom: (n,) counts and (n, most) indices, -1 past
    // the count, and the image of each, 0 to 26 for shifts of -1, 0 or 1
    // boxes along each vector, (s0 + 1) + 3 (s1 + 1) + 9 (s2 + 1); for tests.
    // In the order of the caller's atoms.
    void get(const Atoms& atoms, std::int32_t* counts, std::int32_t* indices,
             std::int32_t* images) const;
    int most() const;

    struct Impl;
    Impl& impl() { return *impl_; }
    const Impl& impl() const { return *impl_; }

private:
    std::unique_ptr<Impl> impl_;
};

// A potential on the device.
class Potential {
public:
    virtual ~Potential() = default;
    virtual double cutoff() const = 0;
    virtual int species() const = 0;
    // The forces on the atoms, and their energies and virials when asked:
    // their totals, and with atomic those of each atom as the processor
    // shares them out; without, a many-body potential may leave the shares of
    // each term with its centre. The list must be up to date.
    virtual void compute(Atoms& atoms, Neighbors& neighbors, const Box& box, bool energy,
                         bool virial, bool atomic) = 0;
};

std::unique_ptr<Potential> make_eam(const calculators::EamPotential& potential,
                                    Precision precision);
std::unique_ptr<Potential> make_pair(const calculators::PairPotential& potential,
                                     Precision precision);
std::unique_ptr<Potential> make_adp(const calculators::AdpPotential& potential,
                                    Precision precision);
std::unique_ptr<Potential> make_stillinger_weber(
    const calculators::StillingerWeberPotential& potential, Precision precision);
std::unique_ptr<Potential> make_tersoff(const calculators::TersoffPotential& potential,
                                        Precision precision);
std::unique_ptr<Potential> make_meam(const calculators::MeamPotential& potential,
                                     Precision precision);

}  // namespace mdapy::cuda
