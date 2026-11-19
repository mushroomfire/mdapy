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
#include "calculators/nep.h"
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
    void kick_drift(double c, double dt);           // kick(c), then drift(dt)
    void scale(double fx, double fy, double fz);    // v *= (fx, fy, fz)
    // v_x -= half (v_y xy + v_z xz), then v_y -= half v_z yz
    void shear(double half, double yz, double xz, double xy);
    // x = lo + (x - old_lo) old_inverse h: the atoms carried with the box
    void remap(const double* old_lo, const double* old_inverse, const double* lo,
               const double* h);
    // v = c1 v + c2 sqrt(kt / m) g, g Gaussian of a seed and a counter; with
    // keep, the noise of each atom less an equal share of the momentum of
    // all the noise, as fix langevin zero yes, so that the momentum is kept
    void langevin(double c1, double c2, double kt, std::uint64_t seed, std::uint64_t counter,
                  bool keep = false);
    void kinetic(double* tensor) const;  // (3, 3), the sum of m v v
    // The directions in which each atom may move, (n, 3) in the axes of the
    // system, whose vectors are those of the atoms times rotation (3, 3)
    // transposed; then restrict_* removes the other components of the forces
    // or the velocities, as simulation/_integrator.py does on the processor.
    void set_movable(const bool* movable, const double* rotation);
    void clear_movable();  // every atom moves freely
    void restrict_forces();
    void restrict_velocities();

    // The controls of a run (simulation/_controls.py): the velocities of the
    // components the movable mask holds, (n, 3) in the frame of the atoms
    // and zero where they move freely, which restrict_velocities sets; forces
    // added to the atoms, (n, 3), which add_forces adds; and the groups of the
    // atoms, (n,), bit k for group k. Null clears each.
    void set_prescribed(const double* velocities);
    void set_added(const double* forces);
    void add_forces();
    void set_groups(const std::int32_t* bits);
    // (count, 3): the sums of the forces of the atoms of groups 0 to count - 1.
    void group_forces(int count, double* sums) const;
    // The sum of m |v|^2 over the components that move freely.
    double kinetic_free() const;
    // A Langevin bath on group bit, v = c1 v + c2 sqrt(kt / m) g in the
    // directions that move freely, g Gaussian of a seed and a counter; the
    // change of the sum of m |v|^2 / 2 of its atoms.
    double bath(int bit, double c1, double c2, double kt, std::uint64_t seed,
                std::uint64_t counter);

    // A Switch (simulation/_controls.py): the forces kept, then those of
    // another interaction computed, then f = (1 - weight) kept + weight f;
    // with other false, f = (1 - weight) f, the other none.
    void stash_forces();
    void mix_forces(double weight, bool other);
    // Springs to sites, (n, 3) in the frame of the atoms, and their
    // constants, (n,); null clears them. springs gives their forces and
    // energies in place of those of the atoms, the positions unwrapped by
    // their images in the box h, and their total energy.
    void set_springs(const double* sites, const double* constants);
    double springs(const double* h);
    // The field of HNEMD, (3,) in the frame of the atoms: each atom pushed by
    // W_i^T field, W_i its virial, less the mean of the pushes; after a
    // computation of the virials of each atom.
    void drive(const double* field);
    // (3,) the potential part of the heat flux times the volume, sum_i W_i v_i.
    void heat_flux(double* flux) const;
    // A flat wall (simulation/_controls.py): each atom with x . normal <
    // offset put back by as much, its velocity along the normal made
    // 2 speed - v . normal; normal (3,) a unit vector in the frame of the
    // atoms, into the side they are kept on.
    void reflect(const double* normal, double offset, double speed);
    // The multiscale shock technique (LAMMPS's fix msst): the sum of the
    // squares of the velocities; and the velocities half a step on,
    // v = exp(d t) (c + d v - c exp(-d t)) / d, or its expansion for a small
    // d t, with c = ftm2v f / m and d = base / m, less axial along the axis,
    // kept unless trial; the sum of their squares.
    double speed_squares() const;
    // A virtual indenter (LAMMPS's fix indent): kind 0 a sphere about at, 1 a
    // cylinder through at along the axis, 2 the region beyond the plane at[0]
    // across the axis; K (R - r)^2 out of a sphere or a cylinder (sign 1) or
    // into it (-1), the displacements in the minimum image of LAMMPS in the
    // box h with pbc, or K d^2 back across the plane on the side sign. The
    // forces added; out (4,) its energy and the force on it.
    void indent(int kind, int axis, double sign, double k, double radius, const double* at,
                const double* h, const bool* pbc, double* out);
    double msst_kick(double half, double base, double axial, int axis, bool trial,
                     double ftm2v);

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
// model prepared (NepModel::prepare)
std::unique_ptr<Potential> make_nep(const calculators::NepModel& model, Precision precision);

}  // namespace mdapy::cuda
