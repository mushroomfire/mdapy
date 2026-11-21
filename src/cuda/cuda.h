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
#include <vector>

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

// The device in use: its name, its memory in bytes, its compute capability
// and its number of multiprocessors; throws std::runtime_error without one.
struct DeviceProperties {
    std::string name;
    std::int64_t memory = 0;
    int major = 0;
    int minor = 0;
    int multiprocessors = 0;
};
DeviceProperties device_properties();

// The number of devices this process sees, 0 without any.
int device_count();

// The device the following allocations and kernels of this process use;
// Throws std::invalid_argument for one it does not see.
void use_device(int device);

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
    // Under MPI: the atoms of this process are the first owned of the
    // caller's, its ghosts after them; the steps of the integrator and every
    // sum over the atoms take the first owned alone, and a neighbor list keeps
    // them first. 0 for all, as by default. owned() gives their number.
    void set_owned(std::int64_t owned);
    std::int64_t owned() const;
    // The index among all the atoms of each, (n,) in the caller's order, by
    // which it draws its noise; null for its index in the caller's order.
    void set_ids(const std::int64_t* ids);
    // The ids in the caller's order, (n,).
    void get_ids(std::int64_t* ids) const;

    // A build of the bricks under MPI (simulation/_domain_device.py), the
    // rows those of the places of the atoms on the device, which settle makes
    // the caller's order. What each atom carries: its position and species,
    // its images, mass, id, velocity and force; keep_rows keeps those rows in
    // their order and drops the others, pack_rows packs what they carry,
    // row_bytes each, which append_rows adds after the atoms, and
    // append_ghosts adds ghosts at positions (count, 3) with their species
    // and ids. Each drops what the controls of a run set on the atoms, which
    // the caller gives again, and the energies and virials computed.
    void settle();
    void keep_rows(const std::int32_t* rows, std::int64_t count);
    std::vector<unsigned char> pack_rows(const std::int32_t* rows, std::int64_t count) const;
    static std::size_t row_bytes();
    void append_rows(const unsigned char* bytes, std::size_t size, std::int64_t count);
    void append_ghosts(const double* positions, const std::int32_t* species,
                       const std::int64_t* ids, std::int64_t count);
    // The first owned atoms into the box h (3, 3) of origin lo (3,) along the
    // periodic directions, their images counted.
    void wrap(const double* h, const double* lo, const bool* pbc);
    // The fractional coordinate along vector d of the box of the rows from
    // begin to end, into out (end - begin,).
    void fractions(int d, const double* h, const double* lo, std::int64_t begin,
                   std::int64_t end, double* out) const;
    // The positions of rows (count,), plus shift (3,), into out (count, 3).
    void gather_positions(const std::int32_t* rows, std::int64_t count, const double* shift,
                          double* out) const;
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
    // The forces of the last step, as a restart kept them, (n, 3).
    void set_forces(const double* forces);
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
    // The two halves of langevin with keep, for a share of the momentum of
    // the noise of every process: draw_noise draws it and gives the sum of
    // m dv, (3,); keep_noise then sets v = c1 v + dv - share / m.
    void draw_noise(double c2, double kt, std::uint64_t seed, std::uint64_t counter,
                    double* total);
    void keep_noise(double c1, const double* share);
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
    // Its two halves, for the mean of every process: push gives the sum of
    // the pushes, (3,), pushed adds them less a mean, (3,).
    void push(const double* field, double* total);
    void pushed(const double* mean);
    // (3,) the potential part of the heat flux times the volume, sum_i W_i v_i.
    void heat_flux(double* flux) const;
    // With the energies each atom carries, after drive: each atom pushed by
    // the field times its energy E_i less the mean of the energies, E_i its
    // own energy and its kinetic energy at the velocity v + kick f / m;
    // mvv2e: amu Å^2/ps^2 in eV.
    void convect(const double* field, double kick, double mvv2e);
    // Its two halves: carried gives the sum of the energies, convected
    // pushes the atoms by the field times their energies less a mean.
    double carried(double kick, double mvv2e);
    void convected(const double* field, double mean);
    // (3,) the part of the heat flux times the volume that the atoms carry,
    // sum_i E_i v_i, of their own energies and velocities now.
    void energy_flux(double mvv2e, double* flux) const;
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
    // The two halves of update, for processes that build together: whether
    // it would build, and the build itself. Under MPI the atoms of the
    // process come first in the list and the ghosts after them, the atoms
    // given just before, in the caller's order.
    bool stale(Atoms& atoms, const Box& box);
    void build(Atoms& atoms, const Box& box);
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

// Electronic stopping, as LAMMPS's fix electron/stopping: the stopping power
// of each species at energies in eV, ascending, linear in between.
class Stopping {
public:
    // energies (count,), values (species, count) in eV/Å.
    Stopping(const double* energies, int count, const double* values, int species);
    ~Stopping();
    Stopping(const Stopping&) = delete;
    Stopping& operator=(const Stopping&) = delete;

    // Each atom of group bit (every atom for -1) with a kinetic energy
    // m v^2 mvv2e / 2 of least and of the first energy or more, and with
    // needed neighbors or more closer than cutoff in the list, which must be
    // up to date, slowed by -S v / |v|, S of its species at its energy; out
    // (2,): the sum of S |v| and the number of them beyond the last energy.
    void apply(Atoms& atoms, Neighbors& neighbors, const Box& box, double least, int needed,
               double cutoff, int bit, double mvv2e, double* out) const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

// Under MPI, the messages of the borders of the bricks of the processes
// (simulation/_domain.py), fixed between two builds of the neighbor list:
// message k sends the atoms at rows of the caller's order and receives
// ghosts into the rows from start on. Positions go out shifted across a
// periodic boundary; the forces on the ghosts come back, added to their atoms.
class Exchange {
public:
    Exchange();
    ~Exchange();
    Exchange(const Exchange&) = delete;
    Exchange& operator=(const Exchange&) = delete;

    // After the list was built, the atoms at their places since.
    void set(const Atoms& atoms, int k, const std::int32_t* rows, std::int64_t count,
             std::int64_t start, std::int64_t received);
    void clear();
    std::int64_t sent(int k) const;
    std::int64_t received(int k) const;
    // The positions message k sends, plus shift (3,), into out (sent, 3); the
    // positions it received, in (received, 3), into the ghosts.
    void pack_positions(const Atoms& atoms, int k, const double* shift, double* out) const;
    void unpack_positions(Atoms& atoms, int k, const double* in) const;
    // The forces on the ghosts message k received, into out (received, 3),
    // and those that come back for the atoms it sent, in (sent, 3), added to
    // theirs.
    void pack_forces(const Atoms& atoms, int k, double* out) const;
    void add_forces(Atoms& atoms, int k, const double* in) const;

    struct Impl;

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
    // The forces, and the energies and virials of the atoms that give the
    // heat flux: those of compute with atomic, but for the many-body
    // potentials whose shares of a term are LAMMPS's, Stillinger-Weber and
    // Tersoff, which leave each term whole with its centre and give the
    // virials of Fan et al., PRB 92, 094301 (2015).
    virtual void compute_heat(Atoms& atoms, Neighbors& neighbors, const Box& box) {
        compute(atoms, neighbors, box, true, true, true);
    }
};

// Potentials added together: each computes in turn, its forces, energies
// and virials added to those of the ones before; the parts must outlive it.
std::unique_ptr<Potential> make_overlay(std::vector<Potential*> parts);

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
