// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// The Kohn-Sham Hamiltonian of norm-conserving pseudopotentials on the plane
// waves of one k-point, applied to a block of bands without a matrix:
//   H psi = |k+G|^2/2 psi + FFT^-1[V_eff(r) FFT[psi]] + sum_ij beta_i D_ij <beta_j|psi>.
// The projectors beta_{I,p,m}(k+G) = (-i)^l Y_lm(k+G) t_p(|k+G|) exp(-i (k+G).tau_I),
// t_p the table of projector p of the species of atom I (radial.projector_table,
// with its 4 pi / sqrt(Omega)), interpolated by Lagrange polynomials of four
// points as QE does, and Y_lm the real spherical harmonics of l up to 3, m
// from -l to l, in the order of the projectors of the species and then of m.
// The overlap S is 1. Hartree atomic units.
// With set_gamma(true), the k-point is Gamma and the bands are real: half a
// sphere of plane waves (dft/hamiltonian/gamma.h), real transforms, and real
// projections p, (projectors, count) doubles at the start of a buffer of
// Complex, wherever a projection is taken or given.

#pragma once

#include "dft/core/portable.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "dft/blas/blas.h"
#include "dft/core/device.h"
#include "dft/fft/fft.h"
#include "dft/lattice/lattice.h"

namespace mdapy::dft {

// The nonlocal part of a species: its projectors tabulated every dq from q =
// 0, their angular momenta and D_ij in hartree.
struct SpeciesProjectors {
    double dq = 0.01;
    int points = 0;                   // of each table
    std::vector<double> tables;       // (projectors, points)
    std::vector<int> angular;         // (projectors)
    std::vector<double> dij;          // (projectors, projectors)
    // the pseudo-orbitals of the free atom tabulated likewise, with the same
    // steps and points, for the first bands; empty for none
    std::vector<double> orbital_tables;  // (orbitals, points)
    std::vector<int> orbital_angular;    // (orbitals)
};

struct Atom {
    Vec3 position;  // fractions of the cell vectors
    int species;
};

namespace detail {
// a column of the projectors: where its table starts, l, m and its atom
struct ProjectorColumn {
    std::int64_t table;
    int l, m;
    double tau[3];  // fractions of the cell vectors
};
// what the kernels of the projectors take of the cell and the k-point
struct ReciprocalCell {
    double b[9];  // the reciprocal vectors, rows
    double k[3];
    double dq;
    int points;
};
}  // namespace detail

class Hamiltonian {
public:
    // basis: the cell that chooses the plane waves of each k-point, for a fixed
    // set of Miller indices as the cell strains; the lattice itself by default
    Hamiltonian(const Lattice& lattice, const Int3& grid, double cutoff,
                std::vector<SpeciesProjectors> species, std::vector<Atom> atoms,
                const Lattice* basis = nullptr, Kernels kernels = Kernels::Mdapy);
    ~Hamiltonian();
    Hamiltonian(const Hamiltonian&) = delete;
    Hamiltonian& operator=(const Hamiltonian&) = delete;

    // Real bands of Gamma from the next set_k on, which then takes Gamma only.
    void set_gamma(bool on) noexcept {
        if (on != gamma_mode_) {  // half spheres or whole ones: those kept are of the other kind
            spheres_.clear();
            k_set_ = false;
        }
        gamma_mode_ = on;
    }
    // Whether the bands of the k-point are real.
    bool gamma() const noexcept { return gamma_; }
    Kernels kernels() const noexcept { return blas_.kernels(); }
    // The plane waves of k (fractions of the reciprocal vectors) and their
    // projectors.
    void set_k(const Vec3& k);
    // The effective potential V_loc + V_H + V_xc on the grid, real, n1 n2 n3
    // values on the device.
    void set_potential(const double* potential);

    // H on count bands, column-major (plane waves, count), on the device.
    void apply(const Complex* psi, Complex* hpsi, int count);
    // S = 1 + sum |beta_a> q_ab <beta_b| on count bands; psi itself without
    // an overlap.
    void apply_overlap(const Complex* psi, Complex* spsi, int count);
    // The parts of H and S from the projections p = <beta|psi> (projectors,
    // count) on the device, for the solver to share them: p of psi; H psi
    // without the projectors; hpsi += beta D p; spsi += beta q p; out = q p.
    void project(const Complex* psi, int count, Complex* p);
    void apply_local(const Complex* psi, Complex* hpsi, int count);
    void add_nonlocal(const Complex* p, int count, Complex* hpsi);
    void add_overlap(const Complex* p, int count, Complex* spsi);
    void overlap_projections(const Complex* p, int count, Complex* out);
    // The same in single precision, for the bands of the single-precision
    // steps: the potential, the projectors and their strengths and overlaps
    // those set, rounded when they change. With the real bands of Gamma, real
    // transforms on the half sphere and real projections, (projectors, count)
    // floats at the start of a buffer of cuComplex.
    void apply(const cuComplex* psi, cuComplex* hpsi, int count);
    void project(const cuComplex* psi, int count, cuComplex* p);
    void apply_local(const cuComplex* psi, cuComplex* hpsi, int count);
    void add_nonlocal(const cuComplex* p, int count, cuComplex* hpsi);
    void add_overlap(const cuComplex* p, int count, cuComplex* spsi);
    void overlap_projections(const cuComplex* p, int count, cuComplex* out);
    // The strengths D and the overlaps q of the projectors of each atom, in the
    // order of the atoms a block (width, width) row by row, width its
    // projector functions; they replace those of the species.
    void set_strengths(const std::vector<double>& blocks);
    void set_overlaps(const std::vector<double>& blocks);
    bool has_overlap() const noexcept { return has_overlap_; }
    // becsum (packed pairs a <= b of each atom, on the host) += sum_n w_n
    // Re conj(<beta_a|psi_n>) <beta_b|psi_n>, twice off the diagonal
    void add_becsum(const Complex* psi, int count, const double* weights, std::vector<double>& becsum);
    // the same into becsum on the device; becsum_size() pairs
    void add_becsum_device(const Complex* psi, int count, const double* weights, double* becsum);
    std::size_t becsum_size() const;
    // The diagonal of S on the device, 1 without an overlap.
    void overlap_diagonal(double* out) const;
    // rho (the grid, on the device) += sum_b w_b |psi_b(r)|^2 / Omega of count
    // bands (plane waves, count) on the device, weights on the host.
    void add_density(const Complex* psi, int count, const double* weights, double* rho);
    // forces (atoms, 3) on the host += those of the projectors on count bands
    // (plane waves, count) on the device of weights w_b (k-point weight, f_max
    // and occupation) on the host: -sum_b w_b d<psi_b|V_nl|psi_b>/dtau_I.
    // With an overlap, of D - e_n q, eigenvalues on the host.
    void nonlocal_forces(const Complex* psi, int count, const double* weights, std::vector<double>& forces,
                         const double* eigenvalues = nullptr);
    // out (xx yy zz yz xz xy) += dE/d epsilon of count bands at a fixed set of
    // plane waves, their weights w_b on the host: the kinetic energy and the
    // projectors, hartree.
    // With an overlap, of D - e_n q, eigenvalues on the host.
    void strain_derivative(const Complex* psi, int count, const double* weights, std::array<double, 6>& out,
                           const double* eigenvalues = nullptr);
    // The diagonal H_GG on the device, for the preconditioner: the kinetic
    // energy, the mean potential and the projectors.
    void diagonal(double* out) const;

    const Sphere& sphere() const noexcept { return sphere_; }
    std::size_t plane_waves() const noexcept { return sphere_.size(); }
    int projectors() const noexcept { return projectors_; }
    const DeviceArray<Complex>& beta() const noexcept { return beta_; }
    const DeviceArray<double>& kinetic() const noexcept { return kinetic_; }
    // The pseudo-orbitals of every atom at the k-point, (plane waves,
    // atomic_functions()) on the device, as the projectors are built.
    int atomic_functions() const noexcept { return atomic_; }
    void atomic(Complex* out) const;

private:
    // kinetic + mean + sum_ij d_ij Re(b_i conj(b_j)) of each plane wave, kinetic optional
    void diagonal_of(const double* kinetic, double mean, const Complex* d, double* out) const;
    mutable DeviceArray<double> diagonal_partial_;  // (atoms, plane waves): the share of each atom
    int batch() const noexcept;  // bands per batch of FFTs
    void single();               // the single-precision copies, when stale

    Lattice lattice_;
    bool has_basis_ = false;
    Lattice basis_;
    Int3 grid_;
    double cutoff_;
    std::vector<SpeciesProjectors> species_;
    std::vector<Atom> atoms_;
    Fft fft_;
    Blas blas_;
    int projectors_ = 0;

    // of the k-point
    Sphere sphere_;
    // the spheres of the k-points set so far: each is made once, and the
    // steps of an SCF come back to the same k-points
    std::vector<std::pair<Vec3, Sphere>> spheres_;
    int window_ = -1;  // the largest |m| of the first Miller index of the sphere: the planes of its boxes that matter
    // the k-point set last; while k_set_, what follows is of it
    Vec3 k_{};
    bool k_set_ = false;
    DeviceArray<double> kinetic_, q_;  // |k+G|^2 / 2 and the Cartesian k+G (n, 3)
    DeviceArray<std::int64_t> box_;
    DeviceArray<Complex> beta_;     // (plane waves, projectors)
    DeviceArray<Complex> dmatrix_;  // (projectors, projectors), block diagonal
    DeviceArray<Complex> qmatrix_;  // the overlaps, likewise
    DeviceArray<double> dreal_, qreal_;  // the same, real, for Gamma
    bool has_overlap_ = false;
    bool gamma_mode_ = false, gamma_ = false;
    DeviceArray<std::int64_t> partner_;  // of the half sphere, for the real transforms
    DeviceArray<double> values_;         // real boxes
    // the potential and the work arrays
    DeviceArray<double> potential_;
    DeviceArray<Complex> boxes_, projections_, scaled_;
    // the tables of every species on the device, and where each starts
    DeviceArray<double> tables_, orbital_tables_;
    int atomic_ = 0;
    // the block of D of each atom, (first, width)
    std::vector<int> blocks_;
    DeviceArray<int> blocks_device_;
    double mean_potential_ = 0.0;
    DeviceArray<double> weights_, total_;
    // of the k-point, for the stress of the projectors: the Miller indices, the
    // columns of the projectors and the reciprocal cell of their kernel
    DeviceArray<int> miller_;
    DeviceArray<detail::ProjectorColumn> columns_;
    detail::ReciprocalCell reciprocal_{};
    DeviceArray<std::size_t> becsum_offsets_;  // where the pairs of each atom start
    // single precision: what single() rounds, and its work arrays
    bool single_stale_ = true;
    DeviceArray<float> kinetic_f_, potential_f_;
    DeviceArray<float> dreal_f_, qreal_f_, values_f_;  // of Gamma, as dreal_, qreal_ and values_
    DeviceArray<cuComplex> beta_f_, dmatrix_f_, qmatrix_f_, boxes_f_, projections_f_, scaled_f_;
};

}  // namespace mdapy::dft
