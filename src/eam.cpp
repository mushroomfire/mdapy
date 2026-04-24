// Copyright (c) 2022-2026, Yongchao Wu in Aalto University
// This file is from the mdapy project, released under the BSD 3-Clause License.
#include "eam.h"
#include <cmath>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <omp.h>

EAM::EAM(const double rc,
         const double dr,
         const double drho,
         const RTwoArrayD F_rho_py,
         const RTwoArrayD rho_r_py,
         const RThreeArrayD rphi_r_py)
    : rc_(rc)
{
    const int nrho = F_rho_py.shape(1);
    const int nr = rho_r_py.shape(1);
    Nelements_ = F_rho_py.shape(0);

    auto F_rho_data = F_rho_py.view();
    auto rho_r_data = rho_r_py.view();
    auto rphi_r_data = rphi_r_py.view();

    F_rho_spline_.resize(Nelements_);
    rho_r_spline_.resize(Nelements_);
    rphi_r_spline_.resize(Nelements_, std::vector<UniformCubicSpline>(Nelements_));

    std::vector<double> tmp;
    tmp.resize(nrho);
    for (int i = 0; i < Nelements_; ++i)
    {
        for (int j = 0; j < nrho; ++j) tmp[j] = F_rho_data(i, j);
        F_rho_spline_[i] = UniformCubicSpline(drho, tmp);
    }

    tmp.resize(nr);
    for (int i = 0; i < Nelements_; ++i)
    {
        for (int j = 0; j < nr; ++j) tmp[j] = rho_r_data(i, j);
        rho_r_spline_[i] = UniformCubicSpline(dr, tmp);
    }

    for (int i = 0; i < Nelements_; ++i)
    {
        for (int j = 0; j < Nelements_; ++j)
        {
            for (int k = 0; k < nr; ++k) tmp[k] = rphi_r_data(i, j, k);
            rphi_r_spline_[i][j] = UniformCubicSpline(dr, tmp);
        }
    }
}

void EAM::calculate(
    const ROneArrayD x_py,
    const ROneArrayD y_py,
    const ROneArrayD z_py,
    const ROneArrayI type_list_py,
    const RTwoArrayD box_py,
    const ROneArrayD origin,
    const ROneArrayI boundary,
    const RTwoArrayI verlet_list_py,
    const RTwoArrayD distance_list_py,
    const ROneArrayI neighbor_number_py,
    TwoArrayD force_py,
    TwoArrayD virial_py,
    OneArrayD energy_py)
{
    const int N = x_py.shape(0);
    const Box box = get_box(box_py, origin, boundary);

    auto x = x_py.view();
    auto y = y_py.view();
    auto z = z_py.view();
    auto type_list = type_list_py.view();
    auto verlet_list = verlet_list_py.view();
    auto distance_list = distance_list_py.view();
    auto neighbor_number = neighbor_number_py.view();

    auto force = force_py.view();
    auto virial = virial_py.view();
    auto energy = energy_py.view();

    // Pass 1: electron density at each atom, plus dF/drho.
    std::vector<double> rho(N, 0.0);
    std::vector<double> dF(N, 0.0);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i)
    {
        double rho_i = 0.0;
        const int num_neighbors = neighbor_number(i);
        for (int jj = 0; jj < num_neighbors; ++jj)
        {
            const int j = verlet_list(i, jj);
            const double rij = distance_list(i, jj);
            if (rij <= rc_)
            {
                rho_i += rho_r_spline_[type_list(j)].evaluate(rij);
            }
        }
        rho[i] = rho_i;
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i)
    {
        double F_i, dF_i;
        F_rho_spline_[type_list(i)].evaluate_and_derivative(rho[i], F_i, dF_i);
        energy(i) += F_i;
        dF[i] = dF_i;
    }

    // Pass 2: pair energy, forces, virials.
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i)
    {
        const int type_i = type_list(i);
        const double dF_i = dF[i];

        double e_pair = 0.0;
        double f_x = 0.0, f_y = 0.0, f_z = 0.0;
        double v[9] = {0.0};

        const int num_neighbors = neighbor_number(i);
        for (int jj = 0; jj < num_neighbors; ++jj)
        {
            const int j = verlet_list(i, jj);
            const int type_j = type_list(j);

            double dx = x(j) - x(i);
            double dy = y(j) - y(i);
            double dz = z(j) - z(i);
            box.pbc(dx, dy, dz);
            const double rij = distance_list(i, jj);
            if (rij > rc_) continue;

            // r*phi and its r-derivative
            double z2, dz2_dr;
            rphi_r_spline_[type_i][type_j].evaluate_and_derivative(rij, z2, dz2_dr);
            const double rinv = 1.0 / rij;
            const double phi_ij = z2 * rinv;
            const double dphi_ij = (dz2_dr - phi_ij) * rinv; // d(phi)/dr

            // d(rho_j)/dr and d(rho_i)/dr for cross terms
            const double d_rho_j_dr = rho_r_spline_[type_j].derivative(rij);
            const double d_rho_i_dr = rho_r_spline_[type_i].derivative(rij);
            const double dF_j = dF[j];

            e_pair += 0.5 * phi_ij;

            double pair_force = dphi_ij + dF_i * d_rho_j_dr + dF_j * d_rho_i_dr;
            pair_force *= rinv;

            const double fpx = pair_force * dx;
            const double fpy = pair_force * dy;
            const double fpz = pair_force * dz;
            f_x += fpx;
            f_y += fpy;
            f_z += fpz;

            v[0] -= dx * fpx;
            v[1] -= dx * fpy;
            v[2] -= dx * fpz;
            v[3] -= dy * fpx;
            v[4] -= dy * fpy;
            v[5] -= dy * fpz;
            v[6] -= dz * fpx;
            v[7] -= dz * fpy;
            v[8] -= dz * fpz;
        }

        energy(i) += e_pair;
        force(i, 0) += f_x;
        force(i, 1) += f_y;
        force(i, 2) += f_z;
        for (int k = 0; k < 9; ++k)
        {
            virial(i, k) += v[k] * 0.5;
        }
    }
}

namespace nb = nanobind;

NB_MODULE(_eam, m)
{
    nb::class_<EAM>(m, "EAM")
        .def(nb::init<const double,
                      const double,
                      const double,
                      const RTwoArrayD,
                      const RTwoArrayD,
                      const RThreeArrayD>(),
             nb::arg("rc"),
             nb::arg("dr"),
             nb::arg("drho"),
             nb::arg("F_rho"),
             nb::arg("rho_r"),
             nb::arg("rphi_r"))
        .def("calculate", &EAM::calculate,
             nb::arg("x"),
             nb::arg("y"),
             nb::arg("z"),
             nb::arg("type_list"),
             nb::arg("box"),
             nb::arg("origin"),
             nb::arg("boundary"),
             nb::arg("verlet_list"),
             nb::arg("distance_list"),
             nb::arg("neighbor_number"),
             nb::arg("force"),
             nb::arg("virial"),
             nb::arg("energy"));
}
