#include "eam.h"
#include <cmath>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <omp.h>

EAM::EAM(const double rc,
         const RTwoArrayD F_rho_py,
         const RTwoArrayD rho_r_py,
         const RThreeArrayD phi_r_py,
         const ROneArrayD r_list_py,
         const ROneArrayD rho_list_py)
    : rc_(rc)
{
    auto r_list = r_list_py.view();
    auto rho_list = rho_list_py.view();
    const int nrho = rho_list.shape(0);
    const int nr = r_list.shape(0);
    Nelements_ = F_rho_py.shape(0);

    auto F_rho_data = F_rho_py.view();
    auto rho_r_data = rho_r_py.view();
    auto phi_r_data = phi_r_py.view();

    F_rho_spline_.resize(Nelements_);
    rho_r_spline_.resize(Nelements_);
    phi_r_spline_.resize(Nelements_, std::vector<CubicSpline>(Nelements_));

    std::vector<double> rho_vec(nrho);
    std::vector<double> r_vec(nr);
    for (int i = 0; i < nrho; ++i)
    {
        rho_vec[i] = rho_list(i);
    }
    for (int i = 0; i < nr; ++i)
    {
        r_vec[i] = r_list(i);
    }

    // spline F_rho
    std::vector<double> F_rho_values(nrho);
    for (int i = 0; i < Nelements_; ++i)
    {
        for (int j = 0; j < nrho; ++j)
        {
            F_rho_values[j] = F_rho_data(i, j);
        }
        F_rho_spline_[i] = CubicSpline(rho_vec, F_rho_values);
    }

    // spline rho_r
    std::vector<double> rho_r_values(nr);
    for (int i = 0; i < Nelements_; ++i)
    {
        for (int j = 0; j < nr; ++j)
        {
            rho_r_values[j] = rho_r_data(i, j);
        }
        rho_r_spline_[i] = CubicSpline(r_vec, rho_r_values);
    }

    // spline phi_r
    std::vector<double> phi_r_values(nr);
    for (int i = 0; i < Nelements_; ++i)
    {
        for (int j = 0; j < Nelements_; ++j)
        {
            for (int k = 0; k < nr; ++k)
            {
                phi_r_values[k] = phi_r_data(i, j, k);
            }
            phi_r_spline_[i][j] = CubicSpline(r_vec, phi_r_values);
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

    std::vector<double> rho(N, 0.0);
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i)
    {
        int type_i = type_list(i);
        double rho_i = 0.0;

        int num_neighbors = neighbor_number(i);
        for (int jj = 0; jj < num_neighbors; ++jj)
        {
            int j = verlet_list(i, jj);
            int type_j = type_list(j);
            double rij = distance_list(i, jj);

            if (rij <= rc_)
            {
                double rho_j_val = rho_r_spline_[type_j].evaluate(rij);
                rho_i += rho_j_val;
            }
        }
        rho[i] = rho_i;
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; ++i)
    {
        const int type_i = type_list(i);

        double F_i = F_rho_spline_[type_i].evaluate(rho[i]);
        double dF_i = F_rho_spline_[type_i].derivative(rho[i]);

        double e_i = F_i;
        double f_x = 0.0, f_y = 0.0, f_z = 0.0;
        double v[9] = {0.0};

        int num_neighbors = neighbor_number(i);
        for (int jj = 0; jj < num_neighbors; ++jj)
        {
            int j = verlet_list(i, jj);
            int type_j = type_list(j);

            double dx = x(j) - x(i);
            double dy = y(j) - y(i);
            double dz = z(j) - z(i);
            box.pbc(dx, dy, dz);
            double rij = distance_list(i, jj);

            if (rij <= rc_)
            {
                double phi_ij = phi_r_spline_[type_i][type_j].evaluate(rij);
                double dphi_ij = phi_r_spline_[type_i][type_j].derivative(rij);
                double rho_j_contrib_deriv = rho_r_spline_[type_j].derivative(rij);
                double rho_i_contrib_deriv = rho_r_spline_[type_i].derivative(rij);
                double dF_j = F_rho_spline_[type_j].derivative(rho[j]);

                e_i += 0.5 * phi_ij;

                double pair_force = dphi_ij + dF_i * rho_j_contrib_deriv + dF_j * rho_i_contrib_deriv;
                pair_force /= rij;

                const double pair_force_dx = pair_force * dx;
                f_x += pair_force_dx;
                const double pair_force_dy = pair_force * dy;
                f_y += pair_force_dy;
                const double pair_force_dz = pair_force * dz;
                f_z += pair_force_dz;

                v[0] -= dx * pair_force_dx;
                v[1] -= dx * pair_force_dy;
                v[2] -= dx * pair_force_dz;
                v[3] -= dy * pair_force_dx;
                v[4] -= dy * pair_force_dy;
                v[5] -= dy * pair_force_dz;
                v[6] -= dz * pair_force_dx;
                v[7] -= dz * pair_force_dy;
                v[8] -= dz * pair_force_dz;
            }
        }

        energy(i) += e_i;
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
                      const RTwoArrayD,
                      const RTwoArrayD,
                      const RThreeArrayD,
                      const ROneArrayD,
                      const ROneArrayD>(),
             nb::arg("rc"),
             nb::arg("F_rho"),
             nb::arg("rho_r"),
             nb::arg("phi_r"),
             nb::arg("r_list"),
             nb::arg("rho_list"))
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
