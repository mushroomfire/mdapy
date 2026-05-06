// Copyright (c) 2022-2026, Yongchao Wu in Aalto University
// This file is from the mdapy project, released under the BSD 3-Clause License.
//
// CHILL+ algorithm:
//   Nguyen & Molinero, J. Phys. Chem. B 119 (2015) 9369-9376.
// Implementation cross-checked against OVITO's ChillPlusModifier
// (Henrik Andersen Sveinsson, 2019).
#include "type.h"
#include "box.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <complex>
#include <cmath>
#include <omp.h>

namespace nb = nanobind;

using cf = std::complex<float>;

// Real-valued normalisation constants for Y_3^|m| in the Condon-Shortley
// convention (matches boost::math::spherical_harmonic that OVITO uses).
//   Y_3^0  = N0 * (5 c^3 - 3 c)
//   Y_3^1  = -N1 * s (5 c^2 - 1) e^{ i phi}
//   Y_3^2  =  N2 * s^2 c          e^{2i phi}
//   Y_3^3  = -N3 * s^3            e^{3i phi}
//   Y_3^{-m} = (-1)^m conj(Y_3^m)
inline void compute_y3m(double dx, double dy, double dz, cf y[7])
{
    const float r2 = static_cast<float>(dx * dx + dy * dy + dz * dz);
    if (r2 <= 0.0f)
    {
        for (int i = 0; i < 7; ++i) y[i] = cf(0.0f, 0.0f);
        return;
    }
    const float r = std::sqrt(r2);
    const float ct = static_cast<float>(dz) / r;                                  // cos(theta)
    const float xy = std::sqrt(static_cast<float>(dx * dx + dy * dy));
    const float st = xy / r;                                                      // sin(theta)
    const float phi = std::atan2(static_cast<float>(dy), static_cast<float>(dx));

    constexpr float PI = 3.14159265358979323846f;
    const float N0 = 0.25f       * std::sqrt(7.0f / PI);
    const float N1 = 0.125f      * std::sqrt(21.0f / PI);
    const float N2 = 0.25f       * std::sqrt(105.0f / (2.0f * PI));
    const float N3 = 0.125f      * std::sqrt(35.0f / PI);

    const float ct2 = ct * ct;
    const float ct3 = ct2 * ct;
    const float st2 = st * st;
    const float st3 = st2 * st;

    const cf e1   = std::exp(cf(0.0f, phi));
    const cf e2   = e1 * e1;
    const cf e3   = e2 * e1;
    const cf en1  = std::conj(e1);
    const cf en2  = std::conj(e2);
    const cf en3  = std::conj(e3);

    const float a1 = N1 * st * (5.0f * ct2 - 1.0f);
    const float a2 = N2 * st2 * ct;
    const float a3 = N3 * st3;

    // index = m + 3, so y[0]..y[6] correspond to m = -3..3
    y[0] =  a3 * en3;                                            // m = -3
    y[1] =  a2 * en2;                                            // m = -2
    y[2] =  a1 * en1;                                            // m = -1
    y[3] =  cf(N0 * (5.0f * ct3 - 3.0f * ct), 0.0f);             // m =  0
    y[4] = -a1 * e1;                                             // m =  1
    y[5] =  a2 * e2;                                             // m =  2
    y[6] = -a3 * e3;                                             // m =  3
}

// Structure type codes:
//   0 OTHER, 1 HEXAGONAL_ICE, 2 CUBIC_ICE,
//   3 INTERFACIAL_ICE, 4 HYDRATE, 5 INTERFACIAL_HYDRATE
void compute_chill_plus(
    const ROneArrayD x_py,
    const ROneArrayD y_py,
    const ROneArrayD z_py,
    const RTwoArrayD box_py,
    const ROneArrayD origin_py,
    const ROneArrayI boundary_py,
    const RTwoArrayI verlet_list_py,
    const RTwoArrayD distance_list_py,
    const ROneArrayI neighbor_number_py,
    const double rc,
    OneArrayI pattern_py,
    const int num_t)
{
    const Box box = get_box(box_py, origin_py, boundary_py);
    const double *x = x_py.data();
    const double *y = y_py.data();
    const double *z = z_py.data();
    auto verlet_list   = verlet_list_py.view();
    auto distance_list = distance_list_py.view();
    auto neighbor_num  = neighbor_number_py.view();
    auto pattern       = pattern_py.view();

    const int N = static_cast<int>(x_py.shape(0));

    // Phase 1: q_3m for each particle (m = -3..3 packed as 7 complex floats).
    cf *q = new cf[static_cast<size_t>(N) * 7]{};

#pragma omp parallel for num_threads(num_t) schedule(dynamic, 1024)
    for (int i = 0; i < N; ++i)
    {
        const int n_neigh = neighbor_num(i);
        const double xi = x[i];
        const double yi = y[i];
        const double zi = z[i];
        cf *qi = q + static_cast<size_t>(i) * 7;
        for (int jj = 0; jj < n_neigh; ++jj)
        {
            if (distance_list(i, jj) > rc) continue;
            const int j = verlet_list(i, jj);
            if (j < 0) continue;
            double dx = x[j] - xi;
            double dy = y[j] - yi;
            double dz = z[j] - zi;
            box.pbc(dx, dy, dz);
            cf y3m[7];
            compute_y3m(dx, dy, dz, y3m);
            for (int k = 0; k < 7; ++k) qi[k] += y3m[k];
        }
    }

    // Phase 2: classify each particle from the c_ij correlations.
#pragma omp parallel for num_threads(num_t) schedule(dynamic, 4096)
    for (int i = 0; i < N; ++i)
    {
        const int n_neigh = neighbor_num(i);
        const cf *qi = q + static_cast<size_t>(i) * 7;

        // |q_i|^2 = sum_m q_i^m * conj(q_i^m)
        float qi_norm_sq = 0.0f;
        for (int k = 0; k < 7; ++k) qi_norm_sq += std::norm(qi[k]);

        int num_eclipsed  = 0;
        int num_staggered = 0;
        int coordination  = 0;

        for (int jj = 0; jj < n_neigh; ++jj)
        {
            if (distance_list(i, jj) > rc) continue;
            const int j = verlet_list(i, jj);
            if (j < 0) continue;
            const cf *qj = q + static_cast<size_t>(j) * 7;

            cf c1(0.0f, 0.0f);
            float qj_norm_sq = 0.0f;
            for (int k = 0; k < 7; ++k)
            {
                c1 += qi[k] * std::conj(qj[k]);
                qj_norm_sq += std::norm(qj[k]);
            }

            const float denom = std::sqrt(qi_norm_sq) * std::sqrt(qj_norm_sq);
            float c_re = (denom > 0.0f) ? (std::real(c1) / denom) : 0.0f;

            if (c_re > -0.35f && c_re < 0.25f) ++num_eclipsed;
            if (c_re < -0.8f) ++num_staggered;
            ++coordination;
        }

        int code = 0;  // OTHER
        if (coordination == 4)
        {
            if (num_eclipsed == 4)                                     code = 4;  // HYDRATE
            else if (num_eclipsed == 3)                                code = 5;  // INTERFACIAL_HYDRATE
            else if (num_staggered == 4)                               code = 2;  // CUBIC_ICE
            else if (num_staggered == 3 && num_eclipsed == 1)          code = 1;  // HEXAGONAL_ICE
            else if (num_staggered == 3 && num_eclipsed == 0)          code = 3;  // INTERFACIAL_ICE
            else if (num_staggered == 2)                               code = 3;  // INTERFACIAL_ICE
        }
        pattern(i) = code;
    }

    delete[] q;
}

NB_MODULE(_chill_plus, m)
{
    m.def("compute_chill_plus", &compute_chill_plus);
}
