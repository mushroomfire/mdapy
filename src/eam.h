// Copyright (c) 2022-2026, Yongchao Wu in Aalto University
// This file is from the mdapy project, released under the BSD 3-Clause License.
#pragma once

#include "type.h"
#include "spline.h"
#include "box.h"
#include <vector>

// EAM (Embedded Atom Method) calculator using LAMMPS-compatible cubic Hermite
// interpolation on uniform grids. The pair channel is stored as z(r) = r*phi(r)
// (LAMMPS's ``z2r`` convention), which is smooth through r=0 — this avoids
// the division-by-zero artifacts that interpolating phi(r) directly produces.
class EAM
{
public:
    EAM(const double rc,
        const double dr,
        const double drho,
        const RTwoArrayD F_rho_py,
        const RTwoArrayD rho_r_py,
        const RThreeArrayD rphi_r_py);

    void calculate(
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
        OneArrayD energy_py,
        const int num_t);

private:
    double rc_;
    int Nelements_;
    std::vector<UniformCubicSpline> F_rho_spline_;
    std::vector<UniformCubicSpline> rho_r_spline_;
    // rphi_r_spline_[i][j] holds the spline of r*phi_{ij}(r). Symmetric in i,j.
    std::vector<std::vector<UniformCubicSpline>> rphi_r_spline_;
};
