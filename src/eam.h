#pragma once

#include "type.h"
#include "spline.h"
#include "box.h"
#include <vector>

class EAM
{
public:
    EAM(const double rc,
        const RTwoArrayD F_rho_py,
        const RTwoArrayD rho_r_py,
        const RThreeArrayD phi_r_py,
        const ROneArrayD r_list_py,
        const ROneArrayD rho_list_py);

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
        OneArrayD energy_py);

private:
    double rc_;
    int Nelements_;
    std::vector<CubicSpline> F_rho_spline_;
    std::vector<CubicSpline> rho_r_spline_;
    std::vector<std::vector<CubicSpline>> phi_r_spline_;
};
