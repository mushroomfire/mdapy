#include "type.h"
#include "box.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

void compute_aja(
    const ROneArrayD x,
    const ROneArrayD y,
    const ROneArrayD z,
    const RTwoArrayD box_array,
    const ROneArrayD origin_array,
    const ROneArrayI boundary_array,
    const RTwoArrayI verlet_list,
    const RTwoArrayD distance_list,
    OneArrayI aja)
{
    const int n_atoms = x.shape(0);
    Box box = get_box(box_array, origin_array, boundary_array);

    const double *x_ptr = x.data();
    const double *y_ptr = y.data();
    const double *z_ptr = z.data();
    const int *verlet_ptr = verlet_list.data();
    const double *dist_ptr = distance_list.data();
    int *aja_ptr = aja.data();

    const int verlet_stride = verlet_list.shape(1);
    const int dist_stride = distance_list.shape(1);

#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n_atoms; ++i)
    {

        double r0_sq = 0.0;
        const double *dist_i = dist_ptr + i * dist_stride;
        for (int j = 0; j < 6; ++j)
        {
            double d = dist_i[j];
            r0_sq += d * d;
        }
        r0_sq /= 6.0;

        int N0 = 0, N1 = 0;
        const double r0_sq_145 = 1.45 * r0_sq;
        const double r0_sq_155 = 1.55 * r0_sq;

        for (int j = 0; j < 14; ++j)
        {
            double d = dist_i[j];
            double rij_sq = d * d;
            if (rij_sq < r0_sq_155)
            {
                N1++;
                if (rij_sq < r0_sq_145)
                {
                    N0++;
                }
            }
        }

        int alpha[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        const double xi = x_ptr[i];
        const double yi = y_ptr[i];
        const double zi = z_ptr[i];

        const int *verlet_i = verlet_ptr + i * verlet_stride;

        for (int j = 0; j < N0; ++j)
        {
            int idx_j = verlet_i[j];

            // rij
            double rij_x = x_ptr[idx_j] - xi;
            double rij_y = y_ptr[idx_j] - yi;
            double rij_z = z_ptr[idx_j] - zi;

            box.pbc(rij_x, rij_y, rij_z);

            const double dist_j = dist_i[j];

            for (int k = j + 1; k < N0; ++k)
            {
                int idx_k = verlet_i[k];

                // rik
                double rik_x = x_ptr[idx_k] - xi;
                double rik_y = y_ptr[idx_k] - yi;
                double rik_z = z_ptr[idx_k] - zi;
                box.pbc(rik_x, rik_y, rik_z);

                double dot_product = rij_x * rik_x + rij_y * rik_y + rij_z * rik_z;
                double cos_theta = dot_product / (dist_j * dist_i[k]);

                if (cos_theta < -0.945)
                    alpha[0]++;
                else if (cos_theta < -0.915)
                    alpha[1]++;
                else if (cos_theta < -0.755)
                    alpha[2]++;
                else if (cos_theta < -0.195)
                    alpha[3]++;
                else if (cos_theta < 0.195)
                    alpha[4]++;
                else if (cos_theta < 0.245)
                    alpha[5]++;
                else if (cos_theta < 0.795)
                    alpha[6]++;
                else
                    alpha[7]++;
            }
        }

        double sigma_cp = std::abs(1.0 - alpha[6] / 24.0);
        int s56m4 = alpha[5] + alpha[6] - alpha[4];

        double sigma_bcc = sigma_cp + 1.0;
        if (s56m4 != 0)
        {
            sigma_bcc = 0.35 * alpha[4] / static_cast<double>(s56m4);
        }

        double sigma_fcc = 0.61 * (std::abs(alpha[0] + alpha[1] - 6) + alpha[2]) / 6.0;

        double sigma_hcp = (std::abs(alpha[0] - 3.0) +
                            std::abs(alpha[0] + alpha[1] + alpha[2] + alpha[3] - 9)) /
                           12.0;

        if (alpha[0] == 7)
            sigma_bcc = 0.0;
        else if (alpha[0] == 6)
            sigma_fcc = 0.0;
        else if (alpha[0] <= 3)
            sigma_hcp = 0.0;

        int structure_type;
        if (alpha[7] > 0)
        {
            structure_type = 0; // Other
        }
        else if (alpha[4] < 3)
        {
            if (N1 > 13 || N1 < 11)
                structure_type = 0; // Other
            else
                structure_type = 4; // ICO
        }
        else if (sigma_bcc <= sigma_cp)
        {
            if (N1 < 11)
                structure_type = 0; // Other
            else
                structure_type = 3; // BCC
        }
        else if (N1 > 12 || N1 < 11)
        {
            structure_type = 0; // Other
        }
        else
        {
            if (sigma_fcc < sigma_hcp)
                structure_type = 1; // FCC
            else
                structure_type = 2; // HCP
        }

        aja_ptr[i] = structure_type;
    }
}

NB_MODULE(_aja, m)
{
    m.def("compute_aja", &compute_aja);
}
