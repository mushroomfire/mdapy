#include "type.h"
#include <cmath>
#include <omp.h>

void compute_temp(
    const RTwoArrayI verlet_list,
    const RTwoArrayD distance_list,
    const ROneArrayD vx_py,
    const ROneArrayD vy_py,
    const ROneArrayD vz_py,
    const ROneArrayD mass_list,
    OneArrayD T,
    const double rc)
{
    // 物理常数
    constexpr double kb = 1.380649e-23;                // Boltzmann constant (J/K)
    constexpr double dim = 3.0;                        // Dimensions
    constexpr double afu = 6.022140857e23;             // Avogadro's number (1/mol)
    constexpr double mass_factor = 1.0 / afu / 1000.0; // g/mol -> kg conversion

    const int n_atoms = verlet_list.shape(0);
    const int max_neigh = verlet_list.shape(1);

    // 获取数组指针
    const int *verlet_ptr = verlet_list.data();
    const double *dist_ptr = distance_list.data();
    const double *vx = vx_py.data();
    const double *vy = vy_py.data();
    const double *vz = vz_py.data();
    const double *mass_ptr = mass_list.data();
    double *T_ptr = T.data();

// OpenMP并行
#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n_atoms; ++i)
    {
        const int *verlet_i = verlet_ptr + i * max_neigh;
        const double *dist_i = dist_ptr + i * max_neigh;
        const double mass_i = mass_ptr[i];

        // 计算邻居的质心速度
        double v_neigh_x = vx[i] * mass_i;
        double v_neigh_y = vy[i] * mass_i;
        double v_neigh_z = vz[i] * mass_i;

        int n_neigh = 1;
        double mass_neigh = mass_i;

        // 累加所有邻居的动量
        for (int j_index = 0; j_index < max_neigh; ++j_index)
        {
            int j = verlet_i[j_index];
            if (j < 0)
                break;

            if (j != i && dist_i[j_index] <= rc)
            {
                const double mass_j = mass_ptr[j];

                v_neigh_x += vx[j] * mass_j;
                v_neigh_y += vy[j] * mass_j;
                v_neigh_z += vz[j] * mass_j;

                n_neigh++;
                mass_neigh += mass_j;
            }
        }

        // 计算质心速度
        const double v_mean_x = v_neigh_x / mass_neigh;
        const double v_mean_y = v_neigh_y / mass_neigh;
        const double v_mean_z = v_neigh_z / mass_neigh;

        // 计算相对于质心的动能
        // 速度单位: A/ps, 质量单位: g/mol
        // 需要转换: v^2 (A/ps)^2 = v^2 * 1e-20 m^2 / 1e-24 s^2 = v^2 * 1e4 m^2/s^2
        constexpr double vel_conv = 1e4; // (A/ps)^2 -> (m/s)^2

        double ke_neigh = 0.0;

        // 原子i的贡献
        double dv_x = vx[i] - v_mean_x;
        double dv_y = vy[i] - v_mean_y;
        double dv_z = vz[i] - v_mean_z;
        double v_sq = dv_x * dv_x + dv_y * dv_y + dv_z * dv_z;
        ke_neigh += 0.5 * mass_i * mass_factor * v_sq * vel_conv;

        // 邻居的贡献
        for (int j_index = 0; j_index < max_neigh; ++j_index)
        {
            int j = verlet_i[j_index];
            if (j < 0)
                break;

            if (j != i && dist_i[j_index] <= rc)
            {
                const double mass_j = mass_ptr[j];

                dv_x = vx[j] - v_mean_x;
                dv_y = vy[j] - v_mean_y;
                dv_z = vz[j] - v_mean_z;
                v_sq = dv_x * dv_x + dv_y * dv_y + dv_z * dv_z;

                ke_neigh += 0.5 * mass_j * mass_factor * v_sq * vel_conv;
            }
        }

        // 计算温度: T = 2*KE / (dim * N * kb)
        T_ptr[i] = ke_neigh * 2.0 / (dim * n_neigh * kb);
    }
}

NB_MODULE(_atomtemp, m)
{
    m.def("compute_temp", &compute_temp);
}
