
#include "type.h"
#include "box.h"
#include <cmath>
#include <algorithm>
#include <omp.h>

void compute_cnp(
    const ROneArrayD x,
    const ROneArrayD y,
    const ROneArrayD z,
    const RTwoArrayD box_array,
    const ROneArrayD origin_array,
    const ROneArrayI boundary_array,
    const RTwoArrayI verlet_list,
    const RTwoArrayD distance_list,
    const ROneArrayI neighbor_number,
    OneArrayD cnp,
    const double rc)
{
    const int n_atoms = x.shape(0);

    Box box = get_box(box_array, origin_array, boundary_array);

    const double *x_ptr = x.data();
    const double *y_ptr = y.data();
    const double *z_ptr = z.data();
    const int *verlet_ptr = verlet_list.data();
    const double *dist_ptr = distance_list.data();
    const int *neighbor_num_ptr = neighbor_number.data();
    double *cnp_ptr = cnp.data();

    const int verlet_stride = verlet_list.shape(1);
    const int dist_stride = distance_list.shape(1);

// OpenMP并行
#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n_atoms; ++i)
    {
        int N = 0;
        double r_sum_sq = 0.0;

        const int neighbor_num_i = neighbor_num_ptr[i];
        const int *verlet_i = verlet_ptr + i * verlet_stride;
        const double *dist_i = dist_ptr + i * dist_stride;

        const double xi = x_ptr[i];
        const double yi = y_ptr[i];
        const double zi = z_ptr[i];

        // 遍历原子i的所有邻居
        for (int m = 0; m < neighbor_num_i; ++m)
        {
            int j = verlet_i[m];

            // 检查距离是否在截断半径内
            if (dist_i[m] <= rc)
            {
                N++;

                double rx = 0.0;
                double ry = 0.0;
                double rz = 0.0;

                const int neighbor_num_j = neighbor_num_ptr[j];
                const int *verlet_j = verlet_ptr + j * verlet_stride;
                const double *dist_j = dist_ptr + j * dist_stride;

                // 寻找共同邻居
                for (int s = 0; s < neighbor_num_j; ++s)
                {
                    int k_from_j = verlet_j[s];

                    // 检查k是否也是i的邻居
                    for (int h = 0; h < neighbor_num_i; ++h)
                    {
                        int k_from_i = verlet_i[h];

                        // 找到共同邻居k
                        if (k_from_j == k_from_i)
                        {
                            // 检查两个距离都在截断半径内
                            if (dist_j[s] <= rc && dist_i[h] <= rc)
                            {
                                int k = k_from_j;

                                const double xk = x_ptr[k];
                                const double yk = y_ptr[k];
                                const double zk = z_ptr[k];

                                const double xj = x_ptr[j];
                                const double yj = y_ptr[j];
                                const double zj = z_ptr[j];

                                // 计算rik = pos[i] - pos[k]
                                double rik_x = xi - xk;
                                double rik_y = yi - yk;
                                double rik_z = zi - zk;

                                // 计算rjk = pos[j] - pos[k]
                                double rjk_x = xj - xk;
                                double rjk_y = yj - yk;
                                double rjk_z = zj - zk;

                                // 应用周期边界条件
                                box.pbc(rik_x, rik_y, rik_z);
                                box.pbc(rjk_x, rjk_y, rjk_z);

                                // r += rik + rjk
                                rx += rik_x + rjk_x;
                                ry += rik_y + rjk_y;
                                rz += rik_z + rjk_z;
                            }
                            break; // 找到匹配的k后跳出h循环
                        }
                    }
                }

                // 累加 r.norm_sqr()
                r_sum_sq += rx * rx + ry * ry + rz * rz;
            }
        }

        // 计算最终的CNP值
        if (N > 0)
        {
            cnp_ptr[i] = r_sum_sq / N;
        }
        else
        {
            cnp_ptr[i] = 1000.0;
        }
    }
}

NB_MODULE(_cnp, m)
{
    m.def("compute_cnp", &compute_cnp);
}
