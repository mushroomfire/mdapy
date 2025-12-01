#include "type.h"
#include "box.h"
#include <cmath>
#include <omp.h>

void compute_bond(
    const ROneArrayD x_py,
    const ROneArrayD y_py,
    const ROneArrayD z_py,
    const RTwoArrayD box_py,
    const ROneArrayD origin,
    const ROneArrayI boundary,
    const RTwoArrayI verlet_list,
    const RTwoArrayD distance_list,
    const ROneArrayI neighbor_number,
    OneArrayI bond_length_distribution,
    OneArrayI bond_angle_distribution,
    const double delta_r,
    const double delta_theta,
    const double rc,
    const int nbins)
{
    Box box = get_box(box_py, origin, boundary);
    const double PI = 3.14159265358979323846;
    // 获取数据指针
    const double *x = x_py.data();
    const double *y = y_py.data();
    const double *z = z_py.data();
    const int *verlet = verlet_list.data();
    const double *distances = distance_list.data();
    const int *neigh_num = neighbor_number.data();
    int *bond_len_dist = bond_length_distribution.data();
    int *bond_ang_dist = bond_angle_distribution.data();

    int N = verlet_list.shape(0);
    int max_neigh = verlet_list.shape(1);

    double delta_r_inv = 1.0 / delta_r;
    double delta_theta_inv = 1.0 / delta_theta;

// 并行计算键长和键角分布
#pragma omp parallel
    {
        // 每个线程有自己的局部数组,避免race condition
        int *local_bond_len = new int[nbins]();
        int *local_bond_ang = new int[nbins]();

#pragma omp for schedule(dynamic)
        for (int i = 0; i < N; ++i)
        {
            int i_neigh = neigh_num[i];

            // 计算键长分布
            for (int jj = 0; jj < i_neigh; ++jj)
            {
                int j = verlet[i * max_neigh + jj];
                if (j > i)
                {
                    double r = distances[i * max_neigh + jj];
                    if (r <= rc)
                    {
                        int index = static_cast<int>(std::floor(r * delta_r_inv));
                        if (index > nbins - 1)
                            index = nbins - 1;
                        local_bond_len[index] += 1;
                    }
                }
            }

            // 计算键角分布
            for (int jj = 0; jj < i_neigh; ++jj)
            {
                int j = verlet[i * max_neigh + jj];
                double r_ij = distances[i * max_neigh + jj];

                if (r_ij <= rc)
                {
                    for (int kk = jj + 1; kk < i_neigh; ++kk)
                    {
                        int k = verlet[i * max_neigh + kk];
                        double r_ik = distances[i * max_neigh + kk];

                        if (r_ik <= rc)
                        {
                            // 计算向量 rij 和 rik
                            double rij_x = x[j] - x[i];
                            double rij_y = y[j] - y[i];
                            double rij_z = z[j] - z[i];

                            double rik_x = x[k] - x[i];
                            double rik_y = y[k] - y[i];
                            double rik_z = z[k] - z[i];

                            // 应用周期性边界条件
                            box.pbc(rij_x, rij_y, rij_z);
                            box.pbc(rik_x, rik_y, rik_z);

                            // 计算点积
                            double dot_product = rij_x * rik_x + rij_y * rik_y + rij_z * rik_z;

                            // 计算夹角(转换为度数)
                            double cos_theta = dot_product / (r_ij * r_ik);
                            // 防止数值误差导致cos_theta超出[-1, 1]范围
                            if (cos_theta > 1.0)
                                cos_theta = 1.0;
                            if (cos_theta < -1.0)
                                cos_theta = -1.0;

                            double theta = std::acos(cos_theta) * 180.0 / PI;

                            int index = static_cast<int>(std::floor(theta * delta_theta_inv));
                            if (index > nbins - 1)
                                index = nbins - 1;
                            local_bond_ang[index] += 1;
                        }
                    }
                }
            }
        }

// 将局部结果合并到全局数组
#pragma omp critical
        {
            for (int i = 0; i < nbins; ++i)
            {
                bond_len_dist[i] += local_bond_len[i];
                bond_ang_dist[i] += local_bond_ang[i];
            }
        }

        delete[] local_bond_len;
        delete[] local_bond_ang;
    }
}

void compute_adf(
    const ROneArrayD x_py,
    const ROneArrayD y_py,
    const ROneArrayD z_py,
    const RTwoArrayD box_py,
    const ROneArrayD origin,
    const ROneArrayI boundary,
    const RTwoArrayI verlet_list,
    const RTwoArrayD distance_list,
    const ROneArrayI neighbor_number,
    const double delta_theta,
    const RTwoArrayD rc_list,
    const RTwoArrayI pair_list,
    const ROneArrayI type_list,
    const int nbins,
    TwoArrayI bond_angle_distribution)
{
    Box box = get_box(box_py, origin, boundary);
    const double PI = 3.14159265358979323846;

    const double *x = x_py.data();
    const double *y = y_py.data();
    const double *z = z_py.data();
    const int *verlet = verlet_list.data();
    const double *distances = distance_list.data();
    const int *neigh_num = neighbor_number.data();
    const double *rc = rc_list.data();
    const int *pairs = pair_list.data();
    const int *types = type_list.data();
    int *bond_ang_dist = bond_angle_distribution.data();

    int N = verlet_list.shape(0);
    int max_neigh = verlet_list.shape(1);
    int Npair = pair_list.shape(0);
    const int rc_cols = 4;

    double delta_theta_inv = 1.0 / delta_theta;

#pragma omp parallel
    {
        int *local_dist = new int[Npair * nbins]();

#pragma omp for schedule(dynamic)
        for (int i = 0; i < N; ++i)
        {
            int itype = types[i];

            for (int m = 0; m < Npair; ++m)
            {
                if (itype == pairs[m * 3 + 0])
                {
                    int i_neigh = neigh_num[i];
                    int jtype_target = pairs[m * 3 + 1];
                    int ktype_target = pairs[m * 3 + 2];
                    bool same_type = (jtype_target == ktype_target);

                    for (int jj = 0; jj < i_neigh; ++jj)
                    {
                        int j = verlet[i * max_neigh + jj];
                        int jtype = types[j];

                        if (jtype == jtype_target)
                        {
                            double r_ij = distances[i * max_neigh + jj];

                            if (r_ij <= rc[m * rc_cols + 1] && r_ij >= rc[m * rc_cols + 0])
                            {
                                // 如果类型相同，从 jj+1 开始避免重复计算
                                // 如果类型不同，需要遍历所有 k
                                int kk_start = same_type ? (jj + 1) : 0;

                                for (int kk = kk_start; kk < i_neigh; ++kk)
                                {
                                    if (kk == jj)
                                        continue; // 跳过 j == k 的情况

                                    int k = verlet[i * max_neigh + kk];
                                    int ktype = types[k];

                                    if (ktype == ktype_target)
                                    {
                                        double r_ik = distances[i * max_neigh + kk];

                                        if (r_ik <= rc[m * rc_cols + 3] && r_ik >= rc[m * rc_cols + 2])
                                        {
                                            double rij_x = x[j] - x[i];
                                            double rij_y = y[j] - y[i];
                                            double rij_z = z[j] - z[i];

                                            double rik_x = x[k] - x[i];
                                            double rik_y = y[k] - y[i];
                                            double rik_z = z[k] - z[i];

                                            box.pbc(rij_x, rij_y, rij_z);
                                            box.pbc(rik_x, rik_y, rik_z);

                                            double dot_product = rij_x * rik_x + rij_y * rik_y + rij_z * rik_z;
                                            double cos_theta = dot_product / (r_ij * r_ik);

                                            // 防止数值误差
                                            if (cos_theta > 1.0)
                                                cos_theta = 1.0;
                                            if (cos_theta < -1.0)
                                                cos_theta = -1.0;

                                            double theta = std::acos(cos_theta) * 180.0 / PI;

                                            int index = static_cast<int>(std::floor(theta * delta_theta_inv));

                                            // 边界检查
                                            if (index < 0)
                                                index = 0;
                                            if (index >= nbins)
                                                index = nbins - 1;

                                            local_dist[m * nbins + index] += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

#pragma omp critical
        {
            for (int m = 0; m < Npair; ++m)
            {
                for (int i = 0; i < nbins; ++i)
                {
                    bond_ang_dist[m * nbins + i] += local_dist[m * nbins + i];
                }
            }
        }

        delete[] local_dist;
    }
}

NB_MODULE(_bond_analysis, m)
{
    m.def("compute_bond", &compute_bond);
    m.def("compute_adf", &compute_adf);
}