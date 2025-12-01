#include "box.h"
#include "type.h"
#include <ptm_functions.h>
#include <ptm_constants.h>
#include <ptm_initialize_data.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <omp.h>
#include <stdio.h>

struct PTMNeighborData
{
    const double *x;
    const double *y;
    const double *z;
    const int *verlet_list;
    int verlet_stride;
    Box box;
    int num_atoms;
    const int *atom_types;
    bool has_atom_types;
    const std::vector<uint64_t> *cached_neighbors; // ⭐ 新增：预排序的邻居
};

// ⭐ OVITO 风格的 get_neighbours 回调
// 关键区别：使用 cached_neighbors 和 decode_correspondences
int ptm_get_neighbours(
    void *vdata,
    size_t _unused_lammps_variable,
    size_t atom_index,
    int num_requested,
    ptm_atomicenv_t *env)
{
    PTMNeighborData *data = static_cast<PTMNeighborData *>(vdata);

    if (atom_index >= static_cast<size_t>(data->num_atoms))
    {
        return -1;
    }

    const int *neighbors = data->verlet_list + atom_index * data->verlet_stride;

    // 限制邻居数量
    int max_nbrs = std::min(num_requested - 1, PTM_MAX_INPUT_POINTS - 1);
    max_nbrs = std::min(max_nbrs, data->verlet_stride);

    // 收集邻居
    int num_neighbors = 0;
    double neighbor_points[PTM_MAX_INPUT_POINTS - 1][3];
    int neighbor_indices[PTM_MAX_INPUT_POINTS - 1];

    for (int i = 0; i < max_nbrs; ++i)
    {
        int nbr_idx = neighbors[i];

        if (nbr_idx < 0 || nbr_idx >= data->num_atoms)
        {
            break;
        }

        if (nbr_idx == static_cast<int>(atom_index))
        {
            continue;
        }

        // 计算相对坐标
        double dx = data->x[nbr_idx] - data->x[atom_index];
        double dy = data->y[nbr_idx] - data->y[atom_index];
        double dz = data->z[nbr_idx] - data->z[atom_index];
        data->box.pbc(dx, dy, dz);

        neighbor_points[num_neighbors][0] = dx;
        neighbor_points[num_neighbors][1] = dy;
        neighbor_points[num_neighbors][2] = dz;
        neighbor_indices[num_neighbors] = nbr_idx;
        num_neighbors++;
    }

    // ⭐ 关键：解码预排序的对应关系
    int dummy = 0;
    ptm_decode_correspondences(
        PTM_MATCH_FCC, // 默认行为
        (*data->cached_neighbors)[atom_index],
        env->correspondences,
        &dummy);

    // ⭐ 关键：按照 correspondences 重新排列邻居
    env->atom_indices[0] = atom_index;
    env->points[0][0] = 0;
    env->points[0][1] = 0;
    env->points[0][2] = 0;

    for (int i = 0; i < num_neighbors; i++)
    {
        int p = env->correspondences[i + 1] - 1; // ⭐ 使用对应关系索引
        if (p >= 0 && p < num_neighbors)
        {
            env->atom_indices[i + 1] = neighbor_indices[p];
            env->points[i + 1][0] = neighbor_points[p][0];
            env->points[i + 1][1] = neighbor_points[p][1];
            env->points[i + 1][2] = neighbor_points[p][2];
        }
    }

    // 设置原子类型
    if (data->has_atom_types)
    {
        env->numbers[0] = data->atom_types[atom_index];
        for (int i = 0; i < num_neighbors; i++)
        {
            int p = env->correspondences[i + 1] - 1;
            if (p >= 0 && p < num_neighbors)
            {
                env->numbers[i + 1] = data->atom_types[neighbor_indices[p]];
            }
        }
    }
    else
    {
        for (int i = 0; i < num_neighbors + 1; i++)
        {
            env->numbers[i] = 0;
        }
    }

    env->num = num_neighbors + 1;
    return num_neighbors + 1;
}

void get_ptm(const char *structure,
             const ROneArrayD x_py,
             const ROneArrayD y_py,
             const ROneArrayD z_py,
             const RTwoArrayD box_py,
             const ROneArrayD origin,
             const ROneArrayI boundary,
             const RTwoArrayI verlet_list_py,
             const ROneArrayI atom_types_py,
             const double rmsd_threshold,
             TwoArrayD output_py,
             TwoArrayI ptm_indices_py)
{
    const int N = static_cast<int>(x_py.shape(0));

    PTMNeighborData nbr_data;
    nbr_data.x = x_py.data();
    nbr_data.y = y_py.data();
    nbr_data.z = z_py.data();
    nbr_data.verlet_list = verlet_list_py.data();
    nbr_data.verlet_stride = static_cast<int>(verlet_list_py.shape(1));
    nbr_data.num_atoms = N;
    nbr_data.box = get_box(box_py, origin, boundary);

    nbr_data.has_atom_types = (atom_types_py.size() > 0 &&
                               atom_types_py.shape(0) == static_cast<size_t>(N));
    nbr_data.atom_types = nbr_data.has_atom_types ? atom_types_py.data() : nullptr;

    auto output = output_py.view();
    auto ptm_indices = ptm_indices_py.view();

    // 解析结构标志
    const char *strings[] = {"fcc", "hcp", "bcc", "ico", "sc",
                             "dcub", "dhex", "graphene", "all", "default"};
    const int32_t flags[] = {
        PTM_CHECK_FCC, PTM_CHECK_HCP, PTM_CHECK_BCC, PTM_CHECK_ICO, PTM_CHECK_SC,
        PTM_CHECK_DCUB, PTM_CHECK_DHEX, PTM_CHECK_GRAPHENE, PTM_CHECK_ALL,
        PTM_CHECK_FCC | PTM_CHECK_HCP | PTM_CHECK_BCC | PTM_CHECK_ICO};

    int input_flags = 0;
    const char *ptr = structure;
    while (*ptr != '\0')
    {
        if (*ptr == ' ' || *ptr == ',' || *ptr == '-' || *ptr == '_' || *ptr == '|')
        {
            ptr++;
            continue;
        }
        bool found = false;
        for (int i = 0; i < 10; i++)
        {
            int len = strlen(strings[i]);
            if (strncmp(ptr, strings[i], len) == 0)
            {
                char next_char = ptr[len];
                if (next_char == '\0' || next_char == ' ' || next_char == ',' ||
                    next_char == '-' || next_char == '_' || next_char == '|')
                {
                    input_flags |= flags[i];
                    ptr += len;
                    found = true;
                    break;
                }
            }
        }
        if (!found)
            ptr++;
    }

    if (input_flags == 0)
        input_flags = PTM_CHECK_FCC | PTM_CHECK_HCP | PTM_CHECK_BCC | PTM_CHECK_ICO;

    // 初始化 PTM
    ptm_initialize_global();

    // ⭐ 第一步：预排序所有原子的邻居（像 OVITO 一样）
    std::vector<uint64_t> cached_neighbors(N);
    nbr_data.cached_neighbors = &cached_neighbors;

    // #pragma omp parallel
    {
        ptm_local_handle_t local_handle = ptm_initialize_local();

        // #pragma omp for
        for (int i = 0; i < N; i++)
        {
            const int *neighbors = nbr_data.verlet_list + i * nbr_data.verlet_stride;

            // 收集邻居点
            int num_neighbors = 0;
            double neighbor_points[PTM_MAX_INPUT_POINTS - 1][3];

            for (int j = 0; j < nbr_data.verlet_stride; ++j)
            {
                int nbr_idx = neighbors[j];
                if (nbr_idx < 0 || nbr_idx >= N)
                    break;
                if (nbr_idx == i)
                    continue;

                double dx = nbr_data.x[nbr_idx] - nbr_data.x[i];
                double dy = nbr_data.y[nbr_idx] - nbr_data.y[i];
                double dz = nbr_data.z[nbr_idx] - nbr_data.z[i];
                nbr_data.box.pbc(dx, dy, dz);

                neighbor_points[num_neighbors][0] = dx;
                neighbor_points[num_neighbors][1] = dy;
                neighbor_points[num_neighbors][2] = dz;
                num_neighbors++;

                if (num_neighbors >= PTM_MAX_INPUT_POINTS - 1)
                    break;
            }

            // ⭐ 预排序邻居
            ptm_preorder_neighbours(local_handle, num_neighbors, neighbor_points, &cached_neighbors[i]);
        }

        ptm_uninitialize_local(local_handle);
    }

// ⭐ 第二步：使用预排序的邻居进行 PTM 分析
#pragma omp parallel
    {
        ptm_local_handle_t local_handle = ptm_initialize_local();

#pragma omp for
        for (int i = 0; i < N; i++)
        {
            int output_cols = static_cast<int>(output_py.shape(1));
            for (int k = 0; k < output_cols; k++)
            {
                output(i, k) = 0.0;
            }

            ptm_result_t result;
            memset(&result, 0, sizeof(ptm_result_t));

            ptm_atomicenv_t output_env;
            memset(&output_env, 0, sizeof(ptm_atomicenv_t));

            // 调用 PTM
            int ret = ptm_index(
                local_handle,
                i,
                ptm_get_neighbours,
                &nbr_data,
                input_flags,
                false,
                &result,
                &output_env);

            if (ret != 0 || result.rmsd > rmsd_threshold ||
                result.structure_type == PTM_MATCH_NONE)
            {
                result.structure_type = PTM_MATCH_NONE;
                result.ordering_type = PTM_ALLOY_NONE;
            }

            // 保存邻居索引
            int max_indices = static_cast<int>(ptm_indices_py.shape(1));
            int num_to_save = std::min(output_env.num, max_indices);
            for (int k = 0; k < num_to_save; k++)
            {
                ptm_indices(i, k) = static_cast<int>(output_env.atom_indices[k]);
            }
            for (int k = num_to_save; k < max_indices; k++)
            {
                ptm_indices(i, k) = -1;
            }

            output(i, 0) = static_cast<double>(result.structure_type);
            output(i, 1) = static_cast<double>(result.ordering_type);
            output(i, 2) = result.rmsd;
            output(i, 3) = result.interatomic_distance;
            output(i, 4) = result.orientation[0];
            output(i, 5) = result.orientation[1];
            output(i, 6) = result.orientation[2];
            output(i, 7) = result.orientation[3];
        }

        ptm_uninitialize_local(local_handle);
    }
}

NB_MODULE(_ptm, m)
{
    m.def("get_ptm", &get_ptm,
          nanobind::arg("structure"),
          nanobind::arg("x"),
          nanobind::arg("y"),
          nanobind::arg("z"),
          nanobind::arg("box"),
          nanobind::arg("origin"),
          nanobind::arg("boundary"),
          nanobind::arg("verlet_list"),
          nanobind::arg("atom_types"),
          nanobind::arg("rmsd_threshold"),
          nanobind::arg("output"),
          nanobind::arg("ptm_indices"),
          "Polyhedral Template Matching with OVITO-style neighbor ordering");
}