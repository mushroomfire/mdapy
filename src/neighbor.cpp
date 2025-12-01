#include "box.h"
#include "type.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <cmath>
#include <vector>
#include <atomic>
#include <omp.h>
#include <algorithm>
#include <stdio.h>

namespace nb = nanobind;

// 优化的 mod 函数
inline int mod(const int a, const int b)
{
    int res = a % b;
    return res + (res >> 31 & b);
}

inline int get_index(const int icel, const int jcel, const int kcel, const int *ncell)
{
    return icel * ncell[1] * ncell[2] + jcel * ncell[2] + kcel;
}

// 统一的获取cell索引函数
inline void get_cell_index(double x, double y, double z,
                           const Box &box, const double rc_inverse,
                           const int *ncell, int &icel, int &jcel, int &kcel)
{
    double xlo = box.origin[0];
    double ylo = box.origin[1];
    double zlo = box.origin[2];

    if (box.triclinic)
    {
        double dx = x - xlo;
        double dy = y - ylo;
        double dz = z - zlo;
        double nx = dx * box.data[9] + dy * box.data[12] + dz * box.data[15];
        double ny = dx * box.data[10] + dy * box.data[13] + dz * box.data[16];
        double nz = dx * box.data[11] + dy * box.data[14] + dz * box.data[17];

        icel = static_cast<int>(std::floor(nx * box.thickness[0] * rc_inverse));
        jcel = static_cast<int>(std::floor(ny * box.thickness[1] * rc_inverse));
        kcel = static_cast<int>(std::floor(nz * box.thickness[2] * rc_inverse));
    }
    else
    {
        icel = static_cast<int>(std::floor((x - xlo) * rc_inverse));
        jcel = static_cast<int>(std::floor((y - ylo) * rc_inverse));
        kcel = static_cast<int>(std::floor((z - zlo) * rc_inverse));
    }

    // 钳制到有效范围
    icel = std::max(0, std::min(icel, ncell[0] - 1));
    jcel = std::max(0, std::min(jcel, ncell[1] - 1));
    kcel = std::max(0, std::min(kcel, ncell[2] - 1));
}

void build_cell(const ROneArrayD x_py,
                const ROneArrayD y_py,
                const ROneArrayD z_py,
                int *atom_cell_list,
                int *cell_id_list,
                const Box &box,
                const int *ncell,
                const double rc)
{
    auto x = x_py.view();
    auto y = y_py.view();
    auto z = z_py.view();
    const int N{static_cast<int>(x.shape(0))};
    const double rc_inverse{1.0 / rc};

    for (int i = 0; i < N; ++i)
    {
        double xi = x(i);
        double yi = y(i);
        double zi = z(i);

        // 关键修复：如果有周期边界条件，先将粒子映射回盒子内
        // 然后再计算单元格索引
        if (box.boundary[0] || box.boundary[1] || box.boundary[2])
        {
            box.wrap_into_box(xi, yi, zi);
        }

        int icel, jcel, kcel;
        // 对 wrap 后的坐标调用 get_cell_index
        get_cell_index(xi, yi, zi, box, rc_inverse, ncell, icel, jcel, kcel);

        int index = get_index(icel, jcel, kcel, ncell);
        atom_cell_list[i] = cell_id_list[index];
        cell_id_list[index] = i;
    }
}

void build_verlet_list(const ROneArrayD x_py,
                       const ROneArrayD y_py,
                       const ROneArrayD z_py,
                       const int *atom_cell_list,
                       const int *cell_id_list,
                       const Box &box,
                       const int *ncell,
                       const double rc,
                       TwoArrayI verlet_list_py,
                       TwoArrayD distance_list_py,
                       OneArrayI neighbor_number_py)
{
    auto x = x_py.view();
    auto y = y_py.view();
    auto z = z_py.view();
    auto verlet_list = verlet_list_py.view();
    auto distance_list = distance_list_py.view();
    auto neighbor_number = neighbor_number_py.view();
    const int N{static_cast<int>(x.shape(0))};
    const double rc_inverse{1.0 / rc};
    const double rcsq{rc * rc};

#pragma omp parallel for firstprivate(x, y, z, verlet_list, distance_list, neighbor_number)
    for (int i = 0; i < N; ++i)
    {
        int nindex{0};
        double xi = x(i);
        double yi = y(i);
        double zi = z(i);

        // 也要 wrap i 的坐标
        if (box.boundary[0] || box.boundary[1] || box.boundary[2])
        {
            box.wrap_into_box(xi, yi, zi);
        }

        int icel, jcel, kcel;
        get_cell_index(xi, yi, zi, box, rc_inverse, ncell, icel, jcel, kcel);

        for (int iicel = icel - 1; iicel <= icel + 1; ++iicel)
        {
            for (int jjcel = jcel - 1; jjcel <= jcel + 1; ++jjcel)
            {
                for (int kkcel = kcel - 1; kkcel <= kcel + 1; ++kkcel)
                {
                    int index = get_index(
                        mod(iicel, ncell[0]),
                        mod(jjcel, ncell[1]),
                        mod(kkcel, ncell[2]),
                        ncell);
                    int j = cell_id_list[index];

                    while (j > -1)
                    {
                        if (j != i)
                        {
                            double xij = x(j) - xi;
                            double yij = y(j) - yi;
                            double zij = z(j) - zi;
                            box.pbc(xij, yij, zij);
                            double dis_sq = xij * xij + yij * yij + zij * zij;

                            if (dis_sq <= rcsq)
                            {
                                verlet_list(i, nindex) = j;
                                distance_list(i, nindex) = std::sqrt(dis_sq);
                                ++nindex;
                            }
                        }
                        j = atom_cell_list[j];
                    }
                }
            }
        }
        neighbor_number(i) = nindex;
    }
}

auto build_neighbor_without_max_neigh(
    const ROneArrayD x_py,
    const ROneArrayD y_py,
    const ROneArrayD z_py,
    const RTwoArrayD box_py,
    const ROneArrayD origin,
    const ROneArrayI boundary,
    const double rc)
{
    Box box = get_box(box_py, origin, boundary);
    int ncell[3]{};
    const int N{static_cast<int>(x_py.shape(0))};

    for (int i = 0; i < 3; ++i)
    {
        ncell[i] = std::max(static_cast<int>(std::floor(box.thickness[i] / rc)), 3);
    }

    const int total_cell = ncell[0] * ncell[1] * ncell[2];
    int *atom_cell_list = new int[N]{};
    int *cell_id_list = new int[total_cell];

#pragma omp parallel for
    for (int i = 0; i < total_cell; ++i)
    {
        cell_id_list[i] = -1;
    }

    build_cell(x_py, y_py, z_py, atom_cell_list, cell_id_list, box, ncell, rc);

    // 使用 vector 动态存储每个粒子的邻居
    std::vector<std::vector<int>> neighbor_lists(N);
    std::vector<std::vector<double>> distance_lists(N);

    auto x = x_py.view();
    auto y = y_py.view();
    auto z = z_py.view();
    const double rc_inverse{1.0 / rc};
    const double rcsq{rc * rc};

    // 使用 vector 动态存储
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        double xi = x(i);
        double yi = y(i);
        double zi = z(i);

        // 也要 wrap i 的坐标
        if (box.boundary[0] || box.boundary[1] || box.boundary[2])
        {
            box.wrap_into_box(xi, yi, zi);
        }

        int icel, jcel, kcel;
        get_cell_index(xi, yi, zi, box, rc_inverse, ncell, icel, jcel, kcel);

        // 预分配一些空间以减少重新分配
        neighbor_lists[i].reserve(150);
        distance_lists[i].reserve(150);

        for (int iicel = icel - 1; iicel <= icel + 1; ++iicel)
        {
            for (int jjcel = jcel - 1; jjcel <= jcel + 1; ++jjcel)
            {
                for (int kkcel = kcel - 1; kkcel <= kcel + 1; ++kkcel)
                {
                    int index = get_index(
                        mod(iicel, ncell[0]),
                        mod(jjcel, ncell[1]),
                        mod(kkcel, ncell[2]),
                        ncell);
                    int j = cell_id_list[index];

                    while (j > -1)
                    {
                        if (j != i)
                        {
                            double xij = x(j) - xi;
                            double yij = y(j) - yi;
                            double zij = z(j) - zi;
                            box.pbc(xij, yij, zij);
                            double dis_sq = xij * xij + yij * yij + zij * zij;

                            if (dis_sq <= rcsq)
                            {
                                neighbor_lists[i].push_back(j);
                                distance_lists[i].push_back(std::sqrt(dis_sq));
                            }
                        }
                        j = atom_cell_list[j];
                    }
                }
            }
        }
    }

    delete[] atom_cell_list;
    delete[] cell_id_list;

    // 找到实际的最大邻居数
    int max_neigh = 0;
    for (int i = 0; i < N; ++i)
    {
        int n_neigh = static_cast<int>(neighbor_lists[i].size());
        if (n_neigh > max_neigh)
        {
            max_neigh = n_neigh;
        }
    }

    // 如果没有邻居，设置最小值为1以避免空数组
    if (max_neigh == 0)
    {
        max_neigh = 1;
    }

    // 分配最终的规则数组
    int *neighbor_number_data = new int[N];
    int *verlet_list_data = new int[N * max_neigh];
    double *distance_list_data = new double[N * max_neigh];

    nb::capsule neighbor_owner(neighbor_number_data, [](void *p) noexcept
                               { delete[] (int *)p; });
    nb::capsule verlet_owner(verlet_list_data, [](void *p) noexcept
                             { delete[] (int *)p; });
    nb::capsule distance_owner(distance_list_data, [](void *p) noexcept
                               { delete[] (double *)p; });

    // 初始化数组（填充无效值）
    const double rc_plus = rc + 1.0;
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < max_neigh; ++j)
        {
            verlet_list_data[i * max_neigh + j] = -1;
            distance_list_data[i * max_neigh + j] = rc_plus;
        }
    }

    // 从 vector 复制到最终数组
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        int n_neigh = static_cast<int>(neighbor_lists[i].size());
        neighbor_number_data[i] = n_neigh;

        for (int j = 0; j < n_neigh; ++j)
        {
            verlet_list_data[i * max_neigh + j] = neighbor_lists[i][j];
            distance_list_data[i * max_neigh + j] = distance_lists[i][j];
        }
    }

    return std::make_tuple(
        nb::ndarray<nb::numpy, int>(verlet_list_data, {static_cast<size_t>(N), static_cast<size_t>(max_neigh)}, verlet_owner),
        nb::ndarray<nb::numpy, double>(distance_list_data, {static_cast<size_t>(N), static_cast<size_t>(max_neigh)}, distance_owner),
        nb::ndarray<nb::numpy, int>(neighbor_number_data, {static_cast<size_t>(N)}, neighbor_owner));
}

void build_neighbor(const ROneArrayD x_py,
                    const ROneArrayD y_py,
                    const ROneArrayD z_py,
                    const RTwoArrayD box_py,
                    const ROneArrayD origin,
                    const ROneArrayI boundary,
                    const double rc,
                    TwoArrayI verlet_list_py,
                    TwoArrayD distance_list_py,
                    OneArrayI neighbor_number_py)
{
    Box box = get_box(box_py, origin, boundary);
    int ncell[3]{};
    const int N{static_cast<int>(x_py.shape(0))};

    for (int i = 0; i < 3; ++i)
    {
        ncell[i] = std::max(static_cast<int>(std::floor(box.thickness[i] / rc)), 3);
    }
    const int total_cell = ncell[0] * ncell[1] * ncell[2];
    int *atom_cell_list = new int[N]{};
    int *cell_id_list = new int[total_cell];

#pragma omp parallel for
    for (int i = 0; i < total_cell; ++i)
    {
        cell_id_list[i] = -1;
    }

    build_cell(x_py, y_py, z_py, atom_cell_list, cell_id_list, box, ncell, rc);
    build_verlet_list(x_py, y_py, z_py, atom_cell_list,
                      cell_id_list, box, ncell, rc,
                      verlet_list_py, distance_list_py, neighbor_number_py);

    delete[] atom_cell_list;
    delete[] cell_id_list;
}

auto filter_overlap_atom(const ROneArrayD x_py,
                         const ROneArrayD y_py,
                         const ROneArrayD z_py,
                         const RTwoArrayD box_py,
                         const ROneArrayD origin,
                         const ROneArrayI boundary,
                         const double rc)
{
    Box box = get_box(box_py, origin, boundary);
    int ncell[3]{};
    const int N{static_cast<int>(x_py.shape(0))};

    for (int i = 0; i < 3; ++i)
    {
        ncell[i] = std::max(static_cast<int>(std::floor(box.thickness[i] / rc)), 3);
    }
    const int total_cell = ncell[0] * ncell[1] * ncell[2];
    int *atom_cell_list = new int[N]{};
    int *cell_id_list = new int[total_cell];

#pragma omp parallel for
    for (int i = 0; i < total_cell; ++i)
    {
        cell_id_list[i] = -1;
    }

    build_cell(x_py, y_py, z_py, atom_cell_list, cell_id_list, box, ncell, rc);
    auto x = x_py.view();
    auto y = y_py.view();
    auto z = z_py.view();

    const double rc_inverse{1.0 / rc};
    const double rcsq{rc * rc};
    bool *filter_data = new bool[N];

    nb::capsule filter_owner(filter_data, [](void *p) noexcept
                             { delete[] (bool *)p; });
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        filter_data[i] = true;
    }

#pragma omp parallel for firstprivate(x, y, z)
    for (int i = 0; i < N; ++i)
    {
        double xi = x(i);
        double yi = y(i);
        double zi = z(i);

        // 也要 wrap i 的坐标
        if (box.boundary[0] || box.boundary[1] || box.boundary[2])
        {
            box.wrap_into_box(xi, yi, zi);
        }

        int icel, jcel, kcel;
        get_cell_index(xi, yi, zi, box, rc_inverse, ncell, icel, jcel, kcel);

        for (int iicel = icel - 1; iicel <= icel + 1; ++iicel)
        {
            for (int jjcel = jcel - 1; jjcel <= jcel + 1; ++jjcel)
            {
                for (int kkcel = kcel - 1; kkcel <= kcel + 1; ++kkcel)
                {
                    int index = get_index(
                        mod(iicel, ncell[0]),
                        mod(jjcel, ncell[1]),
                        mod(kkcel, ncell[2]),
                        ncell);
                    int j = cell_id_list[index];

                    while (j > -1)
                    {
                        if (j > i) // 只检查一次，标记较大索引的原子
                        {
                            double xij = x(j) - xi;
                            double yij = y(j) - yi;
                            double zij = z(j) - zi;
                            box.pbc(xij, yij, zij);
                            double dis_sq = xij * xij + yij * yij + zij * zij;
                            if (dis_sq <= rcsq)
                            {
                                filter_data[j] = false;
                            }
                        }
                        j = atom_cell_list[j];
                    }
                }
            }
        }
    }
    delete[] atom_cell_list;
    delete[] cell_id_list;

    return nb::ndarray<nb::numpy, bool>(filter_data, {static_cast<size_t>(N)}, filter_owner);
}

auto filter_overlap_atom_with_grain(
    const ROneArrayD x_py,
    const ROneArrayD y_py,
    const ROneArrayD z_py,
    const ROneArrayI type_py,     // 1=金属, 2=碳
    const ROneArrayI grain_id_py, // 晶粒ID
    const RTwoArrayD box_py,
    const ROneArrayD origin,
    const ROneArrayI boundary,
    const double rc_metal_metal, // Metal-Metal 截断距离
    const double rc_cc,          // C-C 截断距离
    const double rc_metal_c)     // Metal-C 截断距离
{
    Box box = get_box(box_py, origin, boundary);
    int ncell[3]{};
    const int N{static_cast<int>(x_py.shape(0))};

    // 使用最大截断距离构建 cell
    const double rc_max = std::max({rc_metal_metal, rc_cc, rc_metal_c});

    for (int i = 0; i < 3; ++i)
    {
        ncell[i] = std::max(static_cast<int>(std::floor(box.thickness[i] / rc_max)), 3);
    }

    const int total_cell = ncell[0] * ncell[1] * ncell[2];
    int *atom_cell_list = new int[N]{};
    int *cell_id_list = new int[total_cell];

#pragma omp parallel for
    for (int i = 0; i < total_cell; ++i)
    {
        cell_id_list[i] = -1;
    }

    build_cell(x_py, y_py, z_py, atom_cell_list, cell_id_list, box, ncell, rc_max);

    auto x = x_py.view();
    auto y = y_py.view();
    auto z = z_py.view();
    auto type = type_py.view();
    auto grain_id = grain_id_py.view();

    const double rc_max_inverse{1.0 / rc_max};
    const double rcsq_metal_metal{rc_metal_metal * rc_metal_metal};
    const double rcsq_cc{rc_cc * rc_cc};
    const double rcsq_metal_c{rc_metal_c * rc_metal_c};

    bool *filter_data = new bool[N];
    nb::capsule filter_owner(filter_data, [](void *p) noexcept
                             { delete[] (bool *)p; });

    // 初始化全部保留
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        filter_data[i] = true;
    }

    // 使用原子锁防止竞争条件
    std::vector<std::atomic<bool>> atom_removed(N);
    for (int i = 0; i < N; ++i)
    {
        atom_removed[i].store(false);
    }

#pragma omp parallel for firstprivate(x, y, z, type, grain_id)
    for (int i = 0; i < N; ++i)
    {
        // 如果原子已被标记删除，跳过
        if (atom_removed[i].load())
        {
            continue;
        }

        double xi = x(i);
        double yi = y(i);
        double zi = z(i);
        int type_i = type(i);
        int grain_i = grain_id(i);

        if (box.boundary[0] || box.boundary[1] || box.boundary[2])
        {
            box.wrap_into_box(xi, yi, zi);
        }

        int icel, jcel, kcel;
        get_cell_index(xi, yi, zi, box, rc_max_inverse, ncell, icel, jcel, kcel);

        for (int iicel = icel - 1; iicel <= icel + 1; ++iicel)
        {
            for (int jjcel = jcel - 1; jjcel <= jcel + 1; ++jjcel)
            {
                for (int kkcel = kcel - 1; kkcel <= kcel + 1; ++kkcel)
                {
                    int index = get_index(
                        mod(iicel, ncell[0]),
                        mod(jjcel, ncell[1]),
                        mod(kkcel, ncell[2]),
                        ncell);
                    int j = cell_id_list[index];

                    while (j > -1)
                    {
                        if (j > i && !atom_removed[j].load())
                        {
                            int type_j = type(j);
                            int grain_j = grain_id(j);

                            double xij = x(j) - xi;
                            double yij = y(j) - yi;
                            double zij = z(j) - zi;
                            box.pbc(xij, yij, zij);
                            double dis_sq = xij * xij + yij * yij + zij * zij;

                            bool too_close = false;
                            int atom_to_remove = -1;

                            // 判断原子对类型和应用对应的截断距离
                            if (type_i == 1 && type_j == 1)
                            {
                                // Metal-Metal: 删除索引较大的金属原子
                                if (dis_sq <= rcsq_metal_metal)
                                {
                                    too_close = true;
                                    atom_to_remove = j;
                                }
                            }
                            else if (type_i == 2 && type_j == 2)
                            {
                                // C-C: 检查是否来自不同晶粒的石墨烯
                                if (grain_i != grain_j && dis_sq <= rcsq_cc)
                                {
                                    too_close = true;
                                    // 删除 grain_id 较大的碳原子
                                    atom_to_remove = (grain_i > grain_j) ? i : j;
                                }
                                // 如果是同一晶粒的碳原子靠太近，也删除（理论上不应该发生）
                                else if (grain_i == grain_j && dis_sq <= rcsq_cc)
                                {
                                    too_close = true;
                                    atom_to_remove = j; // 删除索引较大的
                                }
                            }
                            else if (type_i != type_j)
                            {
                                // Metal-C: 删除金属原子
                                if (dis_sq <= rcsq_metal_c)
                                {
                                    too_close = true;
                                    if (type_i == 1)
                                    {
                                        atom_to_remove = i;
                                    }
                                    else
                                    {
                                        atom_to_remove = j;
                                    }
                                }
                            }

                            if (too_close && atom_to_remove >= 0)
                            {
                                bool expected = false;
                                if (atom_removed[atom_to_remove].compare_exchange_strong(
                                        expected, true))
                                {
                                    filter_data[atom_to_remove] = false;
                                }
                            }
                        }
                        j = atom_cell_list[j];
                    }
                }
            }
        }
    }

    delete[] atom_cell_list;
    delete[] cell_id_list;

    return nb::ndarray<nb::numpy, bool>(filter_data, {static_cast<size_t>(N)}, filter_owner);
}

// ========== 新增：批量 wrap 坐标到盒子内 ==========
void wrap_positions(OneArrayD x_py,
                    OneArrayD y_py,
                    OneArrayD z_py,
                    const RTwoArrayD box_py,
                    const ROneArrayD origin,
                    const ROneArrayI boundary)
{
    Box box = get_box(box_py, origin, boundary);
    auto x = x_py.view();
    auto y = y_py.view();
    auto z = z_py.view();
    const int N{static_cast<int>(x.shape(0))};

#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        double xi = x(i);
        double yi = y(i);
        double zi = z(i);

        box.wrap_into_box(xi, yi, zi);

        x(i) = xi;
        y(i) = yi;
        z(i) = zi;
    }
}

void average_by_neighbor(const double rc,
                         const RTwoArrayI verlet_list_py,
                         const RTwoArrayD distance_list_py,
                         const ROneArrayI neighbor_number_py,
                         const ROneArrayD value_py,
                         OneArrayD value_ave_py,
                         const bool include_self)
{
    auto verlet_list = verlet_list_py.view();
    auto distance_list = distance_list_py.view();
    auto neighbor_number = neighbor_number_py.view();
    auto value = value_py.view();
    auto value_ave = value_ave_py.view();
    const int N{static_cast<int>(value.shape(0))};

#pragma omp parallel for firstprivate(verlet_list, distance_list, neighbor_number, value, value_ave)
    for (int i = 0; i < N; ++i)
    {
        double sum{0.0};
        int n_neigh{0};

        if (include_self)
        {
            sum += value(i);
            ++n_neigh;
        }

        for (int j = 0; j < neighbor_number(i); ++j)
        {
            if (distance_list(i, j) <= rc)
            {
                sum += value(verlet_list(i, j));
                ++n_neigh;
            }
        }

        value_ave(i) = (n_neigh > 0) ? sum / n_neigh : 0.0;
    }
}

void sort_verlet_by_distance(
    TwoArrayI verlet_list_py,
    TwoArrayD distance_list_py,
    const int sortNum)
{
    auto verlet_list = verlet_list_py.view();
    auto distance_list = distance_list_py.view();

    const int N{static_cast<int>(verlet_list.shape(0))};
    const int NumNeigh{static_cast<int>(verlet_list.shape(1))};

#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        int effective_sort = std::min(sortNum, NumNeigh);
        for (int j = 0; j < effective_sort; ++j)
        {
            int minIndex = j;
            for (int k = j + 1; k < NumNeigh; ++k)
            {
                if (distance_list(i, k) < distance_list(i, minIndex))
                {
                    minIndex = k;
                }
            }
            if (minIndex != j)
            {
                std::swap(distance_list(i, j), distance_list(i, minIndex));
                std::swap(verlet_list(i, j), verlet_list(i, minIndex));
            }
        }
    }
}

auto _fill_cell_for_void(const ROneArrayD x_py,
                         const ROneArrayD y_py,
                         const ROneArrayD z_py,
                         const RTwoArrayD box_py,
                         const ROneArrayD origin,
                         const ROneArrayI boundary,
                         const double rc)
{
    Box box = get_box(box_py, origin, boundary);
    int ncell[3]{};
    const int N{static_cast<int>(x_py.shape(0))};
    auto x = x_py.view();
    auto y = y_py.view();
    auto z = z_py.view();
    const double rc_inverse{1.0 / rc};

    for (int i = 0; i < 3; ++i)
    {
        ncell[i] = std::max(static_cast<int>(std::floor(box.thickness[i] / rc)), 3);
    }

    const int total_cell = ncell[0] * ncell[1] * ncell[2];
    int *cell_id_list = new int[total_cell];

#pragma omp parallel for
    for (int i = 0; i < total_cell; ++i)
    {
        cell_id_list[i] = 0;
    }

#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        double xi = x(i);
        double yi = y(i);
        double zi = z(i);

        if (box.boundary[0] || box.boundary[1] || box.boundary[2])
        {
            box.wrap_into_box(xi, yi, zi);
        }

        int icel, jcel, kcel;
        get_cell_index(xi, yi, zi, box, rc_inverse, ncell, icel, jcel, kcel);

        int index = get_index(icel, jcel, kcel, ncell);
        cell_id_list[index] = 1;
    }

    nb::capsule owner(cell_id_list, [](void *p) noexcept
                      { delete[] (int *)p; });

    return nb::ndarray<nb::numpy, int>(
        cell_id_list,
        {static_cast<size_t>(ncell[0]),
         static_cast<size_t>(ncell[1]),
         static_cast<size_t>(ncell[2])},
        owner);
}

NB_MODULE(_neighbor, m)
{
    m.def("build_neighbor", &build_neighbor,
          "Build neighbor list with given max_neigh");
    m.def("build_neighbor_without_max_neigh", &build_neighbor_without_max_neigh,
          "Build neighbor list with auto-computed max_neigh");
    m.def("wrap_positions", &wrap_positions,
          "Wrap positions into the simulation box (in-place, with OpenMP)");
    m.def("average_by_neighbor", &average_by_neighbor,
          "Average values by neighbors");
    m.def("sort_verlet_by_distance", &sort_verlet_by_distance,
          "Sort neighbor list by distance");
    m.def("filter_overlap_atom", &filter_overlap_atom,
          "filter atom within rc distance");
    m.def("filter_overlap_atom_with_grain", &filter_overlap_atom_with_grain,
          "filter atom for different type");
    m.def("_fill_cell_for_void", &_fill_cell_for_void,
          "fill cell for detect void.");
}