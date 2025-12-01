
#include "aabb_tree.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cmath>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <array>

namespace nb = nanobind;

struct AxisComparator
{
    int axis;
    const double *x_ptr;
    const double *y_ptr;
    const double *z_ptr;

    AxisComparator(int a, const double *x, const double *y, const double *z)
        : axis(a), x_ptr(x), y_ptr(y), z_ptr(z) {}

    inline bool operator()(int i, int j) const
    {
        const double *ptr = (axis == 0) ? x_ptr : ((axis == 1) ? y_ptr : z_ptr);
        return ptr[i] < ptr[j];
    }
};

void AABBTree::build(const ROneArrayD x_view,
                     const ROneArrayD y_view,
                     const ROneArrayD z_view)
{
    size_t N = x_view.shape(0);
    point_indices.resize(N);
    for (size_t i = 0; i < N; ++i)
    {
        point_indices[i] = i;
    }
    nodes.reserve(2 * N);

    const double *x_ptr = x_view.data();
    const double *y_ptr = y_view.data();
    const double *z_ptr = z_view.data();

    build_recursive(0, N, x_ptr, y_ptr, z_ptr);
}

int AABBTree::build_recursive(int start, int end,
                              const double *x_ptr,
                              const double *y_ptr,
                              const double *z_ptr)
{
    int node_idx = nodes.size();
    nodes.emplace_back();
    Node &node = nodes.back();

    int idx = point_indices[start];
    node.aabb.min[0] = node.aabb.max[0] = x_ptr[idx];
    node.aabb.min[1] = node.aabb.max[1] = y_ptr[idx];
    node.aabb.min[2] = node.aabb.max[2] = z_ptr[idx];

    for (int i = start + 1; i < end; ++i)
    {
        idx = point_indices[i];
        double p[3] = {x_ptr[idx], y_ptr[idx], z_ptr[idx]};
        node.aabb.expand(p);
    }

    if (end - start == 1)
    {
        node.point_index = point_indices[start];
        return node_idx;
    }

    double extent[3];
    for (int dim = 0; dim < 3; ++dim)
    {
        extent[dim] = node.aabb.max[dim] - node.aabb.min[dim];
    }
    int longest_axis = (extent[1] > extent[0]) ? 1 : 0;
    if (extent[2] > extent[longest_axis])
        longest_axis = 2;

    int mid = (start + end) / 2;
    std::nth_element(point_indices.begin() + start,
                     point_indices.begin() + mid,
                     point_indices.begin() + end,
                     AxisComparator(longest_axis, x_ptr, y_ptr, z_ptr));

    node.left = build_recursive(start, mid, x_ptr, y_ptr, z_ptr);
    node.right = build_recursive(mid, end, x_ptr, y_ptr, z_ptr);

    return node_idx;
}

void AABBTree::query_recursive_single_image(int node_idx, const double q[3], int query_idx,
                                            double &current_max_dist_sq,
                                            const double *x_ptr,
                                            const double *y_ptr,
                                            const double *z_ptr,
                                            KNNHeap &heap) const
{
    const Node &node = nodes[node_idx];

    double min_d_sq = point_to_aabb_sq_dist(q, node.aabb);
    if (min_d_sq > current_max_dist_sq)
    {
        return;
    }

    if (node.left == -1 && node.right == -1)
    {
        int p_idx = node.point_index;
        if (p_idx == query_idx)
            return;

        double xij = q[0] - x_ptr[p_idx];
        double yij = q[1] - y_ptr[p_idx];
        double zij = q[2] - z_ptr[p_idx];

        double d_sq = xij * xij + yij * yij + zij * zij;

        if (d_sq <= current_max_dist_sq)
        {
            if (heap.try_insert(d_sq, p_idx))
            {
                current_max_dist_sq = heap.max_dist_sq();
            }
        }
        return;
    }

    if (node.left != -1 && node.right != -1)
    {
        double dist_left = point_to_aabb_sq_dist(q, nodes[node.left].aabb);
        double dist_right = point_to_aabb_sq_dist(q, nodes[node.right].aabb);

        if (dist_left < dist_right)
        {
            if (dist_left <= current_max_dist_sq)
            {
                query_recursive_single_image(node.left, q, query_idx, current_max_dist_sq,
                                             x_ptr, y_ptr, z_ptr, heap);
            }
            if (dist_right <= current_max_dist_sq)
            {
                query_recursive_single_image(node.right, q, query_idx, current_max_dist_sq,
                                             x_ptr, y_ptr, z_ptr, heap);
            }
        }
        else
        {
            if (dist_right <= current_max_dist_sq)
            {
                query_recursive_single_image(node.right, q, query_idx, current_max_dist_sq,
                                             x_ptr, y_ptr, z_ptr, heap);
            }
            if (dist_left <= current_max_dist_sq)
            {
                query_recursive_single_image(node.left, q, query_idx, current_max_dist_sq,
                                             x_ptr, y_ptr, z_ptr, heap);
            }
        }
    }
    else if (node.left != -1)
    {
        query_recursive_single_image(node.left, q, query_idx, current_max_dist_sq,
                                     x_ptr, y_ptr, z_ptr, heap);
    }
    else if (node.right != -1)
    {
        query_recursive_single_image(node.right, q, query_idx, current_max_dist_sq,
                                     x_ptr, y_ptr, z_ptr, heap);
    }
}

void AABBTree::query_knn(const double q[3], int query_idx, int k,
                         const BoxCache &box_cache,
                         const double *x_ptr,
                         const double *y_ptr,
                         const double *z_ptr,
                         KNNHeap &heap) const
{
    double current_max_dist_sq = std::numeric_limits<double>::max();

    for (size_t img_idx = 0; img_idx < box_cache.num_images; ++img_idx)
    {
        const double *shift = &box_cache.pbc_shifts[img_idx * 3];

        double q_shifted[3] = {
            q[0] - shift[0],
            q[1] - shift[1],
            q[2] - shift[2]};

        // 优化：如果heap已满，检查这个镜像是否可能包含更近的邻居
        if (heap.is_full())
        {
            double min_possible_dist_sq = point_to_aabb_sq_dist(q_shifted, nodes[0].aabb);
            if (min_possible_dist_sq > current_max_dist_sq)
            {
                continue;
            }
        }

        query_recursive_single_image(0, q_shifted, query_idx, current_max_dist_sq,
                                     x_ptr, y_ptr, z_ptr, heap);
    }
}

// 新增：构建树并保存参考点坐标和PBC信息
void AABBTree::build_with_coords(const ROneArrayD x_view,
                                 const ROneArrayD y_view,
                                 const ROneArrayD z_view,
                                 const RTwoArrayD box_py,
                                 const ROneArrayD origin,
                                 const ROneArrayI boundary)
{
    size_t N = x_view.shape(0);
    
    // 保存参考点坐标
    ref_x.resize(N);
    ref_y.resize(N);
    ref_z.resize(N);
    
    const double *x_ptr = x_view.data();
    const double *y_ptr = y_view.data();
    const double *z_ptr = z_view.data();
    
    // 安全优化：使用memcpy复制数据
    std::memcpy(ref_x.data(), x_ptr, N * sizeof(double));
    std::memcpy(ref_y.data(), y_ptr, N * sizeof(double));
    std::memcpy(ref_z.data(), z_ptr, N * sizeof(double));
    
    // 构建树
    build(x_view, y_view, z_view);
    
    // 清理旧的box_cache
    if (box_cache_ptr) {
        delete box_cache_ptr;
    }
    
    // 设置PBC信息
    Box box = get_box(box_py, origin, boundary);
    box_cache_ptr = new BoxCache(box);
    
    // 根据系统大小和是否有PBC自适应设置镜像数
    int nimages = 1;
    bool has_pbc = box.boundary[0] || box.boundary[1] || box.boundary[2];

    if (has_pbc)
    {
        if (N < 10000)
        {
            nimages = 2;
        }
        else
        {
            nimages = 1;
        }
    }

    std::vector<std::array<double, 3>> pbc_images;
    int nx = box.boundary[0] ? nimages : 0;
    int ny = box.boundary[1] ? nimages : 0;
    int nz = box.boundary[2] ? nimages : 0;

    pbc_images.reserve((2 * nx + 1) * (2 * ny + 1) * (2 * nz + 1));

    for (int iz = -nz; iz <= nz; ++iz)
    {
        for (int iy = -ny; iy <= ny; ++iy)
        {
            for (int ix = -nx; ix <= nx; ++ix)
            {
                std::array<double, 3> shift;
                if (box.triclinic)
                {
                    shift[0] = ix * box.data[0] + iy * box.data[3] + iz * box.data[6];
                    shift[1] = ix * box.data[1] + iy * box.data[4] + iz * box.data[7];
                    shift[2] = ix * box.data[2] + iy * box.data[5] + iz * box.data[8];
                }
                else
                {
                    shift[0] = ix * box.data[0];
                    shift[1] = iy * box.data[4];
                    shift[2] = iz * box.data[8];
                }
                pbc_images.push_back(shift);
            }
        }
    }

    // 按镜像距离排序
    std::sort(pbc_images.begin(), pbc_images.end(),
              [](const std::array<double, 3> &a, const std::array<double, 3> &b)
              {
                  double len_a = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
                  double len_b = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
                  return len_a < len_b;
              });

    box_cache_ptr->num_images = pbc_images.size();
    box_cache_ptr->pbc_shifts.resize(pbc_images.size() * 3);
    box_cache_ptr->image_distances_sq.resize(pbc_images.size());

    for (size_t i = 0; i < pbc_images.size(); ++i)
    {
        box_cache_ptr->pbc_shifts[i * 3 + 0] = pbc_images[i][0];
        box_cache_ptr->pbc_shifts[i * 3 + 1] = pbc_images[i][1];
        box_cache_ptr->pbc_shifts[i * 3 + 2] = pbc_images[i][2];
        box_cache_ptr->image_distances_sq[i] = pbc_images[i][0] * pbc_images[i][0] +
                                               pbc_images[i][1] * pbc_images[i][1] +
                                               pbc_images[i][2] * pbc_images[i][2];
    }
}

// 新增：查找最近点的递归函数（保持原始逻辑，不改变剪枝）
void AABBTree::query_nearest_recursive(int node_idx, const double q[3],
                                       double &best_dist_sq,
                                       int &best_idx) const
{
    if (node_idx < 0 || node_idx >= static_cast<int>(nodes.size()))
    {
        return;
    }

    const Node &node = nodes[node_idx];

    // 计算查询点到当前节点AABB的最小距离
    double min_dist_sq = point_to_aabb_sq_dist(q, node.aabb);
    
    // 如果这个距离已经大于等于当前最佳距离，剪枝
    if (min_dist_sq >= best_dist_sq)
    {
        return;
    }

    // 如果是叶子节点，计算实际距离
    if (node.left == -1 && node.right == -1)
    {
        int p_idx = node.point_index;
        if (p_idx >= 0 && p_idx < static_cast<int>(ref_x.size()))
        {
            double dx = q[0] - ref_x[p_idx];
            double dy = q[1] - ref_y[p_idx];
            double dz = q[2] - ref_z[p_idx];
            double d_sq = dx * dx + dy * dy + dz * dz;

            if (d_sq < best_dist_sq)
            {
                best_dist_sq = d_sq;
                best_idx = p_idx;
            }
        }
        return;
    }

    // 如果不是叶子节点，递归搜索两个子节点
    if (node.left != -1 && node.right != -1)
    {
        double dist_left = point_to_aabb_sq_dist(q, nodes[node.left].aabb);
        double dist_right = point_to_aabb_sq_dist(q, nodes[node.right].aabb);

        if (dist_left < dist_right)
        {
            query_nearest_recursive(node.left, q, best_dist_sq, best_idx);
            query_nearest_recursive(node.right, q, best_dist_sq, best_idx);
        }
        else
        {
            query_nearest_recursive(node.right, q, best_dist_sq, best_idx);
            query_nearest_recursive(node.left, q, best_dist_sq, best_idx);
        }
    }
    else if (node.left != -1)
    {
        query_nearest_recursive(node.left, q, best_dist_sq, best_idx);
    }
    else if (node.right != -1)
    {
        query_nearest_recursive(node.right, q, best_dist_sq, best_idx);
    }
}

// 新增：批量查询最近邻（并行版本，但保持原始逻辑）
void AABBTree::query_nearest_batch(const ROneArrayD query_x_py,
                                   const ROneArrayD query_y_py,
                                   const ROneArrayD query_z_py,
                                   OneArrayI indices_py) const
{
    size_t N_query = query_x_py.shape(0);

    const double *query_x_ptr = query_x_py.data();
    const double *query_y_ptr = query_y_py.data();
    const double *query_z_ptr = query_z_py.data();

    int *indices_ptr = indices_py.data();

    if (nodes.empty() || box_cache_ptr == nullptr)
    {
        std::fill(indices_ptr, indices_ptr + N_query, -1);
        return;
    }

    // 安全优化：使用OpenMP并行，但保持原始逻辑
    int num_threads = omp_get_max_threads();
    int chunk_size = std::max(32, static_cast<int>(N_query / (num_threads * 8)));

#pragma omp parallel for schedule(dynamic, chunk_size)
    for (int i = 0; i < static_cast<int>(N_query); ++i)
    {
        double q[3] = {query_x_ptr[i], query_y_ptr[i], query_z_ptr[i]};
        
        double best_dist_sq = std::numeric_limits<double>::max();
        int best_idx = -1;

        // 遍历所有PBC镜像（保持原始逻辑）
        for (size_t img_idx = 0; img_idx < box_cache_ptr->num_images; ++img_idx)
        {
            const double *shift = &box_cache_ptr->pbc_shifts[img_idx * 3];

            double q_shifted[3] = {
                q[0] - shift[0],
                q[1] - shift[1],
                q[2] - shift[2]
            };

            // 对每个镜像搜索最近点
            query_nearest_recursive(0, q_shifted, best_dist_sq, best_idx);
        }

        indices_ptr[i] = best_idx;
    }
}

void knn(const ROneArrayD x_py, const ROneArrayD y_py, const ROneArrayD z_py,
         const RTwoArrayD box_py, const ROneArrayD origin,
         const ROneArrayI boundary, int k,
         TwoArrayI indices_py, TwoArrayD distances_py)
{
    size_t N = x_py.shape(0);
    auto indices_view = indices_py.view();
    auto distances_view = distances_py.view();
    Box box = get_box(box_py, origin, boundary);

    // 根据系统大小和是否有PBC自适应设置镜像数
    int nimages = 1;
    bool has_pbc = box.boundary[0] || box.boundary[1] || box.boundary[2];

    if (has_pbc)
    {
        if (N < 10000)
        {
            nimages = 2;
        }

        else
        {
            nimages = 1;
        }
    }

    std::vector<std::array<double, 3>> pbc_images;
    int nx = box.boundary[0] ? nimages : 0;
    int ny = box.boundary[1] ? nimages : 0;
    int nz = box.boundary[2] ? nimages : 0;

    pbc_images.reserve((2 * nx + 1) * (2 * ny + 1) * (2 * nz + 1));

    for (int iz = -nz; iz <= nz; ++iz)
    {
        for (int iy = -ny; iy <= ny; ++iy)
        {
            for (int ix = -nx; ix <= nx; ++ix)
            {
                std::array<double, 3> shift;
                if (box.triclinic)
                {
                    shift[0] = ix * box.data[0] + iy * box.data[3] + iz * box.data[6];
                    shift[1] = ix * box.data[1] + iy * box.data[4] + iz * box.data[7];
                    shift[2] = ix * box.data[2] + iy * box.data[5] + iz * box.data[8];
                }
                else
                {
                    shift[0] = ix * box.data[0];
                    shift[1] = iy * box.data[4];
                    shift[2] = iz * box.data[8];
                }
                pbc_images.push_back(shift);
            }
        }
    }

    // 按镜像距离排序，先搜索近的镜像
    std::sort(pbc_images.begin(), pbc_images.end(),
              [](const std::array<double, 3> &a, const std::array<double, 3> &b)
              {
                  double len_a = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
                  double len_b = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
                  return len_a < len_b;
              });

    BoxCache box_cache(box);
    box_cache.num_images = pbc_images.size();
    box_cache.pbc_shifts.resize(pbc_images.size() * 3);
    box_cache.image_distances_sq.resize(pbc_images.size());

    for (size_t i = 0; i < pbc_images.size(); ++i)
    {
        box_cache.pbc_shifts[i * 3 + 0] = pbc_images[i][0];
        box_cache.pbc_shifts[i * 3 + 1] = pbc_images[i][1];
        box_cache.pbc_shifts[i * 3 + 2] = pbc_images[i][2];
        box_cache.image_distances_sq[i] = pbc_images[i][0] * pbc_images[i][0] +
                                          pbc_images[i][1] * pbc_images[i][1] +
                                          pbc_images[i][2] * pbc_images[i][2];
    }

    AABBTree tree;
    tree.build(x_py, y_py, z_py);

    const double *x_ptr = x_py.data();
    const double *y_ptr = y_py.data();
    const double *z_ptr = z_py.data();

    // 优化的OpenMP调度
    int num_threads = omp_get_max_threads();
    int chunk_size = std::max(64, static_cast<int>(N / (num_threads * 16)));

#pragma omp parallel
    {
        KNNHeap heap(k);

#pragma omp for schedule(dynamic, chunk_size) nowait
        for (int i = 0; i < static_cast<int>(N); ++i)
        {
            double q[3] = {x_ptr[i], y_ptr[i], z_ptr[i]};

            heap.clear();
            tree.query_knn(q, i, k, box_cache, x_ptr, y_ptr, z_ptr, heap);

            int count = heap.size();
            for (int j = 0; j < count; ++j)
            {
                indices_view(i, j) = heap.indices[j];
                distances_view(i, j) = std::sqrt(heap.distances_sq[j]);
            }

            for (int j = count; j < k; ++j)
            {
                indices_view(i, j) = -1;
                distances_view(i, j) = -1.0;
            }
        }
    }
}

NB_MODULE(_aabbtree, m)
{
    m.def("knn", &knn);
    
    // 将AABBTree类暴露给Python
    nb::class_<AABBTree>(m, "AABBTree")
        .def(nb::init<>())
        .def("build_with_coords", &AABBTree::build_with_coords, 
             "Build the tree from reference points and save coordinates with PBC info")
        .def("query_nearest_batch", &AABBTree::query_nearest_batch, 
             "Query nearest neighbors for a batch of points (considering PBC)");
}