
#pragma once
#include "box.h"
#include "type.h"
#include <vector>
#include <limits>
#include <algorithm>
#include <cstring>

struct AABB {
    double min[3]{0.0, 0.0, 0.0};
    double max[3]{0.0, 0.0, 0.0};

    inline bool contains(const double point[3]) const {
        return (point[0] >= min[0] && point[0] <= max[0] &&
                point[1] >= min[1] && point[1] <= max[1] &&
                point[2] >= min[2] && point[2] <= max[2]);
    }

    inline bool intersects(const AABB& other) const {
        return !(max[0] < other.min[0] || min[0] > other.max[0] ||
                 max[1] < other.min[1] || min[1] > other.max[1] ||
                 max[2] < other.min[2] || min[2] > other.max[2]);
    }

    inline void expand(const double point[3]) {
        for (int i = 0; i < 3; ++i) {
            if (point[i] < min[i]) min[i] = point[i];
            if (point[i] > max[i]) max[i] = point[i];
        }
    }

    inline void expand(const AABB& other) {
        for (int i = 0; i < 3; ++i) {
            if (other.min[i] < min[i]) min[i] = other.min[i];
            if (other.max[i] > max[i]) max[i] = other.max[i];
        }
    }
};

struct Node {
    AABB aabb;
    int left = -1;
    int right = -1;
    int point_index = -1;
    int count = 0;
};

struct BoxCache {
    double box_length[3];
    double half_box[3];
    bool has_pbc;
    
    size_t num_images;
    std::vector<double> pbc_shifts;
    std::vector<double> image_distances_sq;
    
    BoxCache(const Box& box) {
        has_pbc = box.boundary[0] || box.boundary[1] || box.boundary[2];
        for (int i = 0; i < 3; ++i) {
            box_length[i] = box.get_box_length(i);
            half_box[i] = box_length[i] * 0.5;
        }
        num_images = 0;
    }
};

// 优化的KNN堆：用固定大小数组代替priority_queue
struct KNNHeap {
    static constexpr int MAX_K = 32;
    double distances_sq[MAX_K];
    int indices[MAX_K];
    int k;
    int count;
    
    KNNHeap(int k_) : k(k_), count(0) {
        for (int i = 0; i < k; ++i) {
            distances_sq[i] = std::numeric_limits<double>::max();
            indices[i] = -1;
        }
    }
    
    inline void clear() {
        count = 0;
        for (int i = 0; i < k; ++i) {
            distances_sq[i] = std::numeric_limits<double>::max();
            indices[i] = -1;
        }
    }
    
    inline bool is_full() const {
        return count >= k;
    }
    
    inline int size() const {
        return count;
    }
    
    inline double max_dist_sq() const {
        return distances_sq[k-1];
    }
    
    // 尝试插入新元素，返回是否修改了最大距离
    inline bool try_insert(double d_sq, int idx) {
        // 如果距离太大，不插入
        if (count >= k && d_sq >= distances_sq[k-1]) {
            return false;
        }
        
        // 找到插入位置（保持升序）
        int pos = count;
        if (count >= k) {
            pos = k - 1;
        }
        
        while (pos > 0 && d_sq < distances_sq[pos-1]) {
            --pos;
        }
        
        // 如果需要插入的位置超出k，不插入
        if (pos >= k) {
            return false;
        }
        
        // 右移元素
        int move_count = (count >= k) ? (k - 1 - pos) : (count - pos);
        for (int i = move_count; i > 0; --i) {
            distances_sq[pos + i] = distances_sq[pos + i - 1];
            indices[pos + i] = indices[pos + i - 1];
        }
        
        // 插入新元素
        distances_sq[pos] = d_sq;
        indices[pos] = idx;
        
        if (count < k) {
            ++count;
        }
        
        return true;
    }
};

class AABBTree {
public:
    AABBTree() : box_cache_ptr(nullptr) {}
    
    ~AABBTree() {
        if (box_cache_ptr) {
            delete box_cache_ptr;
            box_cache_ptr = nullptr;
        }
    }

    void build(const ROneArrayD x_view,
               const ROneArrayD y_view,
               const ROneArrayD z_view);

    void query_knn(const double q[3], int query_idx, int k,
                   const BoxCache& box_cache,
                   const double* x_ptr,
                   const double* y_ptr,
                   const double* z_ptr,
                   KNNHeap& heap) const;

    // 新增：用于保存参考点坐标并构建树
    void build_with_coords(const ROneArrayD x_view,
                          const ROneArrayD y_view,
                          const ROneArrayD z_view,
                          const RTwoArrayD box_py,
                          const ROneArrayD origin,
                          const ROneArrayI boundary);
    
    // 新增：批量查询最近邻（只返回索引）
    void query_nearest_batch(const ROneArrayD query_x_py,
                            const ROneArrayD query_y_py,
                            const ROneArrayD query_z_py,
                            OneArrayI indices_py) const;

private:
    std::vector<Node> nodes;
    std::vector<int> point_indices;
    
    // 新增：保存参考点坐标
    std::vector<double> ref_x;
    std::vector<double> ref_y;
    std::vector<double> ref_z;
    
    // 新增：保存BoxCache用于PBC
    BoxCache* box_cache_ptr;

    int build_recursive(int start, int end,
                        const double* x_ptr,
                        const double* y_ptr,
                        const double* z_ptr);

    void query_recursive_single_image(int node_idx, const double q[3], int query_idx, 
                                     double& current_max_dist_sq,
                                     const double* x_ptr,
                                     const double* y_ptr,
                                     const double* z_ptr,
                                     KNNHeap& heap) const;
    
    // 新增：查找最近点的递归函数
    void query_nearest_recursive(int node_idx, const double q[3],
                                double& best_dist_sq,
                                int& best_idx) const;
    
    inline double point_to_aabb_sq_dist(const double q[3], const AABB& aabb) const {
        double dist_sq = 0.0;
        for (int i = 0; i < 3; ++i) {
            double d = 0.0;
            if (q[i] < aabb.min[i]) {
                d = aabb.min[i] - q[i];
            } else if (q[i] > aabb.max[i]) {
                d = q[i] - aabb.max[i];
            }
            dist_sq += d * d;
        }
        return dist_sq;
    }
};