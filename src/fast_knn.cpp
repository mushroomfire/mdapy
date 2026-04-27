// Copyright (c) 2022-2026, Yongchao Wu in Aalto University
// This file is from the mdapy project, released under the BSD 3-Clause License.
//
// Fast k-nearest-neighbor finder modeled on the algorithm used by OVITO's
// NearestNeighborFinder (src/ovito/particles/util/NearestNeighborFinder.{h,cpp}).
//
// Implementation notes:
//   * Bucket leaves (default ~max(k/2, 8) atoms per leaf) instead of
//     one-atom leaves. Greatly reduces tree depth, traversal overhead and
//     cache misses during leaf scans.
//   * Tree built top-down with three pre-split levels (one per axis), then
//     particles are inserted and leaves are split lazily when they exceed
//     the bucket size — same shape as OVITO's tree.
//   * Splits chosen along the largest *physical* extent
//     (box_vector_length * leaf_size_in_that_dim), not the largest extent
//     in reduced coordinates.
//   * Leaf positions stored in an AOS array (3 doubles, contiguous) so the
//     inner distance loop streams memory linearly without index gathers.
//   * Periodic image shifts sorted by length² and pruned per-image with the
//     current k-th distance.

#include "box.h"
#include "type.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <omp.h>

namespace nb = nanobind;

namespace fast_knn {

// ---------------------------------------------------------------------------
//  Utilities
// ---------------------------------------------------------------------------

struct Vec3 {
    double x{0.0}, y{0.0}, z{0.0};
};

struct Box3 {
    double mn[3]{0.0, 0.0, 0.0};   // min corner (reduced or absolute)
    double mx[3]{1.0, 1.0, 1.0};   // max corner

    inline double size(int d) const { return mx[d] - mn[d]; }
};

// Multiply 3x3 row-major matrix in `m` (the same layout used by Box::data
// for the cell matrix or its inverse) by a 3-vector. mdapy's Box stores box
// vectors as rows (a = data[0..2], b = data[3..5], c = data[6..8]). A point
// in reduced coordinates r = (rx, ry, rz) maps to absolute coordinates as
//   p = rx * a + ry * b + rz * c
// = (rx * data[0] + ry * data[3] + rz * data[6], ...).
inline void reduced_to_absolute(const Box& box, const double r[3], double out[3]) {
    out[0] = r[0] * box.data[0] + r[1] * box.data[3] + r[2] * box.data[6];
    out[1] = r[0] * box.data[1] + r[1] * box.data[4] + r[2] * box.data[7];
    out[2] = r[0] * box.data[2] + r[1] * box.data[5] + r[2] * box.data[8];
}

// Inverse box; for an absolute point p, the reduced coordinate is
//   r = p · inv (with inv stored row-major at data[9..17])
inline void absolute_to_reduced(const Box& box, const double p[3], double out[3]) {
    out[0] = p[0] * box.data[9]  + p[1] * box.data[12] + p[2] * box.data[15];
    out[1] = p[0] * box.data[10] + p[1] * box.data[13] + p[2] * box.data[16];
    out[2] = p[0] * box.data[11] + p[1] * box.data[14] + p[2] * box.data[17];
}

// Outward face normal of the cell face perpendicular to axis `dim`, with
// length 1 / thickness so that normal · (point - corner) gives the signed
// distance in absolute units. (See Ovito's cellNormalVector.)
static Vec3 cell_normal(const Box& box, int dim) {
    const double* a = box.data + 0;
    const double* b = box.data + 3;
    const double* c = box.data + 6;

    Vec3 cross;
    if (dim == 0) {
        cross.x = b[1] * c[2] - b[2] * c[1];
        cross.y = b[2] * c[0] - b[0] * c[2];
        cross.z = b[0] * c[1] - b[1] * c[0];
    } else if (dim == 1) {
        cross.x = c[1] * a[2] - c[2] * a[1];
        cross.y = c[2] * a[0] - c[0] * a[2];
        cross.z = c[0] * a[1] - c[1] * a[0];
    } else {
        cross.x = a[1] * b[2] - a[2] * b[1];
        cross.y = a[2] * b[0] - a[0] * b[2];
        cross.z = a[0] * b[1] - a[1] * b[0];
    }
    double L2 = cross.x * cross.x + cross.y * cross.y + cross.z * cross.z;
    double inv = (L2 > 0.0) ? 1.0 / std::sqrt(L2) : 0.0;
    // Length of the box vector along `dim` projected onto this normal is
    // |volume| / cross_length, so dividing by that length gives a normal
    // with the right magnitude for signed distance ⇒ scale by 1/thickness.
    return Vec3{cross.x * inv, cross.y * inv, cross.z * inv};
}

// ---------------------------------------------------------------------------
//  Bounded priority queue: keeps the k smallest distances seen so far.
//  Uses a sorted small array — for the small k typical in this code path
//  (k ≤ 24) a linear insertion is faster than a binary heap.
// ---------------------------------------------------------------------------
struct TopK {
    std::vector<double> d2;
    std::vector<int> idx;
    int k;
    int n;

    explicit TopK(int kk) : d2(kk, std::numeric_limits<double>::infinity()),
                            idx(kk, -1), k(kk), n(0) {}

    inline void clear() {
        n = 0;
        std::fill(d2.begin(), d2.end(), std::numeric_limits<double>::infinity());
        std::fill(idx.begin(), idx.end(), -1);
    }

    inline bool full() const { return n >= k; }
    inline double worst() const { return d2[k - 1]; }

    inline void push(double dist2, int particle_index) {
        if (full() && dist2 >= d2[k - 1]) return;
        // Insert keeping the array sorted ascending. Binary-search the
        // insertion point; for tiny k this costs almost nothing.
        int lo = 0, hi = std::min(n, k);
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (d2[mid] < dist2) lo = mid + 1; else hi = mid;
        }
        int pos = lo;
        if (pos >= k) return;
        int last = (n < k) ? n : (k - 1);
        for (int i = last; i > pos; --i) {
            d2[i] = d2[i - 1];
            idx[i] = idx[i - 1];
        }
        d2[pos] = dist2;
        idx[pos] = particle_index;
        if (n < k) ++n;
    }
};

// ---------------------------------------------------------------------------
//  Tree
// ---------------------------------------------------------------------------

struct Atom {
    double x, y, z;       // wrapped, absolute coords (cache-friendly AOS)
    int particle_index;   // index into the original input arrays
};

struct Node {
    // For internal nodes: children, splitting plane in reduced coords.
    int left  = -1;
    int right = -1;
    int split_dim = -1;   // -1 ⇒ leaf
    double split_pos = 0.0;

    // For leaf nodes: contiguous range in the global atom array.
    int atom_begin = 0;
    int atom_end   = 0;

    // Bounds of this node in absolute coordinates (axis-aligned).
    double bmn[3]{0.0, 0.0, 0.0};
    double bmx[3]{0.0, 0.0, 0.0};
};

class KdTree {
public:
    KdTree() = default;

    void build(const double* x, const double* y, const double* z,
               size_t N, const Box& box, int bucket_size, int depth_limit)
    {
        box_ = box;
        bucket_ = std::max(8, bucket_size);
        depth_limit_ = depth_limit;

        // Precompute the squared lengths of the cell vectors and the unit
        // outward normals of the cell faces. Used to pick split directions
        // and compute exact box-to-point distances.
        for (int d = 0; d < 3; ++d) {
            const double* v = box_.data + d * 3;
            cell_len2_[d] = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            normals_[d] = cell_normal(box_, d);
        }

        // The inverse-box rows that map an absolute (x,y,z) onto the
        // reduced coordinate along each dim. Pulled into locals so the
        // partition's hot loop can read them from registers.
        for (int d = 0; d < 3; ++d) {
            inv_row_[d][0] = box_.data[9  + d];
            inv_row_[d][1] = box_.data[12 + d];
            inv_row_[d][2] = box_.data[15 + d];
        }

        // -- Pass 1 (fused): wrap atoms into the primary cell, stage them
        //    in `atoms_` (AOS), record their reduced coords in `red_` (used
        //    by the partition hot loop), and accumulate the reduced
        //    bounding box for any open boundary axes — all in one sweep.
        atoms_.resize(N);
        red_.resize(N);
        double rmn[3] = { 0.0, 0.0, 0.0 };
        double rmx[3] = { 1.0, 1.0, 1.0 };
        bool any_open = !box_.boundary[0] || !box_.boundary[1] || !box_.boundary[2];
        if (any_open) {
            for (int d = 0; d < 3; ++d) {
                if (!box_.boundary[d]) {
                    rmn[d] = std::numeric_limits<double>::infinity();
                    rmx[d] = -std::numeric_limits<double>::infinity();
                }
            }
        }

        // Wrap loop is embarrassingly parallel — each atom is independent.
        // We do the open-axis bounding box in a separate (cheap) reduction
        // pass to keep this loop's body branch-free per axis.
        // Threshold below which the wrap loop runs serially. Parallelizing
        // a tiny loop costs more than it saves AND has been observed to
        // trigger OpenMP-runtime crashes on systems where multiple
        // libomp/libgomp images are loaded (e.g. via numpy + polars).
        constexpr long long kWrapParThreshold = 50000;

        if (!box_.triclinic) {
            // Orthogonal fast path: r[d] = p[d] / Ldd; wrap is a single
            // divide and one floor per axis, no full mat-vec required.
            const double Lx = box_.data[0], Ly = box_.data[4], Lz = box_.data[8];
            const double invLx = 1.0 / Lx, invLy = 1.0 / Ly, invLz = 1.0 / Lz;
            const int bx = box_.boundary[0], by = box_.boundary[1], bz = box_.boundary[2];
#pragma omp parallel for schedule(static) if((long long)N > kWrapParThreshold)
            for (long long i = 0; i < (long long)N; ++i) {
                double px = x[i], py = y[i], pz = z[i];
                double rx = px * invLx, ry = py * invLy, rz = pz * invLz;
                if (bx) { double s = std::floor(rx); rx -= s; px -= s * Lx; }
                if (by) { double s = std::floor(ry); ry -= s; py -= s * Ly; }
                if (bz) { double s = std::floor(rz); rz -= s; pz -= s * Lz; }
                atoms_[i] = Atom{px, py, pz, static_cast<int>(i)};
                red_[i] = { rx, ry, rz };
            }
        } else {
#pragma omp parallel for schedule(static) if((long long)N > kWrapParThreshold)
            for (long long i = 0; i < (long long)N; ++i) {
                double p[3] = { x[i], y[i], z[i] };
                double r[3];
                absolute_to_reduced(box_, p, r);
                for (int d = 0; d < 3; ++d) {
                    if (box_.boundary[d]) {
                        double s = std::floor(r[d]);
                        if (s != 0.0) {
                            r[d] -= s;
                            p[0] -= s * box_.data[d * 3 + 0];
                            p[1] -= s * box_.data[d * 3 + 1];
                            p[2] -= s * box_.data[d * 3 + 2];
                        }
                    }
                }
                atoms_[i] = Atom{p[0], p[1], p[2], static_cast<int>(i)};
                red_[i] = { r[0], r[1], r[2] };
            }
        }

        if (any_open) {
            // Open-axis bounding-box reduction (handful of comparisons per
            // atom). Cheap enough that we keep it serial — running it under
            // OMP reduction adds more overhead than it saves at this size.
            for (size_t i = 0; i < N; ++i) {
                for (int d = 0; d < 3; ++d) {
                    if (!box_.boundary[d]) {
                        if (red_[i][d] < rmn[d]) rmn[d] = red_[i][d];
                        if (red_[i][d] > rmx[d]) rmx[d] = red_[i][d];
                    }
                }
            }
            for (int d = 0; d < 3; ++d) {
                if (!box_.boundary[d] && rmx[d] - rmn[d] < 1e-12) {
                    rmx[d] = rmn[d] + 1e-9;
                }
            }
        }

        // -- Build the tree top-down. The recursion partitions `order_`
        //    in place; the atoms are physically permuted in a single
        //    in-place cycle pass at the end. nodes_ is pre-sized to an
        //    upper bound (a binary kd-tree with L leaves has ≤ 2L-1
        //    nodes; ceil(N/bucket) bounds L) so make_node() can be a
        //    lock-free atomic fetch_add and tasks can build sibling
        //    subtrees in parallel.
        // Worst case (degenerate splits with one tiny side per level)
        // can produce significantly more nodes than the balanced 2L-1
        // figure. A safe upper bound is 4L + slack.
        size_t L = (N + bucket_ - 1) / bucket_;
        size_t node_capacity = 4 * L + 32;
        nodes_.assign(node_capacity, Node{});
        next_node_.store(0, std::memory_order_relaxed);

        order_.resize(N);
        for (size_t i = 0; i < N; ++i) order_[i] = static_cast<int>(i);

        int root_id = make_node();
        nodes_[root_id].atom_begin = 0;
        nodes_[root_id].atom_end = static_cast<int>(N);
        copy_reduced_bounds(nodes_[root_id], rmn, rmx);

#pragma omp parallel
        {
#pragma omp single
            build_recursive(root_id, 0, rmn, rmx);
        }
        nodes_.resize(static_cast<size_t>(next_node_.load(std::memory_order_relaxed)));

        // -- Physically permute `atoms_` so that atoms_new[i] = atoms_old[order_[i]],
        //    in place via the saved-value cycle algorithm. One scratch
        //    `Atom` per cycle — no full duplicate buffer. After this loop,
        //    `order_[i] == i` for every i.
        for (int i = 0; i < static_cast<int>(N); ++i) {
            if (order_[i] == i) continue;
            Atom tmp = atoms_[i];
            int j = i;
            while (true) {
                int next = order_[j];
                if (next == i) {
                    atoms_[j] = tmp;
                    order_[j] = j;
                    break;
                }
                atoms_[j] = atoms_[next];
                order_[j] = j;
                j = next;
            }
        }
        order_.clear(); order_.shrink_to_fit();
        red_.clear();   red_.shrink_to_fit();

        // Convert all node bounds from reduced to absolute coordinates.
        for (Node& n : nodes_) {
            double cmn[3] = { n.bmn[0], n.bmn[1], n.bmn[2] };
            double cmx[3] = { n.bmx[0], n.bmx[1], n.bmx[2] };
            // For each of the 8 corners of the reduced AABB, transform to
            // absolute coords and rebuild an axis-aligned absolute AABB.
            double amn[3] = { std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::infinity(),
                              std::numeric_limits<double>::infinity() };
            double amx[3] = { -std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity(),
                              -std::numeric_limits<double>::infinity() };
            for (int corner = 0; corner < 8; ++corner) {
                double r[3] = {
                    (corner & 1) ? cmx[0] : cmn[0],
                    (corner & 2) ? cmx[1] : cmn[1],
                    (corner & 4) ? cmx[2] : cmn[2]
                };
                double a[3];
                reduced_to_absolute(box_, r, a);
                for (int d = 0; d < 3; ++d) {
                    if (a[d] < amn[d]) amn[d] = a[d];
                    if (a[d] > amx[d]) amx[d] = a[d];
                }
            }
            for (int d = 0; d < 3; ++d) { n.bmn[d] = amn[d]; n.bmx[d] = amx[d]; }
        }
    }

    // Return min squared distance between point q and node bounding box,
    // using the cell face normals so non-axis-aligned (triclinic) cells get
    // the correct distance, not an over-conservative one.
    inline double min_dist_sq(int node_id, const double q[3]) const {
        const Node& n = nodes_[node_id];
        // p1 = (bmn - q), p2 = (q - bmx); then for each face axis the
        // signed distance to the bounding slab is normal·p1 (entering from
        // below) or normal·p2 (entering from above), whichever is positive.
        double d = 0.0;
        for (int dim = 0; dim < 3; ++dim) {
            double p1x = n.bmn[0] - q[0];
            double p1y = n.bmn[1] - q[1];
            double p1z = n.bmn[2] - q[2];
            double p2x = q[0] - n.bmx[0];
            double p2y = q[1] - n.bmx[1];
            double p2z = q[2] - n.bmx[2];
            double tmin = normals_[dim].x * p1x + normals_[dim].y * p1y + normals_[dim].z * p1z;
            if (tmin > d) d = tmin;
            double tmax = normals_[dim].x * p2x + normals_[dim].y * p2y + normals_[dim].z * p2z;
            if (tmax > d) d = tmax;
        }
        return d * d;
    }

    void query(const double q_orig[3], int self_index, TopK& heap,
               const std::vector<std::array<double, 3>>& pbc_shifts) const
    {
        for (const auto& shift : pbc_shifts) {
            double q[3] = {
                q_orig[0] - shift[0],
                q_orig[1] - shift[1],
                q_orig[2] - shift[2]
            };
            // Skip this image entirely if its closest-possible reach is
            // already farther than the current k-th best.
            if (heap.full() && min_dist_sq(0, q) >= heap.worst()) continue;
            double qr[3];
            absolute_to_reduced(box_, q, qr);
            visit(0, q, qr, self_index, heap);
        }
    }

private:
    int make_node() {
        return next_node_.fetch_add(1, std::memory_order_relaxed);
    }

    static void copy_reduced_bounds(Node& n, const double mn[3], const double mx[3]) {
        for (int d = 0; d < 3; ++d) { n.bmn[d] = mn[d]; n.bmx[d] = mx[d]; }
    }

    int choose_split_dim(const double rmn[3], const double rmx[3]) const {
        // Largest physical extent: cell_vec_length × reduced size.
        double best = -1.0;
        int best_dim = 0;
        for (int d = 0; d < 3; ++d) {
            double sz = rmx[d] - rmn[d];
            double m = cell_len2_[d] * sz * sz;
            if (m > best) { best = m; best_dim = d; }
        }
        return best_dim;
    }

    void build_recursive(int node_id, int depth, const double rmn[3], const double rmx[3]) {
        Node& n = nodes_[node_id];
        int count = n.atom_end - n.atom_begin;
        // Stop subdividing once the bucket is small or we hit the limit.
        if (count <= bucket_ || depth >= depth_limit_) return;

        int dim = choose_split_dim(rmn, rmx);
        double split = 0.5 * (rmn[dim] + rmx[dim]);

        // Partition order_[atom_begin..atom_end) by reduced-coord on `dim`.
        int lo = n.atom_begin, hi = n.atom_end;
        int i = lo;
        for (int j = lo; j < hi; ++j) {
            if (red_[order_[j]][dim] < split) {
                std::swap(order_[i], order_[j]);
                ++i;
            }
        }
        int mid = i;

        // Degenerate split (everything fell on one side): force a median
        // split using nth_element to guarantee progress.
        if (mid == lo || mid == hi) {
            mid = lo + count / 2;
            std::nth_element(order_.begin() + lo, order_.begin() + mid,
                             order_.begin() + hi,
                             [&](int a, int b) {
                                 return red_[a][dim] < red_[b][dim];
                             });
            split = red_[order_[mid]][dim];
        }

        // Allocating children invalidates `&n` (vector growth).
        int left_id  = make_node();
        int right_id = make_node();
        // Rebind via index after potential reallocation.
        nodes_[node_id].split_dim = dim;
        nodes_[node_id].split_pos = split;
        nodes_[node_id].left  = left_id;
        nodes_[node_id].right = right_id;

        nodes_[left_id].atom_begin  = lo;
        nodes_[left_id].atom_end    = mid;
        nodes_[right_id].atom_begin = mid;
        nodes_[right_id].atom_end   = hi;

        double l_rmx[3] = { rmx[0], rmx[1], rmx[2] }; l_rmx[dim] = split;
        double r_rmn[3] = { rmn[0], rmn[1], rmn[2] }; r_rmn[dim] = split;
        copy_reduced_bounds(nodes_[left_id],  rmn,   l_rmx);
        copy_reduced_bounds(nodes_[right_id], r_rmn, rmx);

        // Sibling subtrees touch disjoint ranges → spawn tasks for big
        // enough subtrees, recurse serially for small ones (to stay below
        // task-creation overhead).
        constexpr int kTaskThreshold = 8192;
        if (count > kTaskThreshold) {
            double l_rmn[3] = { rmn[0], rmn[1], rmn[2] };
            double r_rmx[3] = { rmx[0], rmx[1], rmx[2] };
            int next_depth = depth + 1;
#pragma omp task default(shared) firstprivate(left_id, next_depth, l_rmn, l_rmx)
            build_recursive(left_id, next_depth, l_rmn, l_rmx);
#pragma omp task default(shared) firstprivate(right_id, next_depth, r_rmn, r_rmx)
            build_recursive(right_id, next_depth, r_rmn, r_rmx);
#pragma omp taskwait
        } else {
            build_recursive(left_id, depth + 1, rmn, l_rmx);
            build_recursive(right_id, depth + 1, r_rmn, rmx);
        }
    }

    void visit(int node_id, const double q[3], const double qr[3],
               int self_index, TopK& heap) const {
        const Node& n = nodes_[node_id];
        if (n.split_dim < 0) {
            // Leaf scan — streaming over a contiguous Atom block.
            const Atom* a = atoms_.data() + n.atom_begin;
            int count = n.atom_end - n.atom_begin;
            for (int i = 0; i < count; ++i) {
                double dx = a[i].x - q[0];
                double dy = a[i].y - q[1];
                double dz = a[i].z - q[2];
                double d2 = dx * dx + dy * dy + dz * dz;
                if (a[i].particle_index == self_index && d2 == 0.0) continue;
                heap.push(d2, a[i].particle_index);
            }
            return;
        }

        // Pick near-child cheaply via the splitting plane in reduced coords
        // (same trick OVITO uses) — avoids computing min_dist_sq twice.
        int near_id, far_id;
        if (qr[n.split_dim] < n.split_pos) { near_id = n.left;  far_id = n.right; }
        else                                { near_id = n.right; far_id = n.left;  }
        visit(near_id, q, qr, self_index, heap);
        if (!heap.full() || heap.worst() > min_dist_sq(far_id, q)) {
            visit(far_id, q, qr, self_index, heap);
        }
    }

    Box box_{};
    int bucket_ = 8;
    int depth_limit_ = 20;

    double cell_len2_[3]{};
    Vec3 normals_[3];
    double inv_row_[3][3]{};                             // inverse-box rows

    std::vector<Atom> atoms_;
    std::vector<int> order_;                             // build-time only
    std::vector<std::array<double, 3>> red_;             // build-time only
    std::vector<Node> nodes_;
    std::atomic<int> next_node_{0};                      // build-time only
};

// ===========================================================================
//  OrthoTree — bosque-style implicit kd-tree for orthogonal cells.
//
//  Idea (after https://github.com/cavemanloverboy/Bosque):
//    * Atoms live in one contiguous AOS array, mutated *in place*.
//    * Build = `nth_element` recursively on dim = (level mod 3). After the
//      partition, atoms[..median) all sit "left" of atoms[median] on that
//      dim, and atoms[median+1..) sit "right".
//    * The tree has *zero* metadata: a subtree is simply a contiguous slice,
//      and at each level you split the slice in half — the median is the
//      element at index `len/2` and serves as the splitting "stem".
//    * Build is naturally parallel: sibling slices are disjoint, just spawn
//      a task on the left half and recurse on the right.
//
//  This is dramatically faster than the partition-style top-down build we
//  use for triclinic cells because it has *no* node allocation, *no* node
//  indices, *no* bounding-box bookkeeping — only one nth_element per level.
// ===========================================================================
struct OrthoAtom {
    double x, y, z;
    int idx;             // original input index
};
static_assert(sizeof(OrthoAtom) == 32, "OrthoAtom should be 32B for cache-line packing");

static constexpr int kOrthoBucket = 32;

static inline double ortho_coord(const OrthoAtom& a, int dim) {
    return (&a.x)[dim];
}

static inline double squared_dist_ortho(const OrthoAtom& a, const double q[3]) {
    double dx = a.x - q[0];
    double dy = a.y - q[1];
    double dz = a.z - q[2];
    return dx * dx + dy * dy + dz * dz;
}

// Recursive in-place build. After this returns, `[atoms, atoms+n)` is a kd
// tree in implicit form with split dim = `level % 3` at every level.
static void build_ortho(OrthoAtom* atoms, int n, int level) {
    if (n <= kOrthoBucket) return;
    int dim = level % 3;
    int median = n / 2;
    std::nth_element(atoms, atoms + median, atoms + n,
                     [dim](const OrthoAtom& a, const OrthoAtom& b) {
                         return ortho_coord(a, dim) < ortho_coord(b, dim);
                     });

    // Spawn a task for the left half if the slice is large enough; recurse
    // on the right half on the current thread. Tasks at every level give a
    // wide-and-shallow task graph that the OpenMP runtime balances well.
    constexpr int kParThreshold = 25000;
    if (n > kParThreshold) {
#pragma omp task default(shared) firstprivate(atoms, median, level)
        build_ortho(atoms, median, level + 1);
        build_ortho(atoms + median + 1, n - median - 1, level + 1);
#pragma omp taskwait
    } else {
        build_ortho(atoms, median, level + 1);
        build_ortho(atoms + median + 1, n - median - 1, level + 1);
    }
}

// k-NN query against an implicit-tree slice. Walks the median-split tree
// recursively: descend near-side first, then check the splitting plane and
// recurse into the far side only if it could improve the heap.
static void query_ortho(const OrthoAtom* atoms, int n, const double q[3],
                        int self_idx, TopK& heap, int level) {
    if (n <= kOrthoBucket) {
        for (int i = 0; i < n; ++i) {
            double d2 = squared_dist_ortho(atoms[i], q);
            if (atoms[i].idx == self_idx && d2 == 0.0) continue;
            heap.push(d2, atoms[i].idx);
        }
        return;
    }
    int dim = level % 3;
    int median = n / 2;
    const OrthoAtom& stem = atoms[median];
    double dx = ortho_coord(stem, dim) - q[dim];
    bool go_left = dx > 0.0;

    if (go_left) {
        query_ortho(atoms, median, q, self_idx, heap, level + 1);
    } else {
        query_ortho(atoms + median + 1, n - median - 1, q, self_idx, heap, level + 1);
    }

    // Need to check the stem and the far side only if the splitting plane
    // is closer than the current k-th best.
    if (!heap.full() || heap.worst() > dx * dx) {
        double d2 = squared_dist_ortho(stem, q);
        if (!(stem.idx == self_idx && d2 == 0.0)) {
            heap.push(d2, stem.idx);
        }
        if (go_left) {
            query_ortho(atoms + median + 1, n - median - 1, q, self_idx, heap, level + 1);
        } else {
            query_ortho(atoms, median, q, self_idx, heap, level + 1);
        }
    }
}

class OrthoKdTree {
public:
    void build(const double* x, const double* y, const double* z,
               size_t N, const Box& box)
    {
        box_ = box;
        // Wrap each atom into the primary cell and stage in atoms_. Track
        // the actual bounding box of the wrapped atoms (= primary box for
        // PBC axes, but the actual data extent for open axes).
        const double Lx = box_.data[0], Ly = box_.data[4], Lz = box_.data[8];
        const double invLx = 1.0 / Lx, invLy = 1.0 / Ly, invLz = 1.0 / Lz;
        const int bx = box_.boundary[0], by = box_.boundary[1], bz = box_.boundary[2];
        const double Ox = box_.origin[0], Oy = box_.origin[1], Oz = box_.origin[2];
        atoms_.resize(N);
#pragma omp parallel for schedule(static) if((long long)N > 50000)
        for (long long i = 0; i < (long long)N; ++i) {
            double px = x[i], py = y[i], pz = z[i];
            if (bx) {
                double s = std::floor((px - Ox) * invLx);
                if (s != 0.0) px -= s * Lx;
            }
            if (by) {
                double s = std::floor((py - Oy) * invLy);
                if (s != 0.0) py -= s * Ly;
            }
            if (bz) {
                double s = std::floor((pz - Oz) * invLz);
                if (s != 0.0) pz -= s * Lz;
            }
            atoms_[i] = OrthoAtom{px, py, pz, static_cast<int>(i)};
        }

        // Root bounding box: PBC axes use the primary cell; open axes use
        // the actual data extent. Used by query() for per-image pruning.
        bmn_[0] = bx ? Ox : std::numeric_limits<double>::infinity();
        bmn_[1] = by ? Oy : std::numeric_limits<double>::infinity();
        bmn_[2] = bz ? Oz : std::numeric_limits<double>::infinity();
        bmx_[0] = bx ? Ox + Lx : -std::numeric_limits<double>::infinity();
        bmx_[1] = by ? Oy + Ly : -std::numeric_limits<double>::infinity();
        bmx_[2] = bz ? Oz + Lz : -std::numeric_limits<double>::infinity();
        if (!bx || !by || !bz) {
            for (size_t i = 0; i < N; ++i) {
                if (!bx) {
                    if (atoms_[i].x < bmn_[0]) bmn_[0] = atoms_[i].x;
                    if (atoms_[i].x > bmx_[0]) bmx_[0] = atoms_[i].x;
                }
                if (!by) {
                    if (atoms_[i].y < bmn_[1]) bmn_[1] = atoms_[i].y;
                    if (atoms_[i].y > bmx_[1]) bmx_[1] = atoms_[i].y;
                }
                if (!bz) {
                    if (atoms_[i].z < bmn_[2]) bmn_[2] = atoms_[i].z;
                    if (atoms_[i].z > bmx_[2]) bmx_[2] = atoms_[i].z;
                }
            }
        }

        // Recursive build, parallelized via OpenMP tasks at the top levels.
        if (N > kOrthoBucket) {
#pragma omp parallel
            {
#pragma omp single
                build_ortho(atoms_.data(), static_cast<int>(N), 0);
            }
        }
    }

    void query(const double q_orig[3], int self_idx, TopK& heap,
               const std::vector<std::array<double, 3>>& shifts) const
    {
        for (const auto& shift : shifts) {
            double q[3] = {
                q_orig[0] - shift[0],
                q_orig[1] - shift[1],
                q_orig[2] - shift[2]
            };
            // Per-image whole-domain prune.
            if (heap.full() && bbox_min_dist_sq(q) >= heap.worst()) continue;
            query_ortho(atoms_.data(), static_cast<int>(atoms_.size()),
                        q, self_idx, heap, 0);
        }
    }

    size_t size() const { return atoms_.size(); }

private:
    inline double bbox_min_dist_sq(const double q[3]) const {
        double d = 0.0;
        for (int i = 0; i < 3; ++i) {
            double v = 0.0;
            if (q[i] < bmn_[i])      v = bmn_[i] - q[i];
            else if (q[i] > bmx_[i]) v = q[i] - bmx_[i];
            d += v * v;
        }
        return d;
    }

    Box box_{};
    double bmn_[3]{0.0, 0.0, 0.0};
    double bmx_[3]{0.0, 0.0, 0.0};
    std::vector<OrthoAtom> atoms_;
};

// ---------------------------------------------------------------------------
//  Build the list of periodic image shifts. Sorted by squared length so the
//  query loop visits the closest images first — most images get skipped
//  once the priority queue has converged.
// ---------------------------------------------------------------------------
static std::vector<std::array<double, 3>>
build_pbc_shifts(const Box& box, size_t N)
{
    bool any_pbc = box.boundary[0] || box.boundary[1] || box.boundary[2];

    int nimages = 1;
    if (any_pbc) {
        // OVITO's heuristic: bigger systems need fewer images.
        nimages = static_cast<int>(200 / std::clamp<size_t>(N, 50u, 200u));
        if (nimages < 1) nimages = 1;
        if (nimages < 2 && box.triclinic) nimages = 2;
    }

    int nx = box.boundary[0] ? nimages : 0;
    int ny = box.boundary[1] ? nimages : 0;
    int nz = box.boundary[2] ? nimages : 0;

    std::vector<std::array<double, 3>> shifts;
    shifts.reserve((2 * nx + 1) * (2 * ny + 1) * (2 * nz + 1));
    for (int iz = -nz; iz <= nz; ++iz)
        for (int iy = -ny; iy <= ny; ++iy)
            for (int ix = -nx; ix <= nx; ++ix) {
                std::array<double, 3> s;
                if (box.triclinic) {
                    s[0] = ix * box.data[0] + iy * box.data[3] + iz * box.data[6];
                    s[1] = ix * box.data[1] + iy * box.data[4] + iz * box.data[7];
                    s[2] = ix * box.data[2] + iy * box.data[5] + iz * box.data[8];
                } else {
                    s[0] = ix * box.data[0];
                    s[1] = iy * box.data[4];
                    s[2] = iz * box.data[8];
                }
                shifts.push_back(s);
            }
    std::sort(shifts.begin(), shifts.end(),
              [](const std::array<double, 3>& a, const std::array<double, 3>& b) {
                  return a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
                       < b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
              });
    return shifts;
}

// ---------------------------------------------------------------------------
//  Public entry point
// ---------------------------------------------------------------------------
static void knn(const ROneArrayD x_py, const ROneArrayD y_py, const ROneArrayD z_py,
                const RTwoArrayD box_py, const ROneArrayD origin,
                const ROneArrayI boundary, int k,
                TwoArrayI indices_py, TwoArrayD distances_py)
{
    size_t N = x_py.shape(0);
    auto idx_view = indices_py.view();
    auto dst_view = distances_py.view();
    Box box = get_box(box_py, origin, boundary);

    auto shifts = build_pbc_shifts(box, N);

    const double* xp = x_py.data();
    const double* yp = y_py.data();
    const double* zp = z_py.data();

    int nthreads = omp_get_max_threads();
    int chunk = std::max(64, static_cast<int>(N / (nthreads * 16)));

    if (!box.triclinic) {
        // Orthogonal fast path: bosque-style implicit kd-tree.
        OrthoKdTree tree;
        tree.build(xp, yp, zp, N, box);

#pragma omp parallel
        {
            TopK heap(k);
#pragma omp for schedule(dynamic, chunk) nowait
            for (int i = 0; i < static_cast<int>(N); ++i) {
                double q[3] = { xp[i], yp[i], zp[i] };
                heap.clear();
                tree.query(q, i, heap, shifts);
                for (int j = 0; j < heap.n; ++j) {
                    idx_view(i, j) = heap.idx[j];
                    dst_view(i, j) = std::sqrt(heap.d2[j]);
                }
                for (int j = heap.n; j < k; ++j) {
                    idx_view(i, j) = -1;
                    dst_view(i, j) = -1.0;
                }
            }
        }
        return;
    }

    // Triclinic: use the general partition-based kd-tree (handles
    // non-axis-aligned cell normals correctly).
    int bucket = std::max(8, k / 2);
    KdTree tree;
    tree.build(xp, yp, zp, N, box, bucket, /*depth_limit*/ 20);

#pragma omp parallel
    {
        TopK heap(k);
#pragma omp for schedule(dynamic, chunk) nowait
        for (int i = 0; i < static_cast<int>(N); ++i) {
            double q[3] = { xp[i], yp[i], zp[i] };
            heap.clear();
            tree.query(q, i, heap, shifts);
            for (int j = 0; j < heap.n; ++j) {
                idx_view(i, j) = heap.idx[j];
                dst_view(i, j) = std::sqrt(heap.d2[j]);
            }
            for (int j = heap.n; j < k; ++j) {
                idx_view(i, j) = -1;
                dst_view(i, j) = -1.0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  Tree wrapper for one-off / batched nearest-site queries (used by the
//  Wigner-Seitz analysis). Builds the kd-tree once from a reference set,
//  then `query_nearest_batch` returns the index of the closest reference
//  site for each query point (k = 1, with PBC).
// ---------------------------------------------------------------------------
class Tree {
public:
    Tree() = default;

    void build_with_coords(const ROneArrayD x_py, const ROneArrayD y_py, const ROneArrayD z_py,
                           const RTwoArrayD box_py, const ROneArrayD origin,
                           const ROneArrayI boundary)
    {
        size_t N = x_py.shape(0);
        box_ = get_box(box_py, origin, boundary);
        shifts_ = build_pbc_shifts(box_, N);
        if (!box_.triclinic) {
            ortho_tree_ = std::make_unique<OrthoKdTree>();
            ortho_tree_->build(x_py.data(), y_py.data(), z_py.data(), N, box_);
        } else {
            tri_tree_ = std::make_unique<KdTree>();
            tri_tree_->build(x_py.data(), y_py.data(), z_py.data(), N, box_,
                             /*bucket*/ 8, /*depth_limit*/ 20);
        }
    }

    void query_nearest_batch(const ROneArrayD qx, const ROneArrayD qy, const ROneArrayD qz,
                             OneArrayI indices_py) const
    {
        size_t M = qx.shape(0);
        auto out = indices_py.view();
        const double* xp = qx.data();
        const double* yp = qy.data();
        const double* zp = qz.data();

        int nthreads = omp_get_max_threads();
        int chunk = std::max(64, static_cast<int>(M / (nthreads * 16)));

#pragma omp parallel
        {
            TopK heap(1);
#pragma omp for schedule(dynamic, chunk) nowait
            for (int i = 0; i < static_cast<int>(M); ++i) {
                double q[3] = { xp[i], yp[i], zp[i] };
                heap.clear();
                // self_index = -1 ⇒ never skip self.
                if (ortho_tree_) ortho_tree_->query(q, -1, heap, shifts_);
                else             tri_tree_->query(q, -1, heap, shifts_);
                out(i) = heap.idx[0];
            }
        }
    }

    // k-nearest batch query — used for per-stage benchmarking. When
    // `exclude_self` is true, points that match the query position exactly
    // are skipped (mimics the `knn` free function used by build_nearest_neighbor).
    void query_knn_batch(const ROneArrayD qx, const ROneArrayD qy, const ROneArrayD qz,
                         int k, bool exclude_self,
                         TwoArrayI indices_py, TwoArrayD distances_py) const
    {
        size_t M = qx.shape(0);
        auto idx_view = indices_py.view();
        auto dst_view = distances_py.view();
        const double* xp = qx.data();
        const double* yp = qy.data();
        const double* zp = qz.data();

        int nthreads = omp_get_max_threads();
        int chunk = std::max(64, static_cast<int>(M / (nthreads * 16)));

#pragma omp parallel
        {
            TopK heap(k);
#pragma omp for schedule(dynamic, chunk) nowait
            for (int i = 0; i < static_cast<int>(M); ++i) {
                double q[3] = { xp[i], yp[i], zp[i] };
                heap.clear();
                int self = exclude_self ? i : -1;
                if (ortho_tree_) ortho_tree_->query(q, self, heap, shifts_);
                else             tri_tree_->query(q, self, heap, shifts_);

                for (int j = 0; j < heap.n; ++j) {
                    idx_view(i, j) = heap.idx[j];
                    dst_view(i, j) = std::sqrt(heap.d2[j]);
                }
                for (int j = heap.n; j < k; ++j) {
                    idx_view(i, j) = -1;
                    dst_view(i, j) = -1.0;
                }
            }
        }
    }

private:
    std::unique_ptr<OrthoKdTree> ortho_tree_;
    std::unique_ptr<KdTree> tri_tree_;
    Box box_{};
    std::vector<std::array<double, 3>> shifts_;
};

}  // namespace fast_knn

NB_MODULE(_fast_knn, m) {
    m.def("knn", &fast_knn::knn);
    nb::class_<fast_knn::Tree>(m, "Tree")
        .def(nb::init<>())
        .def("build_with_coords", &fast_knn::Tree::build_with_coords)
        .def("query_nearest_batch", &fast_knn::Tree::query_nearest_batch)
        .def("query_knn_batch", &fast_knn::Tree::query_knn_batch);
}
