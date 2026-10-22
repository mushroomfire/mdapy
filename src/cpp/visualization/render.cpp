// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "visualization/render.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include "visualization/render_gpu.h"

namespace mdapy::visualization {
namespace {

using V = std::array<float, 3>;

inline V operator+(V a, V b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline V operator-(V a, V b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline V operator*(V a, float s) { return {a[0] * s, a[1] * s, a[2] * s}; }
inline V operator*(V a, V b) { return {a[0] * b[0], a[1] * b[1], a[2] * b[2]}; }
inline float dot(V a, V b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline V cross(V a, V b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline V unit(V a) {
    const float length = std::sqrt(dot(a, a));
    return length > 0 ? a * (1.0f / length) : a;
}
inline V load(const float* p) { return {p[0], p[1], p[2]}; }

constexpr float kInfinity = std::numeric_limits<float>::infinity();
constexpr std::uint32_t kSphere = 0, kCylinder = 1, kTriangle = 2;
constexpr std::uint32_t kKindShift = 30;
constexpr std::uint32_t kIndexMask = (1U << kKindShift) - 1;

struct Ray {
    V origin, direction, inverse;
};

Ray make_ray(V origin, V direction) {
    V inverse;
    for (int e = 0; e < 3; ++e) {
        const float d = std::fabs(direction[e]) < 1e-20f ? std::copysign(1e-20f, direction[e])
                                                          : direction[e];
        inverse[e] = 1.0f / d;
    }
    return {origin, direction, inverse};
}

struct Hit {
    float t = kInfinity;
    std::uint32_t ref = 0;  // kind and index
};

struct Bounds {
    V lo{kInfinity, kInfinity, kInfinity}, hi{-kInfinity, -kInfinity, -kInfinity};
    void grow(V p) {
        for (int e = 0; e < 3; ++e) {
            lo[e] = std::min(lo[e], p[e]);
            hi[e] = std::max(hi[e], p[e]);
        }
    }
    void grow(const Bounds& b) {
        grow(b.lo);
        grow(b.hi);
    }
    float area() const {
        const V d = hi - lo;
        if (d[0] < 0) {
            return 0;
        }
        return d[0] * d[1] + d[1] * d[2] + d[2] * d[0];
    }
};

struct Node {
    V lo;
    std::uint32_t first;  // the first reference of a leaf, or the left child
    V hi;
    std::uint32_t count;  // references of a leaf; 0 for an inner node
};

// A bounding volume hierarchy of the primitives of a scene, built with the
// surface area heuristic over binned centroids.
class Bvh {
public:
    explicit Bvh(const Scene& scene) : scene_(scene) {
        const std::size_t spheres = scene.sphere_radius.size();
        const std::size_t cylinders = scene.cylinder_radius.size();
        const std::size_t triangles = scene.triangle_alpha.size();
        const std::size_t n = spheres + cylinders + triangles;
        if (n > kIndexMask) {
            throw std::invalid_argument("too many primitives for a picture");
        }
        refs_.reserve(n);
        bounds_.reserve(n);
        for (std::size_t i = 0; i < spheres; ++i) {
            refs_.push_back((kSphere << kKindShift) | static_cast<std::uint32_t>(i));
        }
        for (std::size_t i = 0; i < cylinders; ++i) {
            refs_.push_back((kCylinder << kKindShift) | static_cast<std::uint32_t>(i));
        }
        for (std::size_t i = 0; i < triangles; ++i) {
            refs_.push_back((kTriangle << kKindShift) | static_cast<std::uint32_t>(i));
        }
        for (const std::uint32_t ref : refs_) {
            bounds_.push_back(bounds_of(ref));
        }
        build();
    }

    bool empty() const { return refs_.empty(); }
    const Bounds& scene_bounds() const { return all_; }
    const std::vector<Node>& nodes() const { return nodes_; }
    const std::vector<std::uint32_t>& refs() const { return refs_; }

    // The nearest hit beyond t_min and below hit.t.
    void closest(const Ray& ray, float t_min, Hit& hit) const {
        if (refs_.empty()) {
            return;
        }
        std::uint32_t stack[96];
        int top = 0;
        stack[top++] = 0;
        while (top > 0) {
            const Node& node = nodes_[stack[--top]];
            if (node.count > 0) {
                for (std::uint32_t k = node.first; k < node.first + node.count; ++k) {
                    const float t = intersect(refs_[k], ray, t_min, hit.t);
                    if (t < hit.t) {
                        hit.t = t;
                        hit.ref = refs_[k];
                    }
                }
                continue;
            }
            const float near_left = enter(nodes_[node.first], ray, t_min, hit.t);
            const float near_right = enter(nodes_[node.first + 1], ray, t_min, hit.t);
            if (near_left <= near_right) {
                if (near_right < kInfinity) {
                    stack[top++] = node.first + 1;
                }
                if (near_left < kInfinity) {
                    stack[top++] = node.first;
                }
            } else {
                if (near_left < kInfinity) {
                    stack[top++] = node.first;
                }
                if (near_right < kInfinity) {
                    stack[top++] = node.first + 1;
                }
            }
        }
    }

    // Whether an opaque primitive lies along the ray between t_min and t_max.
    bool occluded(const Ray& ray, float t_min, float t_max) const {
        if (refs_.empty()) {
            return false;
        }
        std::uint32_t stack[96];
        int top = 0;
        stack[top++] = 0;
        while (top > 0) {
            const Node& node = nodes_[stack[--top]];
            if (enter(node, ray, t_min, t_max) == kInfinity) {
                continue;
            }
            if (node.count > 0) {
                for (std::uint32_t k = node.first; k < node.first + node.count; ++k) {
                    const std::uint32_t ref = refs_[k];
                    if ((ref >> kKindShift) == kTriangle &&
                        scene_.triangle_alpha[ref & kIndexMask] < 1.0f) {
                        continue;  // see-through faces cast no shadow
                    }
                    if (intersect(ref, ray, t_min, t_max) < t_max) {
                        return true;
                    }
                }
                continue;
            }
            stack[top++] = node.first;
            stack[top++] = node.first + 1;
        }
        return false;
    }

    // The distance along the ray to a primitive, or infinity.
    float intersect(std::uint32_t ref, const Ray& ray, float t_min, float t_max) const {
        const std::uint32_t i = ref & kIndexMask;
        switch (ref >> kKindShift) {
            case kSphere: {
                const V oc = ray.origin - load(&scene_.sphere_center[3 * i]);
                const float r = scene_.sphere_radius[i];
                const float b = dot(oc, ray.direction);
                const float c = dot(oc, oc) - r * r;
                const float disc = b * b - c;
                if (disc < 0) {
                    return kInfinity;
                }
                const float s = std::sqrt(disc);
                float t = -b - s;
                if (t <= t_min) {
                    t = -b + s;
                }
                return t > t_min && t < t_max ? t : kInfinity;
            }
            case kCylinder: {
                const V p0 = load(&scene_.cylinder_end[6 * i]);
                const V axis = load(&scene_.cylinder_end[6 * i + 3]) - p0;
                const float length = std::sqrt(dot(axis, axis));
                if (length == 0) {
                    return kInfinity;
                }
                const V a = axis * (1.0f / length);
                const V oc = ray.origin - p0;
                const V d = ray.direction - a * dot(ray.direction, a);
                const V o = oc - a * dot(oc, a);
                const float r = scene_.cylinder_radius[i];
                const float qa = dot(d, d);
                if (qa < 1e-12f) {
                    return kInfinity;
                }
                const float qb = dot(d, o);
                const float qc = dot(o, o) - r * r;
                const float disc = qb * qb - qa * qc;
                if (disc < 0) {
                    return kInfinity;
                }
                const float s = std::sqrt(disc);
                for (const float t : {(-qb - s) / qa, (-qb + s) / qa}) {
                    if (t > t_min && t < t_max) {
                        const float along = dot(oc + ray.direction * t, a);
                        if (along >= 0 && along <= length) {
                            return t;
                        }
                    }
                }
                return kInfinity;
            }
            default: {
                const float* v = &scene_.triangle_vertex[9 * i];
                const V v0 = load(v), e1 = load(v + 3) - v0, e2 = load(v + 6) - v0;
                const V p = cross(ray.direction, e2);
                const float det = dot(e1, p);
                if (std::fabs(det) < 1e-20f) {
                    return kInfinity;
                }
                const float inverse = 1.0f / det;
                const V s = ray.origin - v0;
                const float u = dot(s, p) * inverse;
                if (u < 0 || u > 1) {
                    return kInfinity;
                }
                const V q = cross(s, e1);
                const float w = dot(ray.direction, q) * inverse;
                if (w < 0 || u + w > 1) {
                    return kInfinity;
                }
                const float t = dot(e2, q) * inverse;
                return t > t_min && t < t_max ? t : kInfinity;
            }
        }
    }

    // The unit normal at a point of a primitive, facing the ray.
    V normal(std::uint32_t ref, V point, V direction) const {
        const std::uint32_t i = ref & kIndexMask;
        V n;
        switch (ref >> kKindShift) {
            case kSphere:
                n = unit(point - load(&scene_.sphere_center[3 * i]));
                break;
            case kCylinder: {
                const V p0 = load(&scene_.cylinder_end[6 * i]);
                const V a = unit(load(&scene_.cylinder_end[6 * i + 3]) - p0);
                const V rel = point - p0;
                n = unit(rel - a * dot(rel, a));
                break;
            }
            default: {
                const float* v = &scene_.triangle_vertex[9 * i];
                n = unit(cross(load(v + 3) - load(v), load(v + 6) - load(v)));
                break;
            }
        }
        return dot(n, direction) > 0 ? n * -1.0f : n;
    }

    V color(std::uint32_t ref) const {
        const std::uint32_t i = ref & kIndexMask;
        switch (ref >> kKindShift) {
            case kSphere:
                return load(&scene_.sphere_color[3 * i]);
            case kCylinder:
                return load(&scene_.cylinder_color[3 * i]);
            default:
                return load(&scene_.triangle_color[3 * i]);
        }
    }

    float alpha(std::uint32_t ref) const {
        return (ref >> kKindShift) == kTriangle ? scene_.triangle_alpha[ref & kIndexMask] : 1.0f;
    }

    // An id that differs between primitives meant to be outlined apart.
    std::int64_t id(std::uint32_t ref) const {
        const std::uint32_t i = ref & kIndexMask;
        const std::uint32_t kind = ref >> kKindShift;
        const std::int32_t* ids = kind == kSphere     ? scene_.sphere_id.data()
                                  : kind == kCylinder ? scene_.cylinder_id.data()
                                                      : scene_.triangle_id.data();
        return (static_cast<std::int64_t>(kind) << 32) + ids[i];
    }

private:
    Bounds bounds_of(std::uint32_t ref) const {
        const std::uint32_t i = ref & kIndexMask;
        Bounds b;
        switch (ref >> kKindShift) {
            case kSphere: {
                const V c = load(&scene_.sphere_center[3 * i]);
                const float r = scene_.sphere_radius[i];
                b.grow(c - V{r, r, r});
                b.grow(c + V{r, r, r});
                break;
            }
            case kCylinder: {
                const float r = scene_.cylinder_radius[i];
                for (int end = 0; end < 2; ++end) {
                    const V c = load(&scene_.cylinder_end[6 * i + 3 * end]);
                    b.grow(c - V{r, r, r});
                    b.grow(c + V{r, r, r});
                }
                break;
            }
            default:
                for (int k = 0; k < 3; ++k) {
                    b.grow(load(&scene_.triangle_vertex[9 * i + 3 * k]));
                }
        }
        return b;
    }

    // The distance at which the ray enters a node, or infinity.
    static float enter(const Node& node, const Ray& ray, float t_min, float t_max) {
        float t_in = t_min, t_out = t_max;
        for (int e = 0; e < 3; ++e) {
            float t0 = (node.lo[e] - ray.origin[e]) * ray.inverse[e];
            float t1 = (node.hi[e] - ray.origin[e]) * ray.inverse[e];
            if (t0 > t1) {
                std::swap(t0, t1);
            }
            t_in = t0 > t_in ? t0 : t_in;
            t_out = t1 < t_out ? t1 : t_out;
        }
        return t_in <= t_out ? t_in : kInfinity;
    }

    struct Task {
        std::uint32_t node, begin, end;
    };

    // One node over order[begin, end): its bounds, and its children over the
    // two parts of the range, split by the surface area heuristic over binned
    // centroids; a leaf when it holds few references.
    void split(std::vector<std::uint32_t>& order, const std::vector<V>& centroid,
               std::vector<Node>& nodes, const Task& task, std::vector<Task>& children) const {
        Bounds box, centers;
        for (std::uint32_t k = task.begin; k < task.end; ++k) {
            box.grow(bounds_[order[k]]);
            centers.grow(centroid[order[k]]);
        }
        nodes[task.node].lo = box.lo;
        nodes[task.node].hi = box.hi;
        const std::uint32_t count = task.end - task.begin;
        int axis = 0;
        const V extent = centers.hi - centers.lo;
        if (extent[1] > extent[axis]) {
            axis = 1;
        }
        if (extent[2] > extent[axis]) {
            axis = 2;
        }
        if (count <= 4 || extent[axis] <= 0) {
            nodes[task.node].first = task.begin;
            nodes[task.node].count = count;
            return;
        }
        constexpr int kBins = 16;
        const float low = centers.lo[axis];
        const float scale = kBins / extent[axis];
        const auto bin_of = [&](std::uint32_t k) {
            return std::min(kBins - 1, static_cast<int>((centroid[k][axis] - low) * scale));
        };
        Bounds bin_box[kBins];
        std::uint32_t bin_count[kBins] = {};
        for (std::uint32_t k = task.begin; k < task.end; ++k) {
            const int b = bin_of(order[k]);
            ++bin_count[b];
            bin_box[b].grow(bounds_[order[k]]);
        }
        float right_area[kBins] = {};
        std::uint32_t right_count[kBins] = {};
        Bounds sweep;
        std::uint32_t total = 0;
        for (int b = kBins - 1; b > 0; --b) {
            sweep.grow(bin_box[b]);
            total += bin_count[b];
            right_area[b] = sweep.area();
            right_count[b] = total;
        }
        Bounds left;
        std::uint32_t left_count = 0;
        float best = kInfinity;
        int split_bin = -1;
        for (int b = 0; b < kBins - 1; ++b) {
            left.grow(bin_box[b]);
            left_count += bin_count[b];
            if (left_count == 0 || right_count[b + 1] == 0) {
                continue;
            }
            const float cost = left_count * left.area() + right_count[b + 1] * right_area[b + 1];
            if (cost < best) {
                best = cost;
                split_bin = b;
            }
        }
        std::uint32_t middle;
        if (split_bin < 0) {
            middle = task.begin + count / 2;
            std::nth_element(order.begin() + task.begin, order.begin() + middle,
                             order.begin() + task.end, [&](std::uint32_t a, std::uint32_t b) {
                                 return centroid[a][axis] < centroid[b][axis];
                             });
        } else {
            const auto mid = std::partition(order.begin() + task.begin, order.begin() + task.end,
                                            [&](std::uint32_t k) { return bin_of(k) <= split_bin; });
            middle = static_cast<std::uint32_t>(mid - order.begin());
        }
        const auto left_node = static_cast<std::uint32_t>(nodes.size());
        nodes.push_back(Node{});
        nodes.push_back(Node{});
        nodes[task.node].first = left_node;
        nodes[task.node].count = 0;
        children.push_back({left_node, task.begin, middle});
        children.push_back({left_node + 1, middle, task.end});
    }

    void build() {
        const std::size_t n = refs_.size();
        if (n == 0) {
            return;
        }
        std::vector<V> centroid(n);
        for (std::size_t k = 0; k < n; ++k) {
            centroid[k] = (bounds_[k].lo + bounds_[k].hi) * 0.5f;
        }
        // the references, their bounds and centroids sorted together
        std::vector<std::uint32_t> order(n);
        for (std::size_t k = 0; k < n; ++k) {
            order[k] = static_cast<std::uint32_t>(k);
        }
        nodes_.reserve(2 * n);
        nodes_.push_back(Node{});
        // The top of the tree breadth first in this thread, until enough
        // subtrees are left; the order of the work, and so the tree, does not
        // depend on the threads.
        constexpr std::size_t kSubtrees = 256;
        constexpr std::uint32_t kSmall = 4096;
        std::vector<Task> queue{{0, 0, static_cast<std::uint32_t>(n)}}, subtrees;
        for (std::size_t head = 0; head < queue.size(); ++head) {
            const Task task = queue[head];
            if (task.end - task.begin <= kSmall ||
                queue.size() - head - 1 + subtrees.size() >= kSubtrees) {
                subtrees.push_back(task);
                continue;
            }
            split(order, centroid, nodes_, task, queue);
        }
        // each subtree on its own, its root first, children numbered within
        std::vector<std::vector<Node>> parts(subtrees.size());
        parallel_for(static_cast<std::int64_t>(subtrees.size()), 1,
                     [&](std::int64_t begin, std::int64_t end) {
                         for (std::int64_t k = begin; k < end; ++k) {
                             std::vector<Node>& part = parts[static_cast<std::size_t>(k)];
                             part.push_back(Node{});
                             std::vector<Task> tasks{
                                 {0, subtrees[k].begin, subtrees[k].end}};
                             while (!tasks.empty()) {
                                 const Task task = tasks.back();
                                 tasks.pop_back();
                                 split(order, centroid, part, task, tasks);
                             }
                         }
                     });
        for (std::size_t k = 0; k < subtrees.size(); ++k) {
            // node j > 0 of the part goes to offset + j
            const auto offset = static_cast<std::uint32_t>(nodes_.size()) - 1;
            const auto place = [&](Node node) {
                if (node.count == 0) {
                    node.first += offset;
                }
                return node;
            };
            nodes_[subtrees[k].node] = place(parts[k][0]);
            for (std::size_t j = 1; j < parts[k].size(); ++j) {
                nodes_.push_back(place(parts[k][j]));
            }
        }
        // the references in the order of the leaves
        std::vector<std::uint32_t> refs(n);
        for (std::size_t k = 0; k < n; ++k) {
            refs[k] = refs_[order[k]];
            all_.grow(bounds_[order[k]]);
        }
        refs_.swap(refs);
    }

    const Scene& scene_;
    std::vector<std::uint32_t> refs_;
    std::vector<Bounds> bounds_;
    std::vector<Node> nodes_;
    Bounds all_;
};

// A small fast random number generator, seeded per sample so that the
// picture does not depend on the number of threads.
struct Random {
    std::uint32_t state;
    explicit Random(std::uint32_t seed) : state(seed * 747796405U + 2891336453U) {}
    float next() {
        state = state * 747796405U + 2891336453U;
        std::uint32_t word = ((state >> ((state >> 28U) + 4U)) ^ state) * 277803737U;
        word = (word >> 22U) ^ word;
        return static_cast<float>(word >> 8) * (1.0f / 16777216.0f);
    }
};

std::uint32_t mix(std::uint32_t a, std::uint32_t b) {
    std::uint32_t h = a * 0x9E3779B1U ^ (b + 0x7F4A7C15U + (a << 6) + (a >> 2));
    h ^= h >> 16;
    h *= 0x85EBCA6BU;
    h ^= h >> 13;
    return h;
}

float to_srgb(float linear) {
    linear = std::clamp(linear, 0.0f, 1.0f);
    return linear <= 0.0031308f ? 12.92f * linear
                                : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
}

// The nodes are handed to the GPU as they are.
static_assert(sizeof(Node) == 32, "a node is 8 words");

void copy3(float* to, const V& from) {
    for (int e = 0; e < 3; ++e) {
        to[e] = from[e];
    }
}

}  // namespace

#ifndef MDAPY_METAL
namespace gpu {
bool available() { return false; }
void trace(const Geometry&, const Params&, float*, float*, std::int64_t*, const Control&) {
    throw std::runtime_error("no GPU: this build of mdapy has none");
}
}  // namespace gpu
#endif

bool gpu_available() { return gpu::available(); }

std::vector<std::uint8_t> render(const Scene& scene, const Camera& camera, const Style& style,
                                 int width, int height, const Control& control) {
    if (width < 1 || height < 1 || style.supersample < 1 || style.samples < 1 ||
        style.bounces < 0) {
        throw std::invalid_argument(
            "the picture needs a positive size, supersampling and samples, and bounces of 0 or more");
    }
    const Bvh bvh(scene);
    const int s = style.supersample;
    const int sw = width * s, sh = height * s;
    const V forward = unit(camera.forward);
    const V right = unit(cross(forward, camera.up));
    const V up = cross(right, forward);
    const V center = camera.center;
    // lights in the world, from their directions in the frame of the camera
    const V key = unit(right * style.key[0] + up * style.key[1] - forward * style.key[2]);
    const V fill = unit(right * style.fill[0] + up * style.fill[1] - forward * style.fill[2]);
    // the depth range of the scene along the view, for the fog and the origin
    // of orthographic rays
    float nearest = 0, farthest = 1;
    if (!bvh.empty()) {
        const Bounds& b = bvh.scene_bounds();
        nearest = kInfinity;
        farthest = -kInfinity;
        for (int c = 0; c < 8; ++c) {
            const V corner{(c & 1) ? b.hi[0] : b.lo[0], (c & 2) ? b.hi[1] : b.lo[1],
                           (c & 4) ? b.hi[2] : b.lo[2]};
            const float depth = dot(corner - center, forward);
            nearest = std::min(nearest, depth);
            farthest = std::max(farthest, depth);
        }
    }
    const float start = camera.perspective ? 0.0f : nearest - 1.0f;
    const V background{style.background[0], style.background[1], style.background[2]};
    const std::size_t pixels = static_cast<std::size_t>(sw) * static_cast<std::size_t>(sh);
    std::vector<float> rgba(4 * pixels);
    std::vector<float> depth(pixels, kInfinity);
    std::vector<std::int64_t> ids(pixels, -1);
    const float epsilon = 1e-4f * std::max(1.0f, farthest - nearest);

    const auto shade = [&](const Hit& hit, const Ray& ray, Random& random) -> V {
        const V point = ray.origin + ray.direction * hit.t;
        const V n = bvh.normal(hit.ref, point, ray.direction);
        const V base = bvh.color(hit.ref);
        const V lifted = point + n * epsilon;
        float ao = 1.0f;
        if (style.ao_samples > 0 && style.ao_strength > 0) {
            // cosine-weighted directions about the normal
            const V helper = std::fabs(n[0]) < 0.9f ? V{1, 0, 0} : V{0, 1, 0};
            const V tangent = unit(cross(n, helper));
            const V bitangent = cross(n, tangent);
            int blocked = 0;
            for (int k = 0; k < style.ao_samples; ++k) {
                const float u = random.next(), phi = 6.2831853f * random.next();
                const float r = std::sqrt(u);
                const V d = tangent * (r * std::cos(phi)) + bitangent * (r * std::sin(phi)) +
                            n * std::sqrt(std::max(0.0f, 1.0f - u));
                blocked += bvh.occluded(make_ray(lifted, d), 0.0f, style.ao_distance) ? 1 : 0;
            }
            ao = 1.0f - style.ao_strength * static_cast<float>(blocked) /
                            static_cast<float>(style.ao_samples);
        }
        if (style.flat) {
            return base * ao;
        }
        float lit = std::max(0.0f, dot(n, key));
        float shadow = 1.0f;
        if (style.shadows && lit > 0 && bvh.occluded(make_ray(lifted, key), 0.0f, kInfinity)) {
            shadow = 1.0f - style.shadow_strength;
        }
        float diffuse = lit * shadow + style.fill_intensity * std::max(0.0f, dot(n, fill));
        if (style.toon_levels > 0) {
            const auto levels = static_cast<float>(style.toon_levels);
            diffuse = std::min(1.0f, std::floor(diffuse * levels + 0.5f) / levels);
        }
        const V halfway = unit(key - ray.direction);
        float shine = 0;
        if (style.specular > 0 && lit > 0) {
            shine = style.specular * std::pow(std::max(0.0f, dot(n, halfway)), style.shininess) *
                    shadow;
            if (style.toon_levels > 0) {
                shine = shine > 0.5f * style.specular ? style.specular : 0.0f;
            }
        }
        const float light = style.ambient * ao + style.diffuse * diffuse * (0.5f + 0.5f * ao);
        return base * light + V{shine, shine, shine};
    };

    // directions about a normal, cosine-weighted
    const auto cosine = [](V n, Random& random) -> V {
        const V helper = std::fabs(n[0]) < 0.9f ? V{1, 0, 0} : V{0, 1, 0};
        const V tangent = unit(cross(n, helper));
        const V bitangent = cross(n, tangent);
        const float u = random.next(), phi = 6.2831853f * random.next();
        const float r = std::sqrt(u);
        return tangent * (r * std::cos(phi)) + bitangent * (r * std::sin(phi)) +
               n * std::sqrt(std::max(0.0f, 1.0f - u));
    };
    // a direction toward the disk of the key light, uniform in its cone
    const V key_helper = std::fabs(key[0]) < 0.9f ? V{1, 0, 0} : V{0, 1, 0};
    const V key_tangent = unit(cross(key, key_helper));
    const V key_bitangent = cross(key, key_tangent);
    const float cone = std::cos(std::clamp(style.light_size, 0.0f, 1.5f));
    const auto toward_light = [&](Random& random) -> V {
        if (style.light_size <= 0) {
            return key;
        }
        const float z = 1.0f - random.next() * (1.0f - cone);
        const float phi = 6.2831853f * random.next();
        const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
        return key_tangent * (r * std::cos(phi)) + key_bitangent * (r * std::sin(phi)) + key * z;
    };
    // the light reaching the camera from a hit along one path: at each
    // surface, the key light it sees, then a diffuse bounce that either
    // escapes to the sky or reaches the next surface
    const auto path = [&](const Hit& first, const Ray& primary, Random& random) -> V {
        V radiance{0, 0, 0}, weight{1, 1, 1};
        Hit hit = first;
        Ray ray = primary;
        for (int bounce = 0;; ++bounce) {
            const V point = ray.origin + ray.direction * hit.t;
            const V n = bvh.normal(hit.ref, point, ray.direction);
            const V lifted = point + n * epsilon;
            const V reflected = weight * bvh.color(hit.ref);
            const V l = toward_light(random);
            const float lit = dot(n, l);
            if (lit > 0 && !bvh.occluded(make_ray(lifted, l), 0.0f, kInfinity)) {
                radiance = radiance + reflected * (style.diffuse * lit);
                if (bounce == 0 && style.specular > 0) {
                    const float shine =
                        style.specular *
                        std::pow(std::max(0.0f, dot(n, unit(l - ray.direction))), style.shininess);
                    radiance = radiance + V{shine, shine, shine};
                }
            }
            const Ray next_ray = make_ray(lifted, cosine(n, random));
            Hit next;
            bvh.closest(next_ray, 0.0f, next);
            if (next.t == kInfinity) {
                radiance = radiance + reflected * style.ambient;
                break;
            }
            if (bounce == style.bounces) {
                break;
            }
            weight = reflected;
            hit = next;
            ray = next_ray;
        }
        return radiance;
    };
    const bool global = style.bounces > 0;
    const int samples = global ? style.samples : 1;

    const std::int64_t grain = std::max<std::int64_t>(1, 64 / s);
    if (style.gpu) {
        gpu::Params params{};
        copy3(params.center, center);
        copy3(params.forward, forward);
        copy3(params.right, right);
        copy3(params.up, up);
        params.half_width = camera.half_width;
        params.half_height = camera.half_height;
        params.start = start;
        params.nearest = nearest;
        params.farthest = farthest;
        params.epsilon = epsilon;
        copy3(params.key, key);
        copy3(params.fill, fill);
        copy3(params.key_tangent, key_tangent);
        copy3(params.key_bitangent, key_bitangent);
        params.ambient = style.ambient;
        params.diffuse = style.diffuse;
        params.specular = style.specular;
        params.shininess = style.shininess;
        params.fill_intensity = style.fill_intensity;
        params.ao_distance = style.ao_distance;
        params.ao_strength = style.ao_strength;
        params.shadow_strength = style.shadow_strength;
        params.fog = style.fog;
        params.light_size = style.light_size;
        params.cone = cone;
        copy3(params.background, background);
        params.perspective = camera.perspective ? 1 : 0;
        params.ao_samples = style.ao_samples;
        params.shadows = style.shadows ? 1 : 0;
        params.flat = style.flat ? 1 : 0;
        params.toon_levels = style.toon_levels;
        params.bounces = style.bounces;
        params.samples = samples;
        params.width = sw;
        params.height = sh;
        params.refs = static_cast<std::uint32_t>(bvh.refs().size());
        const gpu::Geometry geometry{scene.sphere_center.data(),   scene.sphere_radius.data(),
                                     scene.sphere_color.data(),    scene.sphere_id.data(),
                                     scene.sphere_radius.size(),   scene.cylinder_end.data(),
                                     scene.cylinder_radius.data(), scene.cylinder_color.data(),
                                     scene.cylinder_id.data(),     scene.cylinder_radius.size(),
                                     scene.triangle_vertex.data(), scene.triangle_color.data(),
                                     scene.triangle_alpha.data(),  scene.triangle_id.data(),
                                     scene.triangle_alpha.size(),  bvh.nodes().data(),
                                     bvh.nodes().size(),           bvh.refs().data()};
        gpu::trace(geometry, params, rgba.data(), depth.data(), ids.data(), control);
    } else {
    parallel_for(
        sh, grain,
        [&](std::int64_t row_begin, std::int64_t row_end) {
            for (std::int64_t y = row_begin; y < row_end; ++y) {
                for (std::int64_t x = 0; x < sw; ++x) {
                    const std::size_t p = static_cast<std::size_t>(y) * static_cast<std::size_t>(sw) +
                                          static_cast<std::size_t>(x);
                    const std::uint32_t seed =
                        mix(static_cast<std::uint32_t>(x), static_cast<std::uint32_t>(y));
                    V color{0, 0, 0};
                    float coverage = 0.0f;
                    for (int sample = 0; sample < samples; ++sample) {
                        Random random(sample == 0 ? seed : mix(seed, static_cast<std::uint32_t>(sample)));
                        // the first sample at the center of the subpixel, the others
                        // anywhere in it
                        const float jx = sample == 0 ? 0.5f : random.next();
                        const float jy = sample == 0 ? 0.5f : random.next();
                        const float px = (2.0f * (static_cast<float>(x) + jx) / static_cast<float>(sw)) - 1.0f;
                        const float py = 1.0f - (2.0f * (static_cast<float>(y) + jy) / static_cast<float>(sh));
                        Ray ray;
                        if (camera.perspective) {
                            ray = make_ray(center, unit(forward + right * (px * camera.half_width) +
                                                        up * (py * camera.half_height)));
                        } else {
                            ray = make_ray(center + right * (px * camera.half_width) +
                                               up * (py * camera.half_height) + forward * start,
                                           forward);
                        }
                        float remaining = 1.0f, t_min = 0.0f;
                        for (int layer = 0; layer < 8 && remaining > 1e-3f; ++layer) {
                            Hit hit;
                            bvh.closest(ray, t_min, hit);
                            if (hit.t == kInfinity) {
                                break;
                            }
                            if (sample == 0 && layer == 0) {
                                depth[p] = hit.t + (camera.perspective ? 0.0f : start);
                                ids[p] = bvh.id(hit.ref);
                            }
                            V c = global ? path(hit, ray, random) : shade(hit, ray, random);
                            if (style.fog > 0 && farthest > nearest) {
                                const float f = style.fog * std::clamp((hit.t + start - nearest) / (farthest - nearest), 0.0f, 1.0f);
                                c = c * (1.0f - f) + background * f;
                            }
                            const float alpha = bvh.alpha(hit.ref);
                            color = color + c * (remaining * alpha);
                            remaining *= 1.0f - alpha;
                            t_min = hit.t + epsilon;
                        }
                        coverage += 1.0f - remaining;
                    }
                    color = color * (1.0f / static_cast<float>(samples));
                    coverage /= static_cast<float>(samples);
                    // premultiplied by the opacity; the background is added later
                    rgba[4 * p + 0] = color[0];
                    rgba[4 * p + 1] = color[1];
                    rgba[4 * p + 2] = color[2];
                    rgba[4 * p + 3] = coverage;
                }
            }
        },
        control);
    }

    // outlines: a subpixel behind a different primitive nearest it, or the
    // background next to a primitive, takes the color of the line
    const int reach = static_cast<int>(std::lround(style.outline_width * static_cast<float>(s)));
    if (reach > 0) {
        std::vector<float> line(pixels, 0.0f);
        parallel_for(
            sh, grain,
            [&](std::int64_t row_begin, std::int64_t row_end) {
                for (std::int64_t y = row_begin; y < row_end; ++y) {
                    for (std::int64_t x = 0; x < sw; ++x) {
                        const std::size_t p = static_cast<std::size_t>(y) * static_cast<std::size_t>(sw) +
                                              static_cast<std::size_t>(x);
                        bool edge = false;
                        for (int dy = -reach; dy <= reach && !edge; ++dy) {
                            for (int dx = -reach; dx <= reach && !edge; ++dx) {
                                if (dx * dx + dy * dy > reach * reach) {
                                    continue;
                                }
                                const std::int64_t qx = x + dx, qy = y + dy;
                                if (qx < 0 || qy < 0 || qx >= sw || qy >= sh) {
                                    continue;
                                }
                                const std::size_t q = static_cast<std::size_t>(qy) * static_cast<std::size_t>(sw) +
                                                      static_cast<std::size_t>(qx);
                                if (ids[q] == ids[p] || ids[q] < 0) {
                                    continue;
                                }
                                edge = depth[q] < depth[p] - style.outline_depth || ids[p] < 0;
                            }
                        }
                        line[p] = edge ? 1.0f : 0.0f;
                    }
                }
            },
            control);
        const V ink{style.outline_color[0], style.outline_color[1], style.outline_color[2]};
        for (std::size_t p = 0; p < pixels; ++p) {
            if (line[p] > 0) {
                rgba[4 * p + 0] = ink[0];
                rgba[4 * p + 1] = ink[1];
                rgba[4 * p + 2] = ink[2];
                rgba[4 * p + 3] = 1.0f;
            }
        }
    }

    // average the subpixels, over the background, into sRGB
    std::vector<std::uint8_t> image(4 * static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
    const float weight = 1.0f / static_cast<float>(s * s);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum[4] = {0, 0, 0, 0};
            for (int j = 0; j < s; ++j) {
                for (int i = 0; i < s; ++i) {
                    const std::size_t p = static_cast<std::size_t>(y * s + j) * static_cast<std::size_t>(sw) +
                                          static_cast<std::size_t>(x * s + i);
                    for (int c = 0; c < 4; ++c) {
                        sum[c] += rgba[4 * p + static_cast<std::size_t>(c)];
                    }
                }
            }
            const float coverage = sum[3] * weight;
            const float alpha = coverage + (1.0f - coverage) * style.background[3];
            const std::size_t o = 4 * (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                       static_cast<std::size_t>(x));
            for (int c = 0; c < 3; ++c) {
                // premultiplied color over the premultiplied background
                const float value = sum[c] * weight + (1.0f - coverage) * style.background[3] * background[c];
                const float straight = alpha > 0 ? value / alpha : 0.0f;
                image[o + static_cast<std::size_t>(c)] =
                    static_cast<std::uint8_t>(std::lround(255.0f * to_srgb(straight)));
            }
            image[o + 3] = static_cast<std::uint8_t>(std::lround(255.0f * std::clamp(alpha, 0.0f, 1.0f)));
        }
    }
    return image;
}

void trace(const Scene& scene, std::int64_t rays, const float* origins, const float* directions,
           float* distance, std::int32_t* kind, std::int64_t* index) {
    const Bvh bvh(scene);
    for (std::int64_t r = 0; r < rays; ++r) {
        const Ray ray = make_ray(load(origins + 3 * r), unit(load(directions + 3 * r)));
        Hit hit;
        bvh.closest(ray, 0.0f, hit);
        distance[r] = hit.t;
        kind[r] = hit.t == kInfinity ? -1 : static_cast<std::int32_t>(hit.ref >> kKindShift);
        index[r] = hit.t == kInfinity ? -1 : static_cast<std::int64_t>(hit.ref & kIndexMask);
    }
}

}  // namespace mdapy::visualization
