// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "analysis/voronoi.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace mdapy::analysis {
namespace {

using Vec = std::array<double, 3>;

constexpr std::int64_t kCube = std::numeric_limits<std::int64_t>::min();

double dot(const Vec& a, const Vec& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Vec cross(const Vec& a, const Vec& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

double norm(const Vec& a) {
    return std::sqrt(dot(a, a));
}

// A face: the plane normal . x = offset (normal of unit length, pointing
// out of the cell) and its convex polygon, points[start, start + count).
struct Face {
    std::int64_t id;
    Vec normal;
    double offset;
    std::size_t start, count;
};

// A convex polyhedron around the origin, cut down plane by plane. The
// buffers are kept from cell to cell to avoid allocations.
class Cell {
public:
    explicit Cell(double eps) : eps_(eps) {}

    // Starts from the cube of half side half.
    void reset(double half) {
        points_.clear();
        faces_.clear();
        const double corners[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
        for (int axis = 0; axis < 3; ++axis) {
            for (double sign : {-1.0, 1.0}) {
                Face face{kCube, {0.0, 0.0, 0.0}, half, points_.size(), 4};
                face.normal[axis] = sign;
                const int u = (axis + 1) % 3, w = (axis + 2) % 3;
                for (const auto& corner : corners) {
                    Vec p;
                    p[axis] = sign * half;
                    p[u] = corner[0] * half;
                    p[w] = corner[1] * half;
                    points_.push_back(p);
                }
                faces_.push_back(face);
            }
        }
        radius_ = std::sqrt(3.0) * half;
    }

    // Keeps the part with normal . x <= offset; the plane's face gets id.
    void cut(const Vec& normal, double offset, std::int64_t id) {
        bool beyond = false;
        for (const Vec& p : points_) {
            if (dot(normal, p) - offset > eps_) {
                beyond = true;
                break;
            }
        }
        if (!beyond) {
            return;
        }
        next_points_.clear();
        next_faces_.clear();
        fresh_.clear();
        for (const Face& face : faces_) {
            const std::size_t first = next_points_.size();
            for (std::size_t k = 0; k < face.count; ++k) {
                const Vec& a = points_[face.start + k];
                const Vec& b = points_[face.start + (k + 1) % face.count];
                const double da = dot(normal, a) - offset, db = dot(normal, b) - offset;
                if (da <= eps_) {
                    keep(a, first);
                    if (da >= -eps_) {
                        fresh_.push_back(a);
                    }
                }
                if ((da <= eps_) != (db <= eps_)) {
                    const double t = da / (da - db);
                    const Vec p = {a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]),
                                   a[2] + t * (b[2] - a[2])};
                    keep(p, first);
                    fresh_.push_back(p);
                }
            }
            // the last point may repeat the first
            while (next_points_.size() - first > 1 &&
                   close(next_points_.back(), next_points_[first])) {
                next_points_.pop_back();
            }
            const std::size_t count = next_points_.size() - first;
            if (count >= 3) {
                next_faces_.push_back({face.id, face.normal, face.offset, first, count});
            } else {
                next_points_.resize(first);
            }
        }
        add_face(normal, offset, id);
        points_.swap(next_points_);
        faces_.swap(next_faces_);
        double largest = 0.0;
        for (const Vec& p : points_) {
            largest = std::max(largest, dot(p, p));
        }
        radius_ = std::sqrt(largest);
    }

    double radius() const { return radius_; }
    const std::vector<Face>& faces() const { return faces_; }

    double area(const Face& face) const {
        Vec sum = {0.0, 0.0, 0.0};
        for (std::size_t k = 0; k < face.count; ++k) {
            const Vec c =
                cross(points_[face.start + k], points_[face.start + (k + 1) % face.count]);
            for (int e = 0; e < 3; ++e) {
                sum[e] += c[e];
            }
        }
        return 0.5 * std::fabs(dot(sum, face.normal));
    }

private:
    bool close(const Vec& a, const Vec& b) const {
        const Vec d = {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
        return dot(d, d) <= eps_ * eps_;
    }

    // Appends p to the polygon that starts at first unless it repeats the
    // previous point.
    void keep(const Vec& p, std::size_t first) {
        if (next_points_.size() == first || !close(p, next_points_.back())) {
            next_points_.push_back(p);
        }
    }

    // The face of the cutting plane: the distinct points on it, in angular
    // order around their center.
    void add_face(const Vec& normal, double offset, std::int64_t id) {
        distinct_.clear();
        for (const Vec& p : fresh_) {
            bool seen = false;
            for (const Vec& q : distinct_) {
                if (close(p, q)) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                distinct_.push_back(p);
            }
        }
        if (distinct_.size() < 3) {
            return;
        }
        Vec center = {0.0, 0.0, 0.0};
        for (const Vec& p : distinct_) {
            for (int e = 0; e < 3; ++e) {
                center[e] += p[e];
            }
        }
        for (int e = 0; e < 3; ++e) {
            center[e] /= static_cast<double>(distinct_.size());
        }
        const Vec helper = std::fabs(normal[0]) < 0.9 ? Vec{1.0, 0.0, 0.0} : Vec{0.0, 1.0, 0.0};
        Vec u = cross(normal, helper);
        const double length = norm(u);
        u = {u[0] / length, u[1] / length, u[2] / length};
        const Vec w = cross(normal, u);
        order_.clear();
        for (std::size_t k = 0; k < distinct_.size(); ++k) {
            const Vec d = {distinct_[k][0] - center[0], distinct_[k][1] - center[1],
                           distinct_[k][2] - center[2]};
            order_.emplace_back(std::atan2(dot(d, w), dot(d, u)), k);
        }
        std::sort(order_.begin(), order_.end());
        next_faces_.push_back({id, normal, offset, next_points_.size(), distinct_.size()});
        for (const auto& item : order_) {
            next_points_.push_back(distinct_[item.second]);
        }
    }

    double eps_;
    double radius_ = 0.0;
    std::vector<Vec> points_, next_points_, fresh_, distinct_;
    std::vector<Face> faces_, next_faces_;
    std::vector<std::pair<double, std::size_t>> order_;
};

// A face of a finished cell.
struct Out {
    std::int64_t id;
    double area;
    std::int32_t vertices;
};

}  // namespace

VoronoiCells voronoi(std::int64_t n, const double* positions, const double* matrix,
                     const double* origin, const bool* pbc, const std::int64_t* offsets,
                     const double* vectors, double reach, const Control& control) {
    // walls of the directions that are not periodic, moved out to every atom
    struct Wall {
        Vec normal;
        double low, high;
        int axis;
    };
    std::vector<Wall> walls;
    for (int axis = 0; axis < 3; ++axis) {
        if (pbc[axis]) {
            continue;
        }
        const Vec a = {matrix[3 * axis], matrix[3 * axis + 1], matrix[3 * axis + 2]};
        const int k = (axis + 1) % 3, l = (axis + 2) % 3;
        Vec normal = cross({matrix[3 * k], matrix[3 * k + 1], matrix[3 * k + 2]},
                           {matrix[3 * l], matrix[3 * l + 1], matrix[3 * l + 2]});
        const double length = norm(normal);
        normal = {normal[0] / length, normal[1] / length, normal[2] / length};
        if (dot(normal, a) < 0.0) {
            normal = {-normal[0], -normal[1], -normal[2]};
        }
        Wall wall{normal, 0.0, dot(normal, a), axis};
        for (std::int64_t i = 0; i < n; ++i) {
            const Vec x = {positions[3 * i] - origin[0], positions[3 * i + 1] - origin[1],
                           positions[3 * i + 2] - origin[2]};
            wall.low = std::min(wall.low, dot(normal, x));
            wall.high = std::max(wall.high, dot(normal, x));
        }
        walls.push_back(wall);
    }

    std::vector<std::vector<Out>> faces(static_cast<std::size_t>(n));
    VoronoiCells cells;
    cells.volumes.assign(static_cast<std::size_t>(n), 0.0);
    cells.radii.assign(static_cast<std::size_t>(n), 0.0);
    cells.complete.assign(static_cast<std::size_t>(n), 0);
    const double eps = 1e-10 * reach;
    parallel_for(
        n, 0,
        [&](std::int64_t begin, std::int64_t end) {
            Cell cell(eps);
            std::vector<std::pair<double, std::int64_t>> nearest;
            for (std::int64_t i = begin; i < end; ++i) {
                cell.reset(reach);
                // the neighbors by distance, ties by their order in the list
                nearest.clear();
                for (std::int64_t p = offsets[i]; p < offsets[i + 1]; ++p) {
                    const Vec v = {vectors[3 * p], vectors[3 * p + 1], vectors[3 * p + 2]};
                    nearest.emplace_back(dot(v, v), p);
                }
                std::sort(nearest.begin(), nearest.end());
                const Vec x = {positions[3 * i] - origin[0], positions[3 * i + 1] - origin[1],
                               positions[3 * i + 2] - origin[2]};
                for (const Wall& wall : walls) {
                    const double s = dot(wall.normal, x);
                    const Vec back = {-wall.normal[0], -wall.normal[1], -wall.normal[2]};
                    cell.cut(back, s - wall.low, -1 - 2 * wall.axis);
                    cell.cut(wall.normal, wall.high - s, -2 - 2 * wall.axis);
                }
                bool complete = false;
                for (const auto& [square, p] : nearest) {
                    const Vec v = {vectors[3 * p], vectors[3 * p + 1], vectors[3 * p + 2]};
                    const double d = std::sqrt(square);
                    if (0.5 * d > cell.radius() + eps) {
                        complete = true;  // this and every farther neighbor miss the cell
                        break;
                    }
                    if (d > 0.0) {
                        cell.cut({v[0] / d, v[1] / d, v[2] / d}, 0.5 * d, p);
                    }
                }
                complete = complete || cell.radius() <= 0.5 * reach + eps;
                double volume = 0.0;
                std::vector<Out>& mine = faces[static_cast<std::size_t>(i)];
                for (const Face& face : cell.faces()) {
                    const double a = cell.area(face);
                    volume += face.offset * a / 3.0;
                    complete = complete && face.id != kCube;
                    if (a > 0.0) {
                        mine.push_back({face.id, a, static_cast<std::int32_t>(face.count)});
                    }
                }
                cells.volumes[i] = volume;
                cells.radii[i] = cell.radius();
                cells.complete[i] = complete;
            }
        },
        control);

    cells.offsets.assign(static_cast<std::size_t>(n) + 1, 0);
    for (std::int64_t i = 0; i < n; ++i) {
        cells.offsets[i + 1] = cells.offsets[i] + static_cast<std::int64_t>(faces[i].size());
    }
    for (std::int64_t i = 0; i < n; ++i) {
        for (const Out& face : faces[i]) {
            cells.pairs.push_back(face.id);
            cells.areas.push_back(face.area);
            cells.vertices.push_back(face.vertices);
        }
    }
    return cells;
}

}  // namespace mdapy::analysis
