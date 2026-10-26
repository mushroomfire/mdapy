// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// A convex cell around the origin cut down plane by plane: the Voronoi cell
// of an atom from the planes halfway to its neighbors.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace mdapy::geometry {

using Vec = std::array<double, 3>;

// The id of the faces of the starting cube.
inline constexpr std::int64_t kCube = std::numeric_limits<std::int64_t>::min();

inline double dot(const Vec& a, const Vec& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline Vec cross(const Vec& a, const Vec& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

inline double norm(const Vec& a) {
    return std::sqrt(dot(a, a));
}

// A number from 0 to 4 that grows with the angle of (x, y) as atan2 does, at
// the cost of a division.
inline double pseudo_angle(double x, double y) {
    const double sum = std::fabs(x) + std::fabs(y);
    if (sum == 0.0) {
        return 0.0;
    }
    const double p = y / sum;  // from -1 to 1
    return x >= 0.0 ? (p < 0.0 ? 4.0 + p : p) : 2.0 - p;
}

// A face: the plane normal . x = offset (normal of unit length, pointing
// out of the cell) and its convex polygon, the points corners[start, start +
// count) in order around it.
struct Face {
    std::int64_t id;
    Vec normal;
    double offset;
    std::size_t start, count;
};

// A convex polyhedron around the origin, cut down plane by plane. Each point
// is stored once and the faces list the points around them, so a point that
// a cut creates on an edge is shared by the two faces of the edge. The
// buffers are kept from cell to cell to avoid allocations.
class Cell {
public:
    explicit Cell(double eps) : eps_(eps) {}

    // Points closer than eps to a cutting plane lie in it.
    void set_tolerance(double eps) { eps_ = eps; }

    // Starts from the cube of half side half.
    void reset(double half) {
        points_.clear();
        corners_.clear();
        faces_.clear();
        // the corner with bit e of its index set lies on the + side of axis e
        for (std::uint32_t c = 0; c < 8; ++c) {
            points_.push_back({(c & 1U) ? half : -half, (c & 2U) ? half : -half,
                               (c & 4U) ? half : -half});
        }
        // counterclockwise seen from the + side of the axis
        const std::uint32_t around[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            for (std::uint32_t side = 0; side < 2; ++side) {
                Face face{kCube, {0.0, 0.0, 0.0}, half, corners_.size(), 4};
                face.normal[axis] = side ? 1.0 : -1.0;
                const std::uint32_t u = (axis + 1) % 3, w = (axis + 2) % 3;
                for (std::uint32_t k = 0; k < 4; ++k) {
                    // every face counterclockwise seen from outside
                    const auto& corner = around[side ? k : 3 - k];
                    corners_.push_back((side << axis) | (corner[0] << u) | (corner[1] << w));
                }
                faces_.push_back(face);
            }
        }
        radius_ = std::sqrt(3.0) * half;
    }

    // Keeps the part with normal . x <= offset; the plane's face gets id.
    void cut(const Vec& normal, double offset, std::int64_t id) {
        const std::size_t n = points_.size();
        side_.resize(n);
        bool beyond = false;
        for (std::size_t v = 0; v < n; ++v) {
            side_[v] = dot(normal, points_[v]) - offset;
            beyond = beyond || side_[v] > eps_;
        }
        if (!beyond) {
            return;
        }
        // the points beyond the plane go; those within eps of it stay and
        // join the new face
        renumber_.resize(n);
        next_points_.clear();
        section_.clear();
        double largest = 0.0;
        for (std::size_t v = 0; v < n; ++v) {
            if (side_[v] <= eps_) {
                renumber_[v] = static_cast<std::uint32_t>(next_points_.size());
                if (side_[v] >= -eps_) {
                    section_.push_back(renumber_[v]);
                }
                next_points_.push_back(points_[v]);
                largest = std::max(largest, dot(points_[v], points_[v]));
            } else {
                renumber_[v] = kUnused;
            }
        }
        // an edge from one side of the plane to the other gets a point on
        // it; a face gains at most one corner, as it loses one for each two
        // it gains
        next_faces_.clear();
        crossings_.clear();
        next_corners_.resize(corners_.size() + faces_.size());
        std::uint32_t* out = next_corners_.data();
        std::size_t size = 0;
        for (const Face& face : faces_) {
            const std::size_t first = size;
            const std::uint32_t* loop = corners_.data() + face.start;
            std::uint32_t a = loop[face.count - 1];
            double da = side_[a];
            for (std::size_t k = 0; k < face.count; ++k) {
                const std::uint32_t b = loop[k];
                const double db = side_[b];
                if ((da < -eps_ && db > eps_) || (da > eps_ && db < -eps_)) {
                    out[size++] = crossing(a, b, largest);
                }
                out[size] = renumber_[b];
                size += static_cast<std::size_t>(db <= eps_);  // no branch: hard to predict
                a = b;
                da = db;
            }
            const std::size_t count = size - first;
            if (count >= 3) {
                next_faces_.push_back({face.id, face.normal, face.offset, first, count});
            } else {
                size = first;
            }
        }
        next_corners_.resize(size);
        if (section_.size() >= 3) {
            add_face(normal, offset, id);
        } else {
            // the plane only grazes the cell: no new face, and the points
            // on the plane may belong to no face any more
            prune();
            largest = 0.0;
            for (const Vec& p : next_points_) {
                largest = std::max(largest, dot(p, p));
            }
        }
        points_.swap(next_points_);
        corners_.swap(next_corners_);
        faces_.swap(next_faces_);
        radius_ = std::sqrt(largest);
    }

    double radius() const { return radius_; }
    const std::vector<Face>& faces() const { return faces_; }
    const std::vector<Vec>& points() const { return points_; }
    // The points of a face, counterclockwise seen from outside the cell.
    const std::uint32_t* corners(const Face& face) const { return corners_.data() + face.start; }

    double area(const Face& face) const {
        Vec sum = {0.0, 0.0, 0.0};
        for (std::size_t k = 0; k < face.count; ++k) {
            const Vec c = cross(points_[corners_[face.start + k]],
                                points_[corners_[face.start + (k + 1) % face.count]]);
            for (int e = 0; e < 3; ++e) {
                sum[e] += c[e];
            }
        }
        return 0.5 * std::fabs(dot(sum, face.normal));
    }

    // The number of edges of a face with a squared length of at least
    // shortest2.
    std::int32_t edges(const Face& face, double shortest2) const {
        if (shortest2 <= 0.0) {
            return static_cast<std::int32_t>(face.count);
        }
        std::int32_t count = 0;
        for (std::size_t k = 0; k < face.count; ++k) {
            const Vec& a = points_[corners_[face.start + k]];
            const Vec& b = points_[corners_[face.start + (k + 1) % face.count]];
            const Vec d = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
            count += static_cast<std::int32_t>(dot(d, d) >= shortest2);
        }
        return count;
    }

private:
    static constexpr std::uint32_t kUnused = std::numeric_limits<std::uint32_t>::max();

    // The point where the plane crosses the edge from a to b, made once for
    // the two faces of the edge.
    std::uint32_t crossing(std::uint32_t a, std::uint32_t b, double& largest) {
        const std::uint32_t low = std::min(a, b), high = std::max(a, b);
        for (const auto& known : crossings_) {
            if (known[0] == low && known[1] == high) {
                return known[2];
            }
        }
        const double t = side_[low] / (side_[low] - side_[high]);
        const Vec& p = points_[low];
        const Vec& q = points_[high];
        const Vec x = {p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]),
                       p[2] + t * (q[2] - p[2])};
        const auto index = static_cast<std::uint32_t>(next_points_.size());
        next_points_.push_back(x);
        largest = std::max(largest, dot(x, x));
        section_.push_back(index);
        crossings_.push_back({low, high, index});
        return index;
    }

    // The face of the cutting plane: the section points in angular order
    // around their center.
    void add_face(const Vec& normal, double offset, std::int64_t id) {
        Vec center = {0.0, 0.0, 0.0};
        for (const std::uint32_t v : section_) {
            for (int e = 0; e < 3; ++e) {
                center[e] += next_points_[v][e];
            }
        }
        for (int e = 0; e < 3; ++e) {
            center[e] /= static_cast<double>(section_.size());
        }
        const Vec helper = std::fabs(normal[0]) < 0.9 ? Vec{1.0, 0.0, 0.0} : Vec{0.0, 1.0, 0.0};
        Vec u = cross(normal, helper);
        const double length = norm(u);
        u = {u[0] / length, u[1] / length, u[2] / length};
        const Vec w = cross(normal, u);
        order_.clear();
        for (const std::uint32_t v : section_) {
            const Vec d = {next_points_[v][0] - center[0], next_points_[v][1] - center[1],
                           next_points_[v][2] - center[2]};
            order_.emplace_back(pseudo_angle(dot(d, u), dot(d, w)), v);
        }
        std::sort(order_.begin(), order_.end());
        next_faces_.push_back({id, normal, offset, next_corners_.size(), section_.size()});
        for (const auto& item : order_) {
            next_corners_.push_back(item.second);
        }
    }

    // Keeps only the new points that some face uses, renumbered in order of
    // use; the old points are no longer needed.
    void prune() {
        renumber_.assign(next_points_.size(), kUnused);
        points_.clear();
        for (std::uint32_t& v : next_corners_) {
            if (renumber_[v] == kUnused) {
                renumber_[v] = static_cast<std::uint32_t>(points_.size());
                points_.push_back(next_points_[v]);
            }
            v = renumber_[v];
        }
        next_points_.swap(points_);
    }

    double eps_;
    double radius_ = 0.0;
    std::vector<Vec> points_, next_points_;
    std::vector<double> side_;
    std::vector<std::uint32_t> corners_, next_corners_, renumber_, section_;
    std::vector<std::array<std::uint32_t, 3>> crossings_;
    std::vector<Face> faces_, next_faces_;
    std::vector<std::pair<double, std::uint32_t>> order_;
};

}  // namespace mdapy::geometry
