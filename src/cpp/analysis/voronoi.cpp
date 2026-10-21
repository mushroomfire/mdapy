// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "analysis/voronoi.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "neighbor/cell_list.h"

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

// A number from 0 to 4 that grows with the angle of (x, y) as atan2 does, at
// the cost of a division.
double pseudo_angle(double x, double y) {
    const double sum = std::fabs(x) + std::fabs(y);
    if (sum == 0.0) {
        return 0.0;
    }
    const double p = y / sum;  // from -1 to 1
    return x >= 0.0 ? (p < 0.0 ? 4.0 + p : p) : 2.0 - p;
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
            order_.emplace_back(pseudo_angle(dot(d, u), dot(d, w)), k);
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

// A wall of a direction that is not periodic: the atoms lie between
// normal . x = low and normal . x = high.
struct Wall {
    Vec normal;
    double low, high;
    int axis;
};

std::vector<Wall> make_walls(std::int64_t n, const double* positions, const double* matrix,
                             const double* origin, const bool* pbc) {
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
    return walls;
}

// A neighbor that may cut the cell.
struct Candidate {
    double r2;
    std::int32_t j;
    std::array<std::int32_t, 3> shift;
    Vec vector;
};

// A face of a finished cell.
struct Out {
    std::int32_t j;
    std::array<std::int32_t, 3> shift;
    double area;
    std::int32_t vertices;
};

// The cell of the atom at x from its candidates sorted by distance, all those
// closer than reach. Returns whether no farther atom can cut the cell; faces
// get the volume and the faces above the threshold.
bool build_cell(Cell& cell, const std::vector<Wall>& walls, const Vec& x,
                const std::vector<Candidate>& candidates, double reach, double eps,
                double area_threshold, double& volume, double& radius, std::vector<Out>& faces) {
    cell.reset(reach);
    for (const Wall& wall : walls) {
        const double s = dot(wall.normal, x);
        const Vec back = {-wall.normal[0], -wall.normal[1], -wall.normal[2]};
        cell.cut(back, s - wall.low, -1 - 2 * wall.axis);
        cell.cut(wall.normal, wall.high - s, -2 - 2 * wall.axis);
    }
    bool complete = false;
    for (std::size_t c = 0; c < candidates.size(); ++c) {
        const double d = std::sqrt(candidates[c].r2);
        if (0.5 * d > cell.radius() + eps) {
            complete = true;  // this and every farther neighbor miss the cell
            break;
        }
        if (d > 0.0) {
            const Vec& v = candidates[c].vector;
            cell.cut({v[0] / d, v[1] / d, v[2] / d}, 0.5 * d, static_cast<std::int64_t>(c));
        }
    }
    complete = complete || cell.radius() <= 0.5 * reach + eps;
    volume = 0.0;
    double surface = 0.0;
    const std::size_t first = faces.size();
    for (const Face& face : cell.faces()) {
        if (face.id == kCube) {
            complete = false;
        }
        const double a = cell.area(face);
        volume += face.offset * a / 3.0;
        surface += a;
        if (a > 0.0) {
            Out out{-1, {0, 0, 0}, a, static_cast<std::int32_t>(face.count)};
            if (face.id >= 0) {
                const Candidate& other = candidates[static_cast<std::size_t>(face.id)];
                out.j = other.j;
                out.shift = other.shift;
            }
            faces.push_back(out);
        }
    }
    // drop the faces below the threshold
    std::size_t kept = first;
    for (std::size_t f = first; f < faces.size(); ++f) {
        if (faces[f].area > area_threshold * surface) {
            faces[kept++] = faces[f];
        }
    }
    faces.resize(kept);
    radius = cell.radius();
    return complete;
}

bool by_distance(const Candidate& p, const Candidate& q) {
    return std::tie(p.r2, p.j, p.shift) < std::tie(q.r2, q.j, q.shift);
}

// Collects the faces of each atom, computed in any order, into CSR form.
struct Collector {
    std::vector<std::vector<Out>> buffers;    // one per chunk of work
    std::vector<std::int64_t> buffer, start;  // per atom: where its faces are
    std::vector<std::int32_t> count;

    explicit Collector(std::int64_t n)
        : buffer(static_cast<std::size_t>(n), -1), start(static_cast<std::size_t>(n), 0),
          count(static_cast<std::size_t>(n), 0) {}

    void assemble(std::int64_t n, VoronoiCells& cells) const {
        cells.offsets.assign(static_cast<std::size_t>(n) + 1, 0);
        cells.coordination.assign(static_cast<std::size_t>(n), 0);
        for (std::int64_t i = 0; i < n; ++i) {
            cells.offsets[i + 1] = cells.offsets[i] + count[i];
        }
        const auto total = static_cast<std::size_t>(cells.offsets[n]);
        cells.neighbors.resize(total);
        cells.shifts.resize(3 * total);
        cells.areas.resize(total);
        cells.vertices.resize(total);
        for (std::int64_t i = 0; i < n; ++i) {
            const Out* faces = buffers[static_cast<std::size_t>(buffer[i])].data() + start[i];
            for (std::int32_t f = 0; f < count[i]; ++f) {
                const auto at = static_cast<std::size_t>(cells.offsets[i] + f);
                cells.neighbors[at] = faces[f].j;
                for (int e = 0; e < 3; ++e) {
                    cells.shifts[3 * at + e] = faces[f].shift[e];
                }
                cells.areas[at] = faces[f].area;
                cells.vertices[at] = faces[f].vertices;
                cells.coordination[i] += faces[f].j >= 0;
            }
        }
    }
};

}  // namespace

VoronoiCells voronoi(const double* positions, std::int64_t n, const double* matrix,
                     const double* origin, const bool* pbc, double area_threshold,
                     const Control& control) {
    const std::vector<Wall> walls = make_walls(n, positions, matrix, origin, pbc);
    VoronoiCells cells;
    cells.volumes.assign(static_cast<std::size_t>(n), 0.0);
    cells.radii.assign(static_cast<std::size_t>(n), 0.0);
    Collector collector(n);
    if (n == 0) {
        collector.assemble(n, cells);
        return cells;
    }
    const double* h = matrix;
    const double volume = std::fabs(dot({h[0], h[1], h[2]}, cross({h[3], h[4], h[5]}, {h[6], h[7], h[8]})));
    double reach = 2.5 * std::cbrt(volume / static_cast<double>(n));
    std::vector<std::int64_t> todo(static_cast<std::size_t>(n));
    std::iota(todo.begin(), todo.end(), 0);
    while (!todo.empty()) {
        const neighbor::CellGrid grid(positions, n, matrix, origin, pbc, reach);
        const auto count = static_cast<std::int64_t>(todo.size());
        const std::int64_t grain = default_grain(count);
        const std::size_t first_buffer = collector.buffers.size();
        collector.buffers.resize(first_buffer + static_cast<std::size_t>((count + grain - 1) / grain));
        std::vector<std::uint8_t> open(todo.size(), 0);
        const double eps = 1e-10 * reach;
        parallel_for(
            count, grain,
            [&](std::int64_t begin, std::int64_t end) {
                const std::size_t id = first_buffer + static_cast<std::size_t>(begin / grain);
                std::vector<Out>& faces = collector.buffers[id];
                Cell cell(eps);
                std::vector<Candidate> candidates;
                for (std::int64_t t = begin; t < end; ++t) {
                    const std::int64_t i = todo[static_cast<std::size_t>(t)];
                    const double* xi = positions + 3 * i;
                    candidates.clear();
                    grid.visit(i, [&](std::int32_t j, const std::int32_t* s, double r2) {
                        if (r2 == 0.0) {
                            throw std::invalid_argument("two atoms are at the same position");
                        }
                        const double* xj = positions + 3 * static_cast<std::int64_t>(j);
                        Vec v;
                        for (int e = 0; e < 3; ++e) {
                            v[e] = xj[e] - xi[e] + s[0] * h[e] + s[1] * h[3 + e] + s[2] * h[6 + e];
                        }
                        candidates.push_back({r2, j, {s[0], s[1], s[2]}, v});
                    });
                    std::sort(candidates.begin(), candidates.end(), by_distance);
                    const std::size_t mark = faces.size();
                    const Vec x = {xi[0] - origin[0], xi[1] - origin[1], xi[2] - origin[2]};
                    if (build_cell(cell, walls, x, candidates, reach, eps, area_threshold,
                                   cells.volumes[i], cells.radii[i], faces)) {
                        collector.buffer[i] = static_cast<std::int64_t>(id);
                        collector.start[i] = static_cast<std::int64_t>(mark);
                        collector.count[i] = static_cast<std::int32_t>(faces.size() - mark);
                    } else {
                        faces.resize(mark);
                        open[static_cast<std::size_t>(t)] = 1;
                    }
                }
            },
            control);
        std::vector<std::int64_t> left;
        for (std::size_t t = 0; t < todo.size(); ++t) {
            if (open[t]) {
                left.push_back(todo[t]);
            }
        }
        todo.swap(left);
        reach *= 1.5;
    }
    collector.assemble(n, cells);
    return cells;
}

}  // namespace mdapy::analysis
