// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "analysis/voronoi.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
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
        const std::uint32_t around[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            for (std::uint32_t side = 0; side < 2; ++side) {
                Face face{kCube, {0.0, 0.0, 0.0}, half, corners_.size(), 4};
                face.normal[axis] = side ? 1.0 : -1.0;
                const std::uint32_t u = (axis + 1) % 3, w = (axis + 2) % 3;
                for (const auto& corner : around) {
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
                size += db <= eps_;  // without a branch: kept or not is hard to predict
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
};

// The neighbors of an atom closer than reach in distance bins: bin b holds
// those with b <= kBins r^2 / reach^2 < b + 1 in the order found. This is
// enough order to stop at the first bin too far to cut the cell, and costs
// less than a sort, whose comparisons the processor cannot predict.
class Neighbors {
public:
    static constexpr std::size_t kBins = 64;

    void clear() { found_.clear(); }

    void add(double r2, std::int32_t j, const std::int32_t* shift) {
        found_.push_back({r2, j, {shift[0], shift[1], shift[2]}});
    }

    void sort(double reach) {
        scale_ = static_cast<double>(kBins) / (reach * reach);
        start_.fill(0);
        bin_.resize(found_.size());
        for (std::size_t c = 0; c < found_.size(); ++c) {
            bin_[c] = static_cast<std::uint32_t>(
                std::min(static_cast<double>(kBins - 1), found_[c].r2 * scale_));
            ++start_[bin_[c] + 1];
        }
        for (std::size_t b = 0; b < kBins; ++b) {
            start_[b + 1] += start_[b];
        }
        std::array<std::uint32_t, kBins> next;
        std::copy(start_.begin(), start_.end() - 1, next.begin());
        sorted_.resize(found_.size());
        for (std::size_t c = 0; c < found_.size(); ++c) {
            sorted_[next[bin_[c]]++] = found_[c];
        }
    }

    // The distance below which no neighbor of bin b lies.
    double low(std::size_t b) const { return std::sqrt(static_cast<double>(b) / scale_); }
    std::size_t begin(std::size_t b) const { return start_[b]; }
    std::size_t end(std::size_t b) const { return start_[b + 1]; }
    const Candidate& operator[](std::size_t k) const { return sorted_[k]; }

private:
    std::vector<Candidate> found_, sorted_;
    std::vector<std::uint32_t> bin_;
    std::array<std::uint32_t, kBins + 1> start_{};
    double scale_ = 1.0;
};

// A face of a finished cell.
struct Out {
    std::int32_t j;
    std::array<std::int32_t, 3> shift;
    double area;
    std::int32_t vertices;
};

// The cell of the atom i from its sorted neighbors, all those closer than
// reach; x is its position from the origin of the box. Returns whether no
// farther atom can cut the cell; faces get the volume and the faces above
// the threshold.
bool build_cell(Cell& cell, const std::vector<Wall>& walls, const Vec& x, const double* positions,
                std::int64_t i, const double* h, const Neighbors& neighbors, double reach,
                double eps, double area_threshold, double& volume, double& radius,
                std::vector<Out>& faces) {
    cell.reset(reach);
    for (const Wall& wall : walls) {
        const double s = dot(wall.normal, x);
        const Vec back = {-wall.normal[0], -wall.normal[1], -wall.normal[2]};
        cell.cut(back, s - wall.low, -1 - 2 * wall.axis);
        cell.cut(wall.normal, wall.high - s, -2 - 2 * wall.axis);
    }
    const double* xi = positions + 3 * i;
    bool complete = false;
    for (std::size_t b = 0; b < Neighbors::kBins && !complete; ++b) {
        if (0.5 * neighbors.low(b) > cell.radius() + eps) {
            complete = true;  // every neighbor from this bin on misses the cell
            break;
        }
        for (std::size_t c = neighbors.begin(b); c < neighbors.end(b); ++c) {
            const Candidate& other = neighbors[c];
            const double d = std::sqrt(other.r2);
            if (0.5 * d > cell.radius() + eps) {
                continue;
            }
            const double* xj = positions + 3 * static_cast<std::int64_t>(other.j);
            const std::int32_t* s = other.shift.data();
            Vec normal;
            for (int e = 0; e < 3; ++e) {
                normal[e] = (xj[e] - xi[e] + s[0] * h[e] + s[1] * h[3 + e] + s[2] * h[6 + e]) / d;
            }
            cell.cut(normal, 0.5 * d, static_cast<std::int64_t>(c));
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
                const Candidate& other = neighbors[static_cast<std::size_t>(face.id)];
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
                Neighbors neighbors;
                for (std::int64_t t = begin; t < end; ++t) {
                    const std::int64_t i = todo[static_cast<std::size_t>(t)];
                    const double* xi = positions + 3 * i;
                    neighbors.clear();
                    grid.visit(i, [&](std::int32_t j, const std::int32_t* s, double r2) {
                        if (r2 == 0.0) {
                            throw std::invalid_argument("two atoms are at the same position");
                        }
                        neighbors.add(r2, j, s);
                    });
                    neighbors.sort(reach);
                    const std::size_t mark = faces.size();
                    const Vec x = {xi[0] - origin[0], xi[1] - origin[1], xi[2] - origin[2]};
                    if (build_cell(cell, walls, x, positions, i, h, neighbors, reach, eps,
                                   area_threshold, cells.volumes[i], cells.radii[i], faces)) {
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
