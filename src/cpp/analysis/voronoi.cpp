// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/voronoi.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

#include "core/voronoi_cell.h"
#include "neighbor/cell_list.h"

namespace mdapy::analysis {
namespace {

using geometry::Cell;
using geometry::cross;
using geometry::dot;
using geometry::Face;
using geometry::kCube;
using geometry::norm;
using geometry::Vec;

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
    index_t j;
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

    void add(double r2, index_t j, const std::int32_t* shift) {
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
    index_t j;
    std::array<std::int32_t, 3> shift;
    double area;
    std::int32_t edges;
};

// A face of the shape of a finished cell.
struct Polygon {
    index_t j;
    double area;
    std::size_t start, count;  // its corners in Shapes::corners
};

// The shapes of finished cells: their vertices and all their faces.
struct Shapes {
    std::vector<Vec> points;  // in the frame of the box
    std::vector<Polygon> polygons;
    std::vector<std::uint32_t> corners;  // vertex within its cell
};

// The cell of the atom i from its sorted neighbors, all those closer than
// reach; x is its position from the origin of the box. Returns whether no
// farther atom can cut the cell; faces get the faces that count, and shapes,
// unless null, the shape of the cell when it is finished.
bool build_cell(Cell& cell, const std::vector<Wall>& walls, const Vec& x, const double* positions,
                std::int64_t i, const double* h, const Neighbors& neighbors, double reach,
                double eps, const VoronoiThresholds& thresholds, double& volume, double& radius,
                std::vector<Out>& faces, Shapes* shapes) {
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
    const std::size_t first_polygon = shapes ? shapes->polygons.size() : 0;
    const std::size_t first_corner = shapes ? shapes->corners.size() : 0;
    const double shortest2 = thresholds.edge * thresholds.edge;
    for (const Face& face : cell.faces()) {
        if (face.id == kCube) {
            complete = false;
        }
        const double a = cell.area(face);
        volume += face.offset * a / 3.0;
        surface += a;
        if (a > 0.0) {
            Out out{-1, {0, 0, 0}, a, cell.edges(face, shortest2)};
            if (face.id >= 0) {
                const Candidate& other = neighbors[static_cast<std::size_t>(face.id)];
                out.j = other.j;
                out.shift = other.shift;
            }
            faces.push_back(out);
            if (shapes != nullptr) {
                shapes->polygons.push_back({out.j, a, shapes->corners.size(), face.count});
                const std::uint32_t* loop = cell.corners(face);
                shapes->corners.insert(shapes->corners.end(), loop, loop + face.count);
            }
        }
    }
    if (shapes != nullptr) {
        if (complete) {
            for (const Vec& point : cell.points()) {
                shapes->points.push_back({xi[0] + point[0], xi[1] + point[1], xi[2] + point[2]});
            }
        } else {
            shapes->polygons.resize(first_polygon);
            shapes->corners.resize(first_corner);
        }
    }
    // keep the faces that count
    const double smallest = std::max(thresholds.face, thresholds.relative_face * surface);
    std::size_t kept = first;
    for (std::size_t f = first; f < faces.size(); ++f) {
        if (faces[f].area > smallest && faces[f].edges >= 3) {
            faces[kept++] = faces[f];
        }
    }
    faces.resize(kept);
    radius = cell.radius();
    return complete;
}

// Collects the faces and shapes of each atom, computed in any order, into
// CSR form.
struct Collector {
    std::vector<std::vector<Out>> buffers;    // one per chunk of work
    std::vector<Shapes> shapes;               // one per chunk, with polyhedra
    std::vector<std::int64_t> buffer;  // per atom: the chunk of its cell
    std::vector<std::int64_t> start;   // with faces: where its faces are
    std::vector<std::int32_t> count;
    std::vector<std::int64_t> point_start, polygon_start;  // with polyhedra
    std::vector<std::int32_t> point_count, polygon_count;

    Collector(std::int64_t n, bool faces, bool polyhedra) {
        if (faces || polyhedra) {
            buffer.assign(static_cast<std::size_t>(n), -1);
        }
        if (faces) {
            start.assign(static_cast<std::size_t>(n), 0);
            count.assign(static_cast<std::size_t>(n), 0);
        }
        if (polyhedra) {
            point_start.assign(static_cast<std::size_t>(n), 0);
            polygon_start.assign(static_cast<std::size_t>(n), 0);
            point_count.assign(static_cast<std::size_t>(n), 0);
            polygon_count.assign(static_cast<std::size_t>(n), 0);
        }
    }

    void assemble_shapes(std::int64_t n, VoronoiCells& cells) const {
        cells.vertex_offsets.assign(static_cast<std::size_t>(n) + 1, 0);
        cells.face_offsets.assign(static_cast<std::size_t>(n) + 1, 0);
        std::int64_t corner_total = 0;
        for (std::int64_t i = 0; i < n; ++i) {
            cells.vertex_offsets[i + 1] = cells.vertex_offsets[i] + point_count[i];
            cells.face_offsets[i + 1] = cells.face_offsets[i] + polygon_count[i];
            const Polygon* polygons = shapes[static_cast<std::size_t>(buffer[i])].polygons.data();
            for (std::int32_t f = 0; f < polygon_count[i]; ++f) {
                corner_total += static_cast<std::int64_t>(polygons[polygon_start[i] + f].count);
            }
        }
        const auto faces = static_cast<std::size_t>(cells.face_offsets[n]);
        cells.vertices.resize(3 * static_cast<std::size_t>(cells.vertex_offsets[n]));
        cells.corner_offsets.assign(faces + 1, 0);
        cells.corners.resize(static_cast<std::size_t>(corner_total));
        cells.face_neighbors.resize(faces);
        cells.face_areas.resize(faces);
        for (std::int64_t i = 0; i < n; ++i) {
            const Shapes& shape = shapes[static_cast<std::size_t>(buffer[i])];
            for (std::int32_t v = 0; v < point_count[i]; ++v) {
                const Vec& point = shape.points[static_cast<std::size_t>(point_start[i] + v)];
                for (int e = 0; e < 3; ++e) {
                    cells.vertices[3 * static_cast<std::size_t>(cells.vertex_offsets[i] + v) + e] =
                        point[e];
                }
            }
            for (std::int32_t f = 0; f < polygon_count[i]; ++f) {
                const Polygon& polygon = shape.polygons[static_cast<std::size_t>(polygon_start[i] + f)];
                const auto at = static_cast<std::size_t>(cells.face_offsets[i] + f);
                cells.face_neighbors[at] = polygon.j;
                cells.face_areas[at] = polygon.area;
                const std::int64_t next = cells.corner_offsets[at] + static_cast<std::int64_t>(polygon.count);
                cells.corner_offsets[at + 1] = next;
                for (std::size_t k = 0; k < polygon.count; ++k) {
                    cells.corners[static_cast<std::size_t>(cells.corner_offsets[at]) + k] =
                        cells.vertex_offsets[i] + shape.corners[polygon.start + k];
                }
            }
        }
    }

    // The faces kept, in CSR form, copied for the atoms in parallel; and the
    // shapes of the cells, with polyhedra.
    void assemble(std::int64_t n, bool faces, VoronoiCells& cells) const {
        if (faces) {
            cells.offsets.assign(static_cast<std::size_t>(n) + 1, 0);
            for (std::int64_t i = 0; i < n; ++i) {
                cells.offsets[i + 1] = count[i];
            }
            prefix_sum(cells.offsets.data(), n + 1);
            const auto total = static_cast<std::size_t>(cells.offsets[n]);
            cells.neighbors.resize(total);
            cells.shifts.resize(3 * total);
            cells.areas.resize(total);
            cells.edges.resize(total);
            parallel_for(n, 0, [&](std::int64_t begin, std::int64_t end) {
                for (std::int64_t i = begin; i < end; ++i) {
                    if (count[i] == 0) {
                        continue;
                    }
                    const Out* out = buffers[static_cast<std::size_t>(buffer[i])].data() + start[i];
                    for (std::int32_t f = 0; f < count[i]; ++f) {
                        const auto at = static_cast<std::size_t>(cells.offsets[i] + f);
                        cells.neighbors[at] = out[f].j;
                        for (int e = 0; e < 3; ++e) {
                            const std::int32_t s = out[f].shift[e];
                            if (s < std::numeric_limits<std::int8_t>::min() ||
                                s > std::numeric_limits<std::int8_t>::max()) {
                                throw std::overflow_error(
                                    "a periodic image of a face is outside int8");
                            }
                            cells.shifts[3 * at + e] = static_cast<std::int8_t>(s);
                        }
                        cells.areas[at] = out[f].area;
                        cells.edges[at] = out[f].edges;
                    }
                }
            });
        }
        if (!point_count.empty()) {
            assemble_shapes(n, cells);
        }
    }
};

}  // namespace

VoronoiCells voronoi(const double* positions, std::int64_t n, const double* matrix,
                     const double* origin, const bool* pbc, const VoronoiThresholds& thresholds,
                     bool faces, bool polyhedra, const Control& control) {
    const std::vector<Wall> walls = make_walls(n, positions, matrix, origin, pbc);
    VoronoiCells cells;
    cells.volumes.assign(static_cast<std::size_t>(n), 0.0);
    cells.radii.assign(static_cast<std::size_t>(n), 0.0);
    cells.coordination.assign(static_cast<std::size_t>(n), 0);
    Collector collector(n, faces, polyhedra);
    if (n == 0) {
        collector.assemble(n, faces, cells);
        return cells;
    }
    const double* h = matrix;
    const double volume = std::fabs(dot({h[0], h[1], h[2]}, cross({h[3], h[4], h[5]}, {h[6], h[7], h[8]})));
    double reach = 2.5 * std::cbrt(volume / static_cast<double>(n));
    std::vector<std::int64_t> todo(static_cast<std::size_t>(n));
    std::vector<std::int64_t> slots(static_cast<std::size_t>(n));  // of the atoms to do
    bool all = true;
    // the monitor sees the passes as one region whose total grows by the
    // atoms of each pass; only the end of the last one reports it done
    std::int64_t before = 0;  // atoms of the earlier passes
    Control pass;
    pass.cancel = control.cancel;
    while (!todo.empty()) {
        const neighbor::CellGrid grid(positions, n, matrix, origin, pbc, reach);
        if (all) {
            // every atom, in cell order
            for (std::int64_t t = 0; t < n; ++t) {
                todo[static_cast<std::size_t>(t)] = grid.atom(t);
                slots[static_cast<std::size_t>(t)] = t;
            }
            all = false;
        } else {
            std::vector<std::int64_t> slot_of(static_cast<std::size_t>(n));
            for (std::int64_t t = 0; t < n; ++t) {
                slot_of[static_cast<std::size_t>(grid.atom(t))] = t;
            }
            slots.resize(todo.size());
            for (std::size_t k = 0; k < todo.size(); ++k) {
                slots[k] = slot_of[static_cast<std::size_t>(todo[k])];
            }
        }
        const auto count = static_cast<std::int64_t>(todo.size());
        const std::int64_t grain = default_grain(count);
        const std::size_t first_buffer = collector.buffers.size();
        collector.buffers.resize(first_buffer + static_cast<std::size_t>((count + grain - 1) / grain));
        if (polyhedra) {
            collector.shapes.resize(collector.buffers.size());
        }
        std::vector<std::uint8_t> open(todo.size(), 0);
        const double eps = 1e-10 * reach;
        if (control.monitor) {
            pass.monitor = [&control, before](std::int64_t done, std::int64_t total) {
                return done == total || control.monitor(before + done, before + total);
            };
        }
        parallel_for(
            count, grain,
            [&](std::int64_t begin, std::int64_t end) {
                const std::size_t id = first_buffer + static_cast<std::size_t>(begin / grain);
                std::vector<Out>& found = collector.buffers[id];
                Shapes* shapes = polyhedra ? &collector.shapes[id] : nullptr;
                Cell cell(eps);
                Neighbors neighbors;
                for (std::int64_t t = begin; t < end; ++t) {
                    const std::int64_t i = todo[static_cast<std::size_t>(t)];
                    const double* xi = positions + 3 * i;
                    neighbors.clear();
                    grid.visit(slots[static_cast<std::size_t>(t)], [&](index_t j,
                                                                        const std::int32_t* s,
                                                                        double r2) {
                        if (r2 == 0.0) {
                            throw std::invalid_argument("two atoms are at the same position");
                        }
                        neighbors.add(r2, j, s);
                    });
                    neighbors.sort(reach);
                    const std::size_t mark = found.size();
                    const std::size_t points = shapes ? shapes->points.size() : 0;
                    const std::size_t polygons = shapes ? shapes->polygons.size() : 0;
                    const Vec x = {xi[0] - origin[0], xi[1] - origin[1], xi[2] - origin[2]};
                    if (build_cell(cell, walls, x, positions, i, h, neighbors, reach, eps,
                                   thresholds, cells.volumes[i], cells.radii[i], found, shapes)) {
                        std::int64_t shared = 0;  // the faces with atoms
                        for (std::size_t f = mark; f < found.size(); ++f) {
                            shared += found[f].j >= 0;
                        }
                        cells.coordination[i] = shared;
                        if (faces) {
                            collector.start[i] = static_cast<std::int64_t>(mark);
                            collector.count[i] = static_cast<std::int32_t>(found.size() - mark);
                        } else {
                            found.resize(mark);  // only the cell's size and shape
                        }
                        if (faces || shapes != nullptr) {
                            collector.buffer[i] = static_cast<std::int64_t>(id);
                        }
                        if (shapes != nullptr) {
                            collector.point_start[i] = static_cast<std::int64_t>(points);
                            collector.point_count[i] =
                                static_cast<std::int32_t>(shapes->points.size() - points);
                            collector.polygon_start[i] = static_cast<std::int64_t>(polygons);
                            collector.polygon_count[i] =
                                static_cast<std::int32_t>(shapes->polygons.size() - polygons);
                        }
                    } else {
                        found.resize(mark);
                        open[static_cast<std::size_t>(t)] = 1;
                    }
                }
            },
            pass);
        before += count;
        std::vector<std::int64_t> left;
        for (std::size_t t = 0; t < todo.size(); ++t) {
            if (open[t]) {
                left.push_back(todo[t]);
            }
        }
        todo.swap(left);
        reach *= 1.5;
    }
    if (control.monitor) {
        control.monitor(before, before);
    }
    collector.assemble(n, faces, cells);
    return cells;
}

}  // namespace mdapy::analysis
