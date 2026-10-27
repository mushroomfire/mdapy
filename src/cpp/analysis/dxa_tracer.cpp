// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "analysis/dxa_tracer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <random>
#include <tuple>
#include <utility>

namespace mdapy::analysis::dxa {
namespace {

// the most edges a walk around a vertex of the mesh takes
constexpr int kFan = 4096;

// lengths of vectors between atoms that count as zero (Å)
constexpr double kAtomTolerance = 1e-4;

Vec add(const Vec& a, const Vec& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }

Vec scaled(const Vec& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }

bool zero(const Vec& v, double tolerance) {
    return std::abs(v[0]) <= tolerance && std::abs(v[1]) <= tolerance &&
           std::abs(v[2]) <= tolerance;
}

// Burgers circuits on the mesh, as lists of half-edges linked by next_,
// swept along the lines they enclose. A half-edge or triangle once taken by
// a circuit stays marked with it, which keeps the others off.
class Tracer {
public:
    Tracer(const InterfaceMesh& mesh, const ClusterInput& atoms, int max_circuit, int extended)
        : mesh_(mesh),
          atoms_(atoms),
          max_(max_circuit),
          extended_(extended),
          edge_circuit_(mesh.start.size(), -1),
          next_(mesh.start.size(), -1),
          face_circuit_(static_cast<std::size_t>(mesh.faces()), -1),
          visit_(static_cast<std::size_t>(mesh.vertices()), -1) {}

    void run(const Control& control);
    Network finish(const Clusters& clusters, int crystal);

private:
    struct Circuit {
        std::int64_t first = -1, last = -1;
        std::int64_t count = 0;
        std::int64_t node = -1;
        std::int64_t preliminary = 0;  // points of the line not kept unless it goes on
        bool stored = false;
        bool blocked = false;
        bool dangling = true;
    };
    // an end of a line: its circuit, the next end in its junction (itself
    // for a free end) and the other end of the line
    struct Node {
        std::int64_t line, circuit, ring, opposite;
    };
    struct Line {
        std::deque<Vec> points;
        std::deque<std::int32_t> cores;
        Vec burgers;
        std::int64_t cluster;
        std::array<std::int64_t, 2> nodes;  // at the last point, at the first
        bool alive = true;
    };
    struct Search {
        std::int64_t vertex;
        Vec coord;
        Mat frame;  // into the frame of the start
        int depth;
        std::int64_t predecessor;
        bool visited;
    };

    Vec physical(std::int64_t e) const {
        const double* p = mesh_.physical.data() + 3 * e;
        return {p[0], p[1], p[2]};
    }
    Vec ideal(std::int64_t e) const {
        const double* p = mesh_.ideal.data() + 3 * e;
        return {p[0], p[1], p[2]};
    }
    const Mat& back(std::int64_t e) const {
        return mesh_.matrices[static_cast<std::size_t>(mesh_.back[static_cast<std::size_t>(e)])];
    }
    bool within(std::int64_t e) const { return mesh_.back[static_cast<std::size_t>(e)] == 0; }
    std::int64_t opposite(std::int64_t e) const {
        return mesh_.opposite[static_cast<std::size_t>(e)];
    }
    std::int64_t& next(std::int64_t e) { return next_[static_cast<std::size_t>(e)]; }
    std::int64_t& circuit_of(std::int64_t e) { return edge_circuit_[static_cast<std::size_t>(e)]; }
    std::int64_t& face_circuit(std::int64_t e) {
        return face_circuit_[static_cast<std::size_t>(InterfaceMesh::face(e))];
    }
    bool forward(std::int64_t node) const {
        return lines_[static_cast<std::size_t>(nodes_[static_cast<std::size_t>(node)].line)]
                   .nodes[0] == node;
    }
    const Vec& position(std::int64_t node) const {
        const Line& line =
            lines_[static_cast<std::size_t>(nodes_[static_cast<std::size_t>(node)].line)];
        return forward(node) ? line.points.back() : line.points.front();
    }
    Vec wrap(Vec v) const;
    Vec shift(const Vec& a, const Vec& b) const;
    Vec center(std::int64_t circuit) const;
    std::int64_t edge_at(std::int64_t circuit, std::int64_t index) const;

    std::int64_t allocate();
    void primary(int length);
    bool create(std::int64_t edge, int length);
    bool intersects(std::int64_t circuit);
    void crossing(std::int64_t a1, std::int64_t a2, std::int64_t b1, std::int64_t b2,
                  int& outside, int& inside) const;
    std::int64_t reverse(std::int64_t forward_circuit);
    void create_line(const Vec& burgers, std::int64_t cluster, std::int64_t forward_circuit,
                     int length);
    void advance(std::int64_t node, int length, bool primary);
    bool remove_two(std::int64_t& e0, std::int64_t& e1, std::int64_t& e2);
    bool remove_three(std::int64_t& e0, std::int64_t& e1, std::int64_t& e2);
    bool remove_one(std::int64_t& e0, std::int64_t& e1, std::int64_t& e2);
    bool sweep_two(std::int64_t& e0, std::int64_t& e1, std::int64_t& e2);
    bool insert_one(std::int64_t& e0, std::int64_t& e1);
    void append(std::int64_t node);
    void join(int length);
    void secondary(std::int64_t first_edge, std::int64_t outer, int length);

    // junction rings
    bool in_ring(std::int64_t node, std::int64_t other) const;
    void connect(std::int64_t node, std::int64_t other);
    void dissolve(std::int64_t node);

    const InterfaceMesh& mesh_;
    const ClusterInput& atoms_;
    int max_, extended_;
    std::vector<std::int64_t> edge_circuit_, next_, face_circuit_;
    std::vector<Circuit> circuits_;
    std::int64_t unused_ = -1;
    std::vector<Node> nodes_;
    std::vector<Line> lines_;
    std::vector<std::int64_t> dangling_;
    std::vector<std::int64_t> visit_;
    std::vector<Search> search_;
    std::mt19937 random_{1};
};

Vec Tracer::wrap(Vec v) const {
    for (int k = 0; k < 3; ++k) {
        if (!atoms_.pbc[k]) {
            continue;
        }
        double f = 0.0;
        for (int j = 0; j < 3; ++j) {
            f += v[static_cast<std::size_t>(j)] * atoms_.inverse[3 * j + k];
        }
        const double r = std::round(f);
        for (int j = 0; j < 3; ++j) {
            v[static_cast<std::size_t>(j)] -= r * atoms_.matrix[3 * k + j];
        }
    }
    return v;
}

Vec Tracer::shift(const Vec& a, const Vec& b) const {
    const Vec d = minus(b, a);
    return minus(d, wrap(d));
}

Vec Tracer::center(std::int64_t circuit) const {
    const Circuit& c = circuits_[static_cast<std::size_t>(circuit)];
    Vec current{}, sum{};
    std::int64_t e = c.first;
    do {
        sum = add(sum, current);
        current = add(current, physical(e));
        e = next_[static_cast<std::size_t>(e)];
    } while (e != c.first);
    const double* p =
        mesh_.positions.data() + 3 * mesh_.start[static_cast<std::size_t>(c.first)];
    return add({p[0], p[1], p[2]}, scaled(sum, 1.0 / static_cast<double>(c.count)));
}

std::int64_t Tracer::edge_at(std::int64_t circuit, std::int64_t index) const {
    std::int64_t e = circuits_[static_cast<std::size_t>(circuit)].first;
    for (; index > 0; --index) {
        e = next_[static_cast<std::size_t>(e)];
    }
    return e;
}

std::int64_t Tracer::allocate() {
    if (unused_ >= 0) {
        const std::int64_t c = unused_;
        unused_ = -1;
        circuits_[static_cast<std::size_t>(c)] = Circuit{};
        return c;
    }
    circuits_.emplace_back();
    return static_cast<std::int64_t>(circuits_.size()) - 1;
}

void Tracer::run(const Control& control) {
    for (int length = 3; length <= extended_; ++length) {
        if (control.cancel != nullptr && control.cancel->cancelled()) {
            throw Cancelled();
        }
        for (std::size_t i = 0; i < dangling_.size(); ++i) {
            advance(dangling_[i], length, length <= max_);
        }
        if (length <= max_ && length % 2 == 1) {
            primary(length);
        }
        join(length);
        if (length >= max_) {
            for (const std::int64_t node : dangling_) {
                Circuit& c = circuits_[static_cast<std::size_t>(
                    nodes_[static_cast<std::size_t>(node)].circuit)];
                if (!c.stored) {
                    c.stored = true;
                    c.preliminary = 0;
                }
            }
        }
        if (control.monitor && !control.monitor(length - 2, extended_ - 2)) {
            throw Cancelled();
        }
    }
}

// From each vertex a breadth first search over the free edges, at most half
// the circuit deep, until two paths meet whose ideal vectors differ: the
// circuit of the two encloses a dislocation.
void Tracer::primary(int length) {
    const int depth = (length - 1) / 2;
    for (std::int64_t v = 0; v < mesh_.vertices(); ++v) {
        search_.clear();
        search_.push_back({v, {0, 0, 0}, kIdentity, 0, -1, false});
        visit_[static_cast<std::size_t>(v)] = 0;
        bool found = false;
        for (std::size_t head = 0; head < search_.size() && !found; ++head) {
            const std::int64_t vertex = search_[head].vertex;
            for (std::int64_t at = mesh_.offsets[static_cast<std::size_t>(vertex)];
                 at < mesh_.offsets[static_cast<std::size_t>(vertex) + 1]; ++at) {
                const std::int64_t e = mesh_.edges[static_cast<std::size_t>(at)];
                if (next(e) >= 0 || next(opposite(e)) >= 0 || face_circuit(e) >= 0) {
                    continue;
                }
                const std::int64_t w = mesh_.end(e);
                const Search current = search_[head];
                const Vec coord = add(current.coord, turn(current.frame, ideal(e)));
                const std::int64_t seen = visit_[static_cast<std::size_t>(w)];
                if (seen >= 0) {
                    const Search& other = search_[static_cast<std::size_t>(seen)];
                    if (!zero(minus(other.coord, coord), kTolerance) &&
                        close(multiply(current.frame, back(e)), other.frame) && create(e, length)) {
                        found = true;
                        break;
                    }
                } else if (current.depth < depth) {
                    visit_[static_cast<std::size_t>(w)] = static_cast<std::int64_t>(search_.size());
                    search_.push_back({w, coord,
                                       within(e) ? current.frame : multiply(current.frame, back(e)),
                                       current.depth + 1, e, false});
                }
            }
        }
        for (const Search& s : search_) {
            visit_[static_cast<std::size_t>(s.vertex)] = -1;
        }
    }
}

// The circuit down the path to the end of edge, back along it and up the
// path to its start; kept when its Burgers vector is not zero, it does not
// wrap around a periodic boundary and it crosses no other circuit. Whether
// to stop the search from this start.
bool Tracer::create(std::int64_t edge, int length) {
    auto search = [&](std::int64_t vertex) -> Search& {
        return search_[static_cast<std::size_t>(visit_[static_cast<std::size_t>(vertex)])];
    };
    auto up = [&](const Search& s) -> Search& { return search(mesh_.start[static_cast<std::size_t>(s.predecessor)]); };
    Search& current = search(mesh_.start[static_cast<std::size_t>(edge)]);
    Search& neighbor = search(mesh_.end(edge));
    const std::int64_t c = allocate();
    {
        Circuit& circuit = circuits_[static_cast<std::size_t>(c)];
        circuit.count = 1;
        circuit.first = circuit.last = opposite(edge);
    }
    circuit_of(opposite(edge)) = c;
    for (Search* a = &neighbor;; a = &up(*a)) {
        a->visited = false;
        if (a->predecessor < 0) {
            break;
        }
    }
    for (Search* a = &current;; a = &up(*a)) {
        a->visited = true;
        if (a->predecessor < 0) {
            break;
        }
    }
    Circuit& circuit = circuits_[static_cast<std::size_t>(c)];
    for (Search* a = &neighbor;; a = &up(*a)) {
        if (a->visited) {
            a->visited = false;
            break;
        }
        const std::int64_t p = a->predecessor;
        next(p) = circuit.first;
        circuit.first = p;
        ++circuit.count;
        circuit_of(p) = c;
    }
    for (Search* a = &current; a->visited; a = &up(*a)) {
        const std::int64_t p = opposite(a->predecessor);
        next(circuit.last) = p;
        circuit.last = p;
        ++circuit.count;
        circuit_of(p) = c;
        a->visited = false;
    }
    next(circuit.last) = circuit.first;
    Vec sum{}, burgers{};
    Mat frame = kIdentity;
    std::int64_t e = circuit.first;
    do {
        sum = add(sum, physical(e));
        burgers = add(burgers, turn(frame, ideal(e)));
        if (!within(e)) {
            frame = multiply(frame, back(e));
        }
        e = next(e);
    } while (e != circuit.first);
    const bool crosses = intersects(c);
    if (zero(burgers, kTolerance) || !zero(sum, kAtomTolerance) || crosses) {
        e = circuit.first;
        do {
            const std::int64_t following = next(e);
            next(e) = -1;
            circuit_of(e) = -1;
            e = following;
        } while (e != circuit.first);
        unused_ = c;
        return crosses;
    }
    create_line(burgers, mesh_.cluster[static_cast<std::size_t>(circuit.first)], c, length);
    return true;
}

// Whether the circuit crosses another one at one of its vertices.
bool Tracer::intersects(std::int64_t circuit) {
    const std::int64_t first = circuits_[static_cast<std::size_t>(circuit)].first;
    std::int64_t e1 = first;
    do {
        const std::int64_t e2 = next(e1);
        if (e1 != opposite(e2)) {
            std::int64_t current = opposite(e1);
            int steps = 0;
            do {
                const std::int64_t in = InterfaceMesh::prev(current);
                if (in != e2 && circuit_of(in) >= 0) {
                    int outside = 0, inside = 0;
                    crossing(opposite(e2), opposite(e1), in, next(in), outside, inside);
                    if (outside != 0) {
                        return true;
                    }
                }
                current = opposite(in);
            } while (current != e2 && ++steps < kFan);
        }
        e1 = e2;
    } while (e1 != first);
    return false;
}

// Whether circuit a, coming in along a1 and leaving along a2, goes out of or
// into circuit b, coming in along b1 and leaving along b2, at the vertex
// where they meet.
void Tracer::crossing(std::int64_t a1, std::int64_t a2, std::int64_t b1, std::int64_t b2,
                      int& outside, int& inside) const {
    bool in1 = false, in2 = false;
    std::int64_t e = b2;
    for (int steps = 0; steps < kFan; ++steps) {
        const std::int64_t o = opposite(e);
        if (o == b1) {
            break;
        }
        if (e != b2) {
            in1 = in1 || o == a1;
            in2 = in2 || e == a2;
        }
        e = InterfaceMesh::next(o);
    }
    bool out1 = false, out2 = false;
    e = b1;
    for (int steps = 0; steps < kFan; ++steps) {
        const std::int64_t following = InterfaceMesh::next(e);
        if (following == b2) {
            break;
        }
        e = opposite(following);
        out1 = out1 || e == a1;
        out2 = out2 || following == a2;
    }
    if (out2 && !out1) {
        ++outside;
    } else if (in2 && !in1) {
        ++inside;
    }
}

// The circuit on the other side of the triangles next to a new one, run
// the other way.
std::int64_t Tracer::reverse(std::int64_t forward_circuit) {
    const std::int64_t c = allocate();
    const std::int64_t start = circuits_[static_cast<std::size_t>(forward_circuit)].first;
    Circuit& circuit = circuits_[static_cast<std::size_t>(c)];
    std::int64_t e1 = start;
    do {
        const std::int64_t e2 = next(e1);
        const std::int64_t o1 = opposite(e1), o2 = opposite(e2);
        if (InterfaceMesh::face(o1) != InterfaceMesh::face(o2)) {
            const std::int64_t inner1 = opposite(InterfaceMesh::prev(o1));
            const std::int64_t inner2 = opposite(InterfaceMesh::next(o2));
            face_circuit(o1) = c;
            face_circuit(o2) = c;
            circuit_of(inner1) = c;
            circuit_of(inner2) = c;
            next(inner2) = inner1;
            if (circuit.last < 0) {
                circuit.last = inner1;
                circuit.first = inner2;
                circuit.count += 2;
            } else if (circuit.last != inner2) {
                if (inner1 != circuit.first) {
                    next(inner1) = circuit.first;
                    circuit.count += 2;
                } else {
                    circuit.count += 1;
                }
                circuit.first = inner2;
            } else if (circuit.first != inner1) {
                next(inner1) = circuit.first;
                circuit.first = inner1;
                circuit.count += 1;
            }
        }
        e1 = e2;
    } while (e1 != start);
    next(circuit.last) = circuit.first;
    return c;
}

void Tracer::create_line(const Vec& burgers, std::int64_t cluster,
                         std::int64_t forward_circuit, int length) {
    const std::int64_t backward_circuit = reverse(forward_circuit);
    const auto l = static_cast<std::int64_t>(lines_.size());
    const auto fn = static_cast<std::int64_t>(nodes_.size()), bn = fn + 1;
    nodes_.push_back({l, forward_circuit, fn, bn});
    nodes_.push_back({l, backward_circuit, bn, fn});
    circuits_[static_cast<std::size_t>(forward_circuit)].node = fn;
    circuits_[static_cast<std::size_t>(backward_circuit)].node = bn;
    dangling_.push_back(fn);
    dangling_.push_back(bn);
    Line line;
    line.burgers = burgers;
    line.cluster = cluster;
    line.nodes = {fn, bn};
    line.points.push_back(center(backward_circuit));
    line.cores.push_back(static_cast<std::int32_t>(
        circuits_[static_cast<std::size_t>(backward_circuit)].count));
    lines_.push_back(std::move(line));
    append(fn);
    advance(fn, length, true);
    advance(bn, length, true);
}

// Sweeps the circuit of the node over the triangles ahead, shortening it
// where it can, then lengthening it by one edge to pass on, until it is
// length edges long or stuck.
void Tracer::advance(std::int64_t node, int length, bool primary) {
    const std::int64_t c = nodes_[static_cast<std::size_t>(node)].circuit;
    (void)primary;
    for (;;) {
        std::int64_t count = circuits_[static_cast<std::size_t>(c)].count;
        std::int64_t first = edge_at(c, static_cast<std::int64_t>(random_() % static_cast<std::uint64_t>(count)));
        std::int64_t e0 = first, e1 = next(e0), e2 = next(e1);
        std::int64_t counter = 0;
        do {
            if (remove_two(e0, e1, e2) || remove_three(e0, e1, e2) || remove_one(e0, e1, e2) ||
                sweep_two(e0, e1, e2)) {
                append(node);
                counter = -1;
            }
            e0 = e1;
            e1 = e2;
            e2 = next(e2);
            ++counter;
        } while (counter <= circuits_[static_cast<std::size_t>(c)].count);
        count = circuits_[static_cast<std::size_t>(c)].count;
        if (count >= length) {
            break;
        }
        first = edge_at(c, static_cast<std::int64_t>(random_() % static_cast<std::uint64_t>(count)));
        e0 = first;
        e1 = next(first);
        bool extended = false;
        do {
            if (insert_one(e0, e1)) {
                extended = true;
                append(node);
                break;
            }
            e0 = e1;
            e1 = next(e1);
        } while (e0 != first);
        if (!extended) {
            break;
        }
    }
}

// Two edges there and back.
bool Tracer::remove_two(std::int64_t& e0, std::int64_t& e1, std::int64_t& e2) {
    if (e1 != opposite(e2)) {
        return false;
    }
    Circuit& c = circuits_[static_cast<std::size_t>(circuit_of(e0))];
    next(e0) = next(e2);
    if (e0 == c.last) {
        c.first = next(c.last);
    } else if (e1 == c.last) {
        c.last = e0;
        c.first = next(e0);
    } else if (e2 == c.last) {
        c.last = e0;
    }
    c.count -= 2;
    e1 = next(e0);
    e2 = next(e1);
    return true;
}

// Three edges around a free triangle.
bool Tracer::remove_three(std::int64_t& e0, std::int64_t& e1, std::int64_t& e2) {
    const std::int64_t f = InterfaceMesh::face(e1);
    if (InterfaceMesh::face(e2) != f || face_circuit(e1) >= 0) {
        return false;
    }
    const std::int64_t id = circuit_of(e0);
    Circuit& c = circuits_[static_cast<std::size_t>(id)];
    const std::int64_t e3 = next(e2);
    if (InterfaceMesh::face(e3) != f) {
        return false;
    }
    next(e0) = next(e3);
    if (e2 == c.first || e3 == c.first) {
        c.first = next(e3);
        c.last = e0;
    } else if (e1 == c.first) {
        c.first = next(e3);
    } else if (e3 == c.last) {
        c.last = e0;
    }
    c.count -= 3;
    e1 = next(e3);
    e2 = next(e1);
    face_circuit_[static_cast<std::size_t>(f)] = id;
    return true;
}

// Two edges of a free triangle replaced by its third.
bool Tracer::remove_one(std::int64_t& e0, std::int64_t& e1, std::int64_t& e2) {
    const std::int64_t f = InterfaceMesh::face(e1);
    if (InterfaceMesh::face(e2) != f || face_circuit(e1) >= 0 || InterfaceMesh::face(e0) == f) {
        return false;
    }
    const std::int64_t id = circuit_of(e0);
    const std::int64_t shortcut = opposite(InterfaceMesh::prev(e1));
    if (circuit_of(shortcut) >= 0) {
        return false;
    }
    Circuit& c = circuits_[static_cast<std::size_t>(id)];
    next(shortcut) = next(e2);
    next(e0) = shortcut;
    if (e0 == c.last) {
        c.first = shortcut;
    }
    if (e2 == c.last) {
        c.last = shortcut;
    } else if (e2 == c.first) {
        c.first = next(shortcut);
        c.last = shortcut;
    }
    c.count -= 1;
    e1 = shortcut;
    e2 = next(shortcut);
    circuit_of(shortcut) = id;
    face_circuit_[static_cast<std::size_t>(f)] = id;
    return true;
}

// Two edges on two free triangles that share an edge replaced by the two
// edges beyond them.
bool Tracer::sweep_two(std::int64_t& e0, std::int64_t& e1, std::int64_t& e2) {
    if (face_circuit(e1) >= 0 || face_circuit(e2) >= 0) {
        return false;
    }
    const std::int64_t id = circuit_of(e0);
    Circuit& c = circuits_[static_cast<std::size_t>(id)];
    if (InterfaceMesh::face(e1) == InterfaceMesh::face(e2) || c.count <= 2) {
        return false;
    }
    const std::int64_t outer1 = opposite(InterfaceMesh::prev(e1));
    const std::int64_t inner1 = InterfaceMesh::next(e1);
    const std::int64_t outer2 = opposite(InterfaceMesh::next(e2));
    const std::int64_t inner2 = InterfaceMesh::prev(e2);
    if (inner1 != opposite(inner2) || circuit_of(outer1) >= 0 || circuit_of(outer2) >= 0) {
        return false;
    }
    next(outer1) = outer2;
    next(outer2) = next(e2);
    next(e0) = outer1;
    if (e0 == c.last) {
        c.first = outer1;
    } else if (e1 == c.last) {
        c.last = outer1;
        c.first = outer2;
    } else if (e2 == c.last) {
        c.last = outer2;
    }
    circuit_of(outer1) = id;
    circuit_of(outer2) = id;
    face_circuit(e1) = id;
    face_circuit(e2) = id;
    e0 = outer1;
    e1 = outer2;
    e2 = next(e1);
    return true;
}

// An edge across a free triangle replaced by the two others.
bool Tracer::insert_one(std::int64_t& e0, std::int64_t& e1) {
    if (face_circuit(e1) >= 0) {
        return false;
    }
    const std::int64_t first = opposite(InterfaceMesh::prev(e1));
    const std::int64_t second = opposite(InterfaceMesh::next(e1));
    if (circuit_of(first) >= 0 || circuit_of(second) >= 0) {
        return false;
    }
    const std::int64_t id = circuit_of(e0);
    Circuit& c = circuits_[static_cast<std::size_t>(id)];
    next(first) = second;
    next(second) = next(e1);
    next(e0) = first;
    if (e0 == c.last) {
        c.first = first;
    } else if (e1 == c.last) {
        c.last = second;
    }
    circuit_of(first) = id;
    circuit_of(second) = id;
    ++c.count;
    face_circuit(e1) = id;
    return true;
}

void Tracer::append(std::int64_t node) {
    const Node& n = nodes_[static_cast<std::size_t>(node)];
    Circuit& c = circuits_[static_cast<std::size_t>(n.circuit)];
    Line& line = lines_[static_cast<std::size_t>(n.line)];
    const bool at_end = line.nodes[0] == node;
    const Vec last = at_end ? line.points.back() : line.points.front();
    const Vec point = add(last, wrap(minus(center(n.circuit), last)));
    if (at_end) {
        line.points.push_back(point);
        line.cores.push_back(static_cast<std::int32_t>(c.count));
    } else {
        line.points.push_front(point);
        line.cores.push_front(static_cast<std::int32_t>(c.count));
    }
    ++c.preliminary;
}

bool Tracer::in_ring(std::int64_t node, std::int64_t other) const {
    std::int64_t n = nodes_[static_cast<std::size_t>(node)].ring;
    do {
        if (n == other) {
            return true;
        }
        n = nodes_[static_cast<std::size_t>(n)].ring;
    } while (n != nodes_[static_cast<std::size_t>(node)].ring);
    return false;
}

void Tracer::connect(std::int64_t node, std::int64_t other) {
    std::swap(nodes_[static_cast<std::size_t>(node)].ring,
              nodes_[static_cast<std::size_t>(other)].ring);
}

void Tracer::dissolve(std::int64_t node) {
    std::int64_t n = nodes_[static_cast<std::size_t>(node)].ring;
    while (n != node) {
        const std::int64_t following = nodes_[static_cast<std::size_t>(n)].ring;
        nodes_[static_cast<std::size_t>(n)].ring = n;
        n = following;
    }
    nodes_[static_cast<std::size_t>(node)].ring = node;
}

// Lines whose circuits close around holes between others start there; the
// circuits a ring of others encloses fully end in a junction, or, two of
// them, join their lines.
void Tracer::join(int length) {
    for (std::size_t i = 0; i < dangling_.size(); ++i) {
        const std::int64_t c = nodes_[static_cast<std::size_t>(dangling_[i])].circuit;
        std::int64_t e = circuits_[static_cast<std::size_t>(c)].first;
        do {
            if (circuit_of(opposite(e)) < 0) {
                secondary(e, c, length);
                while (circuit_of(opposite(e)) < 0 && e != circuits_[static_cast<std::size_t>(c)].first) {
                    e = next(e);
                }
            } else {
                e = next(e);
            }
        } while (e != circuits_[static_cast<std::size_t>(c)].first);
    }
    for (const std::int64_t node : dangling_) {
        const std::int64_t c = nodes_[static_cast<std::size_t>(node)].circuit;
        circuits_[static_cast<std::size_t>(c)].blocked = true;
        const std::int64_t first = circuits_[static_cast<std::size_t>(c)].first;
        std::int64_t e = first;
        do {
            const std::int64_t adjacent = circuit_of(opposite(e));
            if (adjacent < 0) {
                circuits_[static_cast<std::size_t>(c)].blocked = false;
                break;
            }
            if (adjacent != c && circuits_[static_cast<std::size_t>(adjacent)].dangling) {
                const std::int64_t other = circuits_[static_cast<std::size_t>(adjacent)].node;
                if (!in_ring(node, other)) {
                    connect(node, other);
                }
            }
            e = next(e);
        } while (e != first);
    }
    for (const std::int64_t node : dangling_) {
        const std::int64_t c = nodes_[static_cast<std::size_t>(node)].circuit;
        if (!circuits_[static_cast<std::size_t>(c)].dangling) {
            continue;
        }
        if (!circuits_[static_cast<std::size_t>(c)].blocked) {
            dissolve(node);
            continue;
        }
        if (nodes_[static_cast<std::size_t>(node)].ring == node) {
            continue;
        }
        const Vec base = position(node);
        Vec sum{};
        int arms = 1;
        bool blocked = true;
        for (std::int64_t arm = nodes_[static_cast<std::size_t>(node)].ring; arm != node;
             arm = nodes_[static_cast<std::size_t>(arm)].ring) {
            if (!circuits_[static_cast<std::size_t>(nodes_[static_cast<std::size_t>(arm)].circuit)]
                     .blocked) {
                blocked = false;
                break;
            }
            ++arms;
            sum = add(sum, wrap(minus(position(arm), base)));
        }
        if (!blocked) {
            dissolve(node);
            continue;
        }
        if (arms >= 3) {
            const Vec middle = add(base, scaled(sum, 1.0 / arms));
            std::int64_t arm = node;
            do {
                const Node& a = nodes_[static_cast<std::size_t>(arm)];
                circuits_[static_cast<std::size_t>(a.circuit)].dangling = false;
                Line& line = lines_[static_cast<std::size_t>(a.line)];
                if (line.nodes[0] == arm) {
                    line.points.push_back(add(line.points.back(), wrap(minus(middle, line.points.back()))));
                    line.cores.push_back(line.cores.back());
                } else {
                    line.points.push_front(add(line.points.front(), wrap(minus(middle, line.points.front()))));
                    line.cores.push_front(line.cores.front());
                }
                circuits_[static_cast<std::size_t>(a.circuit)].preliminary = 0;
                arm = a.ring;
            } while (arm != node);
            continue;
        }
        // two arms: one line closes on itself, or two lines become one
        const std::int64_t n1 = node, n2 = nodes_[static_cast<std::size_t>(node)].ring;
        Node& node1 = nodes_[static_cast<std::size_t>(n1)];
        Node& node2 = nodes_[static_cast<std::size_t>(n2)];
        circuits_[static_cast<std::size_t>(node1.circuit)].dangling = false;
        circuits_[static_cast<std::size_t>(node2.circuit)].dangling = false;
        circuits_[static_cast<std::size_t>(node1.circuit)].preliminary = 0;
        circuits_[static_cast<std::size_t>(node2.circuit)].preliminary = 0;
        if (node1.opposite == n2) {
            Line& loop = lines_[static_cast<std::size_t>(node1.line)];
            if (!zero(wrap(minus(position(n1), position(n2))), kAtomTolerance)) {
                loop.points.push_back(add(loop.points.back(), wrap(minus(loop.points.front(), loop.points.back()))));
                loop.cores.push_back(loop.cores.back());
            }
            continue;
        }
        const std::int64_t far1 = node1.opposite, far2 = node2.opposite;
        Line& line1 = lines_[static_cast<std::size_t>(node1.line)];
        Line& line2 = lines_[static_cast<std::size_t>(node2.line)];
        const auto size2 = static_cast<std::ptrdiff_t>(line2.points.size());
        const bool first1 = line1.nodes[1] == n1, first2 = line2.nodes[1] == n2;
        if (first1) {
            line1.nodes[1] = far2;
            Vec offset;
            if (first2) {
                offset = shift(line1.points.front(), line2.points.front());
                line1.points.insert(line1.points.begin(), line2.points.rbegin(), line2.points.rend() - 1);
                line1.cores.insert(line1.cores.begin(), line2.cores.rbegin(), line2.cores.rend() - 1);
            } else {
                offset = shift(line1.points.front(), line2.points.back());
                line1.points.insert(line1.points.begin(), line2.points.begin(), line2.points.end() - 1);
                line1.cores.insert(line1.cores.begin(), line2.cores.begin(), line2.cores.end() - 1);
            }
            for (auto p = line1.points.begin(); p != line1.points.begin() + (size2 - 1); ++p) {
                *p = minus(*p, offset);
            }
        } else {
            line1.nodes[0] = far2;
            Vec offset;
            if (first2) {
                offset = shift(line1.points.back(), line2.points.front());
                line1.points.insert(line1.points.end(), line2.points.begin() + 1, line2.points.end());
                line1.cores.insert(line1.cores.end(), line2.cores.begin() + 1, line2.cores.end());
            } else {
                offset = shift(line1.points.back(), line2.points.back());
                line1.points.insert(line1.points.end(), line2.points.rbegin() + 1, line2.points.rend());
                line1.cores.insert(line1.cores.end(), line2.cores.rbegin() + 1, line2.cores.rend());
            }
            for (auto p = line1.points.end() - (size2 - 1); p != line1.points.end(); ++p) {
                *p = minus(*p, offset);
            }
        }
        line2.alive = false;
        nodes_[static_cast<std::size_t>(far2)].line = node1.line;
        nodes_[static_cast<std::size_t>(far2)].opposite = far1;
        nodes_[static_cast<std::size_t>(far1)].opposite = far2;
        node1.opposite = n2;
        node2.opposite = n1;
    }
    dangling_.erase(std::remove_if(dangling_.begin(), dangling_.end(),
                                   [&](std::int64_t node) {
                                       return nodes_[static_cast<std::size_t>(node)].ring != node;
                                   }),
                    dangling_.end());
}

// The circuit around the hole beyond first_edge, along the circuits around
// it; a new line when it goes along more than one of them and encloses a
// Burgers vector.
void Tracer::secondary(std::int64_t first_edge, std::int64_t outer, int length) {
    std::int64_t count = 1;
    Vec burgers{}, sum{};
    std::int64_t base = -1;
    Mat frame = kIdentity;
    int circuits = 1;
    const std::int64_t start = opposite(first_edge);
    std::int64_t end = start, edge = start;
    bool lost = false;
    for (;;) {
        int steps = 0;
        for (;; ++steps) {
            const std::int64_t in = InterfaceMesh::prev(opposite(edge));
            if (circuit_of(in) >= 0) {
                if (circuit_of(in) != outer) {
                    outer = circuit_of(in);
                    ++circuits;
                }
                edge = opposite(in);
                break;
            }
            edge = in;
            if (steps >= kFan) {
                lost = true;
                break;
            }
        }
        if (lost) {
            break;
        }
        next(end) = edge;
        sum = add(sum, physical(edge));
        burgers = add(burgers, turn(frame, ideal(edge)));
        if (base < 0) {
            base = mesh_.cluster[static_cast<std::size_t>(edge)];
        }
        if (!within(edge)) {
            frame = multiply(frame, back(edge));
        }
        if (edge == start) {
            break;
        }
        end = edge;
        ++count;
        if (count > length) {
            break;
        }
    }
    if (lost || circuits == 1 || count > length || zero(burgers, kTolerance) ||
        !zero(sum, kAtomTolerance) || !close(frame, kIdentity)) {
        edge = start;
        for (;;) {
            const std::int64_t following = next(edge);
            next(edge) = -1;
            if (edge == end) {
                break;
            }
            edge = following;
        }
        return;
    }
    const std::int64_t c = allocate();
    Circuit& circuit = circuits_[static_cast<std::size_t>(c)];
    circuit.first = start;
    circuit.last = end;
    circuit.count = count;
    edge = start;
    do {
        circuit_of(edge) = c;
        edge = next(edge);
    } while (edge != start);
    create_line(burgers, base, c, length);
}

Network Tracer::finish(const Clusters& clusters, int crystal) {
    // the points of the stretched circuits beyond the ends are dropped
    for (Line& line : lines_) {
        if (!line.alive) {
            continue;
        }
        const auto back_points = circuits_[static_cast<std::size_t>(
            nodes_[static_cast<std::size_t>(line.nodes[1])].circuit)].preliminary;
        const auto front_points = circuits_[static_cast<std::size_t>(
            nodes_[static_cast<std::size_t>(line.nodes[0])].circuit)].preliminary;
        line.points.erase(line.points.begin(), line.points.begin() + back_points);
        line.cores.erase(line.cores.begin(), line.cores.begin() + back_points);
        line.points.erase(line.points.end() - front_points, line.points.end());
        line.cores.erase(line.cores.end() - front_points, line.cores.end());
    }
    // the Burgers vectors in the frames of crystal clusters
    std::vector<std::int64_t> touching(clusters.structure.size(), -1), area(clusters.structure.size(), 0);
    std::vector<Mat> into(clusters.structure.size(), kIdentity);
    for (const Transition& t : clusters.transitions) {
        for (const auto& [from, to, m] : {std::tuple{t.from, t.to, t.matrix},
                                          std::tuple{t.to, t.from, transpose(t.matrix)}}) {
            if (clusters.structure[static_cast<std::size_t>(from)] != crystal &&
                clusters.structure[static_cast<std::size_t>(to)] == crystal &&
                t.area > area[static_cast<std::size_t>(from)]) {
                area[static_cast<std::size_t>(from)] = t.area;
                touching[static_cast<std::size_t>(from)] = to;
                into[static_cast<std::size_t>(from)] = m;
            }
        }
    }
    for (Line& line : lines_) {
        const auto c = static_cast<std::size_t>(line.cluster);
        if (line.alive && touching[c] >= 0) {
            line.burgers = turn(into[c], line.burgers);
            line.cluster = touching[c];
        }
    }
    // each line along the axis it runs along most, forward
    for (Line& line : lines_) {
        if (!line.alive) {
            continue;
        }
        const Vec d = minus(line.points.back(), line.points.front());
        if (zero(d, kAtomTolerance)) {
            continue;
        }
        std::size_t axis = 0;
        for (std::size_t k = 1; k < 3; ++k) {
            if (std::abs(d[k]) > std::abs(d[axis])) {
                axis = k;
            }
        }
        if (d[axis] < 0) {
            line.burgers = scaled(line.burgers, -1.0);
            std::swap(line.nodes[0], line.nodes[1]);
            std::reverse(line.points.begin(), line.points.end());
            std::reverse(line.cores.begin(), line.cores.end());
        }
    }
    Network network;
    std::vector<std::int64_t> junction(nodes_.size(), -1);
    for (const Line& line : lines_) {
        if (!line.alive) {
            continue;
        }
        Segment s;
        for (const Vec& p : line.points) {
            s.points.insert(s.points.end(), p.begin(), p.end());
        }
        s.cores.assign(line.cores.begin(), line.cores.end());
        s.burgers = line.burgers;
        s.cluster = line.cluster;
        const std::int64_t last = line.nodes[0], first = line.nodes[1];
        s.closed = nodes_[static_cast<std::size_t>(last)].ring == first &&
                   nodes_[static_cast<std::size_t>(first)].ring == last;
        if (!s.closed) {
            for (int k = 0; k < 2; ++k) {
                const std::int64_t node = k == 0 ? first : last;
                if (nodes_[static_cast<std::size_t>(node)].ring == node) {
                    continue;
                }
                if (junction[static_cast<std::size_t>(node)] < 0) {
                    std::int64_t arm = node;
                    do {
                        junction[static_cast<std::size_t>(arm)] = network.junctions;
                        arm = nodes_[static_cast<std::size_t>(arm)].ring;
                    } while (arm != node);
                    ++network.junctions;
                }
                s.junctions[static_cast<std::size_t>(k)] = junction[static_cast<std::size_t>(node)];
            }
        }
        network.segments.push_back(std::move(s));
    }
    return network;
}

// The points of a line averaged over runs whose number squared is about
// separation times the edges of their circuits; the ends of an open line
// stay, those of a closed one become one point. An infinite line of short
// circuits becomes one straight piece through its mean point.
void coarsen(std::vector<Vec>& points, std::vector<std::int32_t>& cores, double separation,
             bool closed, bool infinite) {
    const auto n = static_cast<std::int64_t>(points.size());
    auto at = [&](std::int64_t k) -> const Vec& { return points[static_cast<std::size_t>(k)]; };
    auto core = [&](std::int64_t k) { return cores[static_cast<std::size_t>(k)]; };
    auto threshold = [&](std::int64_t sum) {
        return static_cast<std::int64_t>(separation * static_cast<double>(sum));
    };
    std::vector<Vec> out;
    std::vector<std::int32_t> out_cores;
    if (infinite && n >= 3) {
        std::int64_t sum = 0;
        Vec mean{};
        for (std::int64_t k = 0; k < n - 1; ++k) {
            sum += core(k);
            mean = add(mean, minus(at(k), at(0)));
        }
        const std::int64_t count = n - 1;
        if (static_cast<double>(sum) * separation > static_cast<double>(count * count)) {
            mean = scaled(mean, 1.0 / static_cast<double>(count));
            points = {add(at(0), mean), add(at(n - 1), mean)};
            cores.assign(2, static_cast<std::int32_t>(sum / count));
            return;
        }
    }
    if (n < 4) {
        return;
    }
    if (!closed) {
        out.push_back(at(0));
        out_cores.push_back(core(0));
    }
    const std::int64_t fewest = zero(minus(at(0), at(n - 1)), kAtomTolerance) ? 4 : 2;
    // the run at the start, and for a closed line the one at the end with it
    std::int64_t sum = 0, count = 0, i = 0;
    Vec mean{};
    do {
        sum += core(i);
        mean = add(mean, minus(at(i), at(0)));
        ++count;
        ++i;
    } while (2 * count * count < threshold(sum) && count + 1 < n / fewest / 2);
    std::int64_t j = n - 1;
    while (count * count < threshold(sum) && count < n / fewest) {
        sum += core(j);
        mean = add(mean, minus(at(j), at(n - 1)));
        ++count;
        --j;
    }
    const Vec shift = scaled(mean, 1.0 / static_cast<double>(count));
    if (closed) {
        out.push_back(add(at(0), shift));
        out_cores.push_back(static_cast<std::int32_t>(sum / count));
    }
    while (i < j) {
        std::int64_t run_sum = 0, run = 0;
        Vec run_mean{};
        do {
            run_sum += core(i);
            run_mean = add(run_mean, at(i));
            ++run;
            ++i;
        } while (run * run < threshold(run_sum) && run + 1 < n / fewest && i != j);
        out.push_back(scaled(run_mean, 1.0 / static_cast<double>(run)));
        out_cores.push_back(static_cast<std::int32_t>(run_sum / run));
    }
    if (!closed) {
        out.push_back(at(n - 1));
        out_cores.push_back(core(n - 1));
    } else {
        out.push_back(add(at(n - 1), shift));
        out_cores.push_back(static_cast<std::int32_t>(sum / count));
    }
    points = std::move(out);
    cores = std::move(out_cores);
}

// Taubin's smoothing of a line, levels times a step of lambda and one of
// mu; the ends of an open line stay.
void smooth(std::vector<Vec>& points, int levels, bool closed) {
    const std::size_t n = points.size();
    if (levels <= 0 || n <= 2 || (n <= 4 && zero(minus(points.front(), points.back()), kAtomTolerance))) {
        return;
    }
    const double lambda = 0.5, mu = 1.0 / (0.1 - 1.0 / lambda);
    std::vector<Vec> moves(n);
    for (int level = 0; level < levels; ++level) {
        for (const double factor : {lambda, mu}) {
            moves[0] = closed ? scaled(add(minus(points[n - 2], points[n - 1]),
                                           minus(points[1], points[0])),
                                       0.5)
                              : Vec{};
            for (std::size_t k = 1; k + 1 < n; ++k) {
                moves[k] = scaled(add(minus(points[k - 1], points[k]), minus(points[k + 1], points[k])), 0.5);
            }
            moves[n - 1] = moves[0];
            for (std::size_t k = 0; k < n; ++k) {
                points[k] = add(points[k], scaled(moves[k], factor));
            }
        }
    }
}

}  // namespace

void smooth_lines(Network& network, double separation, int levels) {
    for (Segment& s : network.segments) {
        std::vector<Vec> points(s.points.size() / 3);
        for (std::size_t k = 0; k < points.size(); ++k) {
            points[k] = {s.points[3 * k], s.points[3 * k + 1], s.points[3 * k + 2]};
        }
        const bool infinite = s.closed && !zero(minus(points.front(), points.back()), kAtomTolerance);
        if (separation > 0) {
            coarsen(points, s.cores, separation, s.closed, infinite);
        }
        smooth(points, levels, s.closed);
        s.points.clear();
        for (const Vec& p : points) {
            s.points.insert(s.points.end(), p.begin(), p.end());
        }
    }
}

Network trace_dislocations(const InterfaceMesh& mesh, const ClusterInput& atoms,
                           const Clusters& clusters, int max_circuit, int stretch,
                           const Control& control) {
    Tracer tracer(mesh, atoms, max_circuit, max_circuit + stretch);
    tracer.run(control);
    return tracer.finish(clusters, atoms.crystal);
}

}  // namespace mdapy::analysis::dxa
