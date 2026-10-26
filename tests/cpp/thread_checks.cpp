// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Checks of the thread pool and the parallel text kernels, without Python.
// CI builds this program with ThreadSanitizer, which cannot run the Python
// test suite on Linux (polars crashes on import under it):
//
//   g++ -std=c++17 -O1 -g -fsanitize=thread -Isrc/cpp -pthread -o thread_checks
//       src/cpp/core/parallel.cpp src/cpp/io/text.cpp src/cpp/neighbor/cell_list.cpp
//       src/cpp/neighbor/kd_tree.cpp src/cpp/calculators/eam.cpp src/cpp/calculators/nep.cpp
//       tests/cpp/thread_checks.cpp
//
// Every check also verifies its result; the program exits with 1 on a failure.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "analysis/distribution.h"
#include "analysis/geometry.h"
#include "analysis/steinhardt.h"
#include "analysis/strain.h"
#include "analysis/voronoi.h"
#include "calculators/eam.h"
#include "calculators/nep.h"
#include "core/parallel.h"
#include "identify/planar_faults.h"
#include "identify/ptm.h"
#include "identify/structure.h"
#include "io/text.h"
#include "neighbor/cell_list.h"
#include "neighbor/kd_tree.h"
#include "neighbor/list_view.h"
#include "visualization/render.h"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAILED: %s\n", what);
        ++failures;
    }
}

const int kThreads[] = {1, 2, 3, 8};

void loops_and_reductions() {
    const std::int64_t n = 100000;
    std::vector<double> values(n);
    std::mt19937_64 random(1);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    for (double& v : values) {
        v = uniform(random);
    }
    double reference = 0.0;
    bool first = true;
    for (int threads : kThreads) {
        mdapy::set_num_threads(threads);
        for (std::int64_t grain : {std::int64_t{0}, std::int64_t{7}, n}) {
            std::vector<std::int64_t> out(n, -1);
            mdapy::parallel_for(n, grain, [&](std::int64_t begin, std::int64_t end) {
                for (std::int64_t i = begin; i < end; ++i) {
                    out[i] = i * i;
                }
            });
            bool ok = true;
            for (std::int64_t i = 0; i < n; ++i) {
                ok = ok && out[i] == i * i;
            }
            check(ok, "parallel_for writes every item once");
        }
        const double sum = mdapy::parallel_reduce(
            n, 0, 0.0,
            [&](std::int64_t begin, std::int64_t end) {
                double s = 0.0;
                for (std::int64_t i = begin; i < end; ++i) {
                    s += values[i];
                }
                return s;
            },
            [](double a, double b) { return a + b; });
        if (first) {
            reference = sum;
            first = false;
        }
        check(std::memcmp(&sum, &reference, sizeof(sum)) == 0,
              "parallel_reduce gives the same bits for any thread count");
        for (std::int64_t count : {std::int64_t{0}, std::int64_t{1}, std::int64_t{1000}, n, std::int64_t{300007}}) {
            std::vector<std::int64_t> sums(static_cast<std::size_t>(count));
            for (std::int64_t i = 0; i < count; ++i) {
                sums[i] = i % 7 + 1;
            }
            std::vector<std::int64_t> expected = sums;
            for (std::int64_t i = 1; i < count; ++i) {
                expected[i] += expected[i - 1];
            }
            mdapy::prefix_sum(sums.data(), count);
            check(sums == expected, "prefix_sum gives the running sums for any thread count");
        }
    }
}

void idle_and_partial_pools() {
    // Regions right after each other, after the workers fall asleep, with
    // fewer chunks than workers, and with the pool resized in between.
    for (int threads : {8, 3, 8}) {
        mdapy::set_num_threads(threads);
        for (int round = 0; round < 300; ++round) {
            const std::int64_t n = 1 + round % 17;
            std::vector<std::int64_t> out(static_cast<std::size_t>(n), -1);
            mdapy::parallel_for(n, 1 + round % 3, [&](std::int64_t begin, std::int64_t end) {
                for (std::int64_t i = begin; i < end; ++i) {
                    out[static_cast<std::size_t>(i)] = i + round;
                }
            });
            bool ok = true;
            for (std::int64_t i = 0; i < n; ++i) {
                ok = ok && out[static_cast<std::size_t>(i)] == i + round;
            }
            check(ok, "regions of a few chunks write every item once");
            if (round % 50 == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        }
    }
}

void progress_cancellation_and_errors() {
    mdapy::set_num_threads(4);
    const auto pause = std::chrono::microseconds(200);
    auto slow = [&](std::int64_t begin, std::int64_t end) {
        for (std::int64_t i = begin; i < end; ++i) {
            std::this_thread::sleep_for(pause);
        }
    };

    // The monitor runs on the calling thread and ends with done == total.
    std::int64_t calls = 0, last_done = -1, last_total = -1;
    const std::thread::id caller = std::this_thread::get_id();
    bool on_caller = true;
    mdapy::Control control;
    control.monitor = [&](std::int64_t done, std::int64_t total) {
        ++calls;
        on_caller = on_caller && std::this_thread::get_id() == caller;
        last_done = done;
        last_total = total;
        return true;
    };
    mdapy::parallel_for(2000, 1, slow, control);
    check(calls > 0 && on_caller, "the monitor runs on the calling thread");
    check(last_done == last_total && last_total == 2000, "the last report is complete");

    // A token set from another thread cancels the region.
    mdapy::CancelToken token;
    std::thread canceller([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        token.cancel();
    });
    mdapy::Control cancelled;
    cancelled.cancel = &token;
    bool thrown = false;
    try {
        mdapy::parallel_for(100000, 1, slow, cancelled);
    } catch (const mdapy::Cancelled&) {
        thrown = true;
    }
    canceller.join();
    check(thrown, "a cancelled region throws Cancelled");

    // A monitor that returns false cancels too.
    mdapy::Control stop;
    stop.monitor = [](std::int64_t, std::int64_t) { return false; };
    thrown = false;
    try {
        mdapy::parallel_for(100000, 1, slow, stop);
    } catch (const mdapy::Cancelled&) {
        thrown = true;
    }
    check(thrown, "a monitor returning false cancels");

    // Exceptions of the body reach the caller.
    thrown = false;
    try {
        mdapy::parallel_for(1000, 3, [](std::int64_t begin, std::int64_t end) {
            if (begin <= 500 && 500 < end) {
                throw std::runtime_error("item 500");
            }
        });
    } catch (const std::runtime_error& e) {
        thrown = std::string(e.what()) == "item 500";
    }
    check(thrown, "an exception of the body is rethrown");
}

void nested_and_concurrent_regions() {
    mdapy::set_num_threads(4);
    const std::int64_t outer = 50, inner = 200;
    std::int64_t expected = 0;
    for (std::int64_t i = 0; i < outer; ++i) {
        for (std::int64_t j = 0; j < inner; ++j) {
            expected += i * j;
        }
    }
    auto nested = [&] {
        return mdapy::parallel_reduce(
            outer, 1, std::int64_t{0},
            [&](std::int64_t begin, std::int64_t end) {
                std::int64_t sum = 0;
                for (std::int64_t i = begin; i < end; ++i) {
                    sum += mdapy::parallel_reduce(
                        inner, 1, std::int64_t{0},
                        [&](std::int64_t b, std::int64_t e) {
                            std::int64_t s = 0;
                            for (std::int64_t j = b; j < e; ++j) {
                                s += i * j;
                            }
                            return s;
                        },
                        [](std::int64_t a, std::int64_t b) { return a + b; });
                }
                return sum;
            },
            [](std::int64_t a, std::int64_t b) { return a + b; });
    };
    check(nested() == expected, "nested regions run inline and are correct");

    // Two threads start regions at once; the busy pool runs one inline.
    std::int64_t results[2] = {0, 0};
    std::thread a([&] { results[0] = nested(); });
    std::thread b([&] { results[1] = nested(); });
    a.join();
    b.join();
    check(results[0] == expected && results[1] == expected,
          "concurrent regions from two threads are correct");
}

void text_tables() {
    using mdapy::io::Column;
    using mdapy::io::Kind;
    const std::int64_t rows = 20000;
    std::mt19937_64 random(2);
    std::uniform_real_distribution<double> uniform(-1e3, 1e3);
    std::vector<double> floats(rows * 3);
    std::vector<std::int64_t> ints(rows);
    std::vector<std::uint32_t> codes(rows);
    for (std::int64_t i = 0; i < rows; ++i) {
        for (int k = 0; k < 3; ++k) {
            floats[i * 3 + k] = uniform(random);
        }
        ints[i] = static_cast<std::int64_t>(random() % 1000000) - 500000;
        codes[i] = static_cast<std::uint32_t>(random() % 3);
    }
    std::vector<Column> out(3);
    out[0].kind = Kind::String;
    out[0].data = codes.data();
    out[0].categories = {"Cu", "Zr", "Al"};
    out[1].kind = Kind::Float;
    out[1].width = 3;
    out[1].data = floats.data();
    out[1].shift = -3;
    out[2].kind = Kind::Int;
    out[2].data = ints.data();
    std::string text;
    for (int threads : kThreads) {
        mdapy::set_num_threads(threads);
        const std::string formatted = mdapy::io::format_table(rows, out);
        if (text.empty()) {
            text = formatted;
        }
        check(formatted == text, "format_table gives the same text for any thread count");
    }

    std::vector<std::string> first_categories;
    std::vector<std::uint32_t> first_codes;
    for (int threads : kThreads) {
        mdapy::set_num_threads(threads);
        std::vector<double> f(rows * 3);
        std::vector<std::int64_t> n(rows);
        std::vector<std::uint32_t> c(rows);
        std::vector<Column> in(3);
        in[0].kind = Kind::String;
        in[0].data = c.data();
        in[1].kind = Kind::Float;
        in[1].width = 3;
        in[1].data = f.data();
        in[1].shift = 3;
        in[2].kind = Kind::Int;
        in[2].data = n.data();
        const std::int64_t end = mdapy::io::parse_table(
            text.data(), static_cast<std::int64_t>(text.size()), 0, rows, in, '\0');
        check(end == static_cast<std::int64_t>(text.size()), "parse_table reads every line");
        check(std::memcmp(f.data(), floats.data(), f.size() * sizeof(double)) == 0,
              "floats read back exactly");
        check(n == ints, "integers read back");
        if (first_categories.empty()) {
            first_categories = in[0].categories;
            first_codes = c;
        }
        check(in[0].categories == first_categories && c == first_codes,
              "string codes are the same for any thread count");
        bool same = true;
        for (std::int64_t i = 0; i < rows; ++i) {
            same = same && in[0].categories[c[i]] == out[0].categories[codes[i]];
        }
        check(same, "strings read back");
    }

    // The first bad line is reported, whichever thread finds a bad line
    // first: lines that start with the comment character have no values.
    std::string bad = text;
    for (std::int64_t row : {std::int64_t{15000}, std::int64_t{4000}, std::int64_t{4001}}) {
        std::size_t position = 0;
        for (std::int64_t r = 0; r < row; ++r) {
            position = bad.find('\n', position) + 1;
        }
        bad[position] = '?';
    }
    for (int threads : kThreads) {
        mdapy::set_num_threads(threads);
        std::vector<double> f(rows * 3);
        std::vector<std::int64_t> n(rows);
        std::vector<std::uint32_t> c(rows);
        std::vector<Column> in(3);
        in[0].kind = Kind::String;
        in[0].data = c.data();
        in[1].kind = Kind::Float;
        in[1].width = 3;
        in[1].data = f.data();
        in[2].kind = Kind::Int;
        in[2].data = n.data();
        std::int64_t row = -1;
        try {
            mdapy::io::parse_table(bad.data(), static_cast<std::int64_t>(bad.size()), 0, rows,
                                   in, '?');
        } catch (const mdapy::io::ParseError& e) {
            row = e.row;
        }
        check(row == 4000, "the first bad row is reported for any thread count");
    }
}

void neighbor_search() {
    const std::int64_t n = 3000;
    std::mt19937_64 random(3);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double matrix[9] = {20.0, 0.0, 0.0, 3.0, 18.0, 0.0, -2.0, 1.5, 22.0};
    const double origin[3] = {1.0, -2.0, 0.5};
    const bool pbc[3] = {true, true, false};
    std::vector<double> positions(n * 3);
    for (std::int64_t i = 0; i < n; ++i) {
        const double f[3] = {uniform(random), uniform(random), uniform(random)};
        for (int e = 0; e < 3; ++e) {
            positions[3 * i + e] =
                origin[e] + f[0] * matrix[e] + f[1] * matrix[3 + e] + f[2] * matrix[6 + e];
        }
    }
    // the pairs of each atom as a sorted list, to compare lists in any order
    using Key = std::tuple<mdapy::index_t, int, int, int, double>;
    const auto keys = [&](const mdapy::neighbor::NeighborPairs& pairs) {
        std::vector<std::vector<Key>> atoms(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i) {
            for (std::int64_t p = pairs.offsets[i]; p < pairs.offsets[i + 1]; ++p) {
                const std::int8_t* s = &pairs.shifts[3 * p];
                atoms[i].emplace_back(pairs.neighbors[p], s[0], s[1], s[2], pairs.distances[p]);
            }
            std::sort(atoms[i].begin(), atoms[i].end());
        }
        return atoms;
    };
    mdapy::neighbor::NeighborPairs first, first_near;
    for (int threads : kThreads) {
        mdapy::set_num_threads(threads);
        mdapy::neighbor::NeighborPairs pairs = mdapy::neighbor::build(
            positions.data(), n, matrix, origin, pbc, 3.5);
        if (threads == kThreads[0]) {
            first = pairs;
        }
        check(pairs.offsets == first.offsets && pairs.neighbors == first.neighbors &&
                  pairs.shifts == first.shifts && pairs.distances == first.distances,
              "the neighbor search gives the same pairs for any thread count");
        check(pairs.offsets.back() > 0, "the neighbor search finds pairs");
        std::vector<std::int64_t> order, kept;
        mdapy::neighbor::order_by_distance(positions.data(), n, pairs.offsets.data(),
                                           pairs.neighbors.data(), pairs.shifts.data(),
                                           matrix, 12, order, kept);
        bool sorted = kept.back() == 12 * n;
        for (std::int64_t i = 0; i < n && sorted; ++i) {
            for (std::int64_t k = kept[i] + 1; k < kept[i + 1]; ++k) {
                sorted = sorted && pairs.distances[order[k - 1]] <= pairs.distances[order[k]];
            }
        }
        check(sorted, "the nearest neighbors come in order of distance");
        const mdapy::neighbor::NeighborPairs near =
            mdapy::neighbor::nearest(positions.data(), n, matrix, origin, pbc, 12);
        if (threads == kThreads[0]) {
            first_near = near;
        }
        check(near.neighbors == first_near.neighbors && near.shifts == first_near.shifts &&
                  near.distances == first_near.distances,
              "the k-d tree gives the same neighbors for any thread count");
        bool same = true;
        for (std::int64_t i = 0; i < n && same; ++i) {
            for (int k = 0; k < 12; ++k) {
                same = same && near.distances[12 * i + k] == pairs.distances[order[kept[i] + k]];
            }
        }
        check(same, "the k-d tree finds the first pairs of the ordered list");
        std::vector<double> vectors(3 * pairs.neighbors.size());
        mdapy::neighbor::pair_vectors(positions.data(), n, pairs.offsets.data(),
                                      pairs.neighbors.data(), pairs.shifts.data(), matrix,
                                      vectors.data());
        bool lengths = true;
        for (std::size_t p = 0; p < pairs.neighbors.size(); ++p) {
            const double* v = &vectors[3 * p];
            lengths = lengths && std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) ==
                                     pairs.distances[p];
        }
        check(lengths, "the pair vectors have the lengths of the distances");
        // the pairs within 3 of a list within 3.5, as a new search finds them
        const mdapy::neighbor::NeighborPairs followed = mdapy::neighbor::follow(
            positions.data(), n, pairs.offsets.data(), pairs.neighbors.data(),
            pairs.shifts.data(), matrix, 3.0);
        const mdapy::neighbor::NeighborPairs shorter =
            mdapy::neighbor::build(positions.data(), n, matrix, origin, pbc, 3.0);
        check(followed.offsets == shorter.offsets && keys(followed) == keys(shorter),
              "following a longer list gives the pairs of a new search");
        // every atom is the nearest atom to its own position
        const mdapy::neighbor::KdTree tree(positions.data(), n, matrix, origin, pbc);
        std::vector<mdapy::index_t> j(static_cast<std::size_t>(n));
        std::vector<std::int8_t> shifts(3 * static_cast<std::size_t>(n));
        std::vector<double> distances(static_cast<std::size_t>(n));
        tree.nearest_points(positions.data(), n, 1, j.data(), shifts.data(), distances.data());
        bool own = true;
        for (std::int64_t i = 0; i < n; ++i) {
            own = own && j[i] == i && distances[i] == 0.0;
        }
        check(own, "every atom is the nearest atom to its position");
    }
    {
        // a tree wide enough that its top levels are split by all threads together
        const std::int64_t m = 40000;
        std::vector<double> many(m * 3);
        for (std::int64_t i = 0; i < m; ++i) {
            const double f[3] = {uniform(random), uniform(random), uniform(random)};
            for (int e = 0; e < 3; ++e) {
                many[3 * i + e] =
                    origin[e] + f[0] * matrix[e] + f[1] * matrix[3 + e] + f[2] * matrix[6 + e];
            }
        }
        mdapy::neighbor::NeighborPairs reference;
        for (int threads : kThreads) {
            mdapy::set_num_threads(threads);
            const mdapy::neighbor::NeighborPairs near =
                mdapy::neighbor::nearest(many.data(), m, matrix, origin, pbc, 6);
            if (threads == kThreads[0]) {
                reference = near;
            }
            check(near.neighbors == reference.neighbors && near.shifts == reference.shifts &&
                      near.distances == reference.distances,
                  "a wide k-d tree gives the same neighbors for any thread count");
        }
    }
    bool thrown = sizeof(mdapy::index_t) == 8;
    try {
        mdapy::check_atom_count(std::int64_t{1} << 31);
    } catch (const std::overflow_error&) {
        thrown = true;
    }
    check(thrown, "2^31 atoms need 64-bit indices");
}

// A small binary alloy and its full neighbor list as pair vectors.
struct Pairs {
    std::int64_t n = 0;
    std::vector<std::int32_t> species;
    std::vector<double> positions;
    double matrix[9];
    mdapy::neighbor::NeighborPairs pairs;
};

Pairs alloy_pairs(double cutoff) {
    Pairs p;
    const int cells = 6;
    const double a = 3.0;
    const double matrix[9] = {cells * a, 0.0, 0.0, 0.0, cells * a, 0.0, 0.0, 0.0, cells * a};
    std::copy(matrix, matrix + 9, p.matrix);
    const double origin[3] = {0.0, 0.0, 0.0};
    const bool pbc[3] = {true, true, true};
    const double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
    std::mt19937_64 random(5);
    std::normal_distribution<double> jitter(0.0, 0.2);
    for (int x = 0; x < cells; ++x) {
        for (int y = 0; y < cells; ++y) {
            for (int z = 0; z < cells; ++z) {
                for (const auto& b : basis) {
                    p.positions.push_back((x + b[0]) * a + jitter(random));
                    p.positions.push_back((y + b[1]) * a + jitter(random));
                    p.positions.push_back((z + b[2]) * a + jitter(random));
                    p.species.push_back(static_cast<std::int32_t>(random() % 2));
                }
            }
        }
    }
    p.n = static_cast<std::int64_t>(p.species.size());
    p.pairs = mdapy::neighbor::build(p.positions.data(), p.n, matrix, origin, pbc, cutoff);
    return p;
}

void calculators() {
    // EAM with smooth analytic tables
    const double cutoff = 4.5;
    const Pairs p = alloy_pairs(cutoff);
    mdapy::calculators::EamPotential eam;
    eam.elements = 2;
    eam.cutoff = cutoff;
    const int nr = 500, nrho = 500;
    const double dr = 0.01, drho = 0.05;
    eam.rho_max = (nrho - 1) * drho;
    for (int e = 0; e < 2; ++e) {
        std::vector<double> f(nrho), rho(nr);
        for (int k = 0; k < nrho; ++k) {
            f[k] = -std::sqrt(k * drho + 0.1 * e);
        }
        for (int k = 0; k < nr; ++k) {
            const double r = k * dr;
            rho[k] = r < cutoff ? (1.0 + e) * std::exp(-r) * (cutoff - r) * (cutoff - r) : 0.0;
        }
        eam.embedding.emplace_back(f.data(), nrho, drho);
        eam.density.emplace_back(rho.data(), nr, dr);
    }
    for (int k = 0; k < 3; ++k) {
        std::vector<double> z(nr);
        for (int m = 0; m < nr; ++m) {
            const double r = m * dr;
            z[m] = r < cutoff ? (1.0 + 0.1 * k) * std::exp(-2.0 * r) * (cutoff - r) : 0.0;
        }
        eam.pair.emplace_back(z.data(), nr, dr);
    }

    // NEP with random parameters and ZBL
    mdapy::calculators::NepModel nep;
    nep.types = 2;
    nep.n_radial = 4;
    nep.n_angular = 3;
    nep.k_radial = 6;
    nep.k_angular = 5;
    nep.l_max = 4;
    nep.four_body = true;
    nep.five_body = true;
    nep.neurons = 5;
    nep.dim = nep.n_radial + 1 + (nep.n_angular + 1) * (nep.l_max + 2);
    std::mt19937_64 random(7);
    std::normal_distribution<double> normal(0.0, 0.3);
    auto fill = [&](std::vector<double>& v, std::size_t size) {
        v.resize(size);
        for (double& x : v) {
            x = normal(random);
        }
    };
    nep.rc_radial = {4.5, 4.0};
    nep.rc_angular = {3.5, 3.0};
    fill(nep.w0, 2 * 5 * static_cast<std::size_t>(nep.dim));
    fill(nep.b0, 10);
    fill(nep.w1, 10);
    fill(nep.c_radial, 4 * 5 * 7);
    fill(nep.c_angular, 4 * 4 * 6);
    nep.q_scaler.assign(static_cast<std::size_t>(nep.dim), 0.5);
    for (int pair = 0; pair < 4; ++pair) {
        const double row[12] = {1.0, 2.0, 14.4 * 29 * 28, 2.1 * 4.3, 0.18175, 3.1998,
                                0.50986, 0.94229, 0.28022, 0.4029, 0.02817, 0.20162};
        nep.zbl.insert(nep.zbl.end(), row, row + 12);
    }
    nep.prepare();

    const std::size_t n = static_cast<std::size_t>(p.n);
    std::vector<double> first[2][3];
    for (int threads : kThreads) {
        mdapy::set_num_threads(threads);
        std::vector<double> results[2][3];
        for (int c = 0; c < 2; ++c) {
            results[c][0].resize(n);
            results[c][1].resize(3 * n);
            results[c][2].resize(9 * n);
        }
        const mdapy::neighbor::NeighborPairs& q = p.pairs;
        mdapy::calculators::compute_eam(eam, p.n, p.species.data(), p.positions.data(), p.matrix,
                                        q.offsets.data(), q.neighbors.data(), q.shifts.data(),
                                        results[0][0].data(), results[0][1].data(),
                                        results[0][2].data());
        std::vector<double> full(static_cast<std::size_t>(n) * nep.dim);
        std::vector<double> alone(static_cast<std::size_t>(n) * nep.dim);
        mdapy::calculators::compute_nep(nep, p.n, p.species.data(), p.positions.data(), p.matrix,
                                        q.offsets.data(), q.neighbors.data(), q.shifts.data(),
                                        results[1][0].data(), results[1][1].data(),
                                        results[1][2].data(), full.data());
        mdapy::calculators::compute_nep(nep, p.n, p.species.data(), p.positions.data(), p.matrix,
                                        q.offsets.data(), q.neighbors.data(), q.shifts.data(),
                                        nullptr, nullptr, nullptr, alone.data());
        check(alone == full, "NEP descriptors alone are those of the full computation");
        for (int c = 0; c < 2; ++c) {
            if (threads == kThreads[0]) {
                for (int k = 0; k < 3; ++k) {
                    first[c][k] = results[c][k];
                }
            }
            check(results[c][0] == first[c][0] && results[c][1] == first[c][1] &&
                      results[c][2] == first[c][2],
                  c == 0 ? "EAM gives the same results for any thread count"
                         : "NEP gives the same results for any thread count");
        }
    }
}

void structure_identification() {
    // a jittered fcc crystal: some atoms stay fcc, others do not
    const int cells = 6;
    const double a = 3.6;
    const double matrix[9] = {cells * a, 0.0, 0.0, 0.0, cells * a, 0.0, 0.0, 0.0, cells * a};
    const double origin[3] = {0.0, 0.0, 0.0};
    const bool pbc[3] = {true, true, true};
    const double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
    std::mt19937_64 random(9);
    std::normal_distribution<double> jitter(0.0, 0.12);
    std::vector<double> positions;
    for (int x = 0; x < cells; ++x) {
        for (int y = 0; y < cells; ++y) {
            for (int z = 0; z < cells; ++z) {
                for (const auto& b : basis) {
                    positions.push_back((x + b[0]) * a + jitter(random));
                    positions.push_back((y + b[1]) * a + jitter(random));
                    positions.push_back((z + b[2]) * a + jitter(random));
                }
            }
        }
    }
    const auto n = static_cast<std::int64_t>(positions.size() / 3);
    const mdapy::neighbor::NeighborPairs pairs =
        mdapy::neighbor::build(positions.data(), n, matrix, origin, pbc, 3.07);
    // the list read where it is, and the same with a cutoff, as within keeps it
    const mdapy::neighbor::ListView all{positions.data(),        matrix,
                                        pairs.offsets.data(),    pairs.neighbors.data(),
                                        pairs.shifts.data(),     pairs.distances.data()};
    mdapy::neighbor::ListView within = all;
    within.cutoff = 3.07;
    // a wider list with the 14 nearest neighbors of every atom first, nearest
    // first, sorted in place: in the order order_by_distance gives
    mdapy::neighbor::NeighborPairs wide =
        mdapy::neighbor::build(positions.data(), n, matrix, origin, pbc, 4.2);
    std::vector<std::int64_t> nearest_order, nearest_offsets;
    mdapy::neighbor::order_by_distance(positions.data(), n, wide.offsets.data(),
                                       wide.neighbors.data(), wide.shifts.data(), matrix, 14,
                                       nearest_order, nearest_offsets);
    const std::vector<mdapy::index_t> unsorted(wide.neighbors.begin(), wide.neighbors.end());
    const std::vector<double> unsorted_distances(wide.distances.begin(), wide.distances.end());
    mdapy::neighbor::sort_nearest(positions.data(), n, wide.offsets.data(), wide.neighbors.data(),
                                  wide.shifts.data(), wide.distances.data(), matrix, 14);
    bool in_order = true;
    for (std::int64_t i = 0; i < n; ++i) {
        for (std::int64_t q = 0; q < 14; ++q) {
            const std::int64_t p = nearest_order[nearest_offsets[i] + q];
            in_order = in_order && wide.neighbors[wide.offsets[i] + q] == unsorted[p] &&
                       wide.distances[wide.offsets[i] + q] == unsorted_distances[p];
        }
    }
    check(in_order, "a list sorted in place has the nearest neighbors of order_by_distance");
    const auto nearest = [&](std::int64_t count) {
        return mdapy::neighbor::ListView{positions.data(),       matrix,
                                         wide.offsets.data(),    wide.neighbors.data(),
                                         wide.shifts.data(),     wide.distances.data(),
                                         count};
    };
    std::vector<double> masses(static_cast<std::size_t>(n)), velocities(3 * static_cast<std::size_t>(n));
    std::normal_distribution<double> thermal(0.0, 3.0);
    for (std::int64_t i = 0; i < n; ++i) {
        masses[i] = i % 2 == 0 ? 63.5 : 58.7;
        for (int e = 0; e < 3; ++e) {
            velocities[3 * i + e] = thermal(random);
        }
    }
    std::vector<double> first_temperature, first_searched, first_searched_temperature;
    std::vector<std::int8_t> first_types, first_water, first_aja;
    std::vector<double> first_csp, first_q, first_entropy, first_factor;
    std::vector<double> first_cnp;
    std::vector<double> first_volumes, first_areas, first_vertices;
    std::vector<double> first_strain;
    std::vector<std::int64_t> first_angles, first_histogram;
    std::vector<std::int32_t> first_images;
    const std::vector<int> degrees = {4, 6, 12};
    for (int threads : kThreads) {
        mdapy::set_num_threads(threads);
        std::vector<std::int8_t> types(static_cast<std::size_t>(n));
        std::vector<double> csp(static_cast<std::size_t>(n));
        mdapy::identify::common_neighbor_analysis(within, n, types.data());
        std::vector<std::int8_t> searched_types(static_cast<std::size_t>(n));
        mdapy::identify::common_neighbor_analysis_search(positions.data(), n, matrix, origin, pbc,
                                                         3.07, searched_types.data());
        check(searched_types == types, "common neighbor analysis on the grid is that of the list");
        mdapy::neighbor::ListView first_eight = all;
        first_eight.count = 8;
        mdapy::identify::centro_symmetry(first_eight, n, 8, csp.data());
        std::vector<double> cnp(static_cast<std::size_t>(n));
        mdapy::identify::common_neighbor_parameter(all, n, 3.07, cnp.data());
        if (threads == kThreads[0]) {
            first_cnp = cnp;
        }
        check(cnp == first_cnp, "the common neighbor parameter is the same for any thread count");
        std::vector<double> searched_cnp(static_cast<std::size_t>(n));
        mdapy::identify::common_neighbor_parameter_search(positions.data(), n, matrix, origin, pbc,
                                                          3.07, searched_cnp.data());
        bool cnp_near = true;
        for (std::int64_t i = 0; i < n; ++i) {
            cnp_near = cnp_near &&
                       std::abs(searched_cnp[i] - cnp[i]) <= 1e-12 * std::abs(cnp[i]) + 1e-12;
        }
        check(cnp_near, "the common neighbor parameter on the grid is that of the list");
        std::vector<double> temperature(static_cast<std::size_t>(n));
        mdapy::analysis::atomic_temperature(all, n, masses.data(), velocities.data(), 1.0, 1.0,
                                            temperature.data());
        if (threads == kThreads[0]) {
            first_temperature = temperature;
        }
        check(temperature == first_temperature,
              "the atomic temperature is the same for any thread count");
        std::vector<double> searched_temperature(static_cast<std::size_t>(n));
        mdapy::analysis::atomic_temperature_search(positions.data(), n, matrix, origin, pbc, 3.07,
                                                   masses.data(), velocities.data(), 1.0, 1.0,
                                                   searched_temperature.data());
        if (threads == kThreads[0]) {
            first_searched_temperature = searched_temperature;
        }
        bool near = searched_temperature == first_searched_temperature;
        for (std::int64_t i = 0; i < n; ++i) {
            near = near && std::abs(searched_temperature[i] - temperature[i]) <=
                               1e-12 * std::abs(temperature[i]);
        }
        check(near, "the atomic temperature on the grid is the same for any thread count");
        std::vector<double> q(static_cast<std::size_t>(n) * degrees.size());
        mdapy::analysis::steinhardt(all, n, pairs.distances.data(), degrees, true, q.data());
        std::vector<std::int8_t> water(static_cast<std::size_t>(n));
        mdapy::identify::chill_plus(all, n, water.data());
        if (threads == kThreads[0]) {
            first_water = water;
        }
        check(water == first_water, "CHILL+ gives the same types for any thread count");
        std::vector<std::int8_t> aja(static_cast<std::size_t>(n));
        mdapy::identify::ackland_jones(nearest(14), n, aja.data());
        if (threads == kThreads[0]) {
            first_aja = aja;
        }
        check(aja == first_aja && std::count(aja.begin(), aja.end(), mdapy::identify::kFcc) > 0,
              "Ackland-Jones gives the same types for any thread count, some of them fcc");
        // the nearest neighbors found as the analyses go give what the list gives
        std::vector<std::int8_t> found_types(static_cast<std::size_t>(n)),
            listed_types(static_cast<std::size_t>(n));
        mdapy::identify::ackland_jones_nearest(positions.data(), n, matrix, origin, pbc, 14,
                                               found_types.data());
        check(found_types == aja, "Ackland-Jones finds the nearest neighbors as the list has them");
        mdapy::identify::common_neighbor_analysis(nearest(14), n, listed_types.data());
        mdapy::identify::common_neighbor_analysis_nearest(positions.data(), n, matrix, origin,
                                                          pbc, 14, found_types.data());
        check(found_types == listed_types,
              "adaptive common neighbor analysis finds the nearest neighbors as the list has them");
        std::vector<double> listed_csp(static_cast<std::size_t>(n)),
            found_csp(static_cast<std::size_t>(n));
        mdapy::identify::centro_symmetry(nearest(8), n, 8, listed_csp.data());
        mdapy::identify::centro_symmetry_nearest(positions.data(), n, matrix, origin, pbc, 8,
                                                 found_csp.data());
        check(found_csp == listed_csp,
              "the centro-symmetry parameter finds the nearest neighbors as the list has them");
        std::vector<double> listed_q(static_cast<std::size_t>(n) * degrees.size()),
            found_q(listed_q.size());
        mdapy::analysis::steinhardt(nearest(12), n, nullptr, degrees, false, listed_q.data());
        mdapy::analysis::steinhardt_nearest(positions.data(), n, matrix, origin, pbc, 12, degrees,
                                            found_q.data());
        check(found_q == listed_q,
              "Steinhardt parameters find the nearest neighbors as the list has them");
        std::vector<std::int8_t> diamond(static_cast<std::size_t>(n));
        mdapy::identify::identify_diamond(nearest(4), n, diamond.data());
        check(diamond == std::vector<std::int8_t>(static_cast<std::size_t>(n), 0),
              "no atom of fcc is diamond");
        std::vector<double> entropy(static_cast<std::size_t>(n));
        mdapy::analysis::pair_entropy(all, n, 3.07, 0.2, 0.08, entropy.data());
        if (threads == kThreads[0]) {
            first_entropy = entropy;
        }
        check(entropy == first_entropy, "the pair entropy is the same for any thread count");
        std::vector<double> searched(static_cast<std::size_t>(n));
        mdapy::analysis::pair_entropy_search(positions.data(), n, matrix, origin, pbc, 3.07, 0.2,
                                             0.08, searched.data());
        if (threads == kThreads[0]) {
            first_searched = searched;
        }
        bool close = searched == first_searched;
        for (std::int64_t i = 0; i < n; ++i) {
            close = close && std::abs(searched[i] - entropy[i]) <= 1e-12 * std::abs(entropy[i]);
        }
        check(close, "the pair entropy on the grid is the same for any thread count and as listed");
        const std::vector<std::int32_t> species(static_cast<std::size_t>(n), 0);
        const std::vector<std::int64_t> histogram = mdapy::analysis::pair_histogram(
            positions.data(), n, matrix, origin, pbc, 6.0, 60, species.data(), 1);
        // the neighbors within 3.07 counted on the grid are those of the list
        std::vector<std::int64_t> counted(static_cast<std::size_t>(n));
        const double limit = 3.07;
        mdapy::analysis::count_neighbors(positions.data(), n, matrix, origin, pbc,
                                         species.data(), 1, &limit, false, counted.data());
        bool counts_match = true;
        for (std::int64_t i = 0; i < n; ++i) {
            counts_match = counts_match && counted[i] == pairs.offsets[i + 1] - pairs.offsets[i];
        }
        check(counts_match, "neighbors counted on the grid are those of the list");
        std::vector<std::int64_t> counted_listed(static_cast<std::size_t>(n));
        mdapy::analysis::count_neighbors_list(n, pairs.offsets.data(), pairs.neighbors.data(),
                                              pairs.distances.data(), species.data(), 1, &limit,
                                              counted_listed.data());
        check(counted_listed == counted, "neighbors counted from a list are those of the grid");
        // a wider list: its histogram is that of the grid, and the pairs of it
        // within 3.07 are those of a search within 3.07
        const mdapy::neighbor::NeighborPairs longer =
            mdapy::neighbor::build(positions.data(), n, matrix, origin, pbc, 6.5);
        check(mdapy::analysis::pair_histogram_list(n, longer.offsets.data(), longer.neighbors.data(),
                                                   longer.distances.data(), 6.0, 60,
                                                   species.data(), 1) == histogram,
              "the pair histogram of a list is that of the grid");
        const mdapy::neighbor::NeighborPairs cut = mdapy::neighbor::within(
            n, longer.offsets.data(), longer.neighbors.data(), longer.shifts.data(),
            longer.distances.data(), nullptr, 1, &limit);
        check(cut.offsets == pairs.offsets, "the pairs of a list within a cutoff are found");
        // the wider list read with the cutoff gives what the list of the cutoff gives
        mdapy::neighbor::ListView wider{positions.data(),       matrix,
                                        longer.offsets.data(),  longer.neighbors.data(),
                                        longer.shifts.data(),   longer.distances.data()};
        wider.cutoff = 3.07;
        std::vector<std::int8_t> wider_types(static_cast<std::size_t>(n));
        mdapy::identify::common_neighbor_analysis(wider, n, wider_types.data());
        std::vector<double> wider_entropy(static_cast<std::size_t>(n)),
            wider_temperature(static_cast<std::size_t>(n));
        mdapy::analysis::pair_entropy(wider, n, 3.07, 0.2, 0.08, wider_entropy.data());
        mdapy::analysis::atomic_temperature(wider, n, masses.data(), velocities.data(), 1.0, 1.0,
                                            wider_temperature.data());
        bool wider_close = wider_types == types;
        for (std::int64_t i = 0; i < n; ++i) {
            wider_close = wider_close &&
                          std::abs(wider_entropy[i] - entropy[i]) <= 1e-12 * std::abs(entropy[i]) &&
                          std::abs(wider_temperature[i] - temperature[i]) <=
                              1e-12 * std::abs(temperature[i]);
        }
        check(wider_close, "a wider list read with a cutoff gives what the list of it gives");
        // the sorted wider list stops at the first sorted pair beyond the cutoff
        mdapy::neighbor::ListView ordered = nearest(0);
        ordered.sorted = 14;
        ordered.cutoff = 3.07;
        std::vector<std::int8_t> ordered_types(static_cast<std::size_t>(n));
        mdapy::identify::common_neighbor_analysis(ordered, n, ordered_types.data());
        std::vector<double> ordered_temperature(static_cast<std::size_t>(n));
        mdapy::analysis::atomic_temperature(ordered, n, masses.data(), velocities.data(), 1.0,
                                            1.0, ordered_temperature.data());
        bool same_set = ordered_types == types;
        for (std::int64_t i = 0; i < n; ++i) {
            same_set = same_set && std::abs(ordered_temperature[i] - temperature[i]) <=
                                       1e-12 * std::abs(temperature[i]);
        }
        check(same_set, "a sorted list read with a cutoff stops where the cutoff ends its pairs");
        // clusters of bonds shorter than 2.6, on the grid and from the list
        const double bond = 2.6;
        const mdapy::neighbor::NeighborPairs bonded =
            mdapy::neighbor::build(positions.data(), n, matrix, origin, pbc, bond);
        std::vector<std::int64_t> listed(static_cast<std::size_t>(n)),
            searched_clusters(static_cast<std::size_t>(n)), listed_sizes(static_cast<std::size_t>(n)),
            searched_sizes(static_cast<std::size_t>(n));
        std::vector<std::int32_t> listed_images(3 * static_cast<std::size_t>(n)),
            searched_images(3 * static_cast<std::size_t>(n));
        std::vector<std::uint8_t> listed_spanning(static_cast<std::size_t>(n)),
            searched_spanning(static_cast<std::size_t>(n));
        const mdapy::neighbor::ListView bonds{positions.data(),         matrix,
                                              bonded.offsets.data(),    bonded.neighbors.data(),
                                              bonded.shifts.data(),     bonded.distances.data()};
        const std::int64_t listed_count =
            mdapy::analysis::clusters(bonds, n, listed.data(), listed_sizes.data(),
                                      listed_images.data(), listed_spanning.data());
        const std::int64_t searched_count = mdapy::analysis::clusters_search(
            positions.data(), n, matrix, origin, pbc, species.data(), 1, &bond, false,
            searched_clusters.data(), searched_sizes.data(), searched_images.data(),
            searched_spanning.data());
        check(listed_count == searched_count && listed == searched_clusters &&
                  listed_sizes == searched_sizes && listed_spanning == searched_spanning,
              "clusters on the grid are those of the list for any thread count");
        if (threads == kThreads[0]) {
            first_images = searched_images;
        }
        check(searched_images == first_images,
              "the images of the clusters on the grid are the same for any thread count");
        // bond angles on the grid and from the list
        std::vector<std::int64_t> listed_angles(90), searched_angles(90);
        mdapy::analysis::bond_angles(bonds, n, species.data(), 1, 90, listed_angles.data());
        mdapy::analysis::bond_angles_search(positions.data(), n, matrix, origin, pbc,
                                            species.data(), 1, &bond, false, 90,
                                            searched_angles.data());
        check(listed_angles == searched_angles, "bond angles on the grid are those of the list");
        if (threads == kThreads[0]) {
            first_histogram = histogram;
        }
        check(histogram == first_histogram, "the pair histogram is the same for any thread count");
        std::vector<double> factor;
        std::vector<std::int64_t> shells;
        mdapy::analysis::structure_factor(positions.data(), n, matrix, species.data(), 1, 3.0, 30,
                                          factor, shells, 100);
        if (threads == kThreads[0]) {
            first_factor = factor;
        }
        check(factor == first_factor, "the structure factor is the same for any thread count");
        // the strain of the crystal against itself moved by its jitter again
        std::vector<double> current(positions.size()), inverse(9, 0.0);
        for (std::size_t k = 0; k < positions.size(); ++k) {
            current[k] = positions[k] + 0.01 * (cells * a);
        }
        inverse[0] = inverse[4] = inverse[8] = 1.0 / (cells * a);
        const mdapy::analysis::StrainInput strain_input{
            n, positions.data(), inverse.data(), origin, current.data(), inverse.data(),
            origin, pbc, matrix, matrix};
        const auto n_size = static_cast<std::size_t>(n);
        std::vector<double> strain(9 * n_size), shear(n_size), volumetric(n_size), d2(n_size);
        std::vector<double> searched_strain(9 * n_size), searched_shear(n_size),
            searched_volumetric(n_size), searched_d2(n_size), bare_shear(n_size),
            bare_volumetric(n_size);
        mdapy::analysis::atomic_strain_list(
            pairs.offsets.data(), pairs.neighbors.data(), pairs.shifts.data(),
            pairs.distances.data(), 3.07, strain_input,
            {strain.data(), shear.data(), volumetric.data(), d2.data()});
        mdapy::analysis::atomic_strain_search(
            3.07, strain_input,
            {searched_strain.data(), searched_shear.data(), searched_volumetric.data(),
             searched_d2.data()});
        mdapy::analysis::atomic_strain_search(
            3.07, strain_input, {nullptr, bare_shear.data(), bare_volumetric.data(), nullptr});
        if (threads == kThreads[0]) {
            first_strain = strain;
        }
        check(strain == first_strain, "the atomic strain is the same for any thread count");
        check(searched_strain == strain && searched_d2 == d2,
              "the atomic strain on the grid is that of the list");
        check(bare_shear == shear && bare_volumetric == volumetric,
              "the shear and volumetric strain do not depend on the other outputs");
        std::vector<std::int32_t> two_types(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i) {
            two_types[i] = static_cast<std::int32_t>(i % 2);
        }
        std::vector<std::int64_t> angles(2 * 3 * 90);
        mdapy::analysis::bond_angles(all, n, two_types.data(), 2, 90, angles.data());
        const mdapy::analysis::VoronoiCells voronoi =
            mdapy::analysis::voronoi(positions.data(), n, matrix, origin, pbc, {0.0, 0.01, 0.1},
                                     true, true);
        if (threads == kThreads[0]) {
            first_volumes = voronoi.volumes;
            first_areas = voronoi.areas;
            first_vertices = voronoi.vertices;
        }
        const mdapy::analysis::VoronoiCells bare = mdapy::analysis::voronoi(
            positions.data(), n, matrix, origin, pbc, {0.0, 0.01, 0.1}, false, false);
        check(bare.volumes == voronoi.volumes && bare.coordination == voronoi.coordination &&
                  bare.neighbors.empty() && bare.vertices.empty(),
              "Voronoi cells without faces have the same volumes and coordination");
        check(voronoi.volumes == first_volumes && voronoi.areas == first_areas &&
                  voronoi.vertices == first_vertices,
              "the Voronoi cells are the same for any thread count");
        if (threads == kThreads[0]) {
            first_types = types;
            first_csp = csp;
            first_q = q;
            first_angles = angles;
        }
        check(types == first_types && csp == first_csp && q == first_q && angles == first_angles,
              "structure identification gives the same results for any thread count");
        check(std::count(types.begin(), types.end(), mdapy::identify::kFcc) > 0,
              "common neighbor analysis finds fcc atoms");
    }
}

// Polyhedral template matching: the library keeps data of its own, made
// once, and a handle per chunk.
void polyhedral_template_matching() {
    // fcc of 5 x 5 x 5 cubic cells, displaced at random
    const double a = 3.6;
    std::vector<double> positions;
    std::mt19937 rng(11);
    std::normal_distribution<double> noise(0.0, 0.08);
    const double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
    for (int x = 0; x < 5; ++x) {
        for (int y = 0; y < 5; ++y) {
            for (int z = 0; z < 5; ++z) {
                for (const auto& b : basis) {
                    positions.push_back(a * (x + b[0]) + noise(rng));
                    positions.push_back(a * (y + b[1]) + noise(rng));
                    positions.push_back(a * (z + b[2]) + noise(rng));
                }
            }
        }
    }
    const auto n = static_cast<std::int64_t>(positions.size() / 3);
    const double matrix[9] = {5 * a, 0, 0, 0, 5 * a, 0, 0, 0, 5 * a};
    const double origin[3] = {0, 0, 0};
    const bool pbc[3] = {true, true, true};
    const mdapy::neighbor::NeighborPairs nearest =
        mdapy::neighbor::nearest(positions.data(), n, matrix, origin, pbc, 18);
    mdapy::identify::PtmInput input;
    input.n = n;
    input.positions = positions.data();
    input.matrix = matrix;
    input.offsets = nearest.offsets.data();
    input.j = nearest.neighbors.data();
    input.shifts = nearest.shifts.data();
    input.structures = 0xff;
    input.deformation = true;
    std::vector<std::int8_t> first_types;
    std::vector<double> first_rmsd, first_deformation;
    for (int threads : kThreads) {
        mdapy::set_num_threads(threads);
        std::vector<std::int8_t> types(static_cast<std::size_t>(n));
        std::vector<double> rmsd(static_cast<std::size_t>(n)),
            deformation(9 * static_cast<std::size_t>(n));
        mdapy::identify::PtmOutput output;
        output.structure = types.data();
        output.rmsd = rmsd.data();
        output.deformation = deformation.data();
        mdapy::identify::polyhedral_template_matching(input, output);
        if (threads == kThreads[0]) {
            first_types = types;
            first_rmsd = rmsd;
            first_deformation = deformation;
        }
        check(types == first_types && rmsd == first_rmsd && deformation == first_deformation,
              "polyhedral template matching is the same for any thread count");
        check(std::count(types.begin(), types.end(), mdapy::identify::kPtmFcc) > n / 2,
              "polyhedral template matching finds fcc");
    }
    // a slip of a/6 [11-2] above a (111) plane: stacking faults of hcp atoms
    for (std::int64_t i = 0; i < n; ++i) {
        double* x = positions.data() + 3 * i;
        if (x[0] + x[1] + x[2] > 7.5 * a) {
            x[0] += a / 6;
            x[1] += a / 6;
            x[2] -= a / 3;
        }
    }
    const mdapy::neighbor::NeighborPairs slipped =
        mdapy::neighbor::nearest(positions.data(), n, matrix, origin, pbc, 18);
    input.offsets = slipped.offsets.data();
    input.j = slipped.neighbors.data();
    input.shifts = slipped.shifts.data();
    input.deformation = false;
    std::vector<std::int8_t> types(static_cast<std::size_t>(n));
    std::vector<double> rmsd(static_cast<std::size_t>(n));
    std::vector<mdapy::index_t> neighbors(16 * static_cast<std::size_t>(n));
    mdapy::identify::PtmOutput output;
    output.structure = types.data();
    output.rmsd = rmsd.data();
    output.neighbors = neighbors.data();
    mdapy::identify::polyhedral_template_matching(input, output);
    check(std::count(types.begin(), types.end(), mdapy::identify::kPtmHcp) > 0,
          "polyhedral template matching finds hcp atoms of a slip");
    std::vector<std::int8_t> first_faults;
    for (int threads : kThreads) {
        mdapy::set_num_threads(threads);
        std::vector<std::int8_t> faults(static_cast<std::size_t>(n));
        mdapy::identify::planar_faults(types.data(), neighbors.data(), n, true, faults.data());
        if (threads == kThreads[0]) {
            first_faults = faults;
        }
        check(faults == first_faults, "planar faults are the same for any thread count");
    }
}

void trajectories() {
    // atoms vibrating about random sites, the trajectory of each in a row
    const std::int64_t n = 60, frames = 12;
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> site(0.0, 10.0);
    std::normal_distribution<double> noise(0.0, 0.1);
    std::vector<double> positions(static_cast<std::size_t>(n * frames * 3));
    for (std::int64_t i = 0; i < n; ++i) {
        const double x[3] = {site(rng), site(rng), site(rng)};
        for (std::int64_t t = 0; t < frames; ++t) {
            for (int e = 0; e < 3; ++e) {
                positions[static_cast<std::size_t>((i * frames + t) * 3 + e)] = x[e] + noise(rng);
            }
        }
    }
    std::vector<double> first_atomic, first_running;
    for (int threads : kThreads) {
        mdapy::set_num_threads(threads);
        std::vector<double> atomic(static_cast<std::size_t>(n));
        std::vector<double> running(static_cast<std::size_t>(frames));
        mdapy::analysis::lindemann(n, frames, positions.data(), atomic.data(), running.data());
        if (threads == kThreads[0]) {
            first_atomic = atomic;
            first_running = running;
        }
        check(atomic == first_atomic && running == first_running,
              "the Lindemann index is the same for any thread count");
    }
    check(first_running[0] == 0.0 && first_running[frames - 1] > 0.0,
          "the Lindemann index starts at zero and grows");
}

void pictures() {
    // random spheres, cylinders and see-through triangles, with every
    // effect of the renderer on
    std::mt19937_64 rng(12);
    std::uniform_real_distribution<float> place(0.0f, 10.0f), unit(0.0f, 1.0f);
    mdapy::visualization::Scene scene;
    for (int i = 0; i < 300; ++i) {
        for (int e = 0; e < 3; ++e) {
            scene.sphere_center.push_back(place(rng));
            scene.sphere_color.push_back(unit(rng));
        }
        scene.sphere_radius.push_back(0.3f + unit(rng));
        scene.sphere_id.push_back(i);
    }
    for (int i = 0; i < 40; ++i) {
        for (int e = 0; e < 6; ++e) {
            scene.cylinder_end.push_back(place(rng));
        }
        for (int e = 0; e < 3; ++e) {
            scene.cylinder_color.push_back(unit(rng));
        }
        scene.cylinder_radius.push_back(0.2f);
        scene.cylinder_id.push_back(1000);
    }
    for (int i = 0; i < 20; ++i) {
        for (int e = 0; e < 9; ++e) {
            scene.triangle_vertex.push_back(place(rng));
        }
        for (int e = 0; e < 3; ++e) {
            scene.triangle_color.push_back(unit(rng));
        }
        scene.triangle_alpha.push_back(0.5f);
        scene.triangle_id.push_back(2000 + i);
    }
    mdapy::visualization::Camera camera;
    camera.center = {5, 5, 5};
    camera.forward = {-1, -1, -1};
    camera.up = {0, 0, 1};
    camera.half_width = camera.half_height = 9;
    mdapy::visualization::Style style;
    style.outline_width = 1.5f;
    style.fog = 0.3f;
    std::vector<std::uint8_t> first;
    for (int threads : kThreads) {
        mdapy::set_num_threads(threads);
        const std::vector<std::uint8_t> image =
            mdapy::visualization::render(scene, camera, style, 96, 64);
        if (threads == kThreads[0]) {
            first = image;
        }
        check(image == first, "a picture is the same for any thread count");
    }
}

}  // namespace

int main() {
    loops_and_reductions();
    idle_and_partial_pools();
    progress_cancellation_and_errors();
    nested_and_concurrent_regions();
    text_tables();
    neighbor_search();
    structure_identification();
    polyhedral_template_matching();
    calculators();
    trajectories();
    pictures();
    if (failures == 0) {
        std::printf("all checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
