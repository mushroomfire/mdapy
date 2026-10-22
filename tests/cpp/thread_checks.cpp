// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Checks of the thread pool and the parallel text kernels, without Python.
// CI builds this program with ThreadSanitizer, which cannot run the Python
// test suite on Linux (polars crashes on import under it):
//
//   g++ -std=c++17 -O1 -g -fsanitize=thread -Isrc/cpp -pthread -o thread_checks
//       src/cpp/core/parallel.cpp src/cpp/io/text.cpp tests/cpp/thread_checks.cpp
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
#include <vector>

#include "analysis/distribution.h"
#include "analysis/geometry.h"
#include "analysis/steinhardt.h"
#include "analysis/strain.h"
#include "core/parallel.h"
#include "identify/structure.h"
#include "io/text.h"
#include "neighbor/cell_list.h"

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
    mdapy::neighbor::NeighborPairs first;
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
        mdapy::neighbor::order_by_distance(n, pairs.offsets.data(), pairs.distances.data(), 12,
                                           order, kept);
        bool sorted = kept.back() == 12 * n;
        for (std::int64_t i = 0; i < n && sorted; ++i) {
            for (std::int64_t k = kept[i] + 1; k < kept[i + 1]; ++k) {
                sorted = sorted && pairs.distances[order[k - 1]] <= pairs.distances[order[k]];
            }
        }
        check(sorted, "the nearest neighbors come in order of distance");
        const mdapy::neighbor::NeighborPairs near =
            mdapy::neighbor::nearest(positions.data(), n, matrix, origin, pbc, 12);
        bool same = true;
        for (std::int64_t i = 0; i < n && same; ++i) {
            for (int k = 0; k < 12; ++k) {
                same = same && near.distances[12 * i + k] == pairs.distances[order[kept[i] + k]];
            }
        }
        check(same, "the nearest search finds the first pairs of the ordered list");
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
        // every atom is its own nearest site
        std::vector<std::int64_t> site(static_cast<std::size_t>(n));
        std::vector<double> distance2(static_cast<std::size_t>(n));
        mdapy::neighbor::nearest_sites(positions.data(), n, matrix, origin, pbc, positions.data(),
                                       n, site.data(), distance2.data());
        bool own = true;
        for (std::int64_t i = 0; i < n; ++i) {
            own = own && site[i] == i && distance2[i] == 0.0;
        }
        check(own, "every atom is its own nearest site");
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
    std::vector<double> vectors;
    for (std::int64_t i = 0; i < n; ++i) {
        for (std::int64_t k = pairs.offsets[i]; k < pairs.offsets[i + 1]; ++k) {
            const std::int32_t j = pairs.neighbors[k];
            for (int e = 0; e < 3; ++e) {
                double v = positions[3 * j + e] - positions[3 * i + e];
                for (int s = 0; s < 3; ++s) {
                    v += pairs.shifts[3 * k + s] * matrix[3 * s + e];
                }
                vectors.push_back(v);
            }
        }
    }
    // the first 8 neighbor vectors of every atom for the centro-symmetry
    std::vector<double> eight;
    for (std::int64_t i = 0; i < n; ++i) {
        for (int k = 0; k < 8; ++k) {
            for (int e = 0; e < 3; ++e) {
                eight.push_back(vectors[3 * (pairs.offsets[i] + k) + e]);
            }
        }
    }
    // the first 4 neighbors of every atom, for the diamond identification
    std::vector<std::int32_t> four;
    std::vector<double> four_vectors;
    for (std::int64_t i = 0; i < n; ++i) {
        for (int k = 0; k < 4; ++k) {
            four.push_back(pairs.neighbors[pairs.offsets[i] + k]);
            for (int e = 0; e < 3; ++e) {
                four_vectors.push_back(vectors[3 * (pairs.offsets[i] + k) + e]);
            }
        }
    }
    std::vector<double> masses(static_cast<std::size_t>(n)), velocities(3 * static_cast<std::size_t>(n));
    std::normal_distribution<double> thermal(0.0, 3.0);
    for (std::int64_t i = 0; i < n; ++i) {
        masses[i] = i % 2 == 0 ? 63.5 : 58.7;
        for (int e = 0; e < 3; ++e) {
            velocities[3 * i + e] = thermal(random);
        }
    }
    std::vector<double> first_temperature;
    std::vector<std::int8_t> first_types, first_water;
    std::vector<double> first_csp, first_q, first_entropy, first_factor;
    std::vector<double> first_strain;
    std::vector<std::int64_t> first_angles, first_histogram;
    const std::vector<int> degrees = {4, 6, 12};
    for (int threads : kThreads) {
        mdapy::set_num_threads(threads);
        std::vector<std::int8_t> types(static_cast<std::size_t>(n));
        std::vector<double> csp(static_cast<std::size_t>(n));
        mdapy::identify::common_neighbor_analysis(n, pairs.offsets.data(), vectors.data(), 3.07,
                                                  types.data());
        mdapy::identify::centro_symmetry(n, 8, eight.data(), csp.data());
        std::vector<double> temperature(static_cast<std::size_t>(n));
        mdapy::analysis::atomic_temperature(n, pairs.offsets.data(), pairs.neighbors.data(),
                                            masses.data(), velocities.data(), 1.0, 1.0,
                                            temperature.data());
        if (threads == kThreads[0]) {
            first_temperature = temperature;
        }
        check(temperature == first_temperature,
              "the atomic temperature is the same for any thread count");
        std::vector<double> q(static_cast<std::size_t>(n) * degrees.size());
        mdapy::analysis::steinhardt(n, pairs.offsets.data(), pairs.neighbors.data(),
                                    vectors.data(), degrees, true, q.data());
        std::vector<std::int8_t> water(static_cast<std::size_t>(n));
        mdapy::identify::chill_plus(n, pairs.offsets.data(), pairs.neighbors.data(),
                                    vectors.data(), water.data());
        if (threads == kThreads[0]) {
            first_water = water;
        }
        check(water == first_water, "CHILL+ gives the same types for any thread count");
        std::vector<std::int8_t> diamond(static_cast<std::size_t>(n));
        mdapy::identify::identify_diamond(n, four.data(), four_vectors.data(), diamond.data());
        check(diamond == std::vector<std::int8_t>(static_cast<std::size_t>(n), 0),
              "no atom of fcc is diamond");
        std::vector<double> entropy(static_cast<std::size_t>(n)),
            density(static_cast<std::size_t>(n), 0.08);
        mdapy::analysis::pair_entropy(n, pairs.offsets.data(), pairs.distances.data(), 3.07,
                                      0.2, density.data(), entropy.data());
        if (threads == kThreads[0]) {
            first_entropy = entropy;
        }
        check(entropy == first_entropy, "the pair entropy is the same for any thread count");
        const std::vector<std::int32_t> species(static_cast<std::size_t>(n), 0);
        const std::vector<std::int64_t> histogram = mdapy::analysis::pair_histogram(
            positions.data(), n, matrix, origin, pbc, 6.0, 60, species.data(), 1);
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
        std::vector<double> fractional(positions.size()), moved(positions.size(), 0.01);
        for (std::size_t k = 0; k < positions.size(); ++k) {
            fractional[k] = positions[k] / (cells * a);
        }
        std::vector<double> strain(9 * static_cast<std::size_t>(n)), d2(static_cast<std::size_t>(n));
        mdapy::analysis::atomic_strain(n, pairs.offsets.data(), pairs.neighbors.data(),
                                       pairs.shifts.data(), fractional.data(), moved.data(), matrix,
                                       matrix, strain.data(), d2.data());
        if (threads == kThreads[0]) {
            first_strain = strain;
        }
        check(strain == first_strain, "the atomic strain is the same for any thread count");
        std::vector<std::int32_t> two_types(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i) {
            two_types[i] = static_cast<std::int32_t>(i % 2);
        }
        std::vector<std::int64_t> angles(2 * 3 * 90);
        mdapy::analysis::bond_angles(n, pairs.offsets.data(), pairs.neighbors.data(), vectors.data(),
                                     two_types.data(), 2, 90, angles.data());
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

}  // namespace

int main() {
    loops_and_reductions();
    progress_cancellation_and_errors();
    nested_and_concurrent_regions();
    text_tables();
    neighbor_search();
    structure_identification();
    trajectories();
    if (failures == 0) {
        std::printf("all checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
