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

#include "calculators/eam.h"
#include "calculators/nep.h"
#include "core/parallel.h"
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
    }
}

// A small binary alloy and its full neighbor list as pair vectors.
struct Pairs {
    std::int64_t n = 0;
    std::vector<std::int32_t> species;
    std::vector<std::int64_t> offsets;
    std::vector<std::int32_t> neighbors;
    std::vector<std::int32_t> shifts;
    std::vector<double> vectors;
};

Pairs alloy_pairs(double cutoff) {
    Pairs p;
    const int cells = 6;
    const double a = 3.0;
    const double matrix[9] = {cells * a, 0.0, 0.0, 0.0, cells * a, 0.0, 0.0, 0.0, cells * a};
    const double origin[3] = {0.0, 0.0, 0.0};
    const bool pbc[3] = {true, true, true};
    const double basis[4][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}, {0, 0.5, 0.5}};
    std::mt19937_64 random(5);
    std::normal_distribution<double> jitter(0.0, 0.2);
    std::vector<double> positions;
    for (int x = 0; x < cells; ++x) {
        for (int y = 0; y < cells; ++y) {
            for (int z = 0; z < cells; ++z) {
                for (const auto& b : basis) {
                    positions.push_back((x + b[0]) * a + jitter(random));
                    positions.push_back((y + b[1]) * a + jitter(random));
                    positions.push_back((z + b[2]) * a + jitter(random));
                    p.species.push_back(static_cast<std::int32_t>(random() % 2));
                }
            }
        }
    }
    p.n = static_cast<std::int64_t>(p.species.size());
    const mdapy::neighbor::NeighborPairs pairs =
        mdapy::neighbor::build(positions.data(), p.n, matrix, origin, pbc, cutoff);
    p.offsets = pairs.offsets;
    p.neighbors = pairs.neighbors;
    p.shifts = pairs.shifts;
    for (std::int64_t i = 0; i < p.n; ++i) {
        for (std::int64_t k = p.offsets[i]; k < p.offsets[i + 1]; ++k) {
            const std::int32_t j = p.neighbors[k];
            for (int e = 0; e < 3; ++e) {
                double v = positions[3 * j + e] - positions[3 * i + e];
                for (int s = 0; s < 3; ++s) {
                    v += pairs.shifts[3 * k + s] * matrix[3 * s + e];
                }
                p.vectors.push_back(v);
            }
        }
    }
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
        mdapy::calculators::compute_eam(eam, p.n, p.species.data(), p.offsets.data(),
                                        p.neighbors.data(), p.vectors.data(),
                                        results[0][0].data(), results[0][1].data(),
                                        results[0][2].data());
        mdapy::calculators::compute_nep(nep, p.n, p.species.data(), p.offsets.data(),
                                        p.neighbors.data(), p.shifts.data(), p.vectors.data(),
                                        results[1][0].data(), results[1][1].data(),
                                        results[1][2].data(), nullptr);
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

}  // namespace

int main() {
    loops_and_reductions();
    progress_cancellation_and_errors();
    nested_and_concurrent_regions();
    text_tables();
    neighbor_search();
    calculators();
    if (failures == 0) {
        std::printf("all checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
