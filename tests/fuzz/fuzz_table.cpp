// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// A libFuzzer target of the text tables, built as fuzz_numbers.cpp is:
//
//   clang++ -std=c++17 -O1 -g -fsanitize=fuzzer,address,undefined -ffp-contract=off
//       -Isrc/cpp -o fuzz_table tests/fuzz/fuzz_table.cpp
//       src/cpp/core/parallel.cpp src/cpp/io/text.cpp
//
// The first bytes choose the columns (kind and width), the rows and the comment
// character; the rest is the text. Any text must be read or refused with a
// ParseError, without a fault. A table that was read must be written and read
// again to the same values.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/parallel.h"
#include "io/text.h"

namespace {

using mdapy::io::Column;
using mdapy::io::Kind;

constexpr Kind kKinds[] = {Kind::Float, Kind::Int, Kind::Bool, Kind::String, Kind::Skip};

struct Table {
    std::vector<Column> columns;
    std::vector<std::vector<double>> floats;
    std::vector<std::vector<std::int64_t>> ints;
    std::vector<std::vector<std::uint8_t>> bools;
    std::vector<std::vector<std::uint32_t>> codes;
};

Table make(const std::vector<std::pair<Kind, int>>& layout, std::int64_t rows) {
    Table table;
    table.floats.reserve(layout.size());
    table.ints.reserve(layout.size());
    table.bools.reserve(layout.size());
    table.codes.reserve(layout.size());
    for (const auto& [kind, width] : layout) {
        Column column;
        column.kind = kind;
        column.width = width;
        const std::size_t n = static_cast<std::size_t>(rows * width);
        switch (kind) {
            case Kind::Float:
                column.data = table.floats.emplace_back(n).data();
                break;
            case Kind::Int:
                column.data = table.ints.emplace_back(n).data();
                break;
            case Kind::Bool:
                column.data = table.bools.emplace_back(n).data();
                break;
            case Kind::String:
                column.data = table.codes.emplace_back(n).data();
                break;
            case Kind::Skip:
                break;
        }
        table.columns.push_back(column);
    }
    return table;
}

bool same(double a, double b) {
    return (std::isnan(a) && std::isnan(b)) || std::memcmp(&a, &b, sizeof a) == 0;
}

}  // namespace

extern "C" int LLVMFuzzerInitialize(int*, char***) {
    mdapy::set_num_threads(2);
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size < 3) return 0;
    const int count = 1 + data[0] % 4;
    if (size < static_cast<std::size_t>(count) + 2) return 0;
    std::vector<std::pair<Kind, int>> layout;
    for (int c = 0; c < count; ++c) {
        layout.emplace_back(kKinds[data[1 + c] % 5], 1 + (data[1 + c] / 5) % 3);
    }
    const std::int64_t rows = data[1 + count] % 16;
    const char comment = (data[0] & 0x80) != 0 ? '#' : '\0';
    const std::size_t head = static_cast<std::size_t>(count) + 2;
    const char* text = reinterpret_cast<const char*>(data) + head;
    const auto length = static_cast<std::int64_t>(size - head);

    try {
        mdapy::io::skip_lines(text, length, 0, rows);
    } catch (const mdapy::io::ParseError&) {
    }
    Table table = make(layout, rows);
    try {
        mdapy::io::parse_table(text, length, 0, rows, table.columns, comment);
    } catch (const mdapy::io::ParseError&) {
        return 0;
    }

    // Written and read again: the same values and the same categories.
    std::vector<std::pair<Kind, int>> kept;
    for (const auto& item : layout) {
        if (item.first != Kind::Skip) kept.push_back(item);
    }
    if (kept.empty()) return 0;
    Table first = make(kept, rows);
    std::size_t f = 0, i = 0, b = 0, s = 0;
    for (std::size_t c = 0, k = 0; c < layout.size(); ++c) {
        const Column& column = table.columns[c];
        if (column.kind == Kind::Skip) continue;
        Column& copy = first.columns[k++];
        switch (column.kind) {
            case Kind::Float:
                first.floats[f] = table.floats[f];
                copy.data = first.floats[f++].data();
                break;
            case Kind::Int:
                first.ints[i] = table.ints[i];
                copy.data = first.ints[i++].data();
                break;
            case Kind::Bool:
                first.bools[b] = table.bools[b];
                copy.data = first.bools[b++].data();
                break;
            case Kind::String:
                first.codes[s] = table.codes[s];
                copy.data = first.codes[s++].data();
                copy.categories = column.categories;
                break;
            case Kind::Skip:
                break;
        }
    }
    const std::string written = mdapy::io::format_table(rows, first.columns);
    Table again = make(kept, rows);
    try {
        mdapy::io::parse_table(written.data(), static_cast<std::int64_t>(written.size()), 0, rows,
                               again.columns, '\0');
    } catch (const mdapy::io::ParseError& error) {
        std::fprintf(stderr, "written table not read: %s\n[%s]\n", error.what(), written.c_str());
        std::abort();
    }
    for (std::size_t k = 0; k < first.floats.size(); ++k) {
        for (std::size_t n = 0; n < first.floats[k].size(); ++n) {
            if (!same(first.floats[k][n], again.floats[k][n])) std::abort();
        }
    }
    if (first.ints != again.ints || first.bools != again.bools) std::abort();
    for (std::size_t c = 0, k = 0; c < kept.size(); ++c) {
        if (kept[c].first != Kind::String) continue;
        const Column& a = first.columns[c];
        const Column& z = again.columns[c];
        for (std::size_t n = 0; n < first.codes[k].size(); ++n) {
            if (a.categories[first.codes[k][n]] != z.categories[again.codes[k][n]]) std::abort();
        }
        ++k;
    }
    return 0;
}
