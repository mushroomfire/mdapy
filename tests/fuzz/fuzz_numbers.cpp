// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// A libFuzzer target of the number kernels of text files, with clang:
//
//   clang++ -std=c++17 -O1 -g -fsanitize=fuzzer,address,undefined -ffp-contract=off
//       -Isrc/cpp -o fuzz_numbers tests/fuzz/fuzz_numbers.cpp
//       src/cpp/core/parallel.cpp src/cpp/io/text.cpp
//   ./fuzz_numbers -max_total_time=600 corpus/
//
// Any input must be read without a fault. A plain decimal number must give the
// double that strtod gives in the C locale, bit for bit. And any double, given
// as the first eight bytes, must come back unchanged when written and read
// again, with and without a decimal shift, in no more digits than the shortest
// text that std::to_chars writes.

#include <charconv>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <xlocale.h>
#else
#include <locale.h>
#endif

#include "io/text.h"

namespace {

using mdapy::io::Column;
using mdapy::io::Kind;

[[noreturn]] void fail(const char* what, const std::string& text) {
    std::fprintf(stderr, "%s: [%s]\n", what, text.c_str());
    std::abort();
}

// Digits with an optional sign, decimal point and exponent: the numbers that
// both Python's float() and strtod read, and read alike.
bool plain_decimal(const char* p, std::size_t n) {
    std::size_t i = 0;
    if (i < n && (p[i] == '+' || p[i] == '-')) ++i;
    std::size_t digits = 0;
    while (i < n && p[i] >= '0' && p[i] <= '9') ++i, ++digits;
    if (i < n && p[i] == '.') {
        ++i;
        while (i < n && p[i] >= '0' && p[i] <= '9') ++i, ++digits;
    }
    if (digits == 0) return false;
    if (i < n && (p[i] == 'e' || p[i] == 'E')) {
        ++i;
        if (i < n && (p[i] == '+' || p[i] == '-')) ++i;
        std::size_t exponent = 0;
        while (i < n && p[i] >= '0' && p[i] <= '9') ++i, ++exponent;
        if (exponent == 0) return false;
    }
    return i == n;
}

double c_strtod(const std::string& text, bool& whole) {
    static locale_t c = newlocale(LC_ALL_MASK, "C", nullptr);
    char* end = nullptr;
    double value = strtod_l(text.c_str(), &end, c);
    whole = end == text.c_str() + text.size();
    return value;
}

bool same(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return std::memcmp(&a, &b, sizeof a) == 0;
}

// The significant digits of a decimal number written by either writer.
int significant_digits(const std::string& text) {
    std::string digits;
    for (char ch : text) {
        if (ch == 'e' || ch == 'E') break;
        if (ch >= '0' && ch <= '9') digits += ch;
    }
    std::size_t first = digits.find_first_not_of('0');
    if (first == std::string::npos) return 1;
    std::size_t last = digits.find_last_not_of('0');
    return static_cast<int>(last - first + 1);
}

std::string written(double value, int shift) {
    Column column;
    column.kind = Kind::Float;
    column.data = &value;
    column.shift = shift;
    std::vector<Column> columns{column};
    std::string text = mdapy::io::format_table(1, columns);
    if (text.empty() || text.back() != '\n') fail("no line", text);
    text.pop_back();
    return text;
}

void round_trip(double value, int shift) {
    const std::string text = written(value, shift);
    double back = 0;
    if (!mdapy::io::parse_double(text.data(), text.data() + text.size(), back, -shift)) {
        fail("written text not read", text);
    }
    if (!same(value, back)) fail("value changed in a round trip", text);
    if (shift == 0 && std::isfinite(value)) {
        char shortest[64];
        auto result = std::to_chars(shortest, shortest + sizeof shortest, value);
        const std::string reference(shortest, result.ptr);
        if (significant_digits(text) > significant_digits(reference)) {
            fail("more digits than the shortest", text + "] against [" + reference);
        }
    }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const char* text = reinterpret_cast<const char*>(data);
    double value = 0;
    std::int64_t integer = 0;
    const bool read = mdapy::io::parse_double(text, text + size, value);
    mdapy::io::parse_int64(text, text + size, integer);
    if (size > 0) {
        double shifted = 0;
        mdapy::io::parse_double(text + 1, text + size, shifted,
                                static_cast<int>(static_cast<std::int8_t>(data[0])) % 40);
    }
    if (plain_decimal(text, size)) {
        const std::string number(text, size);
        bool whole = false;
        const double reference = c_strtod(number, whole);
        if (read != whole) fail("accepted differently from strtod", number);
        if (read && !same(value, reference)) fail("differs from strtod", number);
    }
    if (size >= 8) {
        double bits;
        std::memcpy(&bits, data, sizeof bits);
        round_trip(bits, 0);
        if (size >= 9) round_trip(bits, static_cast<int>(static_cast<std::int8_t>(data[8])) % 8);
    }
    return 0;
}
