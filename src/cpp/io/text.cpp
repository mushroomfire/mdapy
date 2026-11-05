// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.

#include "io/text.h"

#include "core/buffer.h"
#include "io/powers_of_five.h"

#include <algorithm>
#include <cerrno>
#include <cfloat>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <locale.h>
#include <string_view>
#include <system_error>
#include <unordered_map>

#if defined(__APPLE__)
#include <xlocale.h>
#endif

#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace mdapy::io {
namespace {

// ---------------------------------------------------------------- numbers

// strtod in the "C" locale, whatever the locale of the process is.
#if defined(_WIN32)
double strtod_c(const char* text, char** end) {
    static const _locale_t locale = _create_locale(LC_ALL, "C");
    return _strtod_l(text, end, locale);
}
#else
double strtod_c(const char* text, char** end) {
    static const locale_t locale = newlocale(LC_ALL_MASK, "C", static_cast<locale_t>(0));
    return strtod_l(text, end, locale);
}
#endif

// Without excess precision, m * 10^e and m / 10^e are correctly rounded when
// m <= 2^53 and |e| <= 22, because both operands are exact (Clinger 1990).
constexpr bool kExactArithmetic = FLT_EVAL_METHOD == 0;
constexpr std::uint64_t kMaxExactMantissa = std::uint64_t{1} << 53;
constexpr double kPowersOfTen[] = {1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,
                                   1e8,  1e9,  1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
                                   1e16, 1e17, 1e18, 1e19, 1e20, 1e21, 1e22};

inline bool is_digit(char c) { return c >= '0' && c <= '9'; }

inline char lower(char c) { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c; }

bool equals_ignoring_case(const char* first, const char* last, const char* word) {
    const std::size_t n = std::strlen(word);
    if (static_cast<std::size_t>(last - first) != n) {
        return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (lower(first[i]) != word[i]) {
            return false;
        }
    }
    return true;
}

#if defined(__SIZEOF_INT128__)
__extension__ typedef unsigned __int128 UInt128;
#endif

struct UInt128Parts {
    std::uint64_t high;
    std::uint64_t low;
};

inline UInt128Parts multiply(std::uint64_t a, std::uint64_t b) {
#if defined(__SIZEOF_INT128__)
    const UInt128 product = static_cast<UInt128>(a) * b;
    return {static_cast<std::uint64_t>(product >> 64), static_cast<std::uint64_t>(product)};
#elif defined(_M_X64)
    std::uint64_t high;
    const std::uint64_t low = _umul128(a, b, &high);
    return {high, low};
#elif defined(_M_ARM64)
    return {__umulh(a, b), a * b};
#else
    const std::uint64_t mask = 0xffffffffu;
    const std::uint64_t lo_lo = (a & mask) * (b & mask);
    const std::uint64_t hi_lo = (a >> 32) * (b & mask);
    const std::uint64_t lo_hi = (a & mask) * (b >> 32);
    const std::uint64_t hi_hi = (a >> 32) * (b >> 32);
    const std::uint64_t cross = (lo_lo >> 32) + (hi_lo & mask) + lo_hi;
    return {(hi_lo >> 32) + (cross >> 32) + hi_hi, (cross << 32) | (lo_lo & mask)};
#endif
}

inline int leading_zeros(std::uint64_t x) {  // x != 0
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_clzll(x);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
    unsigned long index;
    _BitScanReverse64(&index, x);
    return 63 - static_cast<int>(index);
#else
    int n = 0;
    for (; (x >> 63) == 0; x <<= 1) {
        ++n;
    }
    return n;
#endif
}

// Correctly rounded w * 10^q for 0 < w < 10^19, by the Eisel-Lemire algorithm
// (D. Lemire, Softw. Pract. Exp. 51, 1700 (2021)). For at most 19 digits it
// needs no fallback (N. Mushtak and D. Lemire, Softw. Pract. Exp. 53, 1467
// (2023)).
double eisel_lemire(std::uint64_t w, std::int64_t q, bool negative) {
    constexpr int kMantissaBits = 52;
    constexpr std::int64_t kMinimumExponent = -1023;
    constexpr std::int64_t kInfinitePower = 0x7ff;
    std::uint64_t mantissa = 0;
    std::int64_t power2 = 0;
    if (q > kLargestPowerOfFive) {
        power2 = kInfinitePower;
    } else if (q >= kSmallestPowerOfFive) {
        const int lz = leading_zeros(w);
        w <<= lz;
        const std::uint64_t* power = kPowersOfFive[q - kSmallestPowerOfFive];
        UInt128Parts product = multiply(w, power[0]);
        constexpr std::uint64_t kPrecisionMask = ~std::uint64_t{0} >> (kMantissaBits + 3);
        if ((product.high & kPrecisionMask) == kPrecisionMask) {
            const UInt128Parts second = multiply(w, power[1]);
            product.low += second.high;
            if (second.high > product.low) {
                ++product.high;
            }
        }
        const int upperbit = static_cast<int>(product.high >> 63);
        const int shift = upperbit + 64 - kMantissaBits - 3;
        mantissa = product.high >> shift;
        // ((152170 + 65536) * q) >> 16 is floor(q * log2(10)) for |q| <= 1650.
        power2 = (((152170 + 65536) * q) >> 16) + 63 + upperbit - lz - kMinimumExponent;
        if (power2 <= 0) {  // subnormal or zero
            if (-power2 + 1 >= 64) {
                mantissa = 0;
                power2 = 0;
            } else {
                mantissa >>= -power2 + 1;
                mantissa += mantissa & 1;
                mantissa >>= 1;
                // Rounding up may reach the smallest normal number.
                power2 = mantissa < (std::uint64_t{1} << kMantissaBits) ? 0 : 1;
            }
        } else {
            // Exactly halfway between two doubles: round to even. This can
            // only happen when 5^q fits in 64 bits.
            if (product.low <= 1 && q >= -4 && q <= 23 && (mantissa & 3) == 1 &&
                (mantissa << shift) == product.high) {
                mantissa &= ~std::uint64_t{1};
            }
            mantissa += mantissa & 1;
            mantissa >>= 1;
            if (mantissa >= (std::uint64_t{2} << kMantissaBits)) {
                mantissa = std::uint64_t{1} << kMantissaBits;
                ++power2;
            }
            mantissa &= ~(std::uint64_t{1} << kMantissaBits);
            if (power2 >= kInfinitePower) {
                power2 = kInfinitePower;
                mantissa = 0;
            }
        }
    }
    std::uint64_t bits = mantissa | (static_cast<std::uint64_t>(power2) << kMantissaBits);
    if (negative) {
        bits |= std::uint64_t{1} << 63;
    }
    double value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Correctly rounded value of a token that is known to be a decimal number:
// [first, mark) is its sign and digits, exponent is the value of its exponent
// part (0 if it has none).
double parse_slow(const char* first, const char* mark, std::int64_t exponent, int shift) {
    std::string text(first, mark);
    text += 'e';
    text += std::to_string(exponent + shift);
    return strtod_c(text.c_str(), nullptr);
}

// --------------------------------------------------------------- tables

inline bool is_blank(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f';
}

std::int64_t count_values(const char* p, const char* end) {
    std::int64_t n = 0;
    while (true) {
        while (p < end && is_blank(*p)) {
            ++p;
        }
        if (p == end) {
            return n;
        }
        ++n;
        while (p < end && !is_blank(*p)) {
            ++p;
        }
    }
}

std::string quoted(const char* first, const char* last) {
    constexpr std::ptrdiff_t kLimit = 40;
    if (last - first > kLimit) {
        return "'" + std::string(first, kLimit) + "...'";
    }
    return "'" + std::string(first, last) + "'";
}

// Codes for the distinct strings of one column, in order of first appearance.
// A column holds few of them, the elements of a system, which are found fastest
// by looking at each; many are hashed.
struct Dictionary {
    static constexpr std::size_t kFew = 16;
    std::unordered_map<std::string_view, std::uint32_t> codes;
    std::vector<std::string_view> values;

    std::uint32_t code(std::string_view value) {
        if (values.size() <= kFew) {
            for (std::size_t i = 0; i < values.size(); ++i) {
                if (values[i] == value) {
                    return static_cast<std::uint32_t>(i);
                }
            }
        } else {
            const auto found = codes.find(value);
            if (found != codes.end()) {
                return found->second;
            }
        }
        const auto next = static_cast<std::uint32_t>(values.size());
        values.push_back(value);
        codes.emplace(value, next);
        return next;
    }
};

// State of one piece of a table in parse_table.
struct Chunk {
    std::int64_t error_row = -1;
    std::string error;
    std::vector<Dictionary> dictionaries;  // one per string column
};

bool parse_bool(const char* first, const char* last, std::uint8_t& value) {
    const std::string_view text(first, static_cast<std::size_t>(last - first));
    if (text == "T" || text == "True" || text == "true") {
        value = 1;
        return true;
    }
    if (text == "F" || text == "False" || text == "false") {
        value = 0;
        return true;
    }
    return false;
}

// The correctly rounded mantissa * 10^exponent, for mantissa < 10^19.
inline double from_decimal(std::uint64_t mantissa, std::int64_t exponent, bool negative) {
    if (mantissa == 0) {
        return negative ? -0.0 : 0.0;
    }
    if (kExactArithmetic && mantissa <= kMaxExactMantissa && exponent >= -22 && exponent <= 22) {
        double result = static_cast<double>(mantissa);
        if (exponent < 0) {
            result /= kPowersOfTen[-exponent];
        } else {
            result *= kPowersOfTen[exponent];
        }
        return negative ? -result : result;
    }
    return eisel_lemire(mantissa, exponent, negative);
}

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
constexpr bool kLittleEndian = false;
#else
constexpr bool kLittleEndian = true;
#endif

inline std::uint64_t load_eight(const char* p) {
    std::uint64_t bytes;
    std::memcpy(&bytes, p, sizeof(bytes));
    return bytes;
}

// Whether all eight bytes are digits.
inline bool eight_digits(std::uint64_t bytes) {
    return (((bytes + 0x4646464646464646u) | (bytes - 0x3030303030303030u)) &
            0x8080808080808080u) == 0;
}

// The number that eight digits spell, the first in the lowest byte.
inline std::uint64_t number_of_eight(std::uint64_t bytes) {
    constexpr std::uint64_t kMask = 0x000000FF000000FFu;
    constexpr std::uint64_t kFirst = 0x000F424000000064u;   // 100 + (1000000 << 32)
    constexpr std::uint64_t kSecond = 0x0000271000000001u;  // 1 + (10000 << 32)
    bytes -= 0x3030303030303030u;
    bytes = (bytes * 10) + (bytes >> 8);
    const std::uint64_t first = (bytes & kMask) * kFirst;
    const std::uint64_t second = ((bytes >> 16) & kMask) * kSecond;
    return (first + second) >> 32;
}

// Reads a number written as nearly every program writes one: an optional minus,
// digits with an optional point, an optional exponent. Stores its value and
// returns where it stops, without looking at what follows; nullptr for anything
// else, which parse_double decides on.
inline const char* read_double(const char* p, const char* end, int shift, double& value) {
    bool negative = false;
    if (p != end && *p == '-') {
        negative = true;
        ++p;
    }
    const char* start = p;
    std::uint64_t mantissa = 0;  // wraps only with more than 19 digits, which are refused
    for (; p != end && is_digit(*p); ++p) {
        mantissa = mantissa * 10 + static_cast<std::uint64_t>(*p - '0');
    }
    std::int64_t digits = p - start;
    std::int64_t exponent = 0;
    if (p != end && *p == '.') {
        ++p;
        const char* fraction = p;
        if (kLittleEndian) {
            while (end - p >= 8 && eight_digits(load_eight(p))) {
                mantissa = mantissa * 100000000u + number_of_eight(load_eight(p));
                p += 8;
            }
        }
        for (; p != end && is_digit(*p); ++p) {
            mantissa = mantissa * 10 + static_cast<std::uint64_t>(*p - '0');
        }
        exponent = fraction - p;
        digits += p - fraction;
    }
    if (digits == 0) {
        return nullptr;
    }
    if (digits > 19) {  // leading zeros do not count
        for (const char* q = start; q != p && (*q == '0' || *q == '.'); ++q) {
            digits -= *q == '0';
        }
        if (digits > 19) {
            return nullptr;
        }
    }
    if (p != end && (*p == 'e' || *p == 'E')) {
        ++p;
        bool negative_exponent = false;
        if (p != end && (*p == '+' || *p == '-')) {
            negative_exponent = *p == '-';
            ++p;
        }
        const char* power = p;
        std::int64_t written = 0;
        for (; p != end && is_digit(*p) && p - power < 6; ++p) {
            written = written * 10 + (*p - '0');
        }
        if (p == power || (p != end && is_digit(*p))) {
            return nullptr;
        }
        exponent += negative_exponent ? -written : written;
    }
    value = from_decimal(mantissa, exponent + shift, negative);
    return p;
}

// Reads an integer of at most 18 digits with an optional minus; nullptr for
// anything else, which parse_int64 decides on.
inline const char* read_int64(const char* p, const char* end, std::int64_t& value) {
    const bool negative = p != end && *p == '-';
    if (negative) {
        ++p;
    }
    const char* start = p;
    std::uint64_t magnitude = 0;
    for (; p != end && is_digit(*p) && p - start < 18; ++p) {
        magnitude = magnitude * 10 + static_cast<std::uint64_t>(*p - '0');
    }
    if (p == start || (p != end && is_digit(*p))) {
        return nullptr;
    }
    value = negative ? -static_cast<std::int64_t>(magnitude) : static_cast<std::int64_t>(magnitude);
    return p;
}

// Where a value ends: at a blank, the end of its line or a comment. stop is the
// character of comments, or a newline when there are none.
inline const char* token_end(const char* p, const char* end, char stop) {
    while (p != end && !is_blank(*p) && *p != '\n' && *p != stop) {
        ++p;
    }
    return p;
}

inline bool ends_token(const char* p, const char* end, char stop) {
    return p == end || is_blank(*p) || *p == '\n' || *p == stop;
}

// The error of a line: the wrong number of values, whatever else is wrong with
// it, or else the given one.
std::string line_error(const char* line, const char* end, char stop, std::int64_t expected,
                       std::string other) {
    const void* newline = std::memchr(line, '\n', static_cast<std::size_t>(end - line));
    const char* line_end = newline ? static_cast<const char*>(newline) : end;
    const void* mark = std::memchr(line, stop, static_cast<std::size_t>(line_end - line));
    const std::int64_t found = count_values(line, mark ? static_cast<const char*>(mark) : line_end);
    if (found != expected) {
        return "expected " + std::to_string(expected) + " values, found " + std::to_string(found);
    }
    return other;
}

// Parses the lines of [p, end), the rows [row, last), into the columns. The
// first line that cannot be read goes into the chunk.
void parse_lines(const char* p, const char* end, std::int64_t row, std::int64_t last,
                 std::vector<Column>& columns, char stop, std::int64_t expected, Chunk& chunk) {
    for (; row < last; ++row) {
        const char* line = p;
        std::string error;
        std::size_t string_column = 0;
        for (Column& column : columns) {
            Dictionary* dictionary =
                column.kind == Kind::String ? &chunk.dictionaries[string_column++] : nullptr;
            for (int k = 0; k < column.width; ++k) {
                while (p != end && is_blank(*p)) {
                    ++p;
                }
                if (p == end || *p == '\n' || *p == stop) {
                    chunk.error_row = row;
                    chunk.error = line_error(line, end, stop, expected, {});
                    return;
                }
                const std::int64_t index = row * column.width + k;
                const char* token = p;
                switch (column.kind) {
                    case Kind::Float: {
                        double& value = static_cast<double*>(column.data)[index];
                        const char* stopped = read_double(p, end, column.shift, value);
                        if (stopped != nullptr && ends_token(stopped, end, stop)) {
                            p = stopped;
                            break;
                        }
                        p = token_end(p, end, stop);
                        if (!parse_double(token, p, value, column.shift)) {
                            error = "cannot read " + quoted(token, p) + " as a number";
                        }
                        break;
                    }
                    case Kind::Int: {
                        std::int64_t& value = static_cast<std::int64_t*>(column.data)[index];
                        const char* stopped = read_int64(p, end, value);
                        if (stopped != nullptr && ends_token(stopped, end, stop)) {
                            p = stopped;
                            break;
                        }
                        p = token_end(p, end, stop);
                        if (!parse_int64(token, p, value)) {
                            error = "cannot read " + quoted(token, p) + " as an integer";
                        }
                        break;
                    }
                    case Kind::Bool:
                        p = token_end(p, end, stop);
                        if (!parse_bool(token, p, static_cast<std::uint8_t*>(column.data)[index])) {
                            error = "cannot read " + quoted(token, p) + " as a boolean (T or F)";
                        }
                        break;
                    case Kind::String:
                        p = token_end(p, end, stop);
                        static_cast<std::uint32_t*>(column.data)[index] = dictionary->code(
                            std::string_view(token, static_cast<std::size_t>(p - token)));
                        break;
                    case Kind::Skip:
                        p = token_end(p, end, stop);
                        break;
                }
                if (!error.empty()) {
                    chunk.error_row = row;
                    chunk.error = line_error(line, end, stop, expected, std::move(error));
                    return;
                }
            }
        }
        while (p != end && is_blank(*p)) {
            ++p;
        }
        if (p != end && *p == stop && stop != '\n') {
            const void* newline = std::memchr(p, '\n', static_cast<std::size_t>(end - p));
            p = newline ? static_cast<const char*>(newline) : end;
        }
        if (p != end && *p != '\n') {
            chunk.error_row = row;
            chunk.error = line_error(line, end, stop, expected, {});
            return;
        }
        if (p != end) {
            ++p;
        }
    }
}

// Offset of the line after the one starting at position, which is line
// `line` of the n lines being read.
std::int64_t next_line(const char* text, std::int64_t size, std::int64_t position,
                       std::int64_t line, std::int64_t n) {
    if (position >= size) {
        throw ParseError(line, "expected " + std::to_string(n) +
                                   " lines, but the text ends after " + std::to_string(line));
    }
    const void* newline =
        std::memchr(text + position, '\n', static_cast<std::size_t>(size - position));
    return newline ? static_cast<const char*>(newline) - text + 1 : size;
}

// How many newlines [first, last) holds. In the sum of eight bytes the high bit
// of a byte stays clear only if that byte is a newline, whatever its neighbors.
std::int64_t count_newlines(const char* first, const char* last) {
    constexpr std::uint64_t kNewlines = 0x0a0a0a0a0a0a0a0au;
    constexpr std::uint64_t kLow = 0x7f7f7f7f7f7f7f7fu;
    constexpr std::uint64_t kHigh = 0x8080808080808080u;
    constexpr std::uint64_t kOnes = 0x0101010101010101u;
    std::int64_t count = 0;
    for (; last - first >= 8; first += 8) {
        const std::uint64_t bytes = load_eight(first) ^ kNewlines;
        const std::uint64_t clear = ~(((bytes & kLow) + kLow) | bytes) & kHigh;
        count += static_cast<std::int64_t>(((clear >> 7) * kOnes) >> 56);
    }
    for (; first != last; ++first) {
        count += *first == '\n';
    }
    return count;
}

// Where n lines lie: pieces of a table that start and end at the start of a
// line, of about one block each so that threads can share them.
struct Lines {
    std::vector<std::int64_t> offsets;  // where each piece starts; and where the last ends
    std::vector<std::int64_t> rows;     // the row each piece starts at; and n
};

Lines find_lines(const char* text, std::int64_t size, std::int64_t offset, std::int64_t n,
                 const Control& control) {
    constexpr std::int64_t kBlock = std::int64_t{1} << 18;
    Lines lines;
    lines.offsets.push_back(offset);
    lines.rows.push_back(0);
    // The first lines one by one: a small table needs no more, and they tell
    // how far the others should reach.
    std::int64_t line = 0;
    std::int64_t start = offset;  // of the next piece
    while (line < n && start - offset < kBlock) {
        start = next_line(text, size, start, line, n);
        ++line;
    }
    if (line > 0) {
        lines.offsets.push_back(start);
        lines.rows.push_back(line);
    }
    if (line >= n) {
        return lines;
    }
    const double mean = static_cast<double>(start - offset) / static_cast<double>(line);
    std::int64_t scanned = start;
    std::vector<std::int64_t> counts;
    std::vector<std::int64_t> ends;
    while (line < n && scanned < size) {
        const double wanted = static_cast<double>(n - line) * mean * 1.02;
        const std::int64_t left = size - scanned;
        const std::int64_t span =
            wanted >= static_cast<double>(left)
                ? left
                : std::min(left, std::max(kBlock, static_cast<std::int64_t>(wanted)));
        const std::int64_t blocks = (span + kBlock - 1) / kBlock;
        counts.assign(static_cast<std::size_t>(blocks), 0);
        ends.assign(static_cast<std::size_t>(blocks), 0);
        parallel_for(
            blocks, 1,
            [&](std::int64_t begin, std::int64_t stop) {
                for (std::int64_t b = begin; b < stop; ++b) {
                    const char* first = text + scanned + b * kBlock;
                    const char* last = text + scanned + std::min(span, (b + 1) * kBlock);
                    const std::int64_t count = count_newlines(first, last);
                    counts[static_cast<std::size_t>(b)] = count;
                    if (count > 0) {
                        while (last[-1] != '\n') {
                            --last;
                        }
                        ends[static_cast<std::size_t>(b)] = last - text;
                    }
                }
            },
            control);
        for (std::int64_t b = 0; b < blocks && line < n; ++b) {
            const std::int64_t count = counts[static_cast<std::size_t>(b)];
            if (count == 0) {
                continue;  // a long line goes on into the next block
            }
            if (line + count >= n) {  // the table ends inside this block
                for (std::int64_t k = line; k < n; ++k) {
                    start = next_line(text, size, start, k, n);
                }
                line = n;
            } else {
                start = ends[static_cast<std::size_t>(b)];
                line += count;
            }
            lines.offsets.push_back(start);
            lines.rows.push_back(line);
        }
        scanned += span;
    }
    // The end of the text: the last line may lack its newline.
    while (line < n) {
        start = next_line(text, size, start, line, n);
        ++line;
        lines.offsets.push_back(start);
        lines.rows.push_back(line);
    }
    return lines;
}

// ------------------------------------------------------------- writing numbers

constexpr char kDigitPairs[] =
    "00010203040506070809101112131415161718192021222324"
    "25262728293031323334353637383940414243444546474849"
    "50515253545556575859606162636465666768697071727374"
    "75767778798081828384858687888990919293949596979899";

inline void write_pair(char* out, std::uint32_t value) {  // value < 100
    std::memcpy(out, kDigitPairs + 2 * value, 2);
}

inline void write_four(char* out, std::uint32_t value) {  // value < 10^4, all four digits
    write_pair(out, value / 100);
    write_pair(out + 2, value % 100);
}

inline void write_eight(char* out, std::uint32_t value) {  // value < 10^8, all eight digits
    write_four(out, value / 10000);
    write_four(out + 4, value % 10000);
}

// Writes the digits of value without leading zeros; returns where they end.
char* write_unsigned(char* out, std::uint64_t value) {
    if (value < 10000) {
        const auto small = static_cast<std::uint32_t>(value);
        if (small < 100) {
            if (small < 10) {
                *out = static_cast<char>('0' + small);
                return out + 1;
            }
            write_pair(out, small);
            return out + 2;
        }
        if (small < 1000) {
            *out = static_cast<char>('0' + small / 100);
            write_pair(out + 1, small % 100);
            return out + 3;
        }
        write_four(out, small);
        return out + 4;
    }
    if (value < 100000000) {
        char* middle = write_unsigned(out, value / 10000);
        write_four(middle, static_cast<std::uint32_t>(value % 10000));
        return middle + 4;
    }
    char* middle = write_unsigned(out, value / 100000000);
    write_eight(middle, static_cast<std::uint32_t>(value % 100000000));
    return middle + 8;
}

inline char* write_int(char* out, std::int64_t value) {
    if (value < 0) {
        *out++ = '-';
        return write_unsigned(out, std::uint64_t{0} - static_cast<std::uint64_t>(value));
    }
    return write_unsigned(out, static_cast<std::uint64_t>(value));
}

// The shortest decimal digits * 10^exponent that read back to a double, the
// closest of them to it, by the Schubfach algorithm (R. Giulietti, "The
// Schubfach way to render doubles", 2020). The double is finite, and at least
// 2^(kLowestFastExponent - 1023), which keeps the powers within the table.
constexpr int kLowestFastExponent = 64;

struct Decimal {
    std::uint64_t digits;
    int exponent;
};

// floor(value * g / 2^128), with its lowest bit set if anything was cut off.
inline std::uint64_t round_to_odd(std::uint64_t high, std::uint64_t low, std::uint64_t value) {
    const UInt128Parts x = multiply(low, value);
    const UInt128Parts y = multiply(high, value);
    const std::uint64_t middle = y.low + x.high;
    return (y.high + (middle < y.low ? 1 : 0)) | (middle > 1 ? 1 : 0);
}

inline Decimal shortest_decimal(std::uint64_t fraction, int biased) {
    constexpr int kFractionBits = 52;
    const std::uint64_t c = (std::uint64_t{1} << kFractionBits) | fraction;
    const int q = biased - 1075;
    if (q <= 0 && q > -kFractionBits - 1 && (c & ((std::uint64_t{1} << -q) - 1)) == 0) {
        return {c >> -q, 0};  // a whole number
    }
    const std::uint64_t even = (c & 1) == 0 ? 1 : 0;
    const bool lower_is_closer = fraction == 0 && biased > 1;
    const std::uint64_t below = 4 * c - 2 + (lower_is_closer ? 1 : 0);
    // floor(log10(2^q)), or floor(log10(3/4 2^q)) when the lower neighbor is closer
    const int k = (q * 1262611 - (lower_is_closer ? 524031 : 0)) >> 22;
    // floor(log2(10^-k))
    const int h = q + ((-k * 1741647) >> 19) + 1;
    // 10^-k rounded up in 128 bits: the table holds it so for -27 <= -k <= 55,
    // and rounded down outside, where it is never exact.
    const int power = -k;
    const std::uint64_t* entry = kPowersOfFive[power - kSmallestPowerOfFive];
    const std::uint64_t up = power < -27 || power > 55 ? 1 : 0;
    const std::uint64_t low = entry[1] + up;
    const std::uint64_t high = entry[0] + (low < up ? 1 : 0);
    const std::uint64_t lower = round_to_odd(high, low, below << h) + (1 - even);
    const std::uint64_t value = round_to_odd(high, low, (4 * c) << h);
    const std::uint64_t upper = round_to_odd(high, low, (4 * c + 2) << h) - (1 - even);
    const std::uint64_t s = value / 4;
    if (s >= 10) {
        const std::uint64_t shorter = s / 10;
        const bool down_inside = lower <= 40 * shorter;
        const bool up_inside = 40 * shorter + 40 <= upper;
        if (down_inside != up_inside) {
            return {shorter + (up_inside ? 1 : 0), k + 1};
        }
    }
    const bool down_inside = lower <= 4 * s;
    const bool up_inside = 4 * s + 4 <= upper;
    if (down_inside != up_inside) {
        return {s + (up_inside ? 1 : 0), k};
    }
    const std::uint64_t middle = 4 * s + 2;
    return {s + ((value > middle || (value == middle && (s & 1) != 0)) ? 1 : 0), k};
}

// Writes what std::to_chars writes for value: the shortest digits that read back
// to it, in the shorter of the plain and the scientific notation, the plain on a
// tie. Beyond 10^15 and at the smallest values std::to_chars does the work. Up to
// 34 bytes after out are overwritten; returns where the text ends.
inline char* write_double(char* out, double value) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const bool negative = (bits >> 63) != 0;
    const int biased = static_cast<int>((bits >> 52) & 0x7ff);
    const double magnitude = std::fabs(value);
    if (magnitude == 0.0) {
        if (negative) {
            *out++ = '-';
        }
        *out = '0';
        return out + 1;
    }
    if (!(magnitude < 1e15) || biased < kLowestFastExponent) {
        return std::to_chars(out, out + 32, value).ptr;
    }
    if (negative) {
        *out++ = '-';
    }
    const Decimal decimal =
        shortest_decimal(bits & ((std::uint64_t{1} << 52) - 1), biased);
    char digits[40] = {};
    char* last;
    const std::uint64_t leading = decimal.digits / 100000000;
    if (leading == 0) {
        last = write_unsigned(digits, decimal.digits);
    } else {
        last = write_unsigned(digits, leading);
        write_eight(last, static_cast<std::uint32_t>(decimal.digits % 100000000));
        last += 8;
    }
    const int point = static_cast<int>(last - digits) + decimal.exponent;
    while (last[-1] == '0') {  // the digits are not all zero
        --last;
    }
    const int n = static_cast<int>(last - digits);
    if (point > 0 && point < n) {  // nearly every number
        std::memcpy(out, digits, 16);
        out[point] = '.';
        std::memcpy(out + point + 1, digits + point, 16);
        return out + n + 1;
    }
    const int power = point - 1;
    const int magnitude_of_power = power < 0 ? -power : power;
    const int scientific = n + (n > 1 ? 1 : 0) + 2 + (magnitude_of_power >= 100 ? 3 : 2);
    if (point <= 0) {
        if (2 - point + n <= scientific) {
            std::memcpy(out, "0.0000", 6);  // at most four zeros win
            std::memcpy(out + 2 - point, digits, 24);
            return out + 2 - point + n;
        }
    } else if (point <= scientific) {  // a whole number
        std::memcpy(out, digits, 16);
        std::memcpy(out + n, "00000000", 8);  // at most five zeros win
        return out + point;
    }
    *out = digits[0];
    if (n > 1) {
        out[1] = '.';
        std::memcpy(out + 2, digits + 1, 16);
        out += n + 1;
    } else {
        ++out;
    }
    *out++ = 'e';
    *out++ = power < 0 ? '-' : '+';
    if (magnitude_of_power >= 100) {
        *out++ = static_cast<char>('0' + magnitude_of_power / 100);
    }
    write_pair(out, static_cast<std::uint32_t>(magnitude_of_power % 100));
    return out + 2;
}

// Writes value * 10^shift for a finite, non-zero value: the shortest digits of
// value with the decimal exponent moved, written as Python's repr writes floats
// (scientific notation for exponents below -4 or above 15).
char* write_shifted(char* out, double value, int shift) {
    char buffer[32];
    const char* end =
        std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::scientific).ptr;
    const char* p = buffer;
    if (*p == '-') {
        *out++ = '-';
        ++p;
    }
    char digits[20];  // at most 17 significant digits
    int n = 0;
    for (; *p != 'e'; ++p) {
        if (*p != '.') {
            digits[n++] = *p;
        }
    }
    int exponent = 0;  // value = d.ddd * 10^exponent
    std::from_chars(p + (p[1] == '+' ? 2 : 1), end, exponent);
    exponent += shift;
    const int point = exponent + 1;  // digits before the decimal point
    if (point <= -4 || point > 16) {
        *out++ = digits[0];
        if (n > 1) {
            *out++ = '.';
            std::memcpy(out, digits + 1, static_cast<std::size_t>(n - 1));
            out += n - 1;
        }
        *out++ = 'e';
        *out++ = exponent < 0 ? '-' : '+';
        const int magnitude = exponent < 0 ? -exponent : exponent;
        if (magnitude < 10) {
            *out++ = '0';
        }
        return write_unsigned(out, static_cast<std::uint64_t>(magnitude));
    }
    if (point <= 0) {
        *out++ = '0';
        *out++ = '.';
        std::memset(out, '0', static_cast<std::size_t>(-point));
        out += -point;
        std::memcpy(out, digits, static_cast<std::size_t>(n));
        return out + n;
    }
    if (point < n) {
        std::memcpy(out, digits, static_cast<std::size_t>(point));
        out += point;
        *out++ = '.';
        std::memcpy(out, digits + point, static_cast<std::size_t>(n - point));
        return out + (n - point);
    }
    std::memcpy(out, digits, static_cast<std::size_t>(n));
    std::memset(out + n, '0', static_cast<std::size_t>(point - n));
    return out + point;
}

// Writes the rows [begin, end) from out on; returns where the text ends.
char* format_rows(std::int64_t begin, std::int64_t end, const std::vector<Column>& columns,
                  char* out) {
    for (std::int64_t row = begin; row < end; ++row) {
        const char* line = out;
        for (const Column& column : columns) {
            const std::int64_t first = row * column.width;
            switch (column.kind) {
                case Kind::Float: {
                    const double* values = static_cast<const double*>(column.data) + first;
                    for (int k = 0; k < column.width; ++k) {
                        out = column.shift != 0 && values[k] != 0.0 && std::isfinite(values[k])
                                  ? write_shifted(out, values[k], column.shift)
                                  : write_double(out, values[k]);
                        *out++ = ' ';
                    }
                    break;
                }
                case Kind::Int: {
                    const auto* values = static_cast<const std::int64_t*>(column.data) + first;
                    for (int k = 0; k < column.width; ++k) {
                        out = write_int(out, values[k]);
                        *out++ = ' ';
                    }
                    break;
                }
                case Kind::Bool: {
                    const auto* values = static_cast<const std::uint8_t*>(column.data) + first;
                    for (int k = 0; k < column.width; ++k) {
                        *out++ = values[k] ? 'T' : 'F';
                        *out++ = ' ';
                    }
                    break;
                }
                case Kind::String: {
                    const auto* values = static_cast<const std::uint32_t*>(column.data) + first;
                    for (int k = 0; k < column.width; ++k) {
                        const std::string& text = column.categories[values[k]];
                        std::memcpy(out, text.data(), text.size());
                        out += text.size();
                        *out++ = ' ';
                    }
                    break;
                }
                case Kind::Skip:
                    for (int k = 0; k < column.width; ++k) {
                        *out++ = ' ';
                    }
                    break;
            }
        }
        if (out == line) {  // no values
            *out++ = '\n';
        } else {
            out[-1] = '\n';
        }
    }
    return out;
}

// The most bytes one row can take.
std::size_t row_bound(const std::vector<Column>& columns) {
    std::size_t bound = 1;
    for (const Column& column : columns) {
        std::size_t value = 1;
        switch (column.kind) {
            case Kind::Float:
                value = 28;
                break;
            case Kind::Int:
                value = 21;
                break;
            case Kind::Bool:
                value = 2;
                break;
            case Kind::String:
                for (const std::string& text : column.categories) {
                    value = std::max(value, text.size() + 1);
                }
                break;
            case Kind::Skip:
                break;
        }
        bound += value * static_cast<std::size_t>(column.width);
    }
    return bound;
}

void check_codes(std::int64_t rows, const std::vector<Column>& columns, const Control& control) {
    for (const Column& column : columns) {
        if (column.kind != Kind::String) {
            continue;
        }
        const auto* codes = static_cast<const std::uint32_t*>(column.data);
        const std::uint32_t largest = parallel_reduce(
            rows * column.width, 0, std::uint32_t{0},
            [&](std::int64_t begin, std::int64_t end) {
                std::uint32_t found = 0;
                for (std::int64_t i = begin; i < end; ++i) {
                    found = std::max(found, codes[i]);
                }
                return found;
            },
            [](std::uint32_t a, std::uint32_t b) { return std::max(a, b); }, control);
        if (rows * column.width > 0 && largest >= column.categories.size()) {
            throw std::invalid_argument("string code " + std::to_string(largest) +
                                        " has no category");
        }
    }
}

void write_all(int descriptor, const char* data, std::size_t size) {
    constexpr std::size_t kMost = std::size_t{1} << 30;
    while (size > 0) {
#if defined(_WIN32)
        const int written = _write(descriptor, data, static_cast<unsigned>(std::min(size, kMost)));
#else
        const auto written = ::write(descriptor, data, std::min(size, kMost));
#endif
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::system_error(errno, std::generic_category());
        }
        data += written;
        size -= static_cast<std::size_t>(written);
    }
}

}  // namespace

bool parse_double(const char* first, const char* last, double& value, int shift) {
    const char* p = first;
    bool negative = false;
    if (p != last && (*p == '+' || *p == '-')) {
        negative = *p == '-';
        ++p;
    }
    const char* digits_begin = p;
    std::uint64_t mantissa = 0;
    int significant = 0;      // digits of mantissa from the first non-zero one
    std::int64_t exponent = 0;
    bool exact = true;        // all digits fit in mantissa
    bool any_digit = false;
    for (; p != last && is_digit(*p); ++p) {
        any_digit = true;
        if (significant < 19) {
            mantissa = mantissa * 10 + static_cast<std::uint64_t>(*p - '0');
            significant += mantissa != 0;
        } else {
            exact = false;
        }
    }
    if (p != last && *p == '.') {
        ++p;
        for (; p != last && is_digit(*p); ++p) {
            any_digit = true;
            if (significant < 19) {
                mantissa = mantissa * 10 + static_cast<std::uint64_t>(*p - '0');
                significant += mantissa != 0;
                --exponent;
            } else {
                exact = false;
            }
        }
    }
    if (!any_digit) {
        if (equals_ignoring_case(digits_begin, last, "nan")) {
            value = negative ? -std::numeric_limits<double>::quiet_NaN()
                             : std::numeric_limits<double>::quiet_NaN();
            return true;
        }
        if (equals_ignoring_case(digits_begin, last, "inf") ||
            equals_ignoring_case(digits_begin, last, "infinity")) {
            value = negative ? -std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::infinity();
            return true;
        }
        return false;
    }
    const char* mark = p;          // start of the exponent part, if any
    std::int64_t written = 0;      // value of the exponent part
    if (p != last && (*p == 'e' || *p == 'E')) {
        ++p;
        bool negative_exponent = false;
        if (p != last && (*p == '+' || *p == '-')) {
            negative_exponent = *p == '-';
            ++p;
        }
        if (p == last || !is_digit(*p)) {
            return false;
        }
        std::int64_t e = 0;
        for (; p != last && is_digit(*p); ++p) {
            if (e < 100000) {
                e = e * 10 + (*p - '0');
            }
        }
        written = negative_exponent ? -e : e;
    }
    if (p != last) {
        return false;
    }
    exponent += written + shift;
    value = exact ? from_decimal(mantissa, exponent, negative)
                  : parse_slow(first, mark, written, shift);
    return true;
}

bool parse_int64(const char* first, const char* last, std::int64_t& value) {
    const char* p = first;
    bool negative = false;
    if (p != last && (*p == '+' || *p == '-')) {
        negative = *p == '-';
        ++p;
    }
    if (p == last) {
        return false;
    }
    // Accumulate the magnitude as unsigned so that INT64_MIN is accepted.
    constexpr std::uint64_t kLimit = std::uint64_t{1} << 63;
    std::uint64_t magnitude = 0;
    for (; p != last; ++p) {
        if (!is_digit(*p)) {
            return false;
        }
        const auto digit = static_cast<std::uint64_t>(*p - '0');
        if (magnitude > (kLimit - digit) / 10) {
            return false;
        }
        magnitude = magnitude * 10 + digit;
    }
    if (!negative && magnitude == kLimit) {
        return false;
    }
    value = negative ? static_cast<std::int64_t>(0 - magnitude)
                     : static_cast<std::int64_t>(magnitude);
    return true;
}

std::int64_t skip_lines(const char* text, std::int64_t size, std::int64_t offset,
                        std::int64_t n) {
    return find_lines(text, size, offset, n, {}).offsets.back();
}

std::int64_t parse_table(const char* text, std::int64_t size, std::int64_t offset,
                         std::int64_t rows, std::vector<Column>& columns, char comment,
                         const Control& control) {
    const Lines lines = find_lines(text, size, offset, rows, control);
    const auto pieces = static_cast<std::int64_t>(lines.offsets.size()) - 1;
    std::int64_t expected = 0;
    std::size_t string_columns = 0;
    for (const Column& column : columns) {
        expected += column.width;
        string_columns += column.kind == Kind::String;
    }
    const char stop = comment != 0 ? comment : '\n';
    std::vector<Chunk> chunks(static_cast<std::size_t>(pieces));
    for (Chunk& chunk : chunks) {
        chunk.dictionaries.resize(string_columns);
    }
    parallel_for(
        pieces, 1,
        [&](std::int64_t begin, std::int64_t end) {
            for (std::int64_t piece = begin; piece < end; ++piece) {
                const auto i = static_cast<std::size_t>(piece);
                parse_lines(text + lines.offsets[i], text + lines.offsets[i + 1], lines.rows[i],
                            lines.rows[i + 1], columns, stop, expected, chunks[i]);
            }
        },
        control);
    for (const Chunk& chunk : chunks) {
        if (chunk.error_row >= 0) {
            throw ParseError(chunk.error_row, chunk.error);
        }
    }

    // Merge the dictionaries of the chunks in chunk order, which gives the
    // order of first appearance in the whole table, and translate the codes.
    std::size_t s = 0;
    for (Column& column : columns) {
        if (column.kind != Kind::String) {
            continue;
        }
        Dictionary merged;
        std::vector<std::vector<std::uint32_t>> translation(chunks.size());
        for (std::size_t c = 0; c < chunks.size(); ++c) {
            for (std::string_view value : chunks[c].dictionaries[s].values) {
                translation[c].push_back(merged.code(value));
            }
        }
        auto* codes = static_cast<std::uint32_t*>(column.data);
        const std::int64_t width = column.width;
        parallel_for(
            pieces, 1,
            [&](std::int64_t begin, std::int64_t end) {
                for (std::int64_t piece = begin; piece < end; ++piece) {
                    const auto c = static_cast<std::size_t>(piece);
                    const auto& table = translation[c];
                    for (std::int64_t i = lines.rows[c] * width; i < lines.rows[c + 1] * width;
                         ++i) {
                        codes[i] = table[codes[i]];
                    }
                }
            },
            control);
        column.categories.assign(merged.values.begin(), merged.values.end());
        ++s;
    }
    return lines.offsets.back();
}

void write_table(std::int64_t rows, const std::vector<Column>& columns, const Sink& sink,
                 const Control& control) {
    check_codes(rows, columns, control);
    // Each chunk of rows has its own place in the memory of a batch, as many
    // bytes as its rows can take, and the threads fill those places. The text of
    // a batch goes out while the threads fill the next one, so that neither
    // waits for the other. A row may take less than it can: the memory it
    // leaves is never touched.
    constexpr std::size_t kChunkBytes = std::size_t{1} << 19;
    constexpr std::int64_t kChunksInBatch = 64;
    constexpr std::size_t kSlack = 64;  // write_double overwrites a little more than its text
    const std::size_t bound = row_bound(columns);
    const auto chunk_rows = std::max<std::int64_t>(1, static_cast<std::int64_t>(kChunkBytes / bound));
    const std::int64_t batch_rows = chunk_rows * kChunksInBatch;
    struct Batch {
        Buffer<char> text;
        std::vector<std::size_t> sizes;
    };
    Batch batches[2];
    int which = 0;
    const bool together = get_num_threads() > 1 && rows > batch_rows;
    std::future<void> writing;  // destroyed first: it waits for the batch it reads
    for (std::int64_t start = 0; start < rows; start += batch_rows) {
        Batch& batch = batches[which];
        which = 1 - which;
        const std::int64_t count = std::min(batch_rows, rows - start);
        const std::int64_t chunks = (count + chunk_rows - 1) / chunk_rows;
        const std::size_t slot =
            static_cast<std::size_t>(std::min(chunk_rows, count)) * bound + kSlack;
        batch.text.resize(static_cast<std::size_t>(chunks) * slot);
        batch.sizes.resize(static_cast<std::size_t>(chunks));
        parallel_for(
            chunks, 1,
            [&](std::int64_t begin, std::int64_t end) {
                for (std::int64_t c = begin; c < end; ++c) {
                    const std::int64_t first = start + c * chunk_rows;
                    char* place = batch.text.data() + static_cast<std::size_t>(c) * slot;
                    const char* stop = format_rows(first, std::min(first + chunk_rows, start + count),
                                                   columns, place);
                    batch.sizes[static_cast<std::size_t>(c)] =
                        static_cast<std::size_t>(stop - place);
                }
            },
            control);
        const auto send = [&batch, &sink, chunks, slot] {
            for (std::int64_t c = 0; c < chunks; ++c) {
                sink(batch.text.data() + static_cast<std::size_t>(c) * slot,
                     batch.sizes[static_cast<std::size_t>(c)]);
            }
        };
        if (writing.valid()) {
            writing.get();
        }
        if (together) {
            writing = std::async(std::launch::async, send);
        } else {
            send();
        }
    }
    if (writing.valid()) {
        writing.get();
    }
}

void write_table(int descriptor, std::int64_t rows, const std::vector<Column>& columns,
                 const Control& control) {
    write_table(
        rows, columns,
        [descriptor](const char* data, std::size_t size) { write_all(descriptor, data, size); },
        control);
}

std::string format_table(std::int64_t rows, const std::vector<Column>& columns,
                         const Control& control) {
    std::string result;
    write_table(
        rows, columns,
        [&result](const char* data, std::size_t size) { result.append(data, size); }, control);
    return result;
}

}  // namespace mdapy::io
