// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.

#include "io/text.h"

#include "io/powers_of_five.h"

#include <cfloat>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <locale.h>
#include <string_view>
#include <unordered_map>

#if defined(__APPLE__)
#include <xlocale.h>
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
struct Dictionary {
    std::unordered_map<std::string_view, std::uint32_t> codes;
    std::vector<std::string_view> values;

    std::uint32_t code(std::string_view value) {
        auto [it, inserted] = codes.try_emplace(value, static_cast<std::uint32_t>(values.size()));
        if (inserted) {
            values.push_back(value);
        }
        return it->second;
    }
};

// State of one chunk of rows in parse_table.
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

// Parses one line, which holds exactly as many values as the columns need,
// into row `row`. Returns an empty string on success, else the error message.
std::string parse_row(const char* p, const char* end, std::int64_t row,
                      std::vector<Column>& columns, std::vector<Dictionary>& dictionaries) {
    std::size_t string_column = 0;
    for (Column& column : columns) {
        Dictionary* dictionary =
            column.kind == Kind::String ? &dictionaries[string_column++] : nullptr;
        for (int k = 0; k < column.width; ++k) {
            while (is_blank(*p)) {
                ++p;
            }
            const char* token = p;
            while (p < end && !is_blank(*p)) {
                ++p;
            }
            const std::int64_t index = row * column.width + k;
            switch (column.kind) {
                case Kind::Float:
                    if (!parse_double(token, p, static_cast<double*>(column.data)[index],
                                      column.shift)) {
                        return "cannot read " + quoted(token, p) + " as a number";
                    }
                    break;
                case Kind::Int:
                    if (!parse_int64(token, p, static_cast<std::int64_t*>(column.data)[index])) {
                        return "cannot read " + quoted(token, p) + " as an integer";
                    }
                    break;
                case Kind::Bool:
                    if (!parse_bool(token, p, static_cast<std::uint8_t*>(column.data)[index])) {
                        return "cannot read " + quoted(token, p) + " as a boolean (T or F)";
                    }
                    break;
                case Kind::String:
                    static_cast<std::uint32_t*>(column.data)[index] = dictionary->code(
                        std::string_view(token, static_cast<std::size_t>(p - token)));
                    break;
                case Kind::Skip:
                    break;
            }
        }
    }
    return {};
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

// Appends value * 10^shift for a finite, non-zero value: the shortest
// digits of value with the decimal exponent moved, written as Python's repr
// writes floats (scientific notation for exponents below -4 or above 15).
void append_shifted(std::string& out, double value, int shift) {
    char buffer[32];
    const char* end =
        std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::scientific).ptr;
    const char* p = buffer;
    if (*p == '-') {
        out.push_back('-');
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
        out.push_back(digits[0]);
        if (n > 1) {
            out.push_back('.');
            out.append(digits + 1, static_cast<std::size_t>(n - 1));
        }
        out.push_back('e');
        out.push_back(exponent < 0 ? '-' : '+');
        const int magnitude = exponent < 0 ? -exponent : exponent;
        if (magnitude < 10) {
            out.push_back('0');
        }
        out.append(std::to_string(magnitude));
    } else if (point <= 0) {
        out.append("0.");
        out.append(static_cast<std::size_t>(-point), '0');
        out.append(digits, static_cast<std::size_t>(n));
    } else if (point < n) {
        out.append(digits, static_cast<std::size_t>(point));
        out.push_back('.');
        out.append(digits + point, static_cast<std::size_t>(n - point));
    } else {
        out.append(digits, static_cast<std::size_t>(n));
        out.append(static_cast<std::size_t>(point - n), '0');
    }
}

void append_value(std::string& out, const Column& column, std::int64_t index) {
    char buffer[32];  // enough for any double or int64
    switch (column.kind) {
        case Kind::Float: {
            const double value = static_cast<const double*>(column.data)[index];
            if (column.shift != 0 && value != 0.0 && std::isfinite(value)) {
                append_shifted(out, value, column.shift);
            } else {
                out.append(buffer, std::to_chars(buffer, buffer + sizeof(buffer), value).ptr);
            }
            break;
        }
        case Kind::Int: {
            const std::int64_t value = static_cast<const std::int64_t*>(column.data)[index];
            out.append(buffer, std::to_chars(buffer, buffer + sizeof(buffer), value).ptr);
            break;
        }
        case Kind::Bool:
            out.push_back(static_cast<const std::uint8_t*>(column.data)[index] ? 'T' : 'F');
            break;
        case Kind::String:
            out.append(column.categories[static_cast<const std::uint32_t*>(column.data)[index]]);
            break;
        case Kind::Skip:
            break;
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
    if (!exact) {  // more than 19 significant digits
        value = parse_slow(first, mark, written, shift);
    } else if (mantissa == 0) {
        value = negative ? -0.0 : 0.0;
    } else if (kExactArithmetic && mantissa <= kMaxExactMantissa && exponent >= -22 &&
               exponent <= 22) {
        double result = static_cast<double>(mantissa);
        if (exponent < 0) {
            result /= kPowersOfTen[-exponent];
        } else {
            result *= kPowersOfTen[exponent];
        }
        value = negative ? -result : result;
    } else {
        value = eisel_lemire(mantissa, exponent, negative);
    }
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
    std::int64_t position = offset;
    for (std::int64_t line = 0; line < n; ++line) {
        position = next_line(text, size, position, line, n);
    }
    return position;
}

std::int64_t parse_table(const char* text, std::int64_t size, std::int64_t offset,
                         std::int64_t rows, std::vector<Column>& columns, char comment,
                         const Control& control) {
    // Start of every line; the line ends where the next one starts.
    std::vector<std::int64_t> starts(static_cast<std::size_t>(rows) + 1);
    starts[0] = offset;
    for (std::int64_t row = 0; row < rows; ++row) {
        starts[row + 1] = next_line(text, size, starts[row], row, rows);
    }
    std::int64_t expected = 0;
    std::size_t string_columns = 0;
    for (const Column& column : columns) {
        expected += column.width;
        string_columns += column.kind == Kind::String;
    }

    const std::int64_t grain = default_grain(rows);
    const std::int64_t nchunks = rows > 0 ? (rows + grain - 1) / grain : 0;
    std::vector<Chunk> chunks(static_cast<std::size_t>(nchunks));
    for (Chunk& chunk : chunks) {
        chunk.dictionaries.resize(string_columns);
    }
    parallel_for(
        rows, grain,
        [&](std::int64_t begin, std::int64_t end) {
            Chunk& chunk = chunks[static_cast<std::size_t>(begin / grain)];
            for (std::int64_t row = begin; row < end; ++row) {
                const char* line = text + starts[row];
                const char* line_end = text + starts[row + 1];
                if (line_end > line && line_end[-1] == '\n') {
                    --line_end;
                }
                if (comment != 0) {
                    const void* mark =
                        std::memchr(line, comment, static_cast<std::size_t>(line_end - line));
                    if (mark) {
                        line_end = static_cast<const char*>(mark);
                    }
                }
                const std::int64_t found = count_values(line, line_end);
                std::string error =
                    found != expected
                        ? "expected " + std::to_string(expected) + " values, found " +
                              std::to_string(found)
                        : parse_row(line, line_end, row, columns, chunk.dictionaries);
                if (!error.empty()) {
                    chunk.error_row = row;
                    chunk.error = std::move(error);
                    return;
                }
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
        const int width = column.width;
        parallel_for(
            rows, grain,
            [&](std::int64_t begin, std::int64_t end) {
                const auto& table = translation[static_cast<std::size_t>(begin / grain)];
                for (std::int64_t i = begin * width; i < end * width; ++i) {
                    codes[i] = table[codes[i]];
                }
            },
            control);
        column.categories.assign(merged.values.begin(), merged.values.end());
        ++s;
    }
    return starts[rows];
}

std::string format_table(std::int64_t rows, const std::vector<Column>& columns,
                         const Control& control) {
    for (const Column& column : columns) {
        if (column.kind != Kind::String) {
            continue;
        }
        const auto* codes = static_cast<const std::uint32_t*>(column.data);
        const std::size_t n = column.categories.size();
        for (std::int64_t i = 0; i < rows * column.width; ++i) {
            if (codes[i] >= n) {
                throw std::invalid_argument("string code " + std::to_string(codes[i]) +
                                            " has no category");
            }
        }
    }
    const std::int64_t grain = default_grain(rows);
    const std::int64_t nchunks = rows > 0 ? (rows + grain - 1) / grain : 0;
    std::vector<std::string> parts(static_cast<std::size_t>(nchunks));
    parallel_for(
        rows, grain,
        [&](std::int64_t begin, std::int64_t end) {
            std::string& out = parts[static_cast<std::size_t>(begin / grain)];
            for (std::int64_t row = begin; row < end; ++row) {
                bool first = true;
                for (const Column& column : columns) {
                    for (int k = 0; k < column.width; ++k) {
                        if (!first) {
                            out.push_back(' ');
                        }
                        first = false;
                        append_value(out, column, row * column.width + k);
                    }
                }
                out.push_back('\n');
            }
        },
        control);
    std::size_t total = 0;
    for (const std::string& part : parts) {
        total += part.size();
    }
    std::string result;
    result.reserve(total);
    for (const std::string& part : parts) {
        result.append(part);
    }
    return result;
}

}  // namespace mdapy::io
