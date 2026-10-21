// Copyright (c) 2022-2026, Yongchao Wu
// This file is part of mdapy, released under the BSD 3-Clause License.
//
// Parsing and formatting of whitespace-separated text tables, shared by all
// text file formats. Numbers are read and written without the C locale, so a
// program that changes LC_NUMERIC (as GUI toolkits do) cannot affect them.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/parallel.h"

namespace mdapy::io {

// Parses all of [first, last) as a decimal number, as Python's float() does:
// an optional sign, digits with an optional decimal point and an optional
// exponent, or nan, inf or infinity in any letter case. The value is the
// number times 10^shift, correctly rounded: the power of ten is applied to
// the decimal number, so it adds no rounding. Returns false if the text is not
// such a number.
bool parse_double(const char* first, const char* last, double& value, int shift = 0);

// Parses all of [first, last) as a decimal integer with an optional sign.
// Returns false if the text is not an integer or does not fit in 64 bits.
bool parse_int64(const char* first, const char* last, std::int64_t& value);

enum class Kind : char {
    Float = 'f',   // double
    Int = 'i',     // std::int64_t
    Bool = 'b',    // std::uint8_t, 0 or 1; text T, F, True, False, true, false
    String = 's',  // std::uint32_t code into categories
    Skip = 'x',    // value is checked to exist but not stored
};

// One column of a table, holding width values per row. data points to
// rows * width values stored row by row (nullptr for Skip).
struct Column {
    Kind kind = Kind::Float;
    int width = 1;
    void* data = nullptr;
    // Float columns: parse_table stores the numbers of the text times
    // 10^shift, and format_table writes the values times 10^shift. Both are
    // exact in decimal, so a value written with shift s and read with -s comes
    // back unchanged.
    int shift = 0;
    // String columns: the text of each code. parse_table fills it in order of
    // first appearance; format_table reads it.
    std::vector<std::string> categories;
};

// A line of a table that cannot be read. row counts from 0 at the first line
// of the table.
class ParseError : public std::runtime_error {
public:
    ParseError(std::int64_t row, const std::string& message)
        : std::runtime_error(message), row(row) {}
    std::int64_t row;
};

// Returns the offset of the line that follows the first n lines starting at
// text[offset]. The last line may lack a final newline. Throws ParseError if
// the text has fewer than n lines.
std::int64_t skip_lines(const char* text, std::int64_t size, std::int64_t offset,
                        std::int64_t n);

// Reads rows lines starting at text[offset]. Each line holds the values of
// the columns in order, separated by blanks (space, tab, CR, VT, FF); text from
// the comment character to the end of a line is ignored (comment == 0: none).
// Returns the offset just past the last line. Throws ParseError for the first
// line in the table that cannot be read.
std::int64_t parse_table(const char* text, std::int64_t size, std::int64_t offset,
                         std::int64_t rows, std::vector<Column>& columns, char comment,
                         const Control& control = {});

// Writes rows lines: the values of the columns in order separated by one
// space, each line ending with a newline. Floats use the shortest text that
// reads back to the same value; with a shift, the same digits with the
// decimal exponent moved. Throws std::invalid_argument for a string code
// without a category.
std::string format_table(std::int64_t rows, const std::vector<Column>& columns,
                         const Control& control = {});

}  // namespace mdapy::io
