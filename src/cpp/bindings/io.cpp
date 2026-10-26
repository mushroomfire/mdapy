// Copyright (c) 2022-2026, Yongchao Wu
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at https://mozilla.org/MPL/2.0/.
//
// Bindings of the text table kernels. A ParseError reaches Python as
// ValueError(message, row). Columns are given as tuples whose optional last
// element is the decimal shift of a float column.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bindings/control.h"
#include "core/parallel.h"
#include "io/text.h"

namespace mdapy::bindings {
namespace {

using AnyArray = nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>;

// Read-only view of an object with the buffer protocol, such as bytes or mmap.
class Buffer {
public:
    explicit Buffer(nb::handle object) {
        if (PyObject_GetBuffer(object.ptr(), &view_, PyBUF_SIMPLE) != 0) {
            throw nb::python_error();
        }
    }
    ~Buffer() { PyBuffer_Release(&view_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    const char* data() const { return static_cast<const char*>(view_.buf); }
    std::int64_t size() const { return static_cast<std::int64_t>(view_.len); }

    void check_offset(std::int64_t offset) const {
        if (offset < 0 || offset > size()) {
            throw std::invalid_argument("offset " + std::to_string(offset) +
                                        " is outside the text of " + std::to_string(size()) +
                                        " bytes");
        }
    }

private:
    Py_buffer view_;
};

[[noreturn]] void raise_parse_error(const io::ParseError& error) {
    PyErr_SetObject(PyExc_ValueError, nb::make_tuple(error.what(), error.row).ptr());
    throw nb::python_error();
}

io::Kind to_kind(const std::string& name) {
    if (name == "f") return io::Kind::Float;
    if (name == "i") return io::Kind::Int;
    if (name == "b") return io::Kind::Bool;
    if (name == "s") return io::Kind::String;
    if (name == "x") return io::Kind::Skip;
    throw std::invalid_argument("unknown column kind '" + name + "'");
}

// Decimal shift of a column: the optional element `index` of its tuple.
int column_shift(const nb::tuple& spec, std::size_t index, io::Kind kind) {
    if (spec.size() <= index) {
        return 0;
    }
    const int shift = nb::cast<int>(spec[index]);
    if (shift != 0 && kind != io::Kind::Float) {
        throw std::invalid_argument("only float columns can have a shift");
    }
    if (shift < -100 || shift > 100) {
        throw std::invalid_argument("shift must be between -100 and 100");
    }
    return shift;
}

// New NumPy array of shape (rows,) or (rows, width); data receives its memory.
template <class T>
nb::object new_array(std::int64_t rows, int width, void*& data) {
    T* values = new T[static_cast<std::size_t>(rows) * static_cast<std::size_t>(width)];
    data = values;
    nb::capsule owner(values, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    const std::size_t shape[2] = {static_cast<std::size_t>(rows), static_cast<std::size_t>(width)};
    return nb::ndarray<nb::numpy, T>(values, width == 1 ? 1 : 2, shape, owner).cast();
}

std::int64_t skip_lines(nb::handle text, std::int64_t offset, std::int64_t n) {
    Buffer buffer(text);
    buffer.check_offset(offset);
    try {
        nb::gil_scoped_release release;
        return io::skip_lines(buffer.data(), buffer.size(), offset, n);
    } catch (const io::ParseError& error) {
        raise_parse_error(error);
    }
}

// spec: (kind, width) or (kind, width, shift) for each column.
nb::tuple parse_table(nb::handle text, std::int64_t offset, std::int64_t rows,
                      const std::vector<nb::tuple>& spec, const std::string& comment) {
    if (rows < 0) {
        throw std::invalid_argument("rows must be at least 0");
    }
    if (comment.size() > 1) {
        throw std::invalid_argument("comment must be one character or empty");
    }
    Buffer buffer(text);
    buffer.check_offset(offset);
    std::vector<io::Column> columns;
    nb::list arrays;
    for (const nb::tuple& item : spec) {
        if (item.size() != 2 && item.size() != 3) {
            throw std::invalid_argument("columns are (kind, width) or (kind, width, shift)");
        }
        const int width = nb::cast<int>(item[1]);
        if (width < 1) {
            throw std::invalid_argument("column width must be at least 1");
        }
        io::Column column;
        column.kind = to_kind(nb::cast<std::string>(item[0]));
        column.width = width;
        column.shift = column_shift(item, 2, column.kind);
        switch (column.kind) {
            case io::Kind::Float:
                arrays.append(new_array<double>(rows, width, column.data));
                break;
            case io::Kind::Int:
                arrays.append(new_array<std::int64_t>(rows, width, column.data));
                break;
            case io::Kind::Bool:
                arrays.append(new_array<bool>(rows, width, column.data));
                break;
            case io::Kind::String:
                arrays.append(new_array<std::uint32_t>(rows, width, column.data));
                break;
            case io::Kind::Skip:
                arrays.append(nb::none());
                break;
        }
        columns.push_back(std::move(column));
    }
    std::int64_t end = 0;
    try {
        run_released(
            [&](const Control& control) {
                end = io::parse_table(buffer.data(), buffer.size(), offset, rows, columns,
                                      comment.empty() ? '\0' : comment[0], control);
            },
            nb::none(), nullptr);
    } catch (const io::ParseError& error) {
        raise_parse_error(error);
    }
    for (std::size_t i = 0; i < columns.size(); ++i) {
        if (columns[i].kind == io::Kind::String) {
            nb::list categories;
            for (const std::string& value : columns[i].categories) {
                categories.append(nb::str(value.data(), value.size()));
            }
            arrays[i] = nb::make_tuple(arrays[i], categories);
        }
    }
    return nb::make_tuple(arrays, end);
}

// spec: (kind, values, categories) or (kind, values, categories, shift) for
// each column.
nb::bytes format_table(const std::vector<nb::tuple>& spec) {
    std::vector<io::Column> columns;
    std::vector<AnyArray> arrays;  // keeps the converted arrays alive
    std::int64_t rows = -1;
    for (const nb::tuple& item : spec) {
        if (item.size() != 3 && item.size() != 4) {
            throw std::invalid_argument(
                "columns are (kind, values, categories) or (kind, values, categories, shift)");
        }
        const std::string name = nb::cast<std::string>(item[0]);
        const AnyArray array = nb::cast<AnyArray>(item[1]);
        arrays.push_back(array);
        io::Column column;
        column.kind = to_kind(name);
        column.shift = column_shift(item, 3, column.kind);
        const bool matches =
            (column.kind == io::Kind::Float && array.dtype() == nb::dtype<double>()) ||
            (column.kind == io::Kind::Int && array.dtype() == nb::dtype<std::int64_t>()) ||
            (column.kind == io::Kind::Bool && array.dtype() == nb::dtype<bool>()) ||
            (column.kind == io::Kind::String && array.dtype() == nb::dtype<std::uint32_t>());
        if (!matches) {
            throw std::invalid_argument("column of kind '" + name + "' has the wrong dtype");
        }
        if (array.ndim() != 1 && array.ndim() != 2) {
            throw std::invalid_argument("columns must be 1- or 2-dimensional");
        }
        const auto n = static_cast<std::int64_t>(array.shape(0));
        if (rows >= 0 && n != rows) {
            throw std::invalid_argument("columns have different numbers of rows");
        }
        rows = n;
        column.width = array.ndim() == 2 ? static_cast<int>(array.shape(1)) : 1;
        column.data = const_cast<void*>(array.data());
        column.categories = nb::cast<std::vector<std::string>>(item[2]);
        columns.push_back(std::move(column));
    }
    std::string text;
    run_released(
        [&](const Control& control) { text = io::format_table(rows < 0 ? 0 : rows, columns, control); },
        nb::none(), nullptr);
    return nb::bytes(text.data(), text.size());
}

}  // namespace

void bind_io(nb::module_& m) {
    m.def("skip_lines", &skip_lines, nb::arg("text"), nb::arg("offset"), nb::arg("n"));
    m.def("parse_table", &parse_table, nb::arg("text"), nb::arg("offset"), nb::arg("rows"),
          nb::arg("columns"), nb::arg("comment") = "");
    m.def("format_table", &format_table, nb::arg("columns"));
}

}  // namespace mdapy::bindings
