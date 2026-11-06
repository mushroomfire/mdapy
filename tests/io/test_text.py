# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import decimal
import errno
import gzip
import io
import locale
import math
import mmap
import os
import random
import re
import sys

import numpy as np
import polars as pl
import pytest
from mdapy._core import io as native

import mdapy
from mdapy.core.threads import get_num_threads, set_num_threads
from mdapy.elements import ELEMENT_DTYPE
from mdapy.errors import FileFormatError
from mdapy.io import _text
from mdapy.io._text import TextReader, categorize, format_rows, strings, write_rows


def parse_floats(tokens):
    text = "\n".join(tokens).encode()
    (values,), _ = native.parse_table(text, 0, len(tokens), [("f", 1)])
    return values


def assert_same_bits(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    bad = np.flatnonzero(actual.view(np.uint64) != expected.view(np.uint64))
    assert bad.size == 0, [(i, actual[i], expected[i]) for i in bad[:5]]


def random_doubles(rng, n):
    bits = rng.integers(0, 2**64, size=n, dtype=np.uint64)
    values = bits.view(np.float64)
    return values[np.isfinite(values)]


def random_decimal(rng):
    """A random decimal string, often with more digits than a double holds."""
    sign = rng.choice(["", "-", "+"])
    whole = "".join(rng.choice(list("0123456789"), size=rng.integers(0, 26)))
    fraction = "".join(rng.choice(list("0123456789"), size=rng.integers(0, 26)))
    if not whole and not fraction:
        whole = "7"
    point = "." if fraction or rng.random() < 0.3 else ""
    exponent = ""
    if rng.random() < 0.7:
        exponent = rng.choice(["e", "E"]) + str(int(rng.integers(-360, 360)))
    return f"{sign}{whole}{point}{fraction}{exponent}"


# ---------------------------------------------------------------- numbers


def test_floats_match_python_for_shortest_representations():
    values = random_doubles(np.random.default_rng(1), 200_000)
    assert_same_bits(parse_floats([repr(float(v)) for v in values]), values)


@pytest.mark.parametrize("spec", ["%.17e", "%.16e", "%.15g", "%.6g", "%.3f", "%.10e"])
def test_floats_match_python_for_printf_formats(spec):
    rng = np.random.default_rng(2)
    values = rng.standard_normal(20_000) * 10.0 ** rng.integers(-30, 30, 20_000)
    tokens = [spec % v for v in values]
    assert_same_bits(parse_floats(tokens), [float(t) for t in tokens])


def test_floats_match_python_for_random_decimal_strings():
    rng = np.random.default_rng(3)
    tokens = [random_decimal(rng) for _ in range(50_000)]
    assert_same_bits(parse_floats(tokens), [float(t) for t in tokens])


@pytest.mark.parametrize(
    "token",
    [
        "0", "-0", "+0", "0.0", "-0.0", "0e999", "-0e-999", "00000.000",
        ".5", "5.", "-.5e-3", "1E5", "1e+05", "1.e5", "007",
        "9007199254740992", "9007199254740993", "9007199254740995",
        "1e22", "1e23", "4.35679e-22", "123456789012345678901234567890",
        "2.2250738585072011e-308", "2.2250738585072014e-308",
        "4.9e-324", "2.4703282292062327e-324", "2.4703282292062328e-324",
        "1e-400", "1.7976931348623157e308", "1.7976931348623158e308",
        "1.7976931348623159e308", "1e400", "-1e400", "1e100000000000000",
        "0.1", "0.30000000000000004", "3.141592653589793238462643383279",
    ],
)  # fmt: skip
def test_float_edge_cases(token):
    assert_same_bits(parse_floats([token]), [float(token)])


@pytest.mark.parametrize(
    "token", ["inf", "+inf", "-inf", "Infinity", "-INFINITY", "iNf"]
)
def test_float_infinities(token):
    assert parse_floats([token])[0] == float(token)


@pytest.mark.parametrize("token", ["nan", "NaN", "-nan", "+NAN"])
def test_float_nans(token):
    value = parse_floats([token])[0]
    assert math.isnan(value)
    assert math.copysign(1.0, value) == (-1.0 if token.startswith("-") else 1.0)


@pytest.mark.parametrize(
    "token",
    [
        "1e", "e5", ".", "+", "-", "1.2.3", "0x10", "1_000", "1d5", "--1",
        "inf5", "nane", "infinit", "1e5.0", "1e+", "١", "1,5", "+-1",
    ],
)  # fmt: skip
def test_invalid_floats(token):
    with pytest.raises(ValueError) as info:
        parse_floats([token])
    assert info.value.args == (f"cannot read '{token}' as a number", 0)


def test_integers():
    values = [0, -0, 1, -1, 2**63 - 1, -(2**63), 1234567890123]
    tokens = [str(v) for v in values] + ["+5", "007", "-007"]
    text = "\n".join(tokens).encode()
    (result,), _ = native.parse_table(text, 0, len(tokens), [("i", 1)])
    assert result.dtype == np.int64
    assert result.tolist() == values + [5, 7, -7]


@pytest.mark.parametrize(
    "token",
    ["1.0", "1e3", "0x1", "+", "-", "9223372036854775808", "-9223372036854775809",
     "99999999999999999999", "1_0", "--1"],
)  # fmt: skip
def test_invalid_integers(token):
    with pytest.raises(ValueError) as info:
        native.parse_table(token.encode(), 0, 1, [("i", 1)])
    assert info.value.args == (f"cannot read '{token}' as an integer", 0)


def test_booleans():
    tokens = ["T", "F", "True", "False", "true", "false"]
    (values,), _ = native.parse_table(" ".join(tokens).encode(), 0, 1, [("b", 6)])
    assert values.dtype == np.bool_
    assert values.tolist() == [[True, False, True, False, True, False]]


@pytest.mark.parametrize("token", ["t", "1", "yes", "TRUE", "Fa"])
def test_invalid_booleans(token):
    with pytest.raises(ValueError, match="as a boolean"):
        native.parse_table(token.encode(), 0, 1, [("b", 1)])


def test_float_parsing_ignores_the_numeric_locale():
    """GUI toolkits set LC_NUMERIC; numbers must still use a decimal point."""
    saved = locale.setlocale(locale.LC_NUMERIC)
    for name in ("de_DE.UTF-8", "de_DE.utf8", "fr_FR.UTF-8", "fr_FR.utf8"):
        try:
            locale.setlocale(locale.LC_NUMERIC, name)
            break
        except locale.Error:
            continue
    else:
        pytest.skip("no locale with a decimal comma is installed")
    try:
        assert locale.localeconv()["decimal_point"] == ","
        # 17 significant digits take the strtod path
        tokens = ["1.2345678901234567", "0.1", "1.5e-300"]
        assert_same_bits(parse_floats(tokens), [float(t) for t in tokens])
    finally:
        locale.setlocale(locale.LC_NUMERIC, saved)


def test_powers_of_five_table():
    """The table of the Eisel-Lemire algorithm, recomputed from its definition."""
    from mdapy._core import testing

    def reference(q):
        if q >= 0:
            value = 5**q
            while value < 1 << 127:
                value *= 2
            while value >= 1 << 128:
                value //= 2
            return value
        z = (5**-q).bit_length()
        b = z + 127 if q >= -27 else 2 * z + 128
        value = 2**b // 5**-q + 1
        while value >= 1 << 128:
            value //= 2
        return value

    table = testing.powers_of_five()
    assert len(table) == 308 + 342 + 1
    for q, (high, low) in zip(range(-342, 309), table):
        assert (high << 64) | low == reference(q), q


def test_floats_with_19_digits_across_all_exponents():
    """Every entry of the table is used: w * 10^q for q from -360 to 330."""
    rng = random.Random(7)
    tokens = []
    for q in range(-360, 331):
        for _ in range(150):
            digits = rng.randint(1, 19)
            tokens.append(f"{rng.randrange(10 ** (digits - 1), 10**digits)}e{q}")
    assert_same_bits(parse_floats(tokens), [float(t) for t in tokens])


def test_floats_halfway_between_doubles_round_to_even():
    """w is an odd multiple of half a unit in the last place, for every scale."""
    rng = np.random.default_rng(8)
    tokens = []
    for shift in range(0, 11):  # w = (2^53 + odd) * 2^shift / 2 stays below 10^19
        odd = rng.integers(0, 2**52, 500) * 2 + 1
        tokens += [str((2**53 + int(k)) << shift >> 1) for k in odd]
        tokens += [str(((2**53 + int(k)) << shift >> 1) + 1) for k in odd]
    tokens += ["9007199254740993", "9007199254740995", "18014398509481987"]
    assert_same_bits(parse_floats(tokens), [float(t) for t in tokens])


def test_floats_near_the_subnormal_boundary():
    smallest_normal = np.finfo(np.float64).smallest_normal
    values = np.nextafter(smallest_normal, 0) * np.arange(1, 2000) / 1000
    tokens = [f"{v:.18e}" for v in values] + [repr(float(v)) for v in values]
    tokens += ["2.2250738585072013e-308", "2.2250738585072012e-308"]
    assert_same_bits(parse_floats(tokens), [float(t) for t in tokens])


# ----------------------------------------------------------------- shifts


def scaled(token, shift):
    """float(token * 10**shift), computed exactly in decimal."""
    return float(decimal.Decimal(token).scaleb(shift))


@pytest.mark.parametrize("shift", [-3, 3, -17, 22, 100, -100])
def test_parse_with_shift_scales_exactly_in_decimal(shift):
    rng = np.random.default_rng(9)
    tokens = [repr(float(v)) for v in random_doubles(rng, 20_000)]
    tokens += [random_decimal(rng) for _ in range(20_000)]  # some above 19 digits
    tokens += ["0", "-0", "1", "0.001", "1e-320", "1.7976931348623157e308"]
    text = "\n".join(tokens).encode()
    (values,), _ = native.parse_table(text, 0, len(tokens), [("f", 1, shift)])
    assert_same_bits(values, [scaled(t, shift) for t in tokens])


def test_parse_with_shift_keeps_nan_and_infinity():
    (values,), _ = native.parse_table(b"nan -inf", 0, 1, [("f", 2, 3)])
    assert math.isnan(values[0, 0]) and values[0, 1] == -math.inf


@pytest.mark.parametrize("shift", [-3, 3, -20, 20])
def test_format_with_shift_moves_the_decimal_exponent(shift):
    values = random_doubles(np.random.default_rng(10), 50_000)
    values = np.concatenate([values, [0.0, -0.0, math.inf, -math.inf, 1.0, 1e-3]])
    lines = native.format_table([("f", values, [], shift)]).decode().splitlines()
    for text, value in zip(lines, values):
        if value == 0 or not math.isfinite(value):
            assert float(text) == value and text in ("0", "-0", "inf", "-inf")
        else:
            expected = decimal.Decimal(repr(float(value))).scaleb(shift)
            assert decimal.Decimal(text) == expected, (text, value)
    (back,), _ = native.parse_table(
        "\n".join(lines).encode(), 0, len(lines), [("f", 1, -shift)]
    )
    assert_same_bits(back, values)


@pytest.mark.parametrize(
    "value, shift, text",
    [
        (1.0, -3, "0.001"),
        (1000.0, -3, "1"),
        (1234.5, -3, "1.2345"),
        (-2.0, -3, "-0.002"),
        (1.5, -5, "1.5e-05"),
        (1.5, -4, "0.00015"),
        (1.0, 16, "1e+16"),
        (1.0, 15, "1000000000000000"),
        (0.1, 3, "100"),
        (1e-300, -30, "1e-330"),
        (1.25e200, 100, "1.25e+300"),
    ],
)
def test_format_with_shift_uses_python_notation(value, shift, text):
    assert (
        native.format_table([("f", np.array([value]), [], shift)])
        == text.encode() + b"\n"
    )


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: native.parse_table(b"1", 0, 1, [("i", 1, 3)]), "only float columns"),
        (
            lambda: native.parse_table(b"1", 0, 1, [("f", 1, 101)]),
            "between -100 and 100",
        ),
        (lambda: native.parse_table(b"1", 0, 1, [("f",)]), "(kind, width)"),
        (
            lambda: native.format_table([("i", np.array([1]), [], 2)]),
            "only float columns",
        ),
        (lambda: native.format_table([("f", np.array([1.0]))]), "(kind, values"),
    ],
)
def test_invalid_shifts(call, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        call()


# ----------------------------------------------------------------- tables


def test_mixed_columns_with_widths():
    text = b"H 0 0 0 1 T\nO 1.5 -2 3e2 2 F\n"
    (codes, categories), positions, ids, flags = native.parse_table(
        text, 0, 2, [("s", 1), ("f", 3), ("i", 1), ("b", 1)]
    )[0]
    assert codes.dtype == np.uint32 and codes.tolist() == [0, 1]
    assert categories == ["H", "O"]
    assert positions.shape == (2, 3)
    np.testing.assert_array_equal(positions, [[0, 0, 0], [1.5, -2, 300]])
    assert ids.tolist() == [1, 2] and flags.tolist() == [True, False]


def test_skipped_columns_must_exist_but_are_not_returned():
    arrays, _ = native.parse_table(b"1 a 2\n", 0, 1, [("i", 1), ("x", 1), ("i", 1)])
    assert arrays[1] is None
    with pytest.raises(ValueError, match="expected 3 values, found 2"):
        native.parse_table(b"1 2\n", 0, 1, [("i", 1), ("x", 1), ("i", 1)])


def test_blanks_line_endings_and_offsets():
    text = b"header\n\t1  2\r\n3\v4\f \n5 6"
    (values,), end = native.parse_table(text, 7, 3, [("i", 2)])
    assert values.tolist() == [[1, 2], [3, 4], [5, 6]]
    assert end == len(text)
    (values,), end = native.parse_table(text, 7, 1, [("i", 2)])
    assert end == text.index(b"3")


def test_comments_are_ignored_to_the_end_of_the_line():
    text = b"1 2 # a b c\n3 4#x\n# 5 6\n"
    with pytest.raises(ValueError) as info:
        native.parse_table(text, 0, 3, [("i", 2)], "#")
    assert info.value.args == ("expected 2 values, found 0", 2)
    (values,), _ = native.parse_table(text, 0, 2, [("i", 2)], "#")
    assert values.tolist() == [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="expected 2 values, found 6"):
        native.parse_table(text, 0, 1, [("i", 2)])


@pytest.mark.parametrize(
    "line, message",
    [
        (b"1 2 3", "expected 2 values, found 3"),
        (b"1", "expected 2 values, found 1"),
        (b"", "expected 2 values, found 0"),
        (b"1 abc", "cannot read 'abc' as a number"),
        (b"1 " + b"9" * 50 + b"x", "cannot read '" + "9" * 40 + "...' as a number"),
    ],
)
def test_bad_lines_report_the_message_and_row(line, message):
    text = b"0 0\n" * 3 + line + b"\n0 0\n"
    with pytest.raises(ValueError) as info:
        native.parse_table(text, 0, 5, [("f", 2)])
    assert info.value.args == (message, 3)


def test_the_first_bad_row_is_reported_whatever_the_thread_count():
    rows = 20_000
    lines = [b"1 2"] * rows
    for row in (19_000, 7_000, 7_001, 6_999):
        lines[row] = b"1 x"
    text = b"\n".join(lines)
    for threads in (1, 4):
        with mdapy.num_threads(threads):
            with pytest.raises(ValueError) as info:
                native.parse_table(text, 0, rows, [("i", 2)])
        assert info.value.args[1] == 6_999


def test_too_few_lines():
    with pytest.raises(ValueError) as info:
        native.parse_table(b"1\n2\n", 0, 3, [("i", 1)])
    assert info.value.args == ("expected 3 lines, but the text ends after 2", 2)
    (values,), end = native.parse_table(b"", 0, 0, [("i", 1)])
    assert values.shape == (0,) and end == 0


def test_strings_keep_the_order_of_first_appearance_for_any_thread_count():
    rng = np.random.default_rng(4)
    words = ["Cu", "α-β", "Zr", "H2O", "x" * 30]
    rows = 50_000
    picks = rng.integers(0, len(words), size=(rows, 2))
    text = "\n".join(f"{words[a]} {words[b]}" for a, b in picks).encode()
    results = []
    for threads in (1, 3, 8):
        with mdapy.num_threads(threads):
            ((codes, categories),), _ = native.parse_table(text, 0, rows, [("s", 2)])
        results.append((codes, categories))
        order = list(dict.fromkeys(words[i] for i in picks.ravel()))
        assert categories == order
        assert [categories[c] for c in codes.ravel()[:100]] == [
            words[i] for i in picks.ravel()[:100]
        ]
    for codes, categories in results[1:]:
        np.testing.assert_array_equal(codes, results[0][0])


@pytest.mark.parametrize(
    "make", [bytes, bytearray, memoryview], ids=["bytes", "bytearray", "memoryview"]
)
def test_buffer_types(make):
    (values,), _ = native.parse_table(make(b"1 2\n"), 0, 1, [("i", 2)])
    assert values.tolist() == [[1, 2]]


def test_mmap_buffer(tmp_path):
    path = tmp_path / "t.txt"
    path.write_bytes(b"1.5 2.5\n")
    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        (values,), end = native.parse_table(m, 0, 1, [("f", 2)])
        assert values.tolist() == [[1.5, 2.5]] and end == 8


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"columns": [("q", 1)]}, "unknown column kind 'q'"),
        ({"columns": [("f", 0)]}, "width must be at least 1"),
        ({"rows": -1}, "rows must be at least 0"),
        ({"offset": 9}, "offset 9 is outside the text of 4 bytes"),
        ({"offset": -1}, "outside the text"),
        ({"comment": "##"}, "one character"),
    ],
)
def test_invalid_arguments(kwargs, message):
    arguments = {"text": b"1 2\n", "offset": 0, "rows": 1, "columns": [("i", 2)]}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        native.parse_table(**arguments)


def test_non_buffer_text_is_rejected():
    with pytest.raises(TypeError):
        native.parse_table("1 2", 0, 1, [("i", 2)])


# ---------------------------------------------------------- tables of many lines


def many_lines(rows=120000, seed=5):
    """A table larger than the blocks it is read in, and its columns."""
    rng = np.random.default_rng(seed)
    columns = [
        ("i", np.arange(rows)),
        ("s", (rng.integers(0, 3, rows).astype(np.uint32), ["Cu", "Zr", "Al"])),
        ("f", rng.normal(0.0, 100.0, (rows, 3))),
    ]
    return b"".join(format_rows(columns)), columns


SPEC = [("i", 1), ("s", 1), ("f", 3)]


def assert_table_read(arrays, columns):
    np.testing.assert_array_equal(arrays[0], columns[0][1])
    codes, categories = arrays[1]
    expected, names = columns[1][1]
    assert [categories[c] for c in codes[:50]] == [names[c] for c in expected[:50]]
    assert_same_bits(arrays[2], columns[2][1])


@pytest.mark.parametrize("threads", [1, 4])
def test_a_table_larger_than_a_block_stops_at_its_end(threads):
    text, columns = many_lines()
    before = get_num_threads()
    try:
        set_num_threads(threads)
        # what follows the table, as the next frame of a trajectory, stays unread
        arrays, end = native.parse_table(text + b"ITEM: TIMESTEP\n0\n", 0, 120000, SPEC)
        assert end == len(text)
        assert_table_read(arrays, columns)
        assert native.skip_lines(text + b"more\n", 0, 120000) == len(text)
        offset = text.index(b"\n") + 1
        assert native.skip_lines(text, offset, 119999) == len(text)
        assert native.skip_lines(text, 0, -1) == 0
    finally:
        set_num_threads(before)


def test_a_large_table_without_a_last_newline():
    text, columns = many_lines()
    arrays, end = native.parse_table(text[:-1], 0, 120000, SPEC)
    assert end == len(text) - 1
    assert_table_read(arrays, columns)
    assert native.skip_lines(text[:-1], 0, 120000) == len(text) - 1


def test_a_large_table_that_ends_too_early():
    text, _ = many_lines()
    for missing, cut in ((5, text), (6, text[:-1] + b" ")):
        rows = 120000 + missing
        with pytest.raises(ValueError) as info:
            native.parse_table(cut, 0, rows, SPEC)
        assert info.value.args == (
            f"expected {rows} lines, but the text ends after 120000",
            120000,
        )
    with pytest.raises(ValueError, match="the text ends after 120000"):
        native.skip_lines(text, 0, 120001)
    with pytest.raises(ValueError, match="the text ends after 1200"):
        native.skip_lines(text[: text.index(b"1200 ")], 0, 120000)


def test_lines_longer_than_a_block():
    long = b"7 Cu 1.5" + b" " * 2**19 + b"2.5 3.5\n"
    text, _ = many_lines(rows=2000)
    lines = text.splitlines(keepends=True)
    # at the start, among the rows, and at the end
    table = long + b"".join(lines[:1000]) + long * 3 + b"".join(lines[1000:]) + long
    arrays, end = native.parse_table(table, 0, 2005, SPEC)
    assert end == len(table)
    np.testing.assert_array_equal(arrays[0][:2], [7, 0])
    np.testing.assert_array_equal(arrays[0][1001:1005], [7, 7, 7, 1000])
    np.testing.assert_array_equal(arrays[2][0], [1.5, 2.5, 3.5])
    assert native.skip_lines(table, 0, 2005) == len(table)
    with pytest.raises(ValueError, match="the text ends after 2005"):
        native.skip_lines(table, 0, 2006)


def test_a_table_with_longer_lines_than_its_first_ones():
    # the first lines tell how far to look, here far too short a way
    first = b"".join(b"%d 1\n" % i for i in range(80000))
    last = b"".join(b"%d 1 %s\n" % (i, b" " * 90) for i in range(80000, 90000))
    text = first + last + b"rest\n"
    (ids, ones), end = native.parse_table(text, 0, 90000, [("i", 1), ("i", 1)])
    assert end == len(first + last)
    np.testing.assert_array_equal(ids, np.arange(90000))
    assert (ones == 1).all()


@pytest.mark.parametrize("threads", [1, 4])
def test_comments_and_line_endings_in_a_large_table(threads):
    rows = 60000
    lines = [
        b"%d 1.5 %s# %d %s"
        % (i, b" " if i % 2 else b"", i, b"\r\n" if i % 3 else b"\n")
        for i in range(rows)
    ]
    text = b"".join(lines) + b"more\n"
    before = get_num_threads()
    try:
        set_num_threads(threads)
        (ids, values), end = native.parse_table(
            text, 0, rows, [("i", 1), ("f", 1)], "#"
        )
    finally:
        set_num_threads(before)
    assert end == len(text) - 5
    np.testing.assert_array_equal(ids, np.arange(rows))
    assert (values == 1.5).all()


@pytest.mark.parametrize(
    "line, message",
    [
        (b"5 Cu 1.0 2.0", "expected 5 values, found 4"),
        (b"5 Cu 1.0 2.0 3.0 4.0", "expected 5 values, found 6"),
        (
            b"x Cu 1.0 2.0 3.0 4.0",
            "expected 5 values, found 6",
        ),  # the count comes first
        (b"5 Cu 1.0 x 3.0", "cannot read 'x' as a number"),
        (b"5.5 Cu 1.0 2.0 3.0", "cannot read '5.5' as an integer"),
        (b"", "expected 5 values, found 0"),
    ],
)
def test_a_bad_line_far_into_a_large_table(line, message):
    text, _ = many_lines()
    lines = text.splitlines(keepends=True)
    lines[90000] = line + b"\n"
    lines[100000] = b"bad\n"
    with pytest.raises(ValueError) as info:
        native.parse_table(b"".join(lines), 0, 120000, SPEC)
    assert info.value.args == (message, 90000)


def test_many_distinct_strings():
    rows = 3000
    names = [f"name{i % 500}" for i in range(rows)]
    text = "".join(f"{n}\n" for n in names).encode()
    ((codes, categories),), _ = native.parse_table(text, 0, rows, [("s", 1)])
    assert categories == [f"name{i}" for i in range(500)]
    assert [categories[c] for c in codes] == names


@pytest.mark.parametrize(
    "token",
    [
        # around each eight digits read at once, with and without a whole part
        *[f"0.{'1234567890' * 3}"[: 2 + n] for n in range(1, 30)],
        *[f"7.{'9' * n}" for n in range(1, 30)],
        *[f"-{'3' * n}.5" for n in range(1, 25)],
        *[f"{'9' * n}" for n in range(1, 25)],
        # leading zeros do not count among the digits a mantissa holds
        "0.000000000000000000000012345",
        "00000000000000000000000012345.5",
        "0.00000000000000000000000000",
        "-0.00000000000000000000000000",
        ".5",
        "5.",
        "1234567.e3",
        "12345678.12345678e-7",
        "12345678.123456789012e+12",
        # exponents, the longest refused by the short way
        "1e5",
        "1E5",
        "1e+5",
        "1e-5",
        "1e000005",
        "1e999999",
        "1e-999999",
        "1e9999999",
        "1e-9999999",
        "0e9999999",
        "1.5e308",
        "1.5e309",
        "4.9e-324",
        "2e-324",
        "9007199254740992",
        "9007199254740993",
        "0.1",
        "-0.1",
    ],
)
def test_floats_as_programs_write_them(token):
    expected = float(token)
    # with a newline, without one at the end of the text, and before more text
    for text in (token + "\n", token, f"{token} 1\n", f"1 {token}\n"):
        words = text.split()
        spec = [("f", len(words))]
        (values,), _ = native.parse_table(text.encode(), 0, 1, spec)
        assert_same_bits(
            values[0, words.index(token)] if len(words) > 1 else values[0], expected
        )


@pytest.mark.parametrize(
    "token",
    ["0", "-0", "7", "-7", "9" * 18, "-" + "9" * 18, "9" * 19, "-" + "9" * 18 + "0"],
)
def test_integers_as_programs_write_them(token):
    if abs(int(token)) >= 2**63:
        with pytest.raises(ValueError, match="as an integer"):
            native.parse_table(token.encode(), 0, 1, [("i", 1)])
        return
    (values,), _ = native.parse_table(token.encode(), 0, 1, [("i", 1)])
    assert values[0] == int(token)


def test_skip_lines():
    text = b"a\nb\r\nc"
    assert native.skip_lines(text, 0, 0) == 0
    assert native.skip_lines(text, 0, 2) == 5
    assert native.skip_lines(text, 0, 3) == len(text)
    with pytest.raises(ValueError) as info:
        native.skip_lines(text, 2, 3)
    assert info.value.args == ("expected 3 lines, but the text ends after 2", 2)
    with pytest.raises(ValueError, match="outside the text"):
        native.skip_lines(text, 99, 1)


# ------------------------------------------------------------- formatting


def near_powers(powers, rng, n):
    """Doubles a few steps from powers."""
    bits = np.asarray(powers, dtype=np.float64).view(np.int64)
    return (bits + rng.integers(-3, 4, n)).view(np.float64)


def float_cases(rng, n):
    bits = rng.integers(0, 2**64, size=n, dtype=np.uint64).view(np.float64)
    tens = 10.0 ** rng.integers(-300, 300, n)
    exponents = rng.integers(0, 2047, n).astype(np.int64)
    fractions = rng.choice([0, 1, 2, 2**52 - 1, 2**52 - 2], n)
    twos = ((exponents << 52) | fractions).view(np.float64)
    digits = rng.integers(0, 10**15, n).astype(np.float64)
    return {
        "all bits": bits[np.isfinite(bits)],
        "positions": rng.uniform(-1000.0, 1000.0, n),
        "velocities": rng.normal(0.0, 1e-3, n),
        "crystals": rng.integers(0, 100000, n) * 3.615 / rng.integers(1, 9, n),
        "whole numbers": rng.integers(-(10**16), 10**16, n).astype(np.float64),
        "round numbers": rng.integers(1, 100, n) * 10.0 ** rng.integers(-8, 18, n),
        "short decimals": digits * 10.0 ** rng.integers(-30, 10, n),
        "near powers of ten": near_powers(tens, rng, n),
        "near powers of two": twos,
        "at the small end": rng.uniform(1.0, 10.0, n)
        * 10.0 ** rng.integers(-323, -280, n),
        "at the large end": rng.uniform(1.0, 10.0, n) * 10.0 ** rng.integers(14, 17, n),
    }


@pytest.mark.parametrize("case", list(float_cases(np.random.default_rng(0), 1)))
def test_floats_are_written_as_std_to_chars_writes_them(case):
    from mdapy._core import testing

    values = float_cases(np.random.default_rng(1), 200000)[case]
    values = values[np.isfinite(values)]
    text = native.format_table([("f", values, [])])
    assert text == testing.to_chars_text(values)
    # the signs of the same numbers, and they read back
    assert native.format_table([("f", -values, [])]) == testing.to_chars_text(-values)
    (back,), _ = native.parse_table(text, 0, len(values), [("f", 1)])
    assert_same_bits(back, values)


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -0.0,
        float("inf"),
        -float("inf"),
        1.0,
        -1.5,
        100000.0,
        10000.0,
        0.001,
        0.0001,
        0.00012,
        1.2e7,
        123456.0,
        1234567.0,
        999999999999999.9,
        1e15,
        1e16,
        5e-324,
        2.2250738585072014e-308,
        1.7976931348623157e308,
        -1.7976931348623157e308,
        2.0**53,
        2.0**53 + 2,
        1.2345678901234567e17,
    ],
)
def test_floats_of_every_notation(value):
    from mdapy._core import testing

    values = np.array([value])
    text = native.format_table([("f", values, [])])
    assert text == testing.to_chars_text(values)
    if not math.isinf(value):
        assert float(text) == value


def test_nan_is_written_as_nan():
    assert b"nan" in native.format_table([("f", np.array([float("nan")]), [])])


def test_every_nan_is_written_as_python_writes_it():
    # a NaN with the sign bit, as x86 computes 0/0, and quiet and signaling NaNs
    # with payloads: std::to_chars writes -nan(ind) or nan(snan) for some of them
    bits = np.array(
        [
            0xFFF8000000000000,
            0x7FF8000000000001,
            0x7FF0000000000001,
            0xFFF4000000000123,
        ],
        dtype=np.uint64,
    )
    values = np.concatenate([bits.view(np.float64), [math.inf, -math.inf]])
    for shift in (0, 3):
        text = native.format_table([("f", values, [], shift)]).decode()
        assert text.split() == ["nan"] * 4 + ["inf", "-inf"]
        (back,), _ = native.parse_table(
            text.encode(), 0, len(values), [("f", 1, -shift)]
        )
        assert np.isnan(back[:4]).all()
        assert list(back[4:]) == [math.inf, -math.inf]


def longest_rows(rows):
    """Every value as long as it can be, in columns that fill any place they get."""
    return [
        ("f", np.full((rows, 3), -1.7976931348623157e308)),
        ("f", np.full(rows, -1.2345678901234567e-300), 100),
        ("f", np.full(rows, -1.2345678901234567e-300), -100),
        ("i", np.full(rows, -(2**63), dtype=np.int64)),
        ("b", np.ones((rows, 2), dtype=bool)),
        ("s", (np.ones(rows, dtype=np.uint32), ["a", "b" * 300])),
    ]


SMALL = repr(1.2345678901234567e-300).split("e")[0]
LONGEST_ROW = (
    b"-1.7976931348623157e+308 -1.7976931348623157e+308 -1.7976931348623157e+308 "
    + f"-{SMALL}e-200 -{SMALL}e-400 ".encode()
    + b"-9223372036854775808 T T "
    + b"b" * 300
    + b"\n"
)


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("rows", [1, 100, 5000])
def test_the_longest_values_fit(threads, rows):
    before = get_num_threads()
    try:
        set_num_threads(threads)
        assert b"".join(format_rows(longest_rows(rows))) == LONGEST_ROW * rows
    finally:
        set_num_threads(before)


def table_in_many_batches(rows=500000):
    rng = np.random.default_rng(6)
    return [
        ("i", np.arange(rows)),
        ("s", (rng.integers(0, 2, rows).astype(np.uint32), ["Cu", "Zr"])),
        ("f", rng.normal(0.0, 100.0, (rows, 3))),
        ("f", rng.normal(0.0, 1.0, rows), -3),
    ]


@pytest.mark.parametrize("threads", [1, 4])
def test_a_file_takes_rows_straight_and_in_order(tmp_path, threads):
    columns = table_in_many_batches()
    path = tmp_path / "table.txt"
    before = get_num_threads()
    try:
        set_num_threads(threads)
        expected = b"".join(format_rows(columns))
        with open(path, "wb") as file:
            file.write(b"header\n")
            write_rows(file, columns)
            assert file.tell() == len(b"header\n") + len(expected)
            file.write(b"middle\n")
            write_rows(file, [("i", np.arange(2))])
            write_rows(file, [])
            file.write(b"footer\n")
    finally:
        set_num_threads(before)
    assert path.read_bytes() == b"header\n" + expected + b"middle\n0\n1\nfooter\n"


def test_a_table_is_the_same_for_any_number_of_threads():
    columns = table_in_many_batches()
    before = get_num_threads()
    try:
        set_num_threads(1)
        one = b"".join(format_rows(columns))
        set_num_threads(4)
        four = b"".join(format_rows(columns))
    finally:
        set_num_threads(before)
    assert one == four


def test_a_compressed_file_gets_compressed_text(tmp_path):
    # such a file has a descriptor too, of the bytes it compresses
    columns = table_in_many_batches(1000)
    path = tmp_path / "table.txt.gz"
    with gzip.open(path, "wb") as file:
        file.write(b"header\n")
        write_rows(file, columns)
    with gzip.open(path, "rb") as file:
        assert file.read() == b"header\n" + b"".join(format_rows(columns))


@pytest.mark.parametrize("rows", [3, 500000])
def test_a_file_that_cannot_be_written_raises(tmp_path, rows):
    path = tmp_path / "table.txt"
    path.write_bytes(b"")
    descriptor = os.open(path, os.O_RDONLY)
    try:
        with pytest.raises(OSError) as info:
            native.write_table(descriptor, [("i", np.arange(rows), [])])
        assert info.value.errno == errno.EBADF
    finally:
        os.close(descriptor)
    assert path.read_bytes() == b""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="no /dev/full")
def test_a_full_disk_raises():
    with open("/dev/full", "wb") as file:
        with pytest.raises(OSError) as info:
            write_rows(file, [("i", np.arange(3))])
    assert info.value.errno == errno.ENOSPC


def test_a_table_of_rows_without_values():
    assert native.format_table([("f", np.empty((3, 0)), [])]) == b"\n\n\n"
    assert native.format_table([("s", np.empty(0, dtype=np.uint32), [])]) == b""


def test_formatted_floats_read_back_exactly():
    values = random_doubles(np.random.default_rng(5), 200_000)
    text = native.format_table([("f", values, [])])
    lines = text.decode().splitlines()
    assert len(lines) == len(values)
    assert_same_bits([float(t) for t in lines], values)
    assert_same_bits(parse_floats(lines), values)
    # shortest representation: never longer than Python's repr
    assert all(len(t) <= len(repr(float(v))) for t, v in zip(lines[:2000], values))


def test_formatted_values():
    columns = [
        ("f", np.array([[1.5, -0.0, 1e20], [0.1, 100.0, 1e-7]]), []),
        ("i", np.array([2**63 - 1, -(2**63)], dtype=np.int64), []),
        ("b", np.array([True, False]), []),
        ("s", np.array([1, 0], dtype=np.uint32), ["O", "H"]),
    ]
    text = native.format_table(columns)
    assert text == (
        b"1.5 -0 1e+20 9223372036854775807 T H\n"
        b"0.1 100 1e-07 -9223372036854775808 F O\n"
    )


def test_format_empty_table():
    assert native.format_table([]) == b""
    assert native.format_table([("f", np.empty(0), [])]) == b""


@pytest.mark.parametrize(
    "columns, message",
    [
        ([("f", np.array([1], dtype=np.int64), [])], "kind 'f' has the wrong dtype"),
        ([("i", np.array([1.0]), [])], "kind 'i' has the wrong dtype"),
        ([("x", np.array([1.0]), [])], "kind 'x' has the wrong dtype"),
        ([("s", np.array([2], dtype=np.uint32), ["a", "b"])], "code 2 has no category"),
        ([("f", np.zeros((1, 1, 1)), [])], "1- or 2-dimensional"),
        ([("f", np.zeros(2), []), ("f", np.zeros(3), [])], "different numbers of rows"),
    ],
)
def test_invalid_format_arguments(columns, message):
    with pytest.raises(ValueError, match=message):
        native.format_table(columns)


def test_parse_and_format_are_inverse_for_mixed_tables():
    rng = np.random.default_rng(6)
    rows = 3000
    columns = [
        ("s", (rng.integers(0, 3, rows).astype(np.uint32), ["Cu", "Zr", "Al"])),
        ("f", random_doubles(rng, 4 * rows)[: 3 * rows].reshape(rows, 3)),
        ("i", rng.integers(-(2**62), 2**62, rows)),
        ("b", rng.random((rows, 2)) < 0.5),
    ]
    text = b"".join(format_rows(columns))
    arrays, end = native.parse_table(
        text, 0, rows, [("s", 1), ("f", 3), ("i", 1), ("b", 2)]
    )
    assert end == len(text)
    codes, categories = arrays[0]
    assert [categories[c] for c in codes] == [
        columns[0][1][1][c] for c in columns[0][1][0]
    ]
    assert_same_bits(arrays[1], columns[1][1])
    np.testing.assert_array_equal(arrays[2], columns[2][1])
    np.testing.assert_array_equal(arrays[3], columns[3][1])


# ------------------------------------------------------- Python helpers


def test_format_rows_writes_in_blocks(monkeypatch):
    monkeypatch.setattr(_text, "WRITE_BLOCK_ROWS", 4)
    blocks = list(format_rows([("i", np.arange(10))]))
    assert [b.count(b"\n") for b in blocks] == [4, 4, 2]
    file = io.BytesIO()
    write_rows(
        file, [("i", np.arange(10)), ("f", np.arange(10) / 2), ("f", np.ones(10), -3)]
    )
    assert file.getvalue().splitlines()[3] == b"3 1.5 0.001"
    assert list(format_rows([])) == []


def test_strings_and_categorize_are_inverse():
    series = pl.Series("element", ["Cu", "Zr", "Cu", "Al"])
    codes, categories = categorize(series)
    assert categories == ["Cu", "Zr", "Al"] and codes.tolist() == [0, 1, 0, 2]
    assert strings(codes, categories, "element").equals(series)
    elements = strings(codes, categories, "element", ELEMENT_DTYPE)
    assert elements.dtype == ELEMENT_DTYPE
    codes, categories = categorize(elements)
    assert codes.tolist() == [29, 40, 29, 13] and categories[29] == "Cu"


def test_text_reader_lines_and_tables():
    text = b"title\r\n2\n1 2\n3 4\nend"
    reader = TextReader(text, "a.txt")
    assert reader.readline() == "title" and reader.line == 1
    assert int(reader.readline()) == 2
    (values,) = reader.table(2, [("f", 2, 3)])
    assert values.tolist() == [[1000, 2000], [3000, 4000]] and reader.line == 4
    assert not reader.at_end()
    assert reader.readline() == "end" and reader.at_end()
    with pytest.raises(FileFormatError, match=r"^a\.txt:6: unexpected end of file$"):
        reader.readline()


def test_text_reader_errors_name_the_file_and_line():
    reader = TextReader(b"a\nb\n1 2\n1 x\n", "a.txt")
    reader.skip(2)
    assert reader.line == 2
    with pytest.raises(FileFormatError, match=r"^a\.txt:4: cannot read 'x'") as info:
        reader.table(2, [("f", 2)])
    assert (info.value.path, info.value.line) == ("a.txt", 4)
    with pytest.raises(FileFormatError, match=r"^a\.txt:5: expected 3 lines"):
        TextReader(b"a\nb\n", "a.txt", line=2).skip(3)
    assert str(TextReader(b"", "f").error("bad", line=7)) == "f:7: bad"


def test_text_reader_rejects_non_utf8_lines():
    reader = TextReader(b"ok\n\xff\xfe\n", "a.txt")
    reader.readline()
    with pytest.raises(FileFormatError, match=r"^a\.txt:2: the line is not UTF-8"):
        reader.readline()


def test_text_reader_passes_other_value_errors_through():
    with pytest.raises(ValueError, match="unknown column kind") as info:
        TextReader(b"1\n", "a.txt").table(1, [("q", 1)])
    assert not isinstance(info.value, FileFormatError)


def test_a_last_line_without_a_line_ending_and_lines_to_skip():
    reader = TextReader(b"2", "a.txt")
    assert reader.readline() == "2"
    with pytest.raises(FileFormatError, match="a.txt"):
        reader.skip(2)


@pytest.mark.parametrize(
    "text, columns",
    [
        (b"\xff\xfe 1.0\n", [("s", 1), ("f", 1)]),  # a value that is not UTF-8
        (b"1 \xff\xfe\n", [("i", 1), ("i", 1)]),  # and the message quotes it
    ],
)
def test_bytes_that_are_not_text_in_a_table(text, columns):
    with pytest.raises(FileFormatError, match="a.txt"):
        TextReader(text, "a.txt").table(1, columns)


def test_more_rows_than_the_file_can_hold_are_not_allocated():
    with pytest.raises(FileFormatError, match="unexpected end of file"):
        TextReader(b"1 2\n", "a.txt").table(4_294_967_296, [("f", 2)])
