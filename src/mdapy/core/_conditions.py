# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""Conditions and sort keys on the columns of a system, written as Python
expressions and turned into polars expressions."""

import ast
import io
import operator
import tokenize
from collections.abc import Mapping
from functools import reduce

import polars as pl

from mdapy.errors import InvalidParameter

# names of the components of array columns by their number; any component
# is also named by its index from 0
COMPONENT_NAMES = {
    3: ("x", "y", "z"),
    4: ("x", "y", "z", "w"),
    9: ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"),
}

FUNCTIONS = {
    "abs": pl.Expr.abs,
    "sqrt": pl.Expr.sqrt,
    "exp": pl.Expr.exp,
    "log": pl.Expr.log,
    "log10": pl.Expr.log10,
    "sin": pl.Expr.sin,
    "cos": pl.Expr.cos,
    "tan": pl.Expr.tan,
    "arcsin": pl.Expr.arcsin,
    "arccos": pl.Expr.arccos,
    "arctan": pl.Expr.arctan,
    "floor": pl.Expr.floor,
    "ceil": pl.Expr.ceil,
    "round": pl.Expr.round,
}

_ARITHMETIC = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
}

_COMPARISONS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

# a name given in backticks, and a component given by its index, stand in
# the Python source for what Python cannot read
_QUOTED = "__quoted_{}__"
_INDEX = "_"


def components(dtype: pl.DataType) -> dict[str, int]:
    """The components of an array column by name and by index."""
    width = dtype.size
    found = {str(k): k for k in range(width)}
    found.update({name: k for k, name in enumerate(COMPONENT_NAMES.get(width, ()))})
    return found


def column(
    name: str, component: str | None, schema: Mapping[str, pl.DataType]
) -> pl.Expr:
    """The column name, or its component, as a polars expression."""
    if name not in schema:
        raise InvalidParameter(
            f"no column {name!r}; the columns are {', '.join(schema)}"
        )
    dtype = schema[name]
    if isinstance(dtype, pl.Array):
        names = components(dtype)
        listed = ", ".join(f"{name}.{c}" for c in names)
        if component is None:
            raise InvalidParameter(
                f"{name} has {dtype.size} components; write one of {listed}"
            )
        if component not in names:
            raise InvalidParameter(
                f"{name} has no component {component!r}; write one of {listed}"
            )
        return pl.col(name).arr.get(names[component])
    if component is not None:
        raise InvalidParameter(f"{name} has no components")
    return pl.col(name)


def key(text: str, schema: Mapping[str, pl.DataType]) -> pl.Expr:
    """A sort key: a column, or a component written column.component."""
    if text in schema or "." not in text:
        return column(text, None, schema)
    name, component = text.rsplit(".", 1)
    return column(name, component, schema)


def condition(text: str, schema: Mapping[str, pl.DataType]) -> pl.Expr:
    """The polars expression of a condition written as a Python expression
    over the columns, such as ``"positions.x > 5 and element == 'Cu'"``."""
    source, quoted = _prepared(text)
    try:
        tree = ast.parse(source.strip(), mode="eval")
    except SyntaxError as exc:
        raise InvalidParameter(
            f"cannot read the condition {text!r}: {exc.msg}"
        ) from None
    found = _Translation(text, schema, quoted).visit(tree.body)
    return found if isinstance(found, pl.Expr) else pl.lit(found)


def _prepared(text: str) -> tuple[str, dict[str, str]]:
    """The condition as Python source: names in backticks replaced by
    placeholders, and components by index, as in ``group.1``, written as
    attributes ``group._1``; and the names of the placeholders."""
    quoted: dict[str, str] = {}
    parts, quote, k = [], None, 0
    while k < len(text):
        char = text[k]
        if quote is None and char == "`":
            end = text.find("`", k + 1)
            if end < 0:
                raise InvalidParameter(f"a backtick is not closed in {text!r}")
            placeholder = _QUOTED.format(len(quoted))
            quoted[placeholder] = text[k + 1 : end]
            parts.append(placeholder)
            k = end + 1
            continue
        if char in "'\"":
            quote = None if quote == char else (char if quote is None else quote)
        parts.append(char)
        k += 1
    source = "".join(parts)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError) as exc:
        raise InvalidParameter(f"cannot read the condition {text!r}: {exc}") from None
    starts = [0]
    for line in source.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    edits = []
    for before, token in zip(tokens, tokens[1:]):
        # a name followed at once by a number such as .1
        if (
            before.type == tokenize.NAME
            and token.type == tokenize.NUMBER
            and before.end == token.start
            and token.string[0] == "."
            and token.string[1:].isdigit()
        ):
            at = starts[token.start[0] - 1] + token.start[1]
            edits.append((at, at + len(token.string), "." + _INDEX + token.string[1:]))
    for begin, end, replacement in reversed(edits):
        source = source[:begin] + replacement + source[end:]
    return source, quoted


class _Translation(ast.NodeVisitor):
    """Python syntax tree of a condition to a polars expression."""

    def __init__(self, text: str, schema: Mapping[str, pl.DataType], quoted: dict):
        self.__text = text
        self.__schema = schema
        self.__quoted = quoted

    def __refuse(self, what: str):
        raise InvalidParameter(f"{what} in the condition {self.__text!r}")

    def __name(self, node: ast.Name) -> str:
        return self.__quoted.get(node.id, node.id)

    def generic_visit(self, node):
        self.__refuse(f"{type(node).__name__} is not supported")

    def visit_BoolOp(self, node):
        values = [self.visit(value) for value in node.values]
        combine = operator.and_ if isinstance(node.op, ast.And) else operator.or_
        return reduce(combine, values)

    def visit_UnaryOp(self, node):
        value = self.visit(node.operand)
        if isinstance(node.op, (ast.Not, ast.Invert)):
            return ~value if isinstance(value, pl.Expr) else not value
        return -value if isinstance(node.op, ast.USub) else value

    def visit_BinOp(self, node):
        op = _ARITHMETIC.get(type(node.op))
        if op is None:
            self.__refuse(f"the operator {type(node.op).__name__} is not supported")
        return op(self.visit(node.left), self.visit(node.right))

    def visit_Compare(self, node):
        parts, left = [], self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)):
                if not isinstance(comparator, (ast.List, ast.Tuple, ast.Set)):
                    self.__refuse("'in' needs a list such as ['Cu', 'Ni']")
                values = [self.visit(item) for item in comparator.elts]
                if not isinstance(left, pl.Expr):
                    self.__refuse("'in' needs a column on its left")
                found = left.is_in(values)
                parts.append(~found if isinstance(op, ast.NotIn) else found)
                continue
            compare = _COMPARISONS.get(type(op))
            if compare is None:
                self.__refuse(f"the comparison {type(op).__name__} is not supported")
            right = self.visit(comparator)
            parts.append(compare(left, right))
            left = right
        return reduce(operator.and_, parts)

    def visit_Call(self, node):
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id not in FUNCTIONS
            or len(node.args) != 1
            or node.keywords
        ):
            self.__refuse(
                "only these functions of one value are supported: "
                + ", ".join(FUNCTIONS)
            )
        value = self.visit(node.args[0])
        if not isinstance(value, pl.Expr):
            value = pl.lit(value)
        return FUNCTIONS[node.func.id](value)

    def visit_Attribute(self, node):
        if not isinstance(node.value, ast.Name):
            self.__refuse("a component belongs to a column, as in positions.x")
        component = node.attr
        if component.startswith(_INDEX) and component[len(_INDEX) :].isdigit():
            component = component[len(_INDEX) :]
        return column(self.__name(node.value), component, self.__schema)

    def visit_Name(self, node):
        return column(self.__name(node), None, self.__schema)

    def visit_Constant(self, node):
        return node.value
