# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import polars as pl
import pytest

from mdapy import Box, System
from mdapy.errors import InvalidParameter


def atoms():
    """Six atoms with columns of every kind of component."""
    rng = np.random.default_rng(3)
    positions = rng.uniform(-5, 5, (6, 3))
    return System(
        positions,
        Box(20.0),
        elements=["Cu", "Ni", "Cu", "O", "H", "Cu"],
    ).with_columns(
        velocities=rng.normal(size=(6, 3)),
        orientation=rng.normal(size=(6, 4)),
        stress=rng.normal(size=(6, 6)),
        atomic_virials=rng.normal(size=(6, 9)),
        group=pl.Series(rng.integers(0, 2, (6, 2)), dtype=pl.Array(pl.Int64, 2)),
        charge=np.array([0.5, -0.5, 1.0, -2.0, 0.25, 0.0]),
        **{"Shear Strain": np.linspace(0.0, 0.5, 6)},
    )


def kept(system, condition):
    """The indices of the atoms that meet a condition."""
    index = system.with_columns(index=np.arange(len(system)))
    return index.filter(condition).table["index"].to_list()


def where(mask):
    return np.flatnonzero(mask).tolist()


def test_comparisons_logic_and_arithmetic():
    s = atoms()
    r, v, q = (
        s.positions,
        s.table["velocities"].to_numpy(),
        s.table["charge"].to_numpy(),
    )
    cases = {
        "charge > 0": q > 0,
        "charge >= 0.5": q >= 0.5,
        "charge < 0": q < 0,
        "charge <= 0": q <= 0,
        "charge == 1": q == 1,
        "charge != 0": q != 0,
        "-1 < charge < 0.75": (-1 < q) & (q < 0.75),
        "0 < charge and positions.x > 0": (q > 0) & (r[:, 0] > 0),
        "charge > 0.9 or charge < -1": (q > 0.9) | (q < -1),
        "(charge > 0) & (positions.y < 0)": (q > 0) & (r[:, 1] < 0),
        "(charge > 0) | (positions.y < 0)": (q > 0) | (r[:, 1] < 0),
        "not charge > 0": ~(q > 0),
        "~(charge > 0)": ~(q > 0),
        "positions.x**2 + positions.y**2 + positions.z**2 < 16": (r**2).sum(1) < 16,
        "positions.x * 2 - 1 > positions.y / 3": r[:, 0] * 2 - 1 > r[:, 1] / 3,
        "positions.x // 1 % 2 == 0": (r[:, 0] // 1) % 2 == 0,
        "-positions.z > +velocities.z": -r[:, 2] > v[:, 2],
        "element == 'Cu'": np.array([1, 0, 1, 0, 0, 1], bool),
        "'Cu' == element": np.array([1, 0, 1, 0, 0, 1], bool),
        "element in ['Ni', 'O']": np.array([0, 1, 0, 1, 0, 0], bool),
        "element not in ('Cu', 'H')": np.array([0, 1, 0, 1, 0, 0], bool),
        "charge in {0.5, 0.0}": np.isin(q, [0.5, 0.0]),
        "True": np.ones(6, bool),
        "1 > 2": np.zeros(6, bool),
        "not False": np.ones(6, bool),
    }
    for condition, expected in cases.items():
        assert kept(s, condition) == where(expected), condition


def test_functions():
    s = atoms()
    r, v = s.positions, s.table["velocities"].to_numpy()
    cases = {
        "abs(velocities.z) > 1": np.abs(v[:, 2]) > 1,
        "sqrt(positions.x**2 + positions.y**2) < 4": np.hypot(r[:, 0], r[:, 1]) < 4,
        "exp(positions.x) > 2": np.exp(r[:, 0]) > 2,
        "log(abs(positions.x)) > 0": np.log(np.abs(r[:, 0])) > 0,
        "log10(abs(positions.x)) > 0.5": np.log10(np.abs(r[:, 0])) > 0.5,
        "sin(positions.x) > 0": np.sin(r[:, 0]) > 0,
        "cos(positions.x) > 0": np.cos(r[:, 0]) > 0,
        "tan(positions.x) > 0": np.tan(r[:, 0]) > 0,
        "arcsin(positions.x / 10) > 0.1": np.arcsin(r[:, 0] / 10) > 0.1,
        "arccos(positions.x / 10) > 1.5": np.arccos(r[:, 0] / 10) > 1.5,
        "arctan(positions.x) > 0.5": np.arctan(r[:, 0]) > 0.5,
        "floor(positions.x) == 1": np.floor(r[:, 0]) == 1,
        "ceil(positions.x) == 1": np.ceil(r[:, 0]) == 1,
        "round(positions.x) == 1": np.round(r[:, 0]) == 1,
        "abs(-2) > 1": np.ones(6, bool),
    }
    for condition, expected in cases.items():
        assert kept(s, condition) == where(expected), condition


def test_components_by_name_and_index():
    s = atoms()
    table = s.table
    for column, names in (
        ("positions", "x y z"),
        ("velocities", "x y z"),
        ("orientation", "x y z w"),
        ("atomic_virials", "xx xy xz yx yy yz zx zy zz"),
        ("stress", None),
        ("group", None),
    ):
        values = table[column].to_numpy()
        names = names.split() if names else []
        for k in range(values.shape[1]):
            middle = float(np.median(values[:, k]))
            expected = where(values[:, k] > middle)
            assert kept(s, f"{column}.{k} > {middle!r}") == expected
            if names:
                assert kept(s, f"{column}.{names[k]} > {middle!r}") == expected
    # a column whose name Python cannot read, in backticks
    shear = table["Shear Strain"].to_numpy()
    assert kept(s, "`Shear Strain` > 0.2 and element == 'Cu'") == where(
        (shear > 0.2) & (table["element"] == "Cu").to_numpy()
    )
    # numbers stay numbers, backticks and dots inside strings stay text
    assert kept(s, "positions.x > 1.5") == where(s.positions[:, 0] > 1.5)
    assert kept(s, "element == '`.1'") == []


@pytest.mark.parametrize(
    "condition, message",
    [
        ("`Shear Strain > 0", "backtick is not closed"),
        ("element == 'Cu", "cannot read"),
        ("charge >", "cannot read"),
        ("charge > 0 if True else False", "IfExp is not supported"),
        ("positions[0] > 0", "Subscript is not supported"),
        ("(lambda: 1)()", "only these functions"),
        ("max(charge, 0) > 0", "only these functions"),
        ("round(charge, 1) > 0", "only these functions"),
        ("abs(x=charge) > 0", "only these functions"),
        ("positions.x.y > 0", "belongs to a column"),
        ("charge is None", "comparison Is is not supported"),
        ("element in 'Cu'", "needs a list"),
        ("'Cu' in ['Cu']", "needs a column on its left"),
        ("charge ^ 1", "operator BitXor is not supported"),
        ("missing > 0", "no column 'missing'; the columns are positions"),
        ("positions > 0", "positions has 3 components; write one of positions.0"),
        ("stress.xx > 0", "stress has no component 'xx'; write one of stress.0"),
        ("group.2 > 0", "group has no component '2'"),
        ("charge.x > 0", "charge has no components"),
    ],
)
def test_errors(condition, message):
    with pytest.raises(InvalidParameter, match=message):
        atoms().filter(condition)
