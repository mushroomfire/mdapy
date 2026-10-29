# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.analysis import OPERATIONS, Histogram, reduce
from mdapy.errors import InvalidParameter


def values_system():
    """Ten atoms with a column of values, one NaN, and a vector column."""
    rng = np.random.default_rng(3)
    value = np.array([1.0, 2.0, np.nan, 4.0, 10.0, 3.0, 0.0, -2.0, 2.0, 5.0])
    return mdapy.System(
        rng.random((10, 3)) * 10, mdapy.Box(10.0), elements="Cu"
    ).with_columns(value=value, force=rng.normal(size=(10, 3)))


@pytest.mark.parametrize("operation", OPERATIONS)
def test_reductions_leave_out_nan(operation):
    system = values_system()
    value = system["value"]
    finite = value[~np.isnan(value)]
    expected = {
        "mean": finite.mean(),
        "sum": finite.sum(),
        "min": finite.min(),
        "max": finite.max(),
        "median": np.median(finite),
        "variance": finite.var(ddof=1),
        "std": finite.std(ddof=1),
        "nonzero": np.count_nonzero(finite),
        "argmin": 7,
        "argmax": 4,
    }[operation]
    assert reduce(system, "value", operation) == pytest.approx(expected, rel=1e-14)


def test_reductions_of_components_and_atoms():
    system = values_system()
    force = system["force"]
    np.testing.assert_allclose(
        reduce(system, "force", "sum"), force.sum(axis=0), rtol=1e-14
    )
    assert reduce(system, "force.y", "max") == force[:, 1].max()
    assert reduce(system, "value * 2", "sum") == 2 * np.nansum(system["value"])
    chosen = np.zeros(10, dtype=bool)
    chosen[[1, 3, 4]] = True
    assert reduce(system, "value", "mean", atoms=chosen) == pytest.approx(16 / 3)
    # the index among all atoms, not among those chosen
    assert reduce(system, "value", "argmin", atoms=[3, 4, 9]) == 3
    np.testing.assert_array_equal(
        reduce(system, "force", "argmax", atoms=[0, 1]), np.argmax(force[:2], axis=0)
    )


def test_reductions_of_no_values():
    system = values_system()
    nothing = np.zeros(10, dtype=bool)
    assert np.isnan(reduce(system, "value", "mean", atoms=nothing))
    assert reduce(system, "value", "sum", atoms=nothing) == 0.0
    assert reduce(system, "value", "nonzero", atoms=nothing) == 0
    assert np.isnan(reduce(system, "value", "std", atoms=[0]))
    with pytest.raises(InvalidParameter, match="argmax of no values"):
        reduce(system, "value", "argmax", atoms=[2])
    with pytest.raises(InvalidParameter, match="operation must be"):
        reduce(system, "value", "mode")
    with pytest.raises(InvalidParameter, match="not numbers"):
        reduce(system, "element", "mean")


def test_histogram_against_numpy():
    system = values_system()
    value = system["value"]
    finite = value[np.isfinite(value)]
    result = Histogram("value", bins=6).compute(system)
    counts, edges = np.histogram(finite, bins=6)
    np.testing.assert_array_equal(result.counts, counts)
    np.testing.assert_allclose(result.edges, edges)
    np.testing.assert_allclose(result.center, 0.5 * (edges[1:] + edges[:-1]))
    np.testing.assert_allclose(result.frequency, counts / counts.sum())
    np.testing.assert_allclose(result.density.sum() * (edges[1] - edges[0]), 1.0)
    assert result.nonfinite == 1 and result.name == "value"
    # a range of our own: the values outside are not counted
    ranged = Histogram("value", bins=4, range=(0, 4)).compute(system, atoms=value < 5)
    np.testing.assert_array_equal(
        ranged.counts, np.histogram(value[value < 5], bins=4, range=(0, 4))[0]
    )
    assert "range=(0.0, 4.0)" in repr(Histogram("value", 4, (0, 4)))


def test_histogram_of_one_value_or_none():
    system = values_system().with_columns(same=np.full(10, 2.5))
    one = Histogram("same", bins=2).compute(system)
    np.testing.assert_allclose(one.edges, [2.5, 3.0, 3.5])
    assert one.counts.tolist() == [10, 0]
    none = Histogram("value", bins=2).compute(system, atoms=[2])
    assert none.counts.tolist() == [0, 0] and none.nonfinite == 1
    assert none.frequency.tolist() == [0.0, 0.0]


def test_histogram_errors():
    system = values_system()
    with pytest.raises(InvalidParameter, match="bins"):
        Histogram("value", bins=0)
    with pytest.raises(InvalidParameter, match="range must increase"):
        Histogram("value", range=(1, 1))
    with pytest.raises(InvalidParameter, match="components"):
        Histogram("force").compute(system)
