# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

from dataclasses import dataclass

import numpy as np
import polars as pl
import pytest

import mdapy
from mdapy.core import Result


@dataclass(frozen=True)
class Counted(Result):
    values: np.ndarray
    curves: dict
    label: str = "x"

    def columns(self):
        return {"counted": self.values}


def test_arrays_are_read_only():
    result = Counted(np.arange(3), {"a": np.zeros(2), "b": "text"})
    assert not result.values.flags.writeable
    assert not result.curves["a"].flags.writeable
    with pytest.raises(AttributeError):
        result.label = "y"


def test_results_become_columns():
    system = mdapy.System(np.zeros((3, 3)), mdapy.Box(5.0), elements="Cu")
    added = system.with_columns(
        Counted(np.arange(3), {}), pl.lit(1.5).alias("weight"), charge=0.5
    )
    np.testing.assert_array_equal(added.table["counted"].to_numpy(), [0, 1, 2])
    assert added.table["weight"].to_list() == [1.5] * 3
    assert added.table["charge"].to_list() == [0.5] * 3
    assert Result().columns() == {}
