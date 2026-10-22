# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest

from mdapy.build import build_crystal, random_alloy
from mdapy.errors import InvalidParameter


def counts(system):
    values, number = np.unique(system.table["element"].to_numpy(), return_counts=True)
    return dict(zip(values.tolist(), number.tolist()))


def test_exact_counts():
    sites = build_crystal("fcc", 3.6, repeat=(5, 5, 5))  # 500 atoms
    alloy = random_alloy(sites, {"Fe": 1, "Ni": 1, "Cr": 1}, seed=1)
    # 500 / 3 = 166.67 each: the two first named get the remaining atoms
    assert counts(alloy) == {"Fe": 167, "Ni": 167, "Cr": 166}
    np.testing.assert_array_equal(alloy.positions, sites.positions)
    assert counts(random_alloy(sites, {"Cu": 0.25, "Zn": 0.75})) == {
        "Cu": 125,
        "Zn": 375,
    }
    assert counts(random_alloy(sites, {"Al": 0.1, "Ni": 0.3, "Co": 0.6})) == {
        "Al": 50,
        "Ni": 150,
        "Co": 300,
    }


def test_seed_and_randomness():
    sites = build_crystal("bcc", 2.87, repeat=(10, 10, 10))
    a = random_alloy(sites, {"Fe": 0.5, "Cr": 0.5}, seed=3)
    b = random_alloy(sites, {"Fe": 0.5, "Cr": 0.5}, seed=np.random.default_rng(3))
    c = random_alloy(sites, {"Fe": 0.5, "Cr": 0.5}, seed=4)
    assert a.table["element"].to_list() == b.table["element"].to_list()
    assert a.table["element"].to_list() != c.table["element"].to_list()
    # no order along the atoms: each half of the sites holds about half the Fe
    fe = a.table["element"].to_numpy() == "Fe"
    assert abs(fe[: len(fe) // 2].mean() - 0.5) < 0.05


@pytest.mark.parametrize(
    "fractions, message",
    [
        ({}, "at least one element"),
        ({"Qq": 1.0}, "unknown element 'Qq'"),
        ({"Fe": 0.0, "Ni": 1.0}, "fraction of Fe must be positive"),
    ],
)
def test_invalid_fractions(fractions, message):
    with pytest.raises(InvalidParameter, match=message):
        random_alloy(build_crystal("fcc", 3.6), fractions)
