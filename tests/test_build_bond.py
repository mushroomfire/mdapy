# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
import polars as pl
from mdapy import System
from mdapy.box import Box


def test_build_bond_with_scalar_cutoff():
    data = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.3, 5.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "type": [1, 2, 2, 1],
        }
    )
    system = System(data=data, box=Box([10.0, 10.0, 10.0], boundary=[0, 0, 0]))

    bond = system.cal_build_bond(1.5)

    expected = np.array([[0, 1], [1, 2]], np.int32)
    assert np.array_equal(bond, expected)
    assert np.array_equal(system.bond, expected)


def test_build_bond_with_pairwise_cutoff():
    data = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.3, 5.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "type": [1, 2, 2, 1],
        }
    )
    system = System(data=data, box=Box([10.0, 10.0, 10.0], boundary=[0, 0, 0]))

    bond = system.cal_build_bond({(1, 1): 0.5, (1, 2): 1.1, (2, 2): 1.2})

    expected = np.array([[0, 1]], np.int32)
    assert np.array_equal(bond, expected)


def test_build_bond_with_element_cutoff():
    data = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.3, 5.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "element": ["Cu", "Zr", "Zr", "Cu"],
        }
    )
    system = System(data=data, box=Box([10.0, 10.0, 10.0], boundary=[0, 0, 0]))

    bond = system.cal_build_bond({("Cu", "Cu"): 0.5, ("Cu", "Zr"): 1.1, ("Zr", "Zr"): 1.2})

    expected = np.array([[0, 1]], np.int32)
    assert np.array_equal(bond, expected)


def test_build_bond_in_small_periodic_box():
    system = System(
        pos=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], float),
        box=Box([2.0, 2.0, 2.0]),
    )

    bond = system.cal_build_bond(1.1)

    expected = np.array([[0, 1]], np.int32)
    assert np.array_equal(bond, expected)
