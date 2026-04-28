# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Voronoi cell volume / cavity radius / coordination — fixture-driven."""

import numpy as np
import pytest

from _fixture_helper import fixtures_with, fixture_ids, system_from_fixture

PATHS = fixtures_with("voronoi_volume")


@pytest.mark.parametrize("path", PATHS, ids=fixture_ids(PATHS))
def test_voronoi_against_fixture(path):
    data = np.load(path)
    system = system_from_fixture(data)
    system.cal_voronoi_volume()

    assert np.allclose(
        data["voronoi_volume"],
        system.data["volume"].to_numpy(allow_copy=False),
        atol=1e-6,
    ), f"{path.name}: atomic volumes differ"

    # OVITO's Cavity Radius is half of mdapy's convention; reference
    # values are stored as OVITO outputs.
    assert np.allclose(
        data["voronoi_cavity_radius"],
        system.data["cavity_radius"].to_numpy(allow_copy=False) * 0.5,
        atol=1e-6,
    ), f"{path.name}: cavity radius differs"

    assert np.array_equal(
        data["voronoi_coord"],
        system.data["neighbor_number"].to_numpy(allow_copy=False),
    ), f"{path.name}: coordination differs"
