# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
Centro-symmetry-parameter (CSP) tests — fixture-driven, no OVITO at runtime.
"""

import numpy as np
import pytest

import mdapy as mp
from mdapy.box import Box

from _fixture_helper import fixtures_with, fixture_ids, system_from_fixture

PATHS = fixtures_with("csp")


@pytest.mark.parametrize("path", PATHS, ids=fixture_ids(PATHS))
def test_csp_against_fixture(path):
    data = np.load(path)
    system = system_from_fixture(data)
    system.cal_centro_symmetry_parameter(int(data["csp_num_neighbors"]))
    got = system.data["csp"].to_numpy(allow_copy=False)
    expected = data["csp"]
    assert np.allclose(got, expected, atol=1e-6, rtol=1e-6), (
        f"{path.name}: max |Δ| = {np.abs(got - expected).max():.3g}"
    )


def test_csp_perfect_fcc_is_zero():
    """No fixture needed: a perfect FCC lattice has CSP = 0 by symmetry."""
    system = mp.build_crystal("Al", "fcc", 4.05, nx=5, ny=5, nz=5)
    system.cal_centro_symmetry_parameter(12)
    assert np.allclose(system.data["csp"].to_numpy(), 0.0, atol=1e-10)


def test_csp_responds_to_displacement():
    system = mp.build_crystal("Al", "fcc", 4.05, nx=4, ny=4, nz=4)
    pos = system.data.select(["x", "y", "z"]).to_numpy().copy()
    pos[0] += np.array([0.2, 0.0, 0.0])
    perturbed = mp.System(pos=pos, box=Box(system.box.box.copy(),
                                           boundary=list(system.box.boundary)))
    perturbed.cal_centro_symmetry_parameter(12)
    assert perturbed.data["csp"].to_numpy()[0] > 1e-3
