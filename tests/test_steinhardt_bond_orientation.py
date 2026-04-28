# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
Steinhardt bond-orientation parameters Q_l — fixture-driven, no
freud at runtime. Reference values were generated once with
`freud.order.Steinhardt` (with and without averaging) and stored as
.npz files; see `tests/_generate_fixtures/generate_structure_analysis.py`.

The fixtures cover the cutoff mode (rc) only. Other modes — nnn,
voronoi, weighted, wl/wlhat — are still freud-coupled in the codebase
and will be migrated in a follow-up.
"""

import numpy as np
import pytest

import mdapy as mp

from _fixture_helper import fixtures_with, fixture_ids, system_from_fixture

PATHS = fixtures_with("q4")    # all qlm fixtures store both q4 and q6
LL = (4, 6)


@pytest.mark.parametrize("path", PATHS, ids=fixture_ids(PATHS))
def test_ql_against_fixture(path):
    data = np.load(path)
    system = system_from_fixture(data)
    rc = float(data["ql_cutoff"])

    # Plain Q_l
    system.cal_steinhardt_bond_orientation(list(LL), rc=rc)
    for l in LL:
        got = system.data[f"ql{l}"].to_numpy(allow_copy=False)
        expected = data[f"q{l}"]
        assert np.allclose(got, expected, atol=1e-6, rtol=1e-6), (
            f"{path.name}: Q_{l} differs (max |Δ|={np.abs(got - expected).max():.3g})"
        )

    # Averaged Q_l
    system.cal_steinhardt_bond_orientation(list(LL), rc=rc, average=True)
    for l in LL:
        got = system.data[f"ql{l}"].to_numpy(allow_copy=False)
        expected = data[f"q{l}_avg"]
        assert np.allclose(got, expected, atol=1e-6, rtol=1e-6), (
            f"{path.name}: <Q_{l}> differs (max |Δ|={np.abs(got - expected).max():.3g})"
        )


def test_ql_perfect_fcc_known_values():
    """Closed-form: a perfect FCC lattice has Q_4 ≈ 0.190941 and
    Q_6 ≈ 0.574524 for every atom (independent of any reference impl)."""
    a = 4.05
    s = mp.build_crystal("Al", "fcc", a, nx=4, ny=4, nz=4)
    s.cal_steinhardt_bond_orientation([4, 6], rc=0.95 * a)
    q4 = s.data["ql4"].to_numpy()
    q6 = s.data["ql6"].to_numpy()
    assert np.allclose(q4, 0.190941, atol=1e-5), f"Q_4 mean={q4.mean():.6f}"
    assert np.allclose(q6, 0.574524, atol=1e-5), f"Q_6 mean={q6.mean():.6f}"
