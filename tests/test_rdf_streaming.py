# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Streaming RDF kernel — exercised on top of the existing AlCrNi fixture and
several built crystals to validate equivalence with the verlet path and to
prove that large cutoffs no longer require a verlet list.
"""

import numpy as np
import pytest

import mdapy as mp
from _fixture_helper import load_misc, input_path


def test_streaming_matches_verlet_on_fixture():
    """On the AlCrNi xyz fixture, the streaming kernel reproduces the verlet
    path bin-for-bin (both go through the small-box replication so they share
    the same enlarged supercell)."""
    data = load_misc("rdf")
    system = mp.System(input_path("AlCrNi.xyz"))
    rc = float(data["cutoff"])
    nbin = int(data["nbins"])
    rdf_v = system.cal_radial_distribution_function(rc, nbin, streaming=False)
    rdf_s = system.cal_radial_distribution_function(rc, nbin, streaming=True)
    for key in rdf_v.g_partial:
        np.testing.assert_allclose(
            rdf_s.g_partial[key], rdf_v.g_partial[key], atol=1e-9,
            err_msg=f"streaming vs verlet mismatch at pair {key}",
        )


def test_streaming_matches_verlet_pure_fcc():
    """Pure fcc Cu, big enough to skip replication — streaming and verlet
    must agree to machine precision."""
    sys_ = mp.build_lattice.build_crystal("Cu", "fcc", 3.615, nx=8, ny=8, nz=8)
    rdf_v = sys_.cal_radial_distribution_function(6.0, 100, streaming=False)
    rdf_s = sys_.cal_radial_distribution_function(6.0, 100, streaming=True)
    np.testing.assert_allclose(rdf_v.g_total, rdf_s.g_total, atol=1e-12)


def test_streaming_matches_verlet_multi_element():
    """Multi-element HEA — partials must match across both kernels."""
    hea = mp.build_hea(
        ("Al", "Cu", "Ni"), (0.34, 0.33, 0.33), "fcc",
        a=3.7, nx=6, ny=6, nz=6, random_seed=42,
    )
    rdf_v = hea.cal_radial_distribution_function(5.0, 80, streaming=False)
    rdf_s = hea.cal_radial_distribution_function(5.0, 80, streaming=True)
    for key in rdf_v.g_partial:
        np.testing.assert_allclose(
            rdf_s.g_partial[key], rdf_v.g_partial[key], atol=1e-12,
            err_msg=f"partial {key} mismatch",
        )


def test_streaming_handles_large_cutoff():
    """Streaming runs at rc near L/2 on a 32k-atom system — the verlet path
    would need ~10 GB for the neighbour list, so this exercises the streaming
    advantage. Just check the kernel runs and produces sensible g(r)."""
    sys_ = mp.build_lattice.build_crystal("Cu", "fcc", 3.615, nx=20, ny=20, nz=20)
    L = sys_.box.get_thickness()[0]
    rc = L / 2.0 - 1e-3  # just inside L/2
    rdf_s = sys_.cal_radial_distribution_function(rc, 200, streaming=True)
    # Bin 0 (r near 0) has no pairs — should be exactly zero.
    assert rdf_s.g_total[0] == 0.0
    # First neighbour shell at ~2.56 Å must show a peak well above 1.
    nn_peak_bin = int(2.56 / rc * 200)
    assert rdf_s.g_total[nn_peak_bin - 5 : nn_peak_bin + 5].max() > 5.0


def test_auto_streaming_picks_streaming_when_rc_large():
    """The streaming=None default should auto-pick streaming when rc exceeds
    L/3 on a small box. We just check the run succeeds — value comparison is
    covered by the equivalence tests above."""
    sys_ = mp.System(input_path("AlCrNi.xyz"))
    rdf = sys_.cal_radial_distribution_function(5.0, 50)  # auto path
    # rc=5 vs L/3 ≈ 2.3 → auto-streaming kicks in
    assert rdf.streaming is True


def test_explicit_verlet_path_still_works():
    """Backward-compat: passing streaming=False keeps the legacy code path."""
    sys_ = mp.build_lattice.build_crystal("Cu", "fcc", 3.615, nx=4, ny=4, nz=4)
    rdf = sys_.cal_radial_distribution_function(4.0, 50, streaming=False)
    assert rdf.streaming is False
    # First-shell peak around 2.56 Å should be present
    nn_bin = int(2.56 / 4.0 * 50)
    assert rdf.g_total[nn_bin - 2 : nn_bin + 2].max() > 1.0
