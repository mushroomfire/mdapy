# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
``mdapy.orthogonal_cell`` — atomsk's ``-orthogonal-cell`` ported to
Python. Tests check both the structural equivalence (a triclinic input
and its orthogonal supercell describe the same infinite crystal) and
direct match against atomsk reference output for a representative HCP
input.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import mdapy as mp
from mdapy.box import Box


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "build_crystal"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lex_sort(pos):
    """Stable sort of a (N, 3) array on Cartesian position to make the
    output of two implementations directly comparable."""
    key = np.round(pos, 6)
    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    return pos[order], order


def _is_diagonal(box, tol=1e-9):
    return np.allclose(box - np.diag(np.diag(box)), 0.0, atol=tol)


# ---------------------------------------------------------------------------
# Direct atomsk match — Mg HCP primitive → 4-atom orthogonal supercell.
# ---------------------------------------------------------------------------

def test_orthogonal_hcp_matches_atomsk():
    hcp = mp.build_crystal("Mg", "hcp", a=3.21, c=5.21)
    ortho = mp.orthogonal_cell(hcp)
    assert ortho.N == 4
    assert _is_diagonal(ortho.box.box)
    np.testing.assert_allclose(np.diag(ortho.box.box),
                               [3.21, 3.21 * np.sqrt(3), 5.21], atol=1e-6)
    pos, _ = _lex_sort(ortho.data.select("x", "y", "z").to_numpy())
    expected = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.853294364099, 2.605],
        [1.605, 2.779941546148, 0.0],
        [1.605, 4.633235910247, 2.605],
    ])
    expected, _ = _lex_sort(expected)
    np.testing.assert_allclose(pos, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Already-orthogonal input is passed through unchanged.
# ---------------------------------------------------------------------------

def test_orthogonal_passthrough_for_cubic():
    fcc = mp.build_crystal("Cu", "fcc", 3.615, nx=2, ny=2, nz=2)
    ortho = mp.orthogonal_cell(fcc)
    assert ortho.N == fcc.N
    np.testing.assert_allclose(ortho.box.box, fcc.box.box, atol=1e-9)


# ---------------------------------------------------------------------------
# Multi-species hexagonal: every basis species survives the transform.
# ---------------------------------------------------------------------------

def test_orthogonal_wurtzite_gan_preserves_species():
    gan = mp.build_crystal(("Ga", "N"), "wurtzite", a=3.19, c=5.18)
    ortho = mp.orthogonal_cell(gan)
    assert _is_diagonal(ortho.box.box)
    # Wurtzite primitive has 4 atoms (2 Ga + 2 N); the orthogonal cell
    # is exactly 2× the primitive volume → 8 atoms (4 of each species).
    assert ortho.N == 8
    eles = ortho.data["element"].to_list()
    assert sorted(eles) == ["Ga", "Ga", "Ga", "Ga", "N", "N", "N", "N"]


# ---------------------------------------------------------------------------
# find_minimal collapses a duplicated supercell back to its primitive.
# ---------------------------------------------------------------------------

def test_orthogonal_find_minimal_collapses_replicated_hcp():
    hcp = mp.build_crystal("Mg", "hcp", a=3.21, c=5.21, nx=2, ny=2, nz=1)
    big = mp.orthogonal_cell(hcp)
    small = mp.orthogonal_cell(hcp, find_minimal=True)
    assert small.N <= big.N
    assert small.N == 4   # the canonical 4-atom orthogonal hcp cell
    np.testing.assert_allclose(np.diag(small.box.box),
                               [3.21, 3.21 * np.sqrt(3), 5.21], atol=1e-6)


def test_orthogonal_find_minimal_keeps_minimum_when_already_minimal():
    hcp = mp.build_crystal("Mg", "hcp", a=3.21, c=5.21)  # 2-atom primitive
    minimal = mp.orthogonal_cell(hcp, find_minimal=True)
    # The 4-atom orthogonal cell IS the smallest orthogonal cell; no
    # further reduction possible.
    assert minimal.N == 4


# ---------------------------------------------------------------------------
# Density / volume invariants — atom density is preserved exactly.
# ---------------------------------------------------------------------------

def test_orthogonal_atom_density_preserved():
    """Triclinic vs orthogonal cells describe the same crystal, so
    atom density (atoms / volume) is invariant under
    `orthogonal_cell`."""
    hcp = mp.build_crystal("Mg", "hcp", a=3.21, c=5.21, nx=3, ny=3, nz=2)
    ortho = mp.orthogonal_cell(hcp)
    den_in = hcp.N / abs(np.linalg.det(hcp.box.box))
    den_out = ortho.N / abs(np.linalg.det(ortho.box.box))
    np.testing.assert_allclose(den_in, den_out, rtol=1e-9)


# ---------------------------------------------------------------------------
# Validation — open boundaries / singular box are rejected.
# ---------------------------------------------------------------------------

def test_orthogonal_rejects_open_boundary():
    pos = np.zeros((1, 3))
    sys_ = mp.System(pos=pos, box=Box(np.eye(3) * 5.0, boundary=[1, 1, 0]))
    with pytest.raises(ValueError, match="periodic"):
        mp.orthogonal_cell(sys_)


def test_orthogonal_extra_columns_round_trip():
    """Per-atom non-position columns (e.g. velocities) follow atoms
    through the transform."""
    hcp = mp.build_crystal("Mg", "hcp", a=3.21, c=5.21)
    rng = np.random.default_rng(0)
    vel = rng.normal(size=(hcp.N, 3))
    hcp.update_data(hcp.data.with_columns(
        vx=pl.Series(vel[:, 0]),
        vy=pl.Series(vel[:, 1]),
        vz=pl.Series(vel[:, 2]),
    ))
    ortho = mp.orthogonal_cell(hcp)
    for col in ("vx", "vy", "vz"):
        assert col in ortho.data.columns
    # Each output velocity must equal one of the input velocities (i.e.
    # the velocity column is just a tiling of the primitive cell's set).
    in_vel = hcp.data.select("vx", "vy", "vz").to_numpy()
    out_vel = ortho.data.select("vx", "vy", "vz").to_numpy()
    for v in out_vel:
        diffs = np.linalg.norm(in_vel - v, axis=1)
        assert diffs.min() < 1e-10
