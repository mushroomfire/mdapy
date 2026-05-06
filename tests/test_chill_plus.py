# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""CHILL+ — fixture-driven, no OVITO at runtime."""

import numpy as np
import pytest

import mdapy as mp
from _fixture_helper import (
    fixtures_with,
    fixture_ids,
    system_from_fixture,
)

PATHS = fixtures_with("chill_plus")


@pytest.mark.parametrize("path", PATHS, ids=fixture_ids(PATHS))
def test_chill_plus_against_fixture(path):
    """mdapy CHILL+ output must match the OVITO reference atom-by-atom."""
    data = np.load(path)
    system = system_from_fixture(data)
    cutoff = float(data["chill_plus_cutoff"])
    system.cal_chill_plus(cutoff=cutoff)
    got = system.data["chill_plus"].to_numpy(allow_copy=False)
    expected = data["chill_plus"]
    n_diff = int(np.sum(got != expected))
    assert n_diff == 0, (
        f"{path.name}: {n_diff}/{len(got)} atoms classified differently; "
        f"counts mdapy={np.bincount(got, minlength=6).tolist()} "
        f"vs ovito={np.bincount(expected, minlength=6).tolist()}"
    )


# Small-box self-tests — perfect ice crystals must give a single uniform
# label even when the cell thickness is below 2 * cutoff.
# Codes: 1 = hexagonal ice, 2 = cubic ice.
@pytest.mark.parametrize(
    "structure,a,nx,ny,nz,expected_label",
    [
        # Cubic ice Ic — diamond lattice on O, a chosen so O-O ~ 2.76 Å.
        ("diamond",     6.37, 1, 1, 1, 2),  # 8 atoms, 6.37 Å box (< 2*3.5)
        ("diamond",     6.37, 2, 2, 2, 2),  # 64 atoms, 12.74 Å box
        # Hexagonal ice Ih — lonsdaleite (hexagonal diamond) on O.
        ("lonsdaleite", 4.50, 1, 1, 1, 1),  # 4 atoms, ~3.9 Å thickness (tiny)
        ("lonsdaleite", 4.50, 2, 2, 2, 1),  # 32 atoms
    ],
)
def test_chill_plus_perfect_ice_small_box(structure, a, nx, ny, nz, expected_label):
    crystal = mp.build_crystal("O", structure, a, nx=nx, ny=ny, nz=nz)
    crystal.cal_chill_plus(cutoff=3.5)
    got = crystal.data["chill_plus"].to_numpy()
    assert np.all(got == expected_label), (
        f"{structure} {nx}x{ny}x{nz}: expected all={expected_label}, got "
        f"{np.unique(got, return_counts=True)}"
    )
