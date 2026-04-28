# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""Identify-Diamond-Structure (IDS) — fixture-driven, no OVITO at runtime."""

from pathlib import Path

import numpy as np
import pytest

from mdapy import System
import mdapy as mp

from _fixture_helper import fixtures_with, fixture_ids, system_from_fixture

INPUT_DIR = Path(__file__).parent / "input_files"

PATHS = fixtures_with("ids")


@pytest.mark.parametrize("path", PATHS, ids=fixture_ids(PATHS))
def test_ids_against_fixture(path):
    data = np.load(path)
    system = system_from_fixture(data)
    system.cal_identify_diamond_structure()
    got = system.data["ids"].to_numpy(allow_copy=False)
    expected = data["ids"]
    n_diff = int(np.sum(got != expected))
    assert n_diff == 0, (
        f"{path.name}: {n_diff}/{len(got)} atoms classified differently"
    )


def test_ids_perfect_crystals():
    """Self-contained sanity: cubic and hexagonal diamond labels."""
    diamond = mp.build_crystal("C", "diamond", 3.5, nx=3, ny=3, nz=3)
    diamond.cal_identify_diamond_structure()
    # 1 = cubic diamond
    assert np.all(diamond.data["ids"].to_numpy() == 1), "cubic diamond -> 1"

    hex_diamond = System(str(INPUT_DIR / "HexDiamond.xyz"))
    hex_diamond.cal_identify_diamond_structure()
    # 4 = hexagonal diamond
    assert np.all(hex_diamond.data["ids"].to_numpy() == 4), "hex diamond -> 4"
