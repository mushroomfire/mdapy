# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
build_crystal — fixture-driven, atomsk as ground truth.

Each fixture is an .npz that captures the canonical
(box, sorted_positions, sorted_elements) triple produced by
``atomsk --create ...``. We rebuild the same structure with mdapy and
demand the same triple. Atomsk is *not* required at test time; running
``tests/_generate_fixtures/generate_build_crystal.py`` regenerates the
references.
"""

from pathlib import Path

import numpy as np
import pytest

import mdapy as mp


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "build_crystal"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonicalize(system):
    pos = system.data.select("x", "y", "z").to_numpy()
    elements = system.data["element"].to_list()
    # Round to 6 decimals for the sort key only; original positions stay
    # untouched so the comparison still uses the unrounded values.
    key = np.round(pos, 6)
    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    return system.box.box.copy(), pos[order], [elements[i] for i in order]


def _assert_matches(name, mdapy_system, atol=1e-6):
    fix = np.load(FIXTURE_DIR / f"{name}.npz")
    ref_box = fix["box"]
    ref_pos = fix["positions"]
    ref_ele = list(fix["elements"])

    box, pos, ele = _canonicalize(mdapy_system)
    np.testing.assert_allclose(box, ref_box, atol=atol,
                               err_msg=f"{name}: box mismatch")
    np.testing.assert_allclose(pos, ref_pos, atol=atol,
                               err_msg=f"{name}: positions mismatch")
    assert ele == ref_ele, f"{name}: element ordering mismatch\n  mdapy: {ele}\n  atomsk: {ref_ele}"


# ---------------------------------------------------------------------------
# Plain (non-Miller) structures
# ---------------------------------------------------------------------------

# (fixture name, mdapy `name` argument, build_crystal kwargs)
PLAIN_CASES = [
    ("Cu_fcc",       "Cu",            dict(structure="fcc", a=3.615)),
    ("Fe_bcc",       "Fe",            dict(structure="bcc", a=2.83)),
    ("C_diamond",    "C",             dict(structure="diamond", a=3.6)),
    ("W_sc",         "W",             dict(structure="sc", a=3.16)),

    ("NaCl_rocksalt", ("Na", "Cl"),   dict(structure="rocksalt", a=5.64)),
    ("NiAl_cscl",    ("Ni", "Al"),    dict(structure="cscl", a=2.86)),
    ("GaAs_zb",      ("Ga", "As"),    dict(structure="zincblende", a=5.65)),
    ("CaF2_fluorite", ("Ca", "F"),    dict(structure="fluorite", a=5.46)),
    ("Ni3Al_l1_2",   ("Ni", "Al"),    dict(structure="l1_2", a=3.57)),

    ("SrTiO3_perovskite", ("Ti", "Sr", "O"),
                                      dict(structure="perovskite", a=3.905)),

    ("GaN_wurtzite", ("Ga", "N"),     dict(structure="wurtzite", a=3.19, c=5.18)),
    ("C_graphite",   "C",             dict(structure="graphite", a=2.46, c=6.71)),
]


@pytest.mark.parametrize("name,elements,kwargs", PLAIN_CASES,
                         ids=[c[0] for c in PLAIN_CASES])
def test_atomsk_match(name, elements, kwargs):
    sys_ = mp.build_crystal(elements, **kwargs)
    _assert_matches(name, sys_)


# ---------------------------------------------------------------------------
# Miller-oriented (cubic) structures
# ---------------------------------------------------------------------------

MILLER_CASES = [
    ("Cu_fcc_111",  "Cu",
     dict(structure="fcc", a=3.615,
          miller1=(1, -1, 0), miller2=(1, 1, -2), miller3=(1, 1, 1))),
    ("Fe_bcc_111",  "Fe",
     dict(structure="bcc", a=2.83,
          miller1=(1, 2, 1), miller2=(-1, 0, 1), miller3=(1, -1, 1))),
    ("NaCl_rocksalt_111", ("Na", "Cl"),
     dict(structure="rocksalt", a=5.64,
          miller1=(1, -1, 0), miller2=(1, 1, -2), miller3=(1, 1, 1))),
]


@pytest.mark.parametrize("name,elements,kwargs", MILLER_CASES,
                         ids=[c[0] for c in MILLER_CASES])
def test_atomsk_match_miller(name, elements, kwargs):
    sys_ = mp.build_crystal(elements, **kwargs)
    _assert_matches(name, sys_)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def test_unsupported_structure_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        mp.build_crystal("X", "no_such_phase", a=1.0)


def test_multispecies_count_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        # rocksalt expects exactly 2 species
        mp.build_crystal(("A", "B", "C"), "rocksalt", a=5.0)


def test_graphite_requires_c():
    with pytest.raises(ValueError, match="graphite"):
        mp.build_crystal("C", "graphite", a=2.46)


def test_miller_unsupported_for_hexagonal():
    with pytest.raises(ValueError, match="Miller"):
        mp.build_crystal(("Ga", "N"), "wurtzite", a=3.19, c=5.18,
                         miller1=(1, 0, 0), miller2=(0, 1, 0), miller3=(0, 0, 1))


def test_replication_scales_atom_count():
    s = mp.build_crystal(("Na", "Cl"), "rocksalt", a=5.64, nx=2, ny=3, nz=4)
    assert s.N == 8 * 2 * 3 * 4
