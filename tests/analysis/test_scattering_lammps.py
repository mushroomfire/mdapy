# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Powder diffraction against LAMMPS compute xrd (the DIFFRACTION package),
which sums over the same wave vectors of the box with the form factors of
the International Tables. Skipped where LAMMPS is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy.analysis import PowderDiffraction
from mdapy.build import build_crystal, random_alloy

lammps = pytest.importorskip("lammps")


def test_the_pattern_as_compute_xrd(tmp_path):
    crystal = build_crystal("fcc", 3.6, elements="Cu", repeat=(3, 3, 3))
    alloy = random_alloy(crystal, {"Cu": 0.6, "Ni": 0.4}, seed=1)
    shake = np.random.default_rng(0).normal(0, 0.1, (len(alloy), 3))
    alloy = alloy.with_columns(positions=alloy["positions"] + shake)
    low, high, bins, wavelength = 20.0, 80.0, 120, 1.5406
    data = tmp_path / "in.data"
    mdapy.io.write(data, alloy, elements=["Cu", "Ni"])
    lmp = lammps.lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        lmp.commands_string(
            f"""units metal
atom_style atomic
boundary p p p
read_data {data}
mass * 60.0
compute xrd all xrd {wavelength} Cu Ni 2Theta {low} {high} c 1 1 1 LP 1
run 0
"""
        )
        nodes = lmp.numpy.extract_compute(
            "xrd", lammps.LMP_STYLE_GLOBAL, lammps.LMP_TYPE_ARRAY
        ).copy()
    finally:
        lmp.close()
    angle, intensity = nodes[:, 0], nodes[:, 1]  # 2θ in degrees, per node
    edges = np.linspace(low, high, bins + 1)
    which = np.searchsorted(edges, angle, side="right") - 1
    inside = (which >= 0) & (which < bins)
    theirs = np.bincount(which[inside], intensity[inside], bins)
    ours = PowderDiffraction(wavelength, (low, high), bins).compute(alloy).intensity
    # the same reflections in the same bins; the intensities differ by the
    # form factors, those of the International Tables there
    np.testing.assert_array_equal(
        ours > 1e-6 * ours.max(), theirs > 1e-6 * theirs.max()
    )
    seen = theirs > 1e-3 * theirs.max()
    np.testing.assert_allclose(ours[seen], theirs[seen], rtol=3e-3)
