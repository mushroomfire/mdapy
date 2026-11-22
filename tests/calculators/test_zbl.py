# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The screened repulsion of Ziegler, Biersack and Littmark: one pair is the
formula, its energy and first two derivatives go to zero at the cutoff,
against LAMMPS pair_style zbl atom by atom, against finite differences, on
the GPU stood in, and its parameters."""

import numpy as np
import pytest

import mdapy
from mdapy.calculators import ZBL, _gpu
from mdapy.calculators.testing import check_calculator
from mdapy.errors import InvalidParameter

from .eam_files import alloy
from .gpu_stand_in import stand_in

WEIGHTS = (0.02817, 0.28022, 0.50986, 0.18175)
EXPONENTS = (0.20162, 0.40290, 0.94229, 3.19980)


def screened(r, zi, zj):
    """Z_i Z_j e^2 / r phi(r / a), a = 0.4685 / (Z_i^0.23 + Z_j^0.23)."""
    a = 0.46850 / (zi**0.23 + zj**0.23)
    phi = sum(c * np.exp(-d * r / a) for c, d in zip(WEIGHTS, EXPONENTS))
    return zi * zj * 14.399645 / r * phi


def pair(r, elements=("Cu", "Ni")):
    return mdapy.System(
        [[0, 0, 0], [r, 0, 0]], mdapy.Box(40.0, pbc=False), elements=list(elements)
    )


def energy(calculator, r, elements=("Cu", "Ni")):
    return calculator.compute(pair(r, elements)).energy


def test_one_pair_is_the_formula():
    """Below the inner cutoff, the screened repulsion less its value at the
    cutoff and the switch's; the force its derivative."""
    zbl = ZBL(["Cu", "Ni"], 1.5, 2.5)
    shift = energy(zbl, 1.0) - screened(1.0, 29, 28)
    for r in (0.3, 0.7, 1.2, 1.49):
        assert energy(zbl, r) == pytest.approx(screened(r, 29, 28) + shift, rel=1e-13)
        h = 1e-6
        slope = (screened(r + h, 29, 28) - screened(r - h, 29, 28)) / (2 * h)
        forces = zbl.compute(pair(r), ("energy", "forces")).forces
        assert forces[1, 0] == pytest.approx(-slope, rel=1e-7)
        np.testing.assert_allclose(forces[0], -forces[1], atol=1e-12)
    # the pairs of two atoms of an element as those of their numbers
    copper = ZBL(["Cu"], 1.5, 2.5)
    assert energy(copper, 0.8, ("Cu", "Cu")) - energy(copper, 1.2, ("Cu", "Cu")) == (
        pytest.approx(screened(0.8, 29, 29) - screened(1.2, 29, 29), rel=1e-12)
    )


def test_it_goes_to_zero_at_the_cutoff():
    """The energy and its first two derivatives vanish at the cutoff, and
    beyond it there is nothing; the switch leaves it smooth at the inner
    cutoff."""
    zbl = ZBL(["Cu", "Ni"], 1.5, 2.5)
    h = 1e-4
    e = [energy(zbl, 2.5 - k * h) for k in range(3)]
    assert abs(e[0]) < 1e-12
    assert abs((e[0] - e[1]) / h) < 1e-6
    assert abs((e[0] - 2 * e[1] + e[2]) / h**2) < 1e-2
    assert energy(zbl, 2.5 + 1e-9) == 0.0
    # continuous force across the inner cutoff
    f = [zbl.compute(pair(1.5 + s), ("forces",)).forces[1, 0] for s in (-1e-9, 1e-9)]
    assert f[0] == pytest.approx(f[1], rel=1e-7)


@pytest.mark.parametrize("pbc", [True, (True, True, False)])
def test_against_lammps(tmp_path, pbc):
    pytest.importorskip("lammps")
    from .lammps_reference import assert_matches, reference

    # fcc of 2.6 Å, nearest neighbors at 1.84 Å, shaken
    system = alloy(seed=7, repeat=4, jitter=0.2, a=2.6, pbc=pbc)
    lines = """
pair_style zbl 1.4 2.3
pair_coeff 1 1 29 29
pair_coeff 1 2 29 28
pair_coeff 2 2 28 28
"""
    expected = reference(system, ["Cu", "Ni"], lines, tmp_path)
    result = ZBL(["Cu", "Ni"], 1.4, 2.3).compute(
        system, ("energy", "atomic_energies", "forces", "virial", "atomic_virials")
    )
    assert_matches(result, expected)


def test_check_calculator():
    system = alloy(seed=2, repeat=2, jitter=0.25, a=2.7)
    check_calculator(ZBL(["Cu", "Ni"], 1.2, 2.4), system, rtol=1e-6, atol=1e-7)


def test_on_the_gpu_stood_in(monkeypatch):
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    system = alloy(seed=3, repeat=6, jitter=0.2, a=2.6)
    cpu = ZBL(["Cu", "Ni"], 1.4, 2.3).compute(system, ("energy", "forces", "virial"))
    gpu = ZBL(["Cu", "Ni"], 1.4, 2.3, device="cuda", precision="double")
    out = gpu.compute(system, ("energy", "forces", "virial"))
    np.testing.assert_allclose(out.forces, cpu.forces, rtol=1e-12, atol=1e-12)
    assert out.energy == pytest.approx(cpu.energy, rel=1e-12)
    assert "device='cuda'" in repr(gpu)


def test_parameters():
    zbl = ZBL(("Ni", "Cu"), 1.0, 2.0)
    assert zbl.elements == ("Ni", "Cu") and zbl.inner == 1.0 and zbl.cutoff == 2.0
    assert repr(zbl) == "ZBL(elements=('Ni', 'Cu'), inner=1.0, cutoff=2.0)"
    for elements in ("Cu", ["Cu", "Cu"], [], ["Xx"], 5):
        with pytest.raises(InvalidParameter):
            ZBL(elements, 1.0, 2.0)
    for inner, cutoff in ((2.0, 2.0), (2.5, 2.0), (-1.0, 2.0), (1.0, np.inf)):
        with pytest.raises(InvalidParameter):
            ZBL(["Cu"], inner, cutoff)
    with pytest.raises(InvalidParameter, match="no parameters for"):
        zbl.compute(pair(1.0, ("Cu", "Ar")))
