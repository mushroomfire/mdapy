# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Collinear spin on the GPU against Quantum ESPRESSO run live: ferromagnetic
bcc iron and fcc nickel, antiferromagnetic chromium. The energy (with the
first-order effect of the rounded gamma of QE's spin-polarized PBE), the total
and absolute magnetization, the Fermi level and the occupied eigenvalues.
Without moments the spins give the paramagnetic result. The first moments
held while the total density settles: for the first steps alone, never from
a density given, with zero moments or with adaptive damping, and the same
ground state with and without."""

import numpy as np
import pytest

from mdapy.calculators._dft import scf
from mdapy.calculators._dft.upf import read_upf

from dft import espresso, gpu

_dft = gpu.module()
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

BOHR = 0.529177210903
CUTOFF = 25.0


def bcc(a):
    h = a / 2 / BOHR
    return np.array([[-h, h, h], [h, -h, h], [h, h, -h]])


def fcc(a):
    h = a / 2 / BOHR
    return np.array([[0, h, h], [h, 0, h], [h, h, 0]])


def compare(cell, positions, symbols, magmoms):
    files = {s: espresso.pseudo(f"{s}_std.upf") for s in dict.fromkeys(symbols)}
    kinds = [list(files).index(s) for s in symbols]
    pseudos = [read_upf(f) for f in files.values()]
    kpoints, weights = _dft.monkhorst_pack([4, 4, 4], [0.0, 0.0, 0.0], True)
    qe = espresso.run(
        cell, positions, symbols, files, CUTOFF, kpoints, weights, "gaussian", 0.01,
        conv_thr=1e-14, magmoms=magmoms, beta=0.3,
    )  # fmt: skip
    ours = scf.run(
        cell, positions, kinds, pseudos, CUTOFF, kgrid=(4, 4, 4), smearing="gaussian",
        sigma=0.01, functional="pbe_pw", grid=qe["grid"], magmoms=magmoms,
        density_tolerance=1e-14, energy_tolerance=1e-12, symmetry=False
    )  # fmt: skip
    assert ours["converged"] and ours["spins"] == 2
    gamma = espresso.rounded_gamma(ours, cell, positions, kinds, pseudos, 4 * CUTOFF)
    n = len(symbols)
    assert ours["energy"] + gamma == pytest.approx(qe["energy"], abs=1e-7 * n)
    assert ours["ewald"] == pytest.approx(qe["ewald"], abs=1e-7 * n)
    for key in ["magnetization", "absolute_magnetization"]:
        assert ours[key] == pytest.approx(qe[key], abs=1e-4), key
    assert ours["fermi"] == pytest.approx(qe["fermi"], abs=1e-6)
    width = qe["eigenvalues"].shape[-1]
    held = ours["occupations"][..., :width] > 1e-6
    np.testing.assert_allclose(
        ours["eigenvalues"][..., :width][held], qe["eigenvalues"][held], atol=1e-6
    )
    # the magnetization of the last step is that of the result
    assert ours["magnetizations"][-1] == pytest.approx(ours["magnetization"], abs=1e-10)
    absolute = ours["absolute_magnetizations"]
    assert absolute[-1] == pytest.approx(ours["absolute_magnetization"], abs=1e-10)
    assert (absolute >= np.abs(ours["magnetizations"]) - 1e-10).all()
    # the moments held for the first steps, at most 12, and no step after
    held = ours["held"]
    count = int(held.sum())
    assert 1 <= count <= 12 and held[:count].all() and not held[count:].any()
    return ours


def test_iron():
    r = compare(bcc(2.87), [[0, 0, 0]], ["Fe"], [2.5])
    assert 2.0 < r["magnetization"] < 2.5


def test_nickel():
    r = compare(fcc(3.52), [[0, 0, 0]], ["Ni"], [0.6])
    assert 0.5 < r["magnetization"] < 0.7


def test_antiferromagnetic_chromium():
    a = 2.88 / BOHR
    r = compare(np.diag([a, a, a]), [[0, 0, 0], [0.5] * 3], ["Cr", "Cr"], [1.0, -1.0])
    assert abs(r["magnetization"]) < 1e-3 and r["absolute_magnetization"] > 1.0


def test_no_moments_is_paramagnetic():
    pp = read_upf(espresso.pseudo("Al_std.upf"))
    common = dict(kgrid=(3, 3, 3), smearing="gaussian", sigma=0.01,
                  density_tolerance=1e-13, energy_tolerance=1e-12)  # fmt: skip
    one = scf.run(fcc(4.05), [[0, 0, 0]], [0], [pp], 12.0, **common, symmetry=False)
    two = scf.run(
        fcc(4.05), [[0, 0, 0]], [0], [pp], 12.0, magmoms=[0.0], **common, symmetry=False
    )
    assert two["spins"] == 2 and one["spins"] == 1
    assert two["energy"] == pytest.approx(one["energy"], abs=1e-9)
    assert abs(two["magnetization"]) < 1e-12 and two["absolute_magnetization"] < 1e-12
    np.testing.assert_allclose(two["eigenvalues"][0], one["eigenvalues"], atol=1e-7)
    np.testing.assert_allclose(two["density"].sum(axis=0), one["density"], atol=1e-8)


def iron(**options):
    pp = read_upf(espresso.pseudo("Fe_std.upf"))
    settings = dict(kgrid=(3, 3, 3), smearing="gaussian", sigma=0.01, functional="pbe_pw",
                    magmoms=[2.5], density_tolerance=1e-12, energy_tolerance=1e-11,
                    symmetry=False, stress=False)  # fmt: skip
    settings.update(options)
    return scf.run(bcc(2.87), [[0, 0, 0]], [0], [pp], 16.0, **settings)


def test_held_moments_reach_the_same_ground_state():
    free = iron(hold_steps=0)
    assert free["converged"] and not free["held"].any()
    # hold 0 never lets go before hold_steps: exactly three steps held
    three = iron(hold=0.0, hold_steps=3)
    assert three["converged"]
    np.testing.assert_array_equal(three["held"][:4], [1, 1, 1, 0])
    assert not three["held"][3:].any()
    default = iron()
    assert default["converged"] and 1 <= default["held"].sum() <= 12
    for r in (three, default):
        assert r["energy"] == pytest.approx(free["energy"], abs=1e-8)
        assert r["magnetization"] == pytest.approx(free["magnetization"], abs=1e-5)
        np.testing.assert_allclose(r["density"], free["density"], atol=1e-6)
    # held, a step counts no residual of the magnetization as converged
    short = iron(hold=0.0, hold_steps=3, max_iterations=3)
    assert not short["converged"] and short["held"].all()
    # a tolerance met at the first step lets go at once: the run without hold
    loose = iron(hold=1e9)
    assert not loose["held"].any()
    np.testing.assert_array_equal(loose["energies"], free["energies"])


def test_what_is_not_held():
    ground = iron(hold_steps=0)
    again = iron(start_density=ground["density"], first_tolerance=1e-4)
    assert again["converged"] and not again["held"].any()
    assert again["energy"] == pytest.approx(ground["energy"], abs=1e-8)
    assert not iron(magmoms=[0.0], max_iterations=2)["held"].any()
    damped = iron(adaptive_damping=True, max_iterations=3)
    assert not damped["held"].any()
    pp = read_upf(espresso.pseudo("Al_std.upf"))
    plain = scf.run(fcc(4.05), [[0, 0, 0]], [0], [pp], 12.0, kgrid=(2, 2, 2),
                    smearing="gaussian", sigma=0.01, symmetry=False, max_iterations=2)  # fmt: skip
    assert "held" not in plain and not plain["absolute_magnetizations"].any()
    with pytest.raises(ValueError, match="hold and hold_steps"):
        iron(hold=-1.0)
    with pytest.raises(ValueError, match="hold and hold_steps"):
        iron(hold_steps=-1)


def test_moments_that_do_not_fit():
    pp = read_upf(espresso.pseudo("Al_std.upf"))
    with pytest.raises(ValueError, match="at most its valence"):
        scf.run(fcc(4.05), [[0, 0, 0]], [0], [pp], 12.0, magmoms=[3.5], symmetry=False)
    with pytest.raises(ValueError, match="a moment for every atom"):
        scf.run(
            fcc(4.05), [[0, 0, 0]], [0], [pp], 12.0, magmoms=[0.1, 0.2], symmetry=False
        )
