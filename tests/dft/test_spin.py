# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Collinear spin on the GPU against Quantum ESPRESSO run live: ferromagnetic
bcc iron and fcc nickel, antiferromagnetic chromium. The energy (with the
first-order effect of the rounded gamma of QE's spin-polarized PBE), the total
and absolute magnetization, the Fermi level and the occupied eigenvalues.
Without moments the spins give the paramagnetic result."""

import numpy as np
import pytest

from mdapy.calculators._dft import scf
from mdapy.calculators._dft.upf import read_upf

from dft import espresso

_dft = pytest.importorskip("mdapy_dft_cuda12._dft")
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
        density_tolerance=1e-14, energy_tolerance=1e-12,
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
    one = scf.run(fcc(4.05), [[0, 0, 0]], [0], [pp], 12.0, **common)
    two = scf.run(fcc(4.05), [[0, 0, 0]], [0], [pp], 12.0, magmoms=[0.0], **common)
    assert two["spins"] == 2 and one["spins"] == 1
    assert two["energy"] == pytest.approx(one["energy"], abs=1e-9)
    assert abs(two["magnetization"]) < 1e-12 and two["absolute_magnetization"] < 1e-12
    np.testing.assert_allclose(two["eigenvalues"][0], one["eigenvalues"], atol=1e-7)
    np.testing.assert_allclose(two["density"].sum(axis=0), one["density"], atol=1e-8)


def test_moments_that_do_not_fit():
    pp = read_upf(espresso.pseudo("Al_std.upf"))
    with pytest.raises(ValueError, match="at most its valence"):
        scf.run(fcc(4.05), [[0, 0, 0]], [0], [pp], 12.0, magmoms=[3.5])
    with pytest.raises(ValueError, match="a moment for every atom"):
        scf.run(fcc(4.05), [[0, 0, 0]], [0], [pp], 12.0, magmoms=[0.1, 0.2])
