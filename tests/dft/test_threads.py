# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Work side by side on one GPU changes nothing but the order of sums: the
k-points of a run solved on several host threads (norm-conserving silicon,
ultrasoft aluminium, PAW iron with spin) give the energy, the forces, the
stress and the eigenvalues of one thread; two runs of other cells on two
Python threads at once give those of the same runs one after the other."""

import threading

import numpy as np
import pytest

from mdapy.calculators._dft import scf
from mdapy.calculators._dft.upf import read_upf

from dft import espresso

_dft = pytest.importorskip("mdapy_dft_cuda12._dft")
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

BOHR = 0.529177210903


def fcc(a):
    h = a / 2 / BOHR
    return np.array([[0, h, h], [h, 0, h], [h, h, 0]])


def bcc(a):
    h = a / 2 / BOHR
    return np.array([[-h, h, h], [h, -h, h], [h, h, -h]])


CASES = {
    # cell, fractions, file, cutoff, options
    "silicon": (fcc(5.431), [[0.01, -0.02, 0.0], [0.27, 0.25, 0.23]], "Si_std.upf", 12.0,
                {"smearing": "fixed"}),
    "aluminium-us": (fcc(4.04), [[0.0, 0.0, 0.0]], "pslibrary/Al.pbe-n-rrkjus_psl.1.0.0.UPF", 12.0,
                     {"smearing": "gaussian", "sigma": 0.01}),
    "iron-paw": (bcc(2.87), [[0.0, 0.0, 0.0]], "pslibrary/Fe.pbe-spn-kjpaw_psl.1.0.0.UPF", 15.0,
                 {"smearing": "gaussian", "sigma": 0.01, "magmoms": [2.5]}),
}  # fmt: skip


def run(case, cell=None, **extra):
    base, fractions, name, cutoff, options = CASES[case]
    pp = read_upf(espresso.pseudo(name))
    return scf.run(
        base if cell is None else cell, fractions, [0] * len(fractions), [pp], cutoff,
        kgrid=(4, 4, 4), functional="pbe_pw", energy_tolerance=1e-12, density_tolerance=1e-14,
        symmetry=False, **options, **extra,
    )  # fmt: skip


@pytest.mark.parametrize("case", list(CASES))
def test_kpoints_on_threads(case):
    one = run(case, kpoint_threads=1)
    four = run(case, kpoint_threads=4)
    assert one["converged"] and four["converged"]
    assert four["energy"] == pytest.approx(one["energy"], abs=1e-10)
    np.testing.assert_allclose(four["forces"], one["forces"], atol=1e-9)
    np.testing.assert_allclose(
        four["strain_derivative"], one["strain_derivative"], atol=1e-9
    )
    np.testing.assert_allclose(four["eigenvalues"], one["eigenvalues"], atol=1e-9)
    if "becsum" in one:
        np.testing.assert_allclose(four["becsum"], one["becsum"], atol=1e-9)


def test_runs_on_threads():
    cells = [fcc(5.431), fcc(5.431) @ np.diag([1.02, 1.0, 0.99])]
    alone = [run("silicon", cell=c, kpoint_threads=1)["energy"] for c in cells]
    together = [None, None]

    def work(i):
        together[i] = run("silicon", cell=cells[i], kpoint_threads=1)["energy"]

    pool = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    np.testing.assert_allclose(together, alone, atol=1e-10)
