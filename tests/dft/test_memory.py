# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The memory of the GPU: the counters of the stream-ordered pool, in use and
reserved, now, at their peaks and at the end of each step of a run; a run
gives back to the device what the pool holds beyond the use of its steps, so
that a small run after work that took much holds the memory of the small one,
and a failed run, on one host thread or several, raises and gives back all;
and the DFT calculator reports them in last["gpu_memory"]."""

import os

import numpy as np
import pytest

import mdapy
from mdapy.calculators import DFT
from mdapy.calculators._dft import scf
from mdapy.calculators._dft.calculator import HARTREE
from mdapy.calculators._dft.upf import read_upf

from dft import espresso, gpu

_dft = gpu.module()
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

BOHR = 0.529177210903
MIB = 1 << 20
KEYS = ("in_use", "in_use_peak", "reserved", "reserved_peak", "free", "total")


def silicon_cell():
    h = 5.431 / 2
    return np.array([[0, h, h], [h, 0, h], [h, h, 0]])


FRACTIONS = np.array([[0.01, -0.02, 0.0], [0.27, 0.25, 0.23]])


def silicon_run(**options):
    pp = read_upf(espresso.pseudo("Si_std.upf"))
    return scf.run(silicon_cell() / BOHR, FRACTIONS, [0, 0], [pp], 12.0, kgrid=(2, 2, 2),
                   functional="pbe_pw", symmetry=False, stress=False, **options)  # fmt: skip


def hold_and_free(megabytes):
    """Arrays of the pool of about that many MiB, taken and freed: the pool
    keeps them reserved."""
    grid = (64, 64, 64)
    count = megabytes * MIB // (16 * 64**3)
    c = np.zeros((count, 1), dtype=np.complex128)
    values = _dft.to_real_space(c, np.zeros(1, dtype=np.int64), grid)
    assert values.shape == (count, *grid)


def test_the_counters():
    m = _dft.memory()
    assert set(m) == set(KEYS)
    assert all(isinstance(m[k], int) and m[k] >= 0 for k in KEYS)
    assert m["in_use"] <= m["reserved"] and 0 < m["free"] <= m["total"]
    assert m["in_use"] <= m["in_use_peak"] and m["reserved"] <= m["reserved_peak"]
    hold_and_free(256)
    held = _dft.memory()
    assert held["reserved"] >= 256 * MIB and held["in_use_peak"] >= 256 * MIB
    _dft.reset_memory_peaks()
    reset = _dft.memory()
    assert reset["in_use_peak"] < 256 * MIB


def test_the_steps():
    r = silicon_run()
    memory = r["memory"]
    steps = len(r["energies"])
    assert set(memory) == {
        "in_use",
        "in_use_peak",
        "reserved",
        "reserved_peak",
        "device_used",
    }
    for k, values in memory.items():
        assert values.shape == (steps,) and values.dtype == np.uint64, k
    assert np.all(memory["in_use"] > 0)
    assert np.all(memory["in_use"] <= memory["reserved"])
    assert np.all(memory["in_use"] <= memory["in_use_peak"])
    assert np.all(memory["reserved"] <= memory["reserved_peak"])
    assert np.all(memory["reserved"] <= memory["device_used"])


def test_a_run_gives_back_what_it_does_not_use():
    hold_and_free(512)
    assert _dft.memory()["reserved"] >= 512 * MIB
    r = silicon_run()
    # what the steps used, and the granule of the pool at most besides
    used = int(r["memory"]["in_use"][-1])
    assert used + 64 * MIB < 512 * MIB
    assert _dft.memory()["reserved"] <= used + 64 * MIB
    # the run's own memory stays for the next run of the cell
    again = silicon_run()
    assert again["energy"] == pytest.approx(r["energy"], abs=1e-10)
    assert _dft.memory()["reserved"] <= int(again["memory"]["in_use"][-1]) + 64 * MIB


@pytest.mark.parametrize("threads", [1, 4])
def test_a_failed_run_gives_back_all(threads):
    # its bands fail after the first bands of the atoms, on one host thread
    # or on several (which waited for ever before)
    hold_and_free(512)
    with pytest.raises(ValueError, match="max_iterations"):
        silicon_run(davidson_iterations=0, kpoint_threads=threads)
    m = _dft.memory()
    assert m["reserved"] <= m["in_use"] + 64 * MIB


def test_the_calculator_reports_them():
    espresso.pseudo("Si_std.upf")
    dft = DFT(encut=12 * HARTREE, kpoints=(2, 2, 2), smearing="fixed", functional="pbe_pw",
              pseudopotentials=os.environ["MDAPY_PSEUDO_DIR"])  # fmt: skip
    cell = silicon_cell()
    dft.compute(mdapy.System(FRACTIONS @ cell, mdapy.Box(cell), elements=["Si", "Si"]))
    m = dft.last["gpu_memory"]
    assert set(m) == {"in_use", "reserved", "device"}
    assert all(isinstance(v, float) for v in m.values())
    assert 0 < m["in_use"] <= m["reserved"] <= m["device"]
