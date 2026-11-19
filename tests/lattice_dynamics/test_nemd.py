# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Nonequilibrium molecular dynamics between two baths: the heat they give
balances, the gradient follows the baths, and solid argon conducts as it
should, in MDAPY, in LAMMPS where it is installed, and on the GPU stood in."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import LAMMPS, LennardJones, _gpu
from mdapy.core.threads import CancelToken
from mdapy.errors import Cancelled, InvalidParameter
from mdapy.lattice_dynamics import nemd
from mdapy.lattice_dynamics._nemd import _gradient

from calculators.gpu_stand_in import stand_in

LJ = (0.0104, 3.405)
FAST = dict(
    difference=10.0,
    timestep=0.004,
    equilibration=1000,
    relaxation=4000,
    production=12000,
    every=50,
    blocks=4,
)


def argon(device="cpu"):
    return LennardJones({"Ar": LJ}, 8.5, shift=True, device=device)


def long_box(repeat=(4, 4, 12)):
    return build_crystal("fcc", 5.28, elements="Ar", repeat=repeat)


def test_the_gradient_of_a_profile():
    positions = (np.arange(20) + 0.5) / 20 * 100.0
    profile = 50.0 - 10.0 * np.abs(positions - 52.5) / 50.0  # hot in the middle
    assert _gradient(profile, positions) == pytest.approx(0.2)


def test_solid_argon_between_two_baths():
    result = nemd(long_box(), argon(), 40.0, **FAST)
    assert 0.2 < result.kappa < 2.0 and result.error > 0
    # the heat the hot bath gave the cold one took, but for the change of the
    # energy of the atoms between, which a short run still sees
    assert result.hot[-1] > 1.0
    assert result.cold[-1] == pytest.approx(-result.hot[-1], rel=0.35)
    t = result.temperatures
    assert t[10] > t[5] > t[0] and t[10] > t[15] > t[0]
    assert t[10] == pytest.approx(45.0, abs=3.0) and t[0] == pytest.approx(
        35.0, abs=3.0
    )
    assert result.gradient > 0 and result.flux > 0
    assert result.length == pytest.approx(12 * 5.28)
    assert result.area == pytest.approx((4 * 5.28) ** 2)
    assert result.temperature == pytest.approx(40.0, rel=0.05)
    assert len(result.times) == len(result.hot) == 240
    assert repr(result).startswith("NEMD(")
    assert not result.temperatures.flags.writeable


def test_in_lammps_and_on_the_gpu(monkeypatch):
    """The same calculation in the other engines, the baths drawing other
    random forces: the conductivity of the same order."""
    pytest.importorskip("lammps")
    lammps = LAMMPS(
        "pair_style lj/cut 8.5\npair_coeff * * 0.0104 3.405\npair_modify shift yes",
        ["Ar"],
    )
    assert 0.2 < nemd(long_box(), lammps, 40.0, **FAST).kappa < 2.0
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    assert 0.2 < nemd(long_box(), argon("cuda"), 40.0, **FAST).kappa < 2.0


def test_progress_and_cancel():
    seen = []
    options = dict(
        FAST,
        equilibration=10,
        relaxation=10,
        production=200,
        every=10,
        blocks=2,
        bins=10,
    )
    nemd(
        long_box((3, 3, 8)),
        argon(),
        40.0,
        progress=lambda d, t: seen.append(d),
        **options,
    )
    assert seen[-1] == 220 and seen == sorted(seen)
    token = CancelToken()
    with pytest.raises(Cancelled):
        nemd(
            long_box((3, 3, 8)),
            argon(),
            40.0,
            cancel=token,
            progress=lambda d, t: token.cancel(),
            **options,
        )


def test_errors():
    box = long_box((3, 3, 8))
    for wrong, match in (
        (dict(axis=3), "axis"),
        (dict(axis=True), "axis"),
        (dict(difference=0.0), "difference"),
        (dict(difference=90.0), "difference"),
        (dict(bins=11), "even"),
        (dict(bins=8), "even"),
        (dict(production=0), "positive"),
        (dict(blocks=1), "blocks"),
        (dict(production=100, every=50), "two samples"),
        (dict(damping=0.0), "damping"),
        (dict(bins=40), "every slab"),
    ):
        with pytest.raises(InvalidParameter, match=match):
            nemd(box, argon(), 40.0, **wrong)
    with pytest.raises(InvalidParameter, match="temperature"):
        nemd(box, argon(), 0.0)
    with pytest.raises(InvalidParameter, match="calculator"):
        nemd(box, "argon", 40.0)
    flat = box.with_box(mdapy.Box(box.box.matrix, pbc=(True, True, False)))
    with pytest.raises(InvalidParameter, match="periodic along"):
        nemd(flat, argon(), 40.0)
