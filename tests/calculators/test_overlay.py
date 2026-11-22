# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Calculators added together: the sum of each one's results, against
LAMMPS pair_style hybrid/overlay atom by atom and in a cascade step by step
with electronic stopping, against finite differences, on the GPU stood in,
and its parameters."""

import numpy as np
import pytest

from mdapy.calculators import EAM, ZBL, LennardJones, Overlay, Springs, _gpu
from mdapy.calculators.testing import check_calculator
from mdapy.errors import InvalidParameter

from .eam_files import alloy
from .gpu_stand_in import stand_in

PROPERTIES = ("energy", "atomic_energies", "forces", "virial", "atomic_virials")


def parts(device="cpu", precision=None):
    return (
        EAM.zhou(["Cu", "Ni"], device=device, precision=precision),
        ZBL(["Cu", "Ni"], 1.0, 2.0, device=device, precision=precision),
    )


def squeezed(seed=0, repeat=4, a=3.0, jitter=0.15, pbc=True):
    """An alloy of copper and nickel squeezed so that pairs come within the
    ZBL's cutoff."""
    return alloy(seed=seed, repeat=repeat, jitter=jitter, a=a, pbc=pbc)


def test_the_sum_of_its_calculators():
    system = squeezed()
    eam, zbl = parts()
    total = Overlay([eam, zbl]).compute(system, PROPERTIES)
    a, b = eam.compute(system, PROPERTIES), zbl.compute(system, PROPERTIES)
    assert abs(b.energy) > 1.0  # the ZBL adds something
    assert total.energy == pytest.approx(a.energy + b.energy, rel=1e-14)
    for name in PROPERTIES[1:]:
        np.testing.assert_allclose(
            getattr(total, name), getattr(a, name) + getattr(b, name), rtol=1e-13,
            atol=1e-13, err_msg=name,
        )  # fmt: skip
    # what one of them leaves out, the sum leaves out
    springs = Springs(system["positions"] + 0.1, np.full(len(system), 1.0))
    out = Overlay([zbl, springs]).compute(system, ("energy", "forces"))
    assert out.virial is None and out.energy == pytest.approx(
        zbl.compute(system).energy + springs.compute(system).energy, rel=1e-14
    )


def test_against_lammps(tmp_path):
    pytest.importorskip("lammps")
    from .lammps_reference import assert_matches, reference

    path = tmp_path / "CuNi.eam.alloy"
    EAM.zhou(["Cu", "Ni"]).write(path)
    system = squeezed(seed=4)
    lines = f"""
pair_style hybrid/overlay eam/alloy zbl 1.0 2.0
pair_coeff * * eam/alloy {path} Cu Ni
pair_coeff 1 1 zbl 29 29
pair_coeff 1 2 zbl 29 28
pair_coeff 2 2 zbl 28 28
"""
    expected = reference(system, ["Cu", "Ni"], lines, tmp_path)
    eam = EAM(path)
    result = Overlay([eam, ZBL(["Cu", "Ni"], 1.0, 2.0)]).compute(system, PROPERTIES)
    assert_matches(result, expected, virials=1e-8)


def test_check_calculator():
    # squeezed, with a virial of 700 eV, which the finite differences give
    # to a few parts in a million
    check_calculator(
        Overlay(parts()), squeezed(seed=2, repeat=2, jitter=0.2), rtol=2e-5, atol=1e-6
    )


def test_a_calculator_given_the_pairs_within_its_cutoff():
    """One that does not pass over the pairs beyond its cutoff gets those
    within it; one without a cutoff builds its own."""

    class Strict(LennardJones):
        _takes_wider_lists = False

    system = squeezed(seed=5, a=3.4)
    strict = Strict({"Cu": (0.1, 2.3), "Ni": (0.1, 2.2)}, 4.0)
    overlay = Overlay([EAM.zhou(["Cu", "Ni"]), strict])
    assert not overlay._takes_wider_lists
    out = overlay.compute(system, ("energy", "forces"))
    alone = strict.compute(system, ("energy", "forces"))
    eam = EAM.zhou(["Cu", "Ni"]).compute(system, ("energy", "forces"))
    np.testing.assert_allclose(out.forces, alone.forces + eam.forces, atol=1e-12)
    springs = Springs(system["positions"], np.ones(len(system)))
    only = Overlay([springs, springs]).compute(system)
    assert only.energy == 0.0


def test_on_the_gpu_stood_in(monkeypatch):
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    system = squeezed(seed=3, repeat=6)
    cpu = Overlay(parts()).compute(system, PROPERTIES)
    gpu = Overlay(parts("cuda", "double"))
    assert gpu._on_device is not None and gpu.device == "cuda"
    out = gpu.compute(system, PROPERTIES)
    np.testing.assert_allclose(out.forces, cpu.forces, rtol=1e-12, atol=1e-11)
    assert out.energy == pytest.approx(cpu.energy, rel=1e-12)
    with pytest.raises(InvalidParameter, match="same elements in the same order"):
        Overlay([EAM.zhou(["Cu", "Ni"], device="cuda"),
                 ZBL(["Ni", "Cu"], 1.0, 2.0, device="cuda")])  # fmt: skip
    with pytest.raises(InvalidParameter, match="all be on the processor or all"):
        Overlay([EAM.zhou(["Cu", "Ni"], device="cuda"), ZBL(["Cu", "Ni"], 1.0, 2.0)])
    springs = Springs(system["positions"], np.ones(len(system)))
    springs.device = "cuda"  # a calculator of the GPU without a potential there
    with pytest.raises(InvalidParameter, match='device="cuda"'):
        Overlay([EAM.zhou(["Cu", "Ni"], device="cuda"), springs])


def test_parameters():
    eam, zbl = parts()
    overlay = Overlay([eam, zbl])
    # splits among processes as its calculators of the same elements
    assert overlay._split is not None and overlay._pairs_once
    assert Overlay([eam, ZBL(["Ni", "Cu"], 1.0, 2.0)])._split is None
    assert overlay.calculators == (eam, zbl)
    assert overlay.cutoff == eam.cutoff and overlay.elements == ("Cu", "Ni")
    assert overlay.implemented == eam.implemented
    assert overlay._heat_flux and overlay._takes_wider_lists
    assert repr(overlay) == f"Overlay([{eam!r}, {zbl!r}])"
    lj = LennardJones((0.1, 2.3), 4.0)  # of any element
    assert Overlay([lj, zbl]).elements is None
    for bad in (eam, [eam], [], [eam, "zbl"], 5):
        with pytest.raises(InvalidParameter):
            Overlay(bad)
