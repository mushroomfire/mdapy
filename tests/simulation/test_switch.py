# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""A potential switched into another along a run: in MDAPY and in LAMMPS
(pair_style hybrid/scaled with ufm, fix ti/spring) the same runs step by
step, with the constants of LAMMPS; the free energy of switching one Einstein
crystal into another; the Langevin noise that keeps the momentum."""

import numpy as np
import pytest

import mdapy
from mdapy.calculators import EAM, LAMMPS, Springs, UhlenbeckFord, _gpu
from mdapy.errors import InvalidParameter
from mdapy.properties._free_energy import _switch
from mdapy.simulation import MolecularDynamics
from mdapy.simulation._controls import smoothstep
from mdapy.simulation.controls import Hold, Move, Switch
from simulation.test_controls import (  # noqa: F401
    gap,
    lammps_constants,
    potentials,
    slab,
)

from calculators.gpu_stand_in import stand_in

UF = (0.5, 1.5)  # eV, Å


def test_the_switching_function():
    t = np.linspace(0, 1, 11)
    values = np.array([smoothstep(x) for x in t])
    assert values[0] == 0.0 and values[-1] == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(values + values[::-1], 1.0, atol=1e-13)
    # flat at both ends: 126 t^5 from either, to the rounding of the sum near 1
    assert smoothstep(1e-6) == pytest.approx(126e-30, rel=1e-4)
    assert abs(1 - smoothstep(1 - 1e-6)) < 1e-12
    switch = Switch(None, start=0.8, stop=0.2)
    assert switch.weight(0.0) == 0.8 and switch.weight(1.0) == pytest.approx(0.2)
    assert Switch(None, start=0.3, stop=0.3).weight(0.5) == 0.3


def test_the_same_runs_as_lammps_with_ufm(potentials, lammps_constants, tmp_path):  # noqa: F811
    """λ from 0 to 1 between EAM and the Uhlenbeck-Ford model: hybrid/scaled
    of LAMMPS with λ a variable of the step."""
    eam, lmp = potentials
    if lmp is None:
        pytest.skip("LAMMPS is not installed")
    from calculators.lammps_reference import has_style

    if not has_style("ufm"):
        pytest.skip("LAMMPS has no pair_style ufm")
    path = tmp_path / "AlCu.eam.alloy"
    EAM.zhou(["Al", "Cu"]).write(path)
    steps = 300
    scaled = LAMMPS(
        "pair_style hybrid/scaled v_mdapy_a eam/alloy v_mdapy_b ufm 6.0\n"
        f"pair_coeff * * eam/alloy {path} Al Cu\n"
        f"pair_coeff * * ufm {UF[0]} {UF[1]}",
        ["Al", "Cu"],
        commands=(
            f"variable mdapy_l equal {_switch(0, steps)}\n"
            "variable mdapy_a equal 1-v_mdapy_l\n"
            "variable mdapy_b equal v_mdapy_l"
        ),
    )
    system, *_ = slab()
    out = []
    with MolecularDynamics(system, scaled, timestep=0.002) as md:
        md.run(steps, thermo=50)
        out.append((md.system, md.thermo))
    switch = Switch(UhlenbeckFord(UF, 6.0), name="uf")
    with MolecularDynamics(system, EAM(path), timestep=0.002) as md:
        md.run(steps, thermo=50, controls=switch)
        out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    dx, dv = gap(a, b)
    assert dx < 1e-9 and dv < 1e-8
    np.testing.assert_allclose(
        ta["potential_energy"], tb["potential_energy"], rtol=1e-10
    )
    lam = tb["uf_lambda"].to_numpy()
    np.testing.assert_allclose(
        lam, [smoothstep(s / steps) for s in tb["step"]], atol=1e-15
    )
    # the difference recorded is that of the two unscaled
    last = b.with_columns(positions=b["positions"])
    difference = (
        UhlenbeckFord(UF, 6.0).compute(last).energy - EAM(path).compute(last).energy
    )
    assert tb["uf_difference"][-1] == pytest.approx(difference, rel=1e-10)


def test_the_same_runs_as_lammps_with_springs(potentials, lammps_constants):  # noqa: F811
    """λ from 0 to 1 between EAM and springs to the places at the start: fix
    ti/spring of LAMMPS with function 2 and no equilibration."""
    eam, lmp = potentials
    if lmp is None:
        pytest.skip("LAMMPS is not installed")
    system, *_ = slab()
    steps, k = 300, 2.0
    out = []
    with MolecularDynamics(system, lmp, timestep=0.002) as md:
        md.command(f"fix mdapy_ti all ti/spring {k} {steps} 0 function 2")
        md.run(steps, thermo=50)
        out.append(md.system)
    springs = Springs(system["positions"], k)
    with MolecularDynamics(system, eam, timestep=0.002) as md:
        md.run(steps, thermo=50, controls=Switch(springs))
        out.append(md.system)
    dx, dv = gap(*out)
    assert dx < 1e-9 and dv < 1e-8


def test_the_free_energy_of_stiffer_springs():
    """One Einstein crystal switched into a stiffer one: the integral of the
    difference of their energies over λ is 3/2 kT ln(k2 / k1) per atom."""
    system, *_ = slab()
    sites = system["positions"]
    first, second, temperature = 1.0, 4.0, 300.0
    with MolecularDynamics(
        system.drop("velocities"), Springs(sites, first), timestep=0.004
    ) as md:
        md.velocities(temperature, seed=1)
        langevin = dict(
            ensemble="nvt",
            temperature=temperature,
            thermostat="langevin",
            thermostat_damping=0.1,
        )
        md.run(1000, seed=2, **langevin)
        start = len(md.thermo)
        md.run(
            8000,
            thermo=10,
            seed=3,
            controls=Switch(Springs(sites, second), name="s"),
            **langevin,
        )
        rows = md.thermo[start:]
    work = np.trapezoid(rows["s_difference"], rows["s_lambda"]) / len(system)
    expected = 1.5 * 8.617333262e-5 * temperature * np.log(second / first)
    assert work == pytest.approx(expected, rel=0.05)


def test_the_kept_momentum(potentials):  # noqa: F811
    eam, lmp = potentials
    system, *_ = slab()
    langevin = dict(
        ensemble="nvt",
        temperature=300.0,
        thermostat="langevin",
        thermostat_damping=0.05,
        seed=4,
    )
    for potential in (eam, lmp):
        if potential is None:
            continue
        momenta = []
        for keep in (False, True):
            with MolecularDynamics(system, potential, timestep=0.002) as md:
                md.velocities(300.0, seed=1)
                md.run(200, keep_momentum=keep, **langevin)
                end = md.system
            masses = np.asarray(end["mass"], dtype=float)
            momenta.append(np.abs(masses @ end["velocities"]).max())
        assert momenta[1] < 1e-9 < momenta[0]


def test_errors(potentials, monkeypatch):  # noqa: F811
    eam, lmp = potentials
    system, bottom, *_ = slab()
    with pytest.raises(InvalidParameter, match="calculator or None"):
        Switch("ufm")
    with pytest.raises(InvalidParameter, match="from 0 to 1"):
        Switch(None, start=1.5)
    assert repr(Switch(None, name="gas")).startswith("Switch(None")
    springs = Springs(system["positions"], 1.0)
    with MolecularDynamics(system, eam) as md:
        with pytest.raises(InvalidParameter, match="one Switch"):
            md.run(1, controls=[Switch(None), Switch(springs)])
        with pytest.raises(InvalidParameter, match="virial"):
            md.run(1, ensemble="nph", pressure=0.0, controls=Switch(springs))
        with pytest.raises(InvalidParameter, match="two controls are named"):
            md.run(
                1, controls=[Switch(None, name="a"), Move(bottom, (0, 0, 0), name="a")]
            )
        md.run(2, controls=Switch(springs, start=0.5, stop=0.5))
        with pytest.raises(InvalidParameter, match="keep_momentum is for"):
            md.run(1, keep_momentum=True)
        with pytest.raises(InvalidParameter, match="bool"):
            md.run(1, keep_momentum=1)
        with pytest.raises(InvalidParameter, match="no momentum to keep"):
            md.run(
                1,
                ensemble="nvt",
                temperature=300.0,
                thermostat="langevin",
                keep_momentum=True,
                controls=Hold(bottom),
            )
    if lmp is not None:
        with MolecularDynamics(system, lmp) as md:
            with pytest.raises(InvalidParameter, match="MDAPY's own dynamics"):
                md.run(1, controls=Switch(None))
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    gpu = EAM.zhou(["Al", "Cu"], device="cuda")
    with MolecularDynamics(big(), gpu) as md:
        with pytest.raises(InvalidParameter, match="compute on the GPU too"):
            md.run(1, controls=Switch(UhlenbeckFord(UF, 6.0)))
        md.run(2)  # usable after the refusal


def big():
    """A crystal thick enough for the GPU."""
    from mdapy.build import build_crystal

    system = build_crystal("fcc", 3.6, elements="Cu", repeat=(5, 5, 5))
    rng = np.random.default_rng(0)
    return system.with_columns(
        positions=system["positions"] + rng.normal(0, 0.05, (len(system), 3)),
        element=list(rng.choice(["Al", "Cu"], len(system), p=[0.2, 0.8])),
    )


@pytest.mark.parametrize("target", ["ufm", "springs", "gas"])
def test_on_the_gpu(monkeypatch, target):
    """A Switch on the GPU, its module stood in for on the processor: the runs
    of the processor, the difference of the energies too."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    system = big()
    out = []
    for device in ("cuda", "cpu"):
        if target == "ufm":  # one kind of atom: species other than those of EAM
            other = UhlenbeckFord(UF, 6.5, device=device)
        elif target == "springs":
            other = Springs(system["positions"], 1.5)
        else:
            other = None
        potential = EAM.zhou(["Al", "Cu"], device=device)
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.velocities(300.0, seed=1)
            md.run(60, thermo=20, controls=Switch(other, name="s"))
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    np.testing.assert_allclose(a["positions"], b["positions"], atol=1e-9)
    np.testing.assert_allclose(a["velocities"], b["velocities"], atol=1e-8)
    np.testing.assert_allclose(ta["s_difference"], tb["s_difference"], rtol=1e-9)
    np.testing.assert_allclose(
        ta["potential_energy"], tb["potential_energy"], rtol=1e-9
    )


def test_the_kept_momentum_on_the_gpu(monkeypatch):
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    with MolecularDynamics(big(), EAM.zhou(["Al", "Cu"], device="cuda")) as md:
        md.velocities(300.0, seed=1)
        md.run(
            100,
            ensemble="nvt",
            temperature=300.0,
            thermostat="langevin",
            keep_momentum=True,
            seed=2,
        )
        end = md.system
    masses = mdapy.elements.atomic_masses(list(end["element"]))
    assert np.abs(masses @ end["velocities"]).max() < 1e-9


def test_springs_in_a_turned_box(potentials):  # noqa: F811
    """The same run in a box turned away from the axes of LAMMPS: the sites
    of the springs turn with the frame of the dynamics."""
    import mdapy

    eam, _ = potentials
    system, *_ = slab()
    angle = 0.7
    turn = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    turned = system.with_box(
        mdapy.Box(system.box.matrix @ turn.T, origin=system.box.origin @ turn.T)
    ).with_columns(
        positions=system["positions"] @ turn.T,
        velocities=system["velocities"] @ turn.T,
    )
    out = []
    for atoms in (system, turned):
        springs = Springs(atoms["positions"], 1.5)
        with MolecularDynamics(atoms, eam, timestep=0.002) as md:
            md.run(100, controls=Switch(springs, name="s"))
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    np.testing.assert_allclose(b["positions"], a["positions"] @ turn.T, atol=1e-9)
    np.testing.assert_allclose(ta["s_difference"], tb["s_difference"], rtol=1e-9)


def test_more_errors(potentials):  # noqa: F811
    eam, _ = potentials
    system, bottom, *_ = slab()
    held = system.with_columns(movable=np.tile(~bottom[:, None], (1, 3)))
    with MolecularDynamics(held, eam) as md:
        with pytest.raises(InvalidParameter, match="no momentum to keep"):
            md.run(
                1,
                ensemble="nvt",
                temperature=300.0,
                thermostat="langevin",
                keep_momentum=True,
            )
    with MolecularDynamics(system, eam) as md:
        with pytest.raises(InvalidParameter, match="two controls are named"):
            md.run(
                1, controls=[Move(bottom, (0, 0, 0), name="a"), Switch(None, name="a")]
            )
