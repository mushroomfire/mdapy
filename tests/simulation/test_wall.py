# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Flat walls that reflect the atoms: in MDAPY and in LAMMPS (fix
wall/reflect, and fix wall/piston for a moving wall) the same runs step by
step; atoms that do not interact as they fly, against still and moving walls
and in a turned box, on the processor and on the GPU stood in; the errors."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM, LennardJones, _gpu
from mdapy.errors import InvalidParameter
from mdapy.simulation import MolecularDynamics
from mdapy.simulation.controls import AdaptiveTimestep, Deform, Wall
from simulation.test_controls import MASSES, lammps_constants, potentials  # noqa: F401

from calculators.gpu_stand_in import stand_in


def open_slab(velocity, repeat=(3, 3, 5)):
    """Al with a fifth of Cu, shaken, with the velocities of 300 K, open
    along z, all moving along z."""
    system = build_crystal("fcc", 4.05, elements="Al", repeat=repeat)
    rng = np.random.default_rng(0)
    elements = rng.choice(["Al", "Cu"], len(system), p=[0.8, 0.2])
    masses = np.array([MASSES[e] for e in elements])
    sigma = np.sqrt(8.617e-5 * 300.0 / (masses * 1.0364e-4))
    velocities = rng.normal(size=(len(system), 3)) * sigma[:, None]
    velocities -= (masses @ velocities) / masses.sum()
    system = system.with_columns(
        element=elements,
        mass=masses,
        positions=system["positions"] + rng.normal(0, 0.03, (len(system), 3)),
        velocities=velocities,
    )
    box = mdapy.Box(
        system.box.matrix, origin=system.box.origin, pbc=(True, True, False)
    )
    system = system.with_box(box)
    return system.with_columns(
        velocities=system["velocities"] + np.array([0.0, 0.0, velocity])
    )


def gap(a, b):
    """The largest differences of the positions, the periodic images along
    x and y taken out, and of the velocities."""
    f = (a["positions"] - b["positions"]) @ np.linalg.inv(a.box.matrix)
    f[:, :2] -= np.round(f[:, :2])
    return np.abs(f @ a.box.matrix).max(), np.abs(
        a["velocities"] - b["velocities"]
    ).max()


CASES = [
    ("mirrors", -5.0, {"ensemble": "nve"}),
    ("mirrors", 5.0, {"ensemble": "nvt", "temperature": 300}),
    ("piston", 0.0, {"ensemble": "nve"}),
    ("piston", 0.0, {"ensemble": "nvt", "temperature": 300}),
    ("face", -5.0, {"ensemble": "nve"}),
]


@pytest.mark.parametrize("walls, velocity, options", CASES)
def test_the_same_runs_as_lammps(
    potentials,  # noqa: F811
    lammps_constants,  # noqa: F811
    walls,
    velocity,
    options,
):
    eam, lmp = potentials
    if lmp is None:
        pytest.skip("LAMMPS is not installed")
    system = open_slab(velocity)
    top = float(system["positions"][:, 2].max())
    controls = {
        "mirrors": [Wall(position=-0.5), Wall(side="high", position=top + 0.5)],
        "piston": [Wall(position=-1.0, velocity=10.0)],
        "face": [Wall()],
    }[walls]
    out = []
    for potential in (lmp, eam):
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.run(
                300, thermo=100, controls=controls, thermostat_damping=0.2, **options
            )
            z = md.system["positions"][:, 2]
            md.run(10, thermo=10)  # without the walls again
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    dx, dv = gap(a, b)
    assert dx < 1e-10 and dv < 1e-9
    for column in ("temperature", "potential_energy"):
        np.testing.assert_allclose(ta[column], tb[column], rtol=1e-10)
    if walls == "piston":  # pushed up by the wall, which has moved 6 Å
        assert z.min() > 4.0
    elif walls == "mirrors":
        assert z.min() > -0.5 and z.max() < top + 0.5


def gas(box, positions, velocities):
    """Argon atoms that do not interact, farther apart than the cutoff."""
    system = mdapy.System(np.array(positions, dtype=float), box, elements="Ar")
    return system.with_columns(velocities=np.array(velocities, dtype=float))


def flown(system, controls, steps, device="cpu"):
    argon = LennardJones({"Ar": (0.0104, 3.405)}, 3.0, device=device)
    with MolecularDynamics(system, argon, timestep=0.01) as md:
        md.run(steps, thermo=steps, controls=controls)
        return md.system


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_atoms_that_fly_against_walls(monkeypatch, device):
    """Free atoms bounce off a still wall as if it were a mirror of their
    path; a moving wall throws the atom it meets at twice its speed."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    box = mdapy.Box([30.0, 30.0, 40.0], pbc=(True, True, False))
    positions = [[5.0, 5.0, 5.0], [20.0, 5.0, 12.0], [5.0, 20.0, 28.0]]
    velocities = [[1.0, 0.0, -10.0], [0.0, 0.0, 0.0], [0.0, -2.0, 8.0]]
    system = gas(box, positions, velocities)
    still = flown(system, [Wall(), Wall(side="high", position=34.0)], 100, device)
    # after 1 ps: the first back where it started, its velocity turned; the
    # third has hit the high wall at 34 Å at 0.75 ps and come back 2 Å
    np.testing.assert_allclose(
        still["positions"],
        [[6.0, 5.0, 5.0], [20.0, 5.0, 12.0], [5.0, 18.0, 32.0]],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        still["velocities"], [[1.0, 0.0, 10.0], [0.0, 0.0, 0.0], [0.0, -2.0, -8.0]]
    )
    # a wall from z = 2 Å at 20 Å/ps meets the second atom at 0.5 ps: then
    # 40 Å/ps, z = 2 + 40 t - 12 Å ahead of the wall's path, unfolded
    piston = flown(system, [Wall(position=2.0, velocity=20.0)], 60, device)
    t = 0.6
    assert piston["positions"][1, 2] == pytest.approx(2 * (2.0 + 20.0 * t) - 12.0)
    assert piston["velocities"][1, 2] == pytest.approx(40.0)
    # the first, met head on at 0.1 ps: 2 × 20 + 10 Å/ps
    assert piston["velocities"][0, 2] == pytest.approx(50.0)


def test_a_turned_box():
    """A wall across x in a box whose open vector is not along x: the
    dynamics turn the box, and the wall with it."""
    box = mdapy.Box([[30.0, 9.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]])
    box = mdapy.Box(box.matrix, pbc=(False, True, True))
    system = gas(
        box, [[8.0, 5.0, 5.0], [20.0, 15.0, 9.0]], [[-10.0, 3.0, 1.0], [0.0, 0.0, 0.0]]
    )
    out = flown(system, [Wall("x", position=3.0)], 100)
    x, v = out["positions"], out["velocities"]
    # 0.5 ps to the wall at 3 Å and 0.5 ps back
    assert x[0, 0] == pytest.approx(8.0, abs=1e-9)
    np.testing.assert_allclose(v[0], [10.0, 3.0, 1.0], atol=1e-9)
    f = box.to_fractional(x)[0]  # y and z periodic
    expected = box.to_fractional(np.array([[8.0, 8.0, 6.0]]))[0]
    np.testing.assert_allclose((f - expected + 0.5) % 1.0 - 0.5, 0, atol=1e-9)


def test_on_the_gpu(monkeypatch):
    """The slab between two walls and before a piston on the GPU, its module
    stood in for on the processor: the runs of the processor."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    system = open_slab(-5.0, (4, 4, 5))
    top = float(system["positions"][:, 2].max())
    controls = [Wall(position=-0.5), Wall(side="high", position=top + 0.5)]
    for walls in (controls, [Wall(position=-1.0, velocity=10.0)]):
        out = []
        for device in ("cuda", "cpu"):
            eam = EAM.zhou(["Al", "Cu"], device=device)
            with MolecularDynamics(system, eam, timestep=0.002) as md:
                md.run(100, thermo=50, controls=walls)
                out.append(md.system)
        a, b = out
        dx, dv = gap(a, b)
        assert dx < 1e-9 and dv < 1e-8


def test_errors(potentials):  # noqa: F811
    eam, lmp = potentials
    for options, match in (
        ({"axis": "w"}, "axis"),
        ({"axis": 2}, "axis"),
        ({"side": "top"}, "side"),
        ({"position": np.nan}, "position"),
        ({"velocity": True}, "velocity"),
        ({"velocity": np.inf}, "velocity"),
    ):
        with pytest.raises(InvalidParameter, match=match):
            Wall(**options)
    assert repr(Wall()) == "Wall(axis='z', side='low', position=None, velocity=0.0)"
    system = open_slab(0.0)
    with MolecularDynamics(system, eam, timestep=0.002) as md:
        for controls, match in (
            ([Wall(), Wall(position=-1.0)], "two walls"),
            ([Wall(), Deform(xx=0.01)], "Deform"),
            ([Wall(velocity=1.0), AdaptiveTimestep(0.1)], "adaptive"),
            ([Wall("x")], "periodic vectors"),
        ):
            with pytest.raises(InvalidParameter, match=match):
                md.run(1, controls=controls)
        md.run(2, controls=[Wall(), Wall(side="high"), AdaptiveTimestep(0.1)])
    if lmp is None:
        return
    with MolecularDynamics(system, lmp, timestep=0.002) as md:
        for wall in (Wall(velocity=-1.0), Wall(side="high", velocity=-1.0), Wall("x")):
            with pytest.raises(InvalidParameter, match="wall"):
                md.run(1, controls=[wall])
        md.run(2)  # still runs
    turned = mdapy.Box([[30.0, 9.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]])
    turned = mdapy.Box(turned.matrix, pbc=(False, True, True))
    flat = mdapy.System([[8.0, 5.0, 5.0]], turned, elements="Al")
    with MolecularDynamics(flat, lmp, timestep=0.002) as md:
        with pytest.raises(InvalidParameter, match="first vector"):
            md.run(1, controls=[Wall("x")])
