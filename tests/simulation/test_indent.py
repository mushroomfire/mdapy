# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Virtual indenters, fix indent: in MDAPY and in LAMMPS the same runs step
by step, spheres, cylinders and planes, still and moving, across a periodic
corner; their forces and energies in closed form on atoms that do not
interact; the GPU stood in; the errors."""

import numpy as np
import pytest

import mdapy
from mdapy.calculators import EAM, LennardJones, _gpu
from mdapy.errors import InvalidParameter
from mdapy.simulation import MolecularDynamics
from mdapy.simulation.controls import AdaptiveTimestep, Deform, Hold, Indenter
from simulation.test_controls import lammps_constants, potentials, slab  # noqa: F401

from calculators.gpu_stand_in import stand_in


def open_slab():
    """The slab of Al and Cu, open along z; its bottom layer, and its top."""
    system, bottom, *_ = slab()
    box = mdapy.Box(
        system.box.matrix, origin=system.box.origin, pbc=(True, True, False)
    )
    system = system.with_box(box)
    return system, bottom, float(system["positions"][:, 2].max())


def gap(a, b):
    f = (a["positions"] - b["positions"]) @ np.linalg.inv(a.box.matrix)
    f[:, :2] -= np.round(f[:, :2])
    return np.abs(f @ a.box.matrix).max(), np.abs(
        a["velocities"] - b["velocities"]
    ).max()


def cases(top, bottom, length):
    middle = length / 2
    return {
        "sphere": (
            [Indenter(center=(middle, middle, top + 10.5), radius=10.0,
                      velocity=(0, 0, -3.0), name="tip")],
            {"ensemble": "nve"},
        ),
        "corner": (
            [Indenter(center=(0.3, 0.7, top + 7.5), radius=8.0,
                      velocity=(0.5, 0, -3.0), name="tip"), Hold(bottom)],
            {"ensemble": "nvt", "temperature": 300},
        ),
        "cylinder": (
            [Indenter("cylinder", axis="y", center=(middle, 0, top + 8.5), radius=8.0,
                      velocity=(0, 0, -3.0), name="tip")],
            {"ensemble": "nve"},
        ),
        "high plane": (
            [Indenter("plane", axis="z", position=top + 0.5, side="high",
                      velocity=-2.0, stiffness=5.0, name="tip")],
            {"ensemble": "nve"},
        ),
        "low plane": (
            [Indenter("plane", axis="z", position=0.5, side="low", velocity=1.5,
                      name="tip")],
            {"ensemble": "nve"},
        ),
        "inside": (
            [Indenter(center=(middle, middle, 10.0), radius=11.0, side="in",
                      name="tip")],
            {"ensemble": "nve"},
        ),
    }  # fmt: skip


@pytest.mark.parametrize(
    "case", ["sphere", "corner", "cylinder", "high plane", "low plane", "inside"]
)
def test_the_same_runs_as_lammps(potentials, lammps_constants, case):  # noqa: F811
    eam, lmp = potentials
    if lmp is None:
        pytest.skip("LAMMPS is not installed")
    system, bottom, top = open_slab()
    controls, options = cases(top, bottom, system.box.matrix[0, 0])[case]
    out = []
    for potential in (lmp, eam):
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.run(300, thermo=50, controls=controls, thermostat_damping=0.2, **options)
            md.run(10, thermo=10)  # without it again
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    dx, dv = gap(a, b)
    assert dx < 1e-10 and dv < 1e-9
    np.testing.assert_allclose(
        ta["potential_energy"], tb["potential_energy"], rtol=1e-10
    )
    rows = ta["step"].to_numpy() <= 300
    force = np.stack(tb["tip_force"].to_numpy()[rows])
    np.testing.assert_allclose(
        np.stack(ta["tip_force"].to_numpy()[rows]), force, rtol=0, atol=1e-9
    )
    np.testing.assert_allclose(
        ta["tip_energy"].to_numpy()[rows],
        tb["tip_energy"].to_numpy()[rows],
        rtol=0,
        atol=1e-10,
    )
    assert np.abs(force).max() > 0.1  # it touched the atoms
    assert tb["tip_force"][-1] is None  # not in the run after it


def gas(positions, pbc=(True, True, False)):
    """Argon atoms that do not interact, farther apart than the cutoff."""
    box = mdapy.Box([20.0, 20.0, 40.0], pbc=pbc)
    return mdapy.System(np.array(positions, dtype=float), box, elements="Ar")


def first_record(system, indenter, device="cpu"):
    argon = LennardJones({"Ar": (0.0104, 3.405)}, 3.0, device=device)
    with MolecularDynamics(system, argon, timestep=0.001) as md:
        md.run(0, controls=indenter)
        row = md.thermo[-1]
        return np.array(row["tip_force"][0]), row["tip_energy"][0]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_forces_in_closed_form(monkeypatch, device):
    """An atom 2 Å inside a sphere of 5 Å feels K (R - r)^2 out along the
    radius, the indenter as much back, its energy K (R - r)^3 / 3; across the
    periodic face of the box as in it; a cylinder leaves out its axis; a
    plane pushes back across it, from either side; a sphere that keeps the
    atoms in pushes the one outside it in."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    k = 4.0
    atoms = gas([[10.0, 10.0, 13.0], [3.0, 3.0, 30.0]])
    on, energy = first_record(
        atoms,
        Indenter(center=(10.0, 10.0, 16.0), radius=5.0, stiffness=k, name="tip"),
        device,
    )
    np.testing.assert_allclose(on, [0.0, 0.0, k * 4.0])  # the atom pushed down
    assert energy == pytest.approx(k * 8.0 / 3.0)
    # the same across the periodic face along x: the center at 19.8, the atom at 0.2
    across = gas([[0.2, 10.0, 13.0], [3.0, 3.0, 30.0]])
    on, energy = first_record(
        across,
        Indenter(center=(19.8, 10.0, 13.0), radius=2.4, stiffness=k, name="tip"),
        device,
    )
    np.testing.assert_allclose(on, [-k * 4.0, 0.0, 0.0], atol=1e-12)
    # and along z in a box periodic in every direction: the center at 39.5,
    # 1 Å from the atom across the face
    on, _ = first_record(
        gas([[10.0, 10.0, 0.5], [3.0, 3.0, 20.0]], pbc=True),
        Indenter(center=(10.0, 10.0, 39.5), radius=4.0, stiffness=k, name="tip"),
        device,
    )
    np.testing.assert_allclose(on, [0.0, 0.0, -k * 9.0], atol=1e-12)
    # a cylinder along y, the atom 3 Å along it from the point given
    on, _ = first_record(
        gas([[10.0, 13.0, 13.0], [3.0, 3.0, 30.0]]),
        Indenter("cylinder", axis="y", center=(10.0, 10.0, 16.0), radius=5.0, stiffness=k,
                 name="tip"),
        device,
    )  # fmt: skip
    np.testing.assert_allclose(on, [0.0, 0.0, k * 4.0], atol=1e-12)
    for side, position, push in (("high", 12.0, -1.0), ("low", 14.0, 1.0)):
        on, energy = first_record(
            gas([[10.0, 10.0, 13.0]]),  # one atom: none other past the plane
            Indenter("plane", axis="z", position=position, side=side, stiffness=k,
                     name="tip"),
            device,
        )  # fmt: skip
        np.testing.assert_allclose(on, [0.0, 0.0, -push * k], atol=1e-12)
        assert energy == pytest.approx(k / 3.0)
    on, energy = first_record(
        gas([[10.0, 10.0, 13.0], [10.0, 10.0, 15.0]]),
        Indenter(center=(10.0, 10.0, 15.0), radius=1.0, side="in", stiffness=k, name="tip"),
        device,
    )  # fmt: skip
    np.testing.assert_allclose(on, [0.0, 0.0, -k * 1.0], atol=1e-12)  # pulled up
    assert energy == pytest.approx(k / 3.0)


def test_a_moving_indenter_follows_its_velocity():
    """A plane at 1 Å/ps reaches an atom at rest 2 Å below it after 2 ps."""
    atoms = gas([[10.0, 10.0, 13.0]])
    argon = LennardJones({"Ar": (0.0104, 3.405)}, 3.0)
    plane = Indenter(
        "plane", axis="z", position=15.0, side="high", velocity=-1.0, name="tip"
    )
    with MolecularDynamics(atoms, argon, timestep=0.01) as md:
        md.run(300, thermo=10, controls=plane)
        thermo = md.thermo
    touched = thermo.filter(thermo["tip_energy"] > 0)["time"].to_numpy()
    assert touched[0] == pytest.approx(2.1)  # the first record after 2 ps


def test_on_the_gpu(monkeypatch):
    """A sphere into the slab on the GPU, its module stood in for on the
    processor: the runs of the processor."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    system, bottom, top = open_slab()
    big = system.with_box(
        mdapy.Box(np.diag([16.2, 16.2, 20.25]), pbc=(True, True, False))
    ).with_columns(positions=system["positions"] * [16.2 / 12.15, 16.2 / 12.15, 1.0])
    out = []
    for device in ("cuda", "cpu"):
        eam = EAM.zhou(["Al", "Cu"], device=device)
        tip = Indenter(center=(0.5, 0.5, top + 7.5), radius=8.0,
                       velocity=(0, 0, -3.0), name="tip")  # fmt: skip
        with MolecularDynamics(big, eam, timestep=0.002) as md:
            md.run(100, thermo=20, controls=[tip, Hold(bottom)])
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    dx, dv = gap(a, b)
    assert dx < 1e-9 and dv < 1e-8
    np.testing.assert_allclose(
        np.stack(ta["tip_force"].to_numpy()),
        np.stack(tb["tip_force"].to_numpy()),
        atol=1e-9,
    )
    assert np.abs(np.stack(tb["tip_force"].to_numpy())).max() > 0.1


def test_errors(potentials):  # noqa: F811
    eam, lmp = potentials
    for options, match in (
        ({"shape": "cone"}, "shape"),
        ({"center": (0, 0, 0), "radius": 1.0, "stiffness": -1.0}, "stiffness"),
        ({"center": (0, 0, 0), "radius": 1.0, "stiffness": True}, "stiffness"),
        ({"radius": 1.0}, "needs a center"),
        ({"center": (0, 0, 0)}, "radius"),
        ({"center": (0, 0, 0), "radius": -1.0}, "radius"),
        ({"center": (0, 0, 0), "radius": 1.0, "axis": "z"}, "no axis"),
        ({"center": (0, 0, 0), "radius": 1.0, "position": 1.0}, "not a position"),
        ({"center": (0, 0, 0), "radius": 1.0, "side": "low"}, "out or in"),
        ({"center": (0, 0, 0), "radius": 1.0, "velocity": 1.0}, "three finite"),
        ({"shape": "cylinder", "center": (0, 0, 0), "radius": 1.0}, "an axis"),
        ({"shape": "plane", "position": 1.0, "side": "low"}, "an axis"),
        ({"shape": "plane", "axis": "z", "side": "low"}, "finite position"),
        ({"shape": "plane", "axis": "z", "position": 1.0}, "low or high"),
        (
            {"shape": "plane", "axis": "z", "position": 1.0, "side": "low",
             "radius": 1.0},
            "not a center",
        ),
        (
            {"shape": "plane", "axis": "z", "position": 1.0, "side": "low",
             "velocity": (0, 0, 1)},
            "one number",
        ),
    ):  # fmt: skip
        with pytest.raises(InvalidParameter, match=match):
            Indenter(**options)
    assert repr(Indenter(center=(1, 2, 3), radius=4.0)) == (
        "Indenter('sphere', center=[1.0, 2.0, 3.0], radius=4.0, side='out', "
        "stiffness=10.0, velocity=[0.0, 0.0, 0.0], name=None)"
    )
    assert repr(
        Indenter("cylinder", axis="y", center=(1, 2, 3), radius=4.0)
    ).startswith("Indenter('cylinder', center=[1.0, 2.0, 3.0], radius=4.0, axis='y',")
    assert repr(Indenter("plane", axis="z", position=5.0, side="low")) == (
        "Indenter('plane', axis='z', position=5.0, side='low', stiffness=10.0, "
        "velocity=0.0, name=None)"
    )
    tip = Indenter(center=(0, 0, 30), radius=1.0, velocity=(0, 0, -1), name="tip")
    still = Indenter(center=(0, 0, 30), radius=1.0, name="tip")
    plane = Indenter("plane", axis="x", position=1.0, side="low")
    system, bottom, _ = open_slab()
    with MolecularDynamics(system, eam) as md:
        for controls, match in (
            ([tip, still], "two controls are named"),
            ([still, Deform(xx=0.01)], "Deform"),
            ([tip, AdaptiveTimestep(0.1)], "moving indenter"),
            ([plane], "not periodic along x"),
        ):
            with pytest.raises(InvalidParameter, match=match):
                md.run(1, controls=controls)
        md.run(2, controls=[still, AdaptiveTimestep(0.1)])
    turned = system.with_box(
        system.box.matrix @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]]),
        scale_positions=True,
    )
    engines = [eam] + ([] if lmp is None else [lmp])
    for potential in engines:
        with MolecularDynamics(turned, potential) as md:
            with pytest.raises(InvalidParameter, match="an indenter needs a box"):
                md.run(1, controls=still)
    if lmp is not None:
        with MolecularDynamics(system, lmp) as md:
            with pytest.raises(InvalidParameter, match="not periodic along x"):
                md.run(1, controls=plane)
            md.run(2, controls=tip)  # still runs, its variables made and deleted
            md.run(2, controls=tip)
