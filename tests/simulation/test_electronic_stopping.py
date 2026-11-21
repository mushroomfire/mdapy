# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Electronic stopping, fix electron/stopping: in MDAPY and in LAMMPS the
same runs step by step, of every atom, of a group, with atoms held, with a
thermostat and with an atom in vacuum; the force and the energy lost in a
step in closed form, the least energy and the neighbors it needs; on the
GPU stood in; restarts; tables read from files; the errors."""

import numpy as np
import pytest

import mdapy
from mdapy.calculators import EAM, LennardJones, Springs, _gpu
from mdapy.elements import atomic_masses
from mdapy.errors import CalculationFailed, InvalidParameter
from mdapy.simulation import MolecularDynamics, _integrator
from mdapy.simulation.controls import ElectronicStopping, Hold
from simulation.test_controls import MASSES, lammps_constants, potentials, slab  # noqa: F401

from calculators.gpu_stand_in import stand_in

ENERGIES = np.geomspace(1.0, 3000.0, 40)


def stopping(**options):
    """Of Al and Cu, as the square root of the energy, below 3 keV."""
    return ElectronicStopping(
        {"Al": (ENERGIES, 0.1 * np.sqrt(ENERGIES)),
         "Cu": (ENERGIES, 0.15 * np.sqrt(ENERGIES))},
        **options,
    )  # fmt: skip


def fast(system, rows, velocities):
    v = system["velocities"].copy()
    v[rows] = velocities
    return system.with_columns(velocities=v)


def shot():
    """The slab of Al and Cu with three atoms of a few hundred eV."""
    system, bottom, top, middle = slab()
    system = fast(system, [10, 50, 90], [[300.0, 120.0, -40.0],
                                         [-150.0, 220.0, 90.0],
                                         [60.0, -80.0, 250.0]])  # fmt: skip
    return system, bottom, top, middle


def in_vacuum():
    """The slab open along z, with an atom of Cu 9 Å above it flying up at
    300 eV, and one inside at 200 eV."""
    system, bottom, *_ = slab()
    matrix = system.box.matrix.copy()
    matrix[2, 2] += 30.0
    x = system["positions"].copy()
    top = int(np.argmax(x[:, 2]))
    x[top, 2] += 9.0
    system = system.with_columns(positions=x, element=np.where(
        np.arange(len(system)) == top, "Cu", system["element"]),
    ).with_box(mdapy.Box(matrix, pbc=(True, True, False)))  # fmt: skip
    system = system.with_columns(mass=[MASSES[e] for e in system["element"]])
    return fast(system, [top, 40], [[20.0, -10.0, 300.0], [-120.0, 150.0, 60.0]])


def thick():
    """Al with a fifth of Cu, 20 Å across, at 300 K, with three atoms of a
    few hundred eV, thick enough for the minimum image of the GPU; its
    bottom and top layers."""
    from mdapy.build import build_crystal

    system = build_crystal("fcc", 4.05, elements="Al", repeat=(5, 5, 5))
    rng = np.random.default_rng(1)
    elements = rng.choice(["Al", "Cu"], len(system), p=[0.8, 0.2])
    system = system.with_columns(element=elements)
    masses = np.array([MASSES[e] for e in elements])
    sigma = np.sqrt(8.617e-5 * 300.0 / (masses * 1.0364e-4))
    v = rng.normal(size=(len(system), 3)) * sigma[:, None]
    v -= (masses @ v) / masses.sum()
    system = fast(system.with_columns(velocities=v), [10, 250, 400],
                  [[300.0, 120.0, -40.0], [-150.0, 220.0, 90.0], [60.0, -80.0, 250.0]])  # fmt: skip
    z = system.box.to_fractional(system["positions"])[:, 2]
    return system, z < 0.15, z > 0.85


def cases():
    system, bottom, top, middle = shot()
    return {
        "every atom": (system, [stopping(min_energy=10.0)], {}),
        "a group": (system, [stopping(mask=middle | (np.arange(len(system)) == 50))],
                    {"ensemble": "nvt", "temperature": 300}),
        "held": (system, [Hold(bottom), stopping(min_energy=30.0)], {}),
        "vacuum": (in_vacuum(), [stopping(min_neighbors=1)], {}),
    }  # fmt: skip


def gap(a, b):
    f = (a["positions"] - b["positions"]) @ np.linalg.inv(a.box.matrix)
    f[:, a.box.pbc] -= np.round(f[:, a.box.pbc])
    return np.abs(f @ a.box.matrix).max(), np.abs(
        a["velocities"] - b["velocities"]
    ).max()


@pytest.mark.parametrize("case", ["every atom", "a group", "held", "vacuum"])
def test_the_same_runs_as_lammps(potentials, lammps_constants, case):  # noqa: F811
    eam, lmp = potentials
    if lmp is None:
        pytest.skip("LAMMPS is not installed")
    system, controls, options = cases()[case]
    out = []
    for potential in (lmp, eam):
        with MolecularDynamics(system, potential, timestep=0.0005) as md:
            md.run(200, thermo=50, controls=controls, **options)
            md.run(10, thermo=10)  # without it again
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    dx, dv = gap(a, b)
    assert dx < 1e-9 and dv < 1e-8
    np.testing.assert_allclose(
        ta["potential_energy"], tb["potential_energy"], rtol=1e-10
    )
    rows = ta["step"].to_numpy() <= 200
    lost = tb["electronic_loss"].to_numpy()[rows]
    np.testing.assert_allclose(
        ta["electronic_loss"].to_numpy()[rows], lost, rtol=1e-10, atol=1e-12
    )
    assert lost[0] == 0.0 and lost[-1] > 1.0  # the fast atoms lost energy
    assert tb["electronic_loss"][-1] is None  # not in the run after it


def alone(velocities, positions=((10.0, 10.0, 10.0),), device="cpu"):
    """Argon atoms that do not interact, in a box of 20 Å."""
    box = mdapy.Box([20.0, 20.0, 20.0])
    system = mdapy.System(np.array(positions, dtype=float), box, elements="Ar")
    system = system.with_columns(velocities=np.array(velocities, dtype=float))
    return system, LennardJones({"Ar": (0.0, 3.405)}, 5.0, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_one_step_in_closed_form(monkeypatch, device):
    """An atom alone at speed v feels -S(E) v / |v| from the forces of the
    first step on, not at the setup, and loses S |v| dt in a step, S linear
    in the table between the energies about E; it feels none below the
    least energy or the first of the table."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    v0 = np.array([30.0, -40.0, 120.0])
    dt = 0.001
    mass = atomic_masses(["Ar"])[0]
    energy = 0.5 * _integrator._MVV2E * mass * float(v0 @ v0)
    table = np.array([10.0, 20.0, 50.0, 100.0])
    values = np.array([1.0, 2.0, 4.0, 3.0])
    drag = np.interp(energy, table, values)
    system, argon = alone([v0], device=device)
    control = ElectronicStopping(
        {"Ar": (table, values)}, min_energy=5.0, min_neighbors=0
    )
    with MolecularDynamics(system, argon, timestep=dt) as md:
        md.run(1, controls=[control])
        v1 = md.system["velocities"][0]
        lost = md.thermo["electronic_loss"].to_numpy()
    speed = float(np.sqrt(v0 @ v0))
    expected = v0 - 0.5 * dt * _integrator._FTM2V * drag / mass * v0 / speed
    np.testing.assert_allclose(v1, expected, rtol=1e-12)
    np.testing.assert_allclose(lost, [0.0, drag * speed * dt], rtol=1e-12)
    for slow in (ElectronicStopping({"Ar": (table, values)}, min_energy=energy + 1.0,
                                    min_neighbors=0),
                 ElectronicStopping({"Ar": (table * 4, values)}, min_energy=1.0,
                                    min_neighbors=0)):  # fmt: skip
        with MolecularDynamics(system, argon, timestep=dt) as md:
            md.run(1, controls=[slow])
            np.testing.assert_array_equal(md.system["velocities"][0], v0)
            assert md.thermo["electronic_loss"][-1] == 0.0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_the_neighbors_it_needs(monkeypatch, device):
    """An atom feels it with min_neighbors others closer than the cutoff of
    the potential, 5 Å, not with them farther, nor alone."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    table = (np.array([1.0, 1000.0]), np.array([2.0, 2.0]))
    v = [[0.0, 0.0, 100.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for gap_, needed, slowed in ((4.0, 1, True), (4.0, 3, False), (5.5, 1, False),
                                 (5.5, 0, True)):  # fmt: skip
        positions = [
            [10.0, 10.0, 10.0],
            [10.0 + gap_, 10.0, 10.0],
            [10.0, 10.0 + gap_, 10.0],
        ]
        system, argon = alone(v, positions, device=device)
        control = ElectronicStopping(
            {"Ar": table}, min_energy=1.0, min_neighbors=needed
        )
        with MolecularDynamics(system, argon, timestep=0.001) as md:
            md.run(1, controls=[control])
            assert bool(md.system["velocities"][0, 2] < 100.0) is slowed, (
                gap_,
                needed,
            )


def test_on_the_gpu_as_on_the_processor(monkeypatch):
    """The GPU, stood in for, gives the run of the processor, with a group
    whose bit comes after those of the watched groups."""
    from mdapy.simulation.controls import Move

    monkeypatch.setattr(_gpu, "cuda", stand_in)
    system, bottom, top = thick()
    out = []
    for device in ("cpu", "cuda"):
        eam = EAM.zhou(["Al", "Cu"], device=device, precision="double")
        controls = [Move(bottom, (0.0, 0.0, 0.0), name="base"),
                    Move(top, (0.0, 0.0, 0.2), name="top"),
                    stopping(mask=~bottom, min_energy=10.0)]  # fmt: skip
        with MolecularDynamics(system, eam, timestep=0.0005) as md:
            md.run(40, thermo=10, controls=controls)
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    dx, dv = gap(a, b)
    assert dx < 1e-9 and dv < 1e-8
    np.testing.assert_allclose(
        ta["electronic_loss"].to_numpy(), tb["electronic_loss"].to_numpy(), rtol=1e-10
    )
    assert ta["electronic_loss"][-1] > 1.0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_a_run_finished_from_its_restart(monkeypatch, tmp_path, device):
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    system, *_ = thick()
    eam = EAM.zhou(["Al", "Cu"], device=device, precision="double")
    options = {"thermo": 10, "controls": [stopping(min_energy=10.0)]}
    with MolecularDynamics(system, eam, timestep=0.0005) as md:
        md.run(40, **options)
        whole = md.system, md.thermo
    path = tmp_path / "md.restart"
    with MolecularDynamics(system, eam, timestep=0.0005) as md:
        frames = md.iterate(
            40, save_every=20, restart=path, restart_every=20, **options
        )
        for frame in frames:
            if frame.info["timestep"] >= 40:
                break
        frames.close()
    again = MolecularDynamics.load(path, eam)
    assert again.step == 20
    again.resume()
    (a, ta), (b, tb) = whole, (again.system, again.thermo)
    np.testing.assert_allclose(b["velocities"], a["velocities"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        tb["electronic_loss"].to_numpy(), ta["electronic_loss"].to_numpy(), rtol=1e-12
    )


def test_tables_read_from_files(tmp_path):
    path = tmp_path / "stopping.txt"
    path.write_text(
        "# energy (eV)  Al  Cu (eV/A)\n1.0 0.1 0.2\n10.0 0.5 0.7  # a comment\n"
        "100.0 1.5 2.0\n"
    )
    control = ElectronicStopping.read(path, ["Al", "Cu"], min_energy=2.0)
    np.testing.assert_array_equal(control.energies, [1.0, 10.0, 100.0])
    np.testing.assert_array_equal(control.stopping["Cu"], [0.2, 0.7, 2.0])
    assert control.min_energy == 2.0 and control.min_neighbors == 1
    assert "ElectronicStopping(['Al', 'Cu'], 1.0 to 100.0 eV" in repr(control)
    with pytest.raises(InvalidParameter, match="not one for each of 3"):
        ElectronicStopping.read(path, ["Al", "Cu", "Ni"])
    with pytest.raises(InvalidParameter, match="cannot read"):
        ElectronicStopping.read(tmp_path / "missing.txt", ["Al"])


def test_errors(potentials):  # noqa: F811
    e = np.array([1.0, 10.0])
    for bad in ({}, [(e, e)], {"Al": e}, {"Al": 5.0}, {"Al": (e, e[:1])}, {"Al": ([1.0], [1.0])},
                {"Al": (e[::-1], e)}, {"Al": (e - 1.0, e)}, {"Al": (e, -e)},
                {"Al": (e, [1.0, np.nan])},
                {"Al": (e, e), "Cu": (e * 2, e)}):  # fmt: skip
        with pytest.raises(InvalidParameter):
            ElectronicStopping(bad)
    for options in ({"min_energy": -1.0}, {"min_neighbors": -1},
                    {"min_neighbors": 1.5}, {"min_neighbors": True}):  # fmt: skip
        with pytest.raises(InvalidParameter):
            ElectronicStopping({"Al": (e, e)}, **options)
    system, *_ = shot()
    eam, lmp = potentials
    with MolecularDynamics(system, eam, timestep=0.0005) as md:
        with pytest.raises(InvalidParameter, match="one ElectronicStopping"):
            md.run(1, controls=[stopping(), stopping()])
        with pytest.raises(InvalidParameter, match="for each of the"):
            md.run(1, controls=[stopping(mask=np.ones(3, dtype=bool))])
        with pytest.raises(InvalidParameter, match="no table for Cu"):
            md.run(1, controls=[ElectronicStopping({"Al": (ENERGIES, ENERGIES)})])
        with pytest.raises(CalculationFailed, match="faster than the stopping table"):
            md.run(10, controls=[ElectronicStopping(
                {"Al": (e * 10, e), "Cu": (e * 10, e)})])  # fmt: skip
    if lmp is not None:
        with MolecularDynamics(system, lmp, timestep=0.0005) as md:
            with pytest.raises(InvalidParameter, match="no table for Cu"):
                md.run(1, controls=[ElectronicStopping({"Al": (ENERGIES, ENERGIES)})])
    springs = Springs(system["positions"], np.full(len(system), 1.0))
    with MolecularDynamics(system, springs, timestep=0.0005) as md:
        with pytest.raises(InvalidParameter, match="min_neighbors=0"):
            md.run(1, controls=[stopping()])
        md.run(1, controls=[stopping(min_neighbors=0)])
    untyped = mdapy.System(table=system.table.drop("element"), box=system.box)
    with MolecularDynamics(untyped, springs, timestep=0.0005) as md:
        with pytest.raises(InvalidParameter, match="an element column"):
            md.run(1, controls=[stopping(min_neighbors=0)])


def test_groups_on_the_gpu(monkeypatch):
    """The GPU has bits for 31 groups, the group of a stopping among them."""
    from mdapy.simulation.controls import Move

    monkeypatch.setattr(_gpu, "cuda", stand_in)
    system, *_ = thick()
    eam = EAM.zhou(["Al", "Cu"], device="cuda", precision="double")
    none = np.zeros(len(system), dtype=bool)
    moves = [Move(none, (0.0, 0.0, 0.0), name=f"m{k}") for k in range(31)]
    with MolecularDynamics(system, eam, timestep=0.0005) as md:
        with pytest.raises(InvalidParameter, match="moved, bathed or stopped"):
            md.run(1, controls=[*moves, stopping(mask=~none)])
        md.run(1, controls=[*moves, stopping()])  # every atom: no group of its own
        md.run(1, controls=[*moves[:30], stopping(mask=~none)])


def test_too_fast_on_the_gpu(monkeypatch):
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    system, *_ = thick()
    eam = EAM.zhou(["Al", "Cu"], device="cuda", precision="double")
    e = np.array([1.0, 10.0])
    with MolecularDynamics(system, eam, timestep=0.0005) as md:
        with pytest.raises(CalculationFailed, match="faster than the stopping table"):
            md.run(10, controls=[ElectronicStopping(
                {"Al": (e * 10, e), "Cu": (e * 10, e)})])  # fmt: skip
        md.run(5, controls=[stopping()])  # it goes on after
