# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Molecular dynamics in MDAPY: analytic averages of an Einstein crystal,
the atoms a movable column holds, and the bookkeeping of the runs. The runs
against LAMMPS are in test_native_dynamics_lammps.py."""

import numpy as np
import pytest

import mdapy
from mdapy import units
from mdapy.build import build_crystal
from mdapy.calculators import EAM, Calculator, CalculatorResult
from mdapy.elements import atomic_masses
from mdapy.errors import CalculationFailed, Cancelled, InvalidParameter, MissingProperty
from mdapy.simulation import MolecularDynamics


class Einstein(Calculator):
    """Each atom on a spring of stiffness k (eV/Å²) to its site."""

    cutoff = 0.0
    implemented = frozenset({"energy", "forces"})

    def __init__(self, sites, box, k=1.0):
        self.sites, self.box, self.k = np.asarray(sites), box, k

    def compute(self, system, properties=("energy", "forces"), *, neighbors=None):
        d = system["positions"] - self.sites
        f = self.box.to_fractional(d + self.box.origin)
        d = (f - np.round(f)) @ self.box.matrix
        return CalculatorResult(
            energy=0.5 * self.k * float((d * d).sum()), forces=-self.k * d
        )


class NotFinite(Calculator):
    cutoff = 0.0
    implemented = frozenset({"energy", "forces"})

    def compute(self, system, properties=("energy", "forces"), *, neighbors=None):
        forces = np.zeros((len(system), 3))
        forces[0, 0] = np.nan
        return CalculatorResult(energy=0.0, forces=forces)


def alloy(seed=0):
    system = build_crystal("fcc", 4.05, elements="Al", repeat=(2, 2, 2))
    kinds = np.random.default_rng(seed).choice(["Al", "Cu"], len(system))
    return system.with_columns(element=kinds)


@pytest.fixture(scope="module")
def eam():
    return EAM.zhou(["Al", "Cu"])


def test_langevin_samples_the_einstein_crystal():
    """Each atom of a crystal of springs has (3/2) kT of kinetic and of
    potential energy on average, whatever its mass."""
    system = alloy()
    springs = Einstein(system["positions"], system.box)
    temperature = 400.0
    with MolecularDynamics(system, springs, timestep=0.002) as md:
        md.velocities(temperature, seed=1)
        md.run(500, ensemble="nvt", temperature=temperature, thermostat="langevin",
               seed=2)  # fmt: skip
        md.run(20000, ensemble="nvt", temperature=temperature, thermostat="langevin",
               seed=3, thermo=10, save_every=20)  # fmt: skip
        thermo, frames = md.thermo[1:], md.frames
    kt = units.BOLTZMANN * temperature
    n = len(system)
    assert thermo["kinetic_energy"].mean() / n == pytest.approx(1.5 * kt, rel=0.03)
    assert thermo["potential_energy"].mean() / n == pytest.approx(1.5 * kt, rel=0.03)
    # equipartition between the light and the heavy atoms
    masses = atomic_masses(system["element"])
    v = np.stack([f["velocities"] for f in frames])
    per_atom = 0.5 * (masses[:, None] * v * v).sum(axis=2).mean(axis=0)
    per_atom *= units.AMU_A2_PER_PS2_TO_EV
    light = np.asarray(system["element"]) == "Al"
    for group in (light, ~light):
        assert per_atom[group].mean() == pytest.approx(1.5 * kt, rel=0.05)


def test_the_energies_of_a_hot_liquid(eam):
    """In a liquid at 3000 K the atoms move much farther than half the skin
    of the kept neighbor list: the energy of every frame is that of the
    potential computed afresh."""
    crystal = build_crystal("fcc", 4.05, elements="Al", repeat=(3, 3, 3))
    start = crystal.with_box(crystal.box.matrix * 1.15 ** (1 / 3), scale_positions=True)
    with MolecularDynamics(start, eam, timestep=0.002) as md:
        md.velocities(3000, seed=1)
        md.run(300, ensemble="nvt", temperature=3000, thermostat="langevin",
               seed=1, thermo=50, save_every=50)  # fmt: skip
        thermo, frames = md.thermo, md.frames
    moved = frames[-1]["positions"] + frames[-1]["image"] @ start.box.matrix
    assert np.linalg.norm(moved - frames[0]["positions"], axis=1).max() > 2.0
    for frame, energy in zip(frames, thermo["potential_energy"], strict=True):
        assert energy == pytest.approx(eam.compute(frame).energy, abs=1e-9)


def test_the_same_seed_the_same_run(eam):
    def last(seed):
        with MolecularDynamics(alloy(), eam, timestep=0.002) as md:
            md.velocities(300, seed=4)
            md.run(50, ensemble="nvt", temperature=300, thermostat="langevin",
                   seed=seed)  # fmt: skip
            return md.system

    a, b, c = last(5), last(5), last(6)
    assert np.array_equal(a["positions"], b["positions"])
    assert not np.array_equal(a["positions"], c["positions"])


@pytest.mark.parametrize("rotated", [False, True])
def test_movable_atoms(eam, rotated):
    """Atoms held by the movable column stay still along the directions
    where it is False, also in a box that LAMMPS would rotate; the
    temperature counts the directions in which the atoms move."""
    system = build_crystal("fcc", 4.05, elements="Al", repeat=(3, 3, 3))
    if rotated:
        matrix = system.box.matrix @ np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]]
        )
        system = system.with_box(matrix, scale_positions=True)
    z = system.box.to_fractional(system["positions"])[:, 2]
    movable = np.ones((len(system), 3), dtype=bool)
    movable[z < 0.1] = False
    movable[50] = [False, False, True]
    held = system.with_columns(movable=movable)
    with MolecularDynamics(held, eam, timestep=0.002) as md:
        md.velocities(300, seed=2)
        start = md.system
        assert np.abs(start["velocities"][~movable]).max() == 0.0
        assert md.thermo.is_empty()
        md.run(300, ensemble="nvt", temperature=300, thermostat="langevin", seed=1)
        md.run(300, ensemble="nvt", temperature=300)
        end = md.system
    moved = (end["positions"] + end["image"] @ end.box.matrix) - start["positions"]
    assert np.abs(moved[~movable]).max() < 1e-12
    assert np.abs(moved[movable]).max() > 0.05
    v = start["velocities"]
    masses = atomic_masses(start["element"])
    twice = units.AMU_A2_PER_PS2_TO_EV * float((masses[:, None] * v * v).sum())
    assert twice / (movable.sum() * units.BOLTZMANN) == pytest.approx(300.0)
    # with every atom movable, the column changes nothing
    free = system.with_columns(movable=np.ones((len(system), 3), dtype=bool))
    with MolecularDynamics(free, eam) as a, MolecularDynamics(system, eam) as b:
        for md in (a, b):
            md.velocities(300, seed=1)
            md.run(20)
        assert np.array_equal(a.system["positions"], b.system["positions"])


def test_velocities(eam):
    system = alloy()
    with MolecularDynamics(system, eam) as md:
        md.velocities(500, seed=3)
        first = md.system
        md.velocities(500, seed=3)
        again = md.system
        assert md.step == 0
        md.velocities(0, seed=3)
        assert np.abs(md.system["velocities"]).max() == 0.0
    v = first["velocities"]
    masses = atomic_masses(first["element"])
    np.testing.assert_allclose(masses @ v, 0.0, atol=1e-10)
    twice = units.AMU_A2_PER_PS2_TO_EV * float((masses[:, None] * v * v).sum())
    assert twice / ((3 * len(system) - 3) * units.BOLTZMANN) == pytest.approx(500.0)
    assert np.array_equal(v, again["velocities"])
    # the velocities of the system start the dynamics
    with MolecularDynamics(first, eam) as md:
        assert np.array_equal(md.system["velocities"], v)


def test_thermo_frames_and_the_system(eam):
    system = alloy()
    with MolecularDynamics(system, eam, timestep=0.002) as md:
        md.velocities(300, seed=1)
        md.run(100, thermo=30, save_every=50)
        md.run(20, ensemble="nvt", temperature=300, thermo=10)
        thermo, frames, final = md.thermo, md.frames, md.system
        assert repr(md) == "MolecularDynamics(32 atoms, step 120)"
    assert repr(md) == "MolecularDynamics(32 atoms, closed)"
    assert thermo["step"].to_list() == [0, 30, 60, 90, 100, 110, 120]
    np.testing.assert_allclose(thermo["time"], thermo["step"] * 0.002)
    total = thermo["total_energy"].to_numpy()
    np.testing.assert_allclose(
        total, thermo["potential_energy"] + thermo["kinetic_energy"]
    )
    assert np.ptp(total[:5]) / len(system) < 1e-4  # nve keeps the energy
    assert [f.info["timestep"] for f in frames] == [0, 50, 100]
    assert final.info == {"timestep": 120, "time": pytest.approx(0.24)}
    assert set(final.table.columns) == {"element", "positions", "velocities", "image"}
    fractional = final.box.to_fractional(final["positions"])
    assert ((fractional >= 0) & (fractional < 1)).all()
    # the pressure and stress of the last record: the virial and the
    # kinetic part
    found = eam.compute(final, ("virial",))
    masses = atomic_masses(final["element"])
    v = final["velocities"]
    kinetic = units.AMU_A2_PER_PS2_TO_EV * np.einsum("i,ij,ik->jk", masses, v, v)
    tensor = (kinetic + found.virial) / final.box.volume * units.EV_PER_A3_TO_GPA
    stress = thermo["stress"][-1].to_numpy()
    np.testing.assert_allclose(stress[:3], -np.diag(tensor), atol=1e-9)
    assert thermo["pressure"][-1] == pytest.approx(np.trace(tensor) / 3, abs=1e-9)
    assert thermo["volume"][-1] == pytest.approx(final.box.volume)
    with pytest.raises(InvalidParameter, match="closed"):
        md.system
    with pytest.raises(InvalidParameter, match="closed"):
        md.run(1)


def test_a_cluster_and_a_sheet(eam):
    crystal = alloy()
    cluster = crystal.with_box(mdapy.Box(30.0, pbc=False))
    with MolecularDynamics(cluster, eam, timestep=0.002) as md:
        md.velocities(300, seed=1)
        md.run(200)
        thermo = md.thermo
        assert md.system.box.pbc.tolist() == [False, False, False]
    assert np.ptp(thermo["total_energy"].to_numpy()) / len(crystal) < 1e-3
    sheet = build_crystal("fcc", 4.05, elements="Al", repeat=(3, 3, 2))
    sheet = sheet.with_box(
        mdapy.Box(np.diag([12.15, 12.15, 30.0]), pbc=(True, True, False))
    )
    with MolecularDynamics(sheet, eam, timestep=0.002) as md:
        md.velocities(300, seed=1)
        md.run(300, ensemble="npt", temperature=300, pressure=0.0, barostat="aniso")
        box = md.system.box.matrix
    assert box[2, 2] == 30.0 and box[0, 0] != 12.15


def test_an_ase_calculator():
    """Any calculator, such as an ASE one with no neighbor list of MDAPY."""
    from ase.calculators.emt import EMT

    from mdapy.interop import from_ase_calculator

    system = build_crystal("fcc", 4.05, elements="Al", repeat=(2, 2, 2))
    with MolecularDynamics(system, from_ase_calculator(EMT()), timestep=0.002) as md:
        md.velocities(300, seed=1)
        md.run(100, ensemble="npt", temperature=300, pressure=0.0, thermo=50)
        thermo = md.thermo
    assert len(thermo) == 3 and np.isfinite(thermo["pressure"].to_numpy()).all()


def test_cancel_and_progress(eam):
    token = mdapy.CancelToken()
    seen = []
    with MolecularDynamics(alloy(), eam) as md:
        md.velocities(300, seed=1)
        with pytest.raises(Cancelled):
            md.run(1000, thermo=10, cancel=token,
                   progress=lambda d, t: seen.append(d) or (d >= 50 and token.cancel()))  # fmt: skip
        assert md.step == 50 and seen[:2] == [10, 20]
        md.run(10)
        assert md.step == 60


def test_the_instance_of_lammps_is_not_there(eam):
    with MolecularDynamics(alloy(), eam) as md:
        with pytest.raises(InvalidParameter, match="LAMMPS calculator"):
            md.command("run 0")
        with pytest.raises(InvalidParameter, match="LAMMPS calculator"):
            md.lammps


def test_invalid(eam):
    system = alloy()
    with pytest.raises(InvalidParameter, match="MDAPY calculator"):
        MolecularDynamics(system, object())
    with pytest.raises(MissingProperty, match="mass"):
        MolecularDynamics(system.drop("element"), eam)
    springs = Einstein(system["positions"], system.box)
    with MolecularDynamics(system, springs) as md:
        md.velocities(300, seed=1)
        with pytest.raises(InvalidParameter, match="virial"):
            md.run(10, ensemble="nph", pressure=0.0)
    with MolecularDynamics(system, eam) as md:
        with pytest.raises(
            InvalidParameter, match="nph needs atoms with a temperature"
        ):
            md.run(10, ensemble="nph", pressure=0.0)
        with pytest.raises(InvalidParameter, match="temperature must be positive"):
            md.run(10, ensemble="nvt", temperature=0.0)
        with pytest.raises(InvalidParameter, match="thermostat must be one of"):
            md.run(10, ensemble="nvt", temperature=300, thermostat="berendsen")
        with pytest.raises(InvalidParameter, match="seed must be positive"):
            md.run(10, ensemble="nvt", temperature=300, thermostat="langevin", seed=0)
    held = system.with_columns(movable=np.zeros((len(system), 3), dtype=bool))
    with MolecularDynamics(held, eam) as md:
        with pytest.raises(InvalidParameter, match="atoms that move"):
            md.run(10, ensemble="nvt", temperature=300)
    with MolecularDynamics(system, NotFinite()) as md:
        with pytest.raises(CalculationFailed, match="not finite"):
            md.run(1)


def same_frames(a, b):
    assert [f.info for f in a] == [f.info for f in b]
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x["positions"], y["positions"])
        np.testing.assert_array_equal(x["velocities"], y["velocities"])
        np.testing.assert_array_equal(x.box.matrix, y.box.matrix)


def test_iterate_gives_the_frames_of_run(eam, tmp_path):
    system = alloy()
    with MolecularDynamics(system, eam, timestep=0.002) as kept:
        kept.velocities(300, seed=1)
        kept.run(100, ensemble="nvt", temperature=300, thermo=30, save_every=25)
        kept.run(50, thermo=30, save_every=25)
        frames, thermo = kept.frames, kept.thermo
    with MolecularDynamics(system, eam, timestep=0.002) as md:
        md.velocities(300, seed=1)
        first = md.iterate(
            100, save_every=25, ensemble="nvt", temperature=300, thermo=30
        )
        assert md.step == 0  # nothing runs until the frames are read
        given = list(first)
        assert [f.info["timestep"] for f in given] == [0, 25, 50, 75, 100]
        # the next run does not give the frame at 100 again
        given += list(md.iterate(50, save_every=25, thermo=30))
        same_frames(given, frames)
        assert md.thermo.equals(thermo) and md.frames == ()
        # written as the run goes, one frame at a time
        mdapy.io.write(tmp_path / "run.xyz", md.iterate(40, save_every=20))
        final = md.system
    with mdapy.io.read_trajectory(tmp_path / "run.xyz") as written:
        assert len(written) == 2  # at 170 and 190: 150 was given before
        np.testing.assert_allclose(
            written[-1]["positions"], final["positions"], atol=1e-6
        )
        np.testing.assert_allclose(
            written[-1]["velocities"], final["velocities"], atol=1e-6
        )


def test_iterate_stops_where_the_reading_ends(eam):
    system = alloy()
    with MolecularDynamics(system, eam, timestep=0.002) as whole:
        whole.velocities(300, seed=1)
        whole.run(40, ensemble="nvt", temperature=300)
        whole.run(20, ensemble="nvt", temperature=300)
        expected = whole.system
    with MolecularDynamics(system, eam, timestep=0.002) as md:
        md.velocities(300, seed=1)
        for frame in md.iterate(100, save_every=10, ensemble="nvt", temperature=300):
            if frame.info["timestep"] == 40:
                break
        assert md.step == 40
        # the thermostat of the broken run is gone: the next run is the same
        md.run(20, ensemble="nvt", temperature=300)
        same_frames([md.system], [expected])


def test_nothing_else_runs_while_frames_are_read(eam):
    token = mdapy.CancelToken()
    with MolecularDynamics(alloy(), eam) as md:
        md.velocities(300, seed=1)
        frames = md.iterate(50, save_every=10)
        next(frames)
        for change in (
            lambda: md.run(1),
            lambda: md.velocities(300, seed=1),
            lambda: next(md.iterate(10, save_every=5)),
        ):
            with pytest.raises(InvalidParameter, match="being read"):
                change()
        assert md.system.info["timestep"] == 0  # reading the state is fine
        frames.close()
        md.run(10)
        assert md.step == 10
        # cancelled between frames, with the progress of the steps run
        seen = []
        with pytest.raises(Cancelled):
            for _ in md.iterate(100, save_every=10, thermo=10, cancel=token,
                                progress=lambda d, t: seen.append(d) or (d >= 30 and token.cancel())):  # fmt: skip
                pass
        assert md.step == 40 and seen == [10, 20, 30]
        md.run(5)
        # closing the dynamics ends the frames being read
        frames = md.iterate(50, save_every=10)
        next(frames)
    with pytest.raises(StopIteration):
        next(frames)
    with pytest.raises(InvalidParameter, match="closed"):
        md.step


def test_invalid_iterate(eam):
    with MolecularDynamics(alloy(), eam) as md:
        # checked at once, before any frame is read
        with pytest.raises(InvalidParameter, match="iterate needs save_every"):
            md.iterate(10, save_every=None)
        with pytest.raises(InvalidParameter, match="save_every must be positive"):
            md.iterate(10, save_every=0)
        with pytest.raises(InvalidParameter, match="steps must not be negative"):
            md.iterate(-1, save_every=1)
        with pytest.raises(InvalidParameter, match="nvt needs a temperature"):
            md.iterate(10, save_every=1, ensemble="nvt")
        assert [f.info["timestep"] for f in md.iterate(0, save_every=5)] == [0]
        assert list(md.iterate(3, save_every=5)) == []  # step 0 was given
    # a run of no steps records the start once
    with MolecularDynamics(alloy(), eam) as md:
        md.run(0, save_every=1)
        assert md.thermo["step"].to_list() == [0]
        assert [f.info["timestep"] for f in md.frames] == [0]
