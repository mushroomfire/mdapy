# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""comm= of molecular dynamics, minimize, neb and the LAMMPS calculator, the
processes stood in for by threads (processes.py): a simulation of its own on
each process with a communicator of one, the same as alone; the processes of
one simulation refusing to run what they do not all hold alike, instead of one
system made of the atoms of each. tests/mpi/check_comm.py checks the same
under mpirun."""

import numpy as np
import pytest

from mdapy.calculators import EAM, LennardJones
from mdapy.errors import InvalidParameter
from mdapy.simulation import MolecularDynamics, minimize, neb
from simulation import processes
from simulation.test_domain import copper, same
from simulation.test_neb import vacancy_hop

TEMPERATURES = (200.0, 400.0, 600.0)


def nvt(temperature, comm=None, beads=1, steps=20):
    potential = EAM.zhou(["Cu"])
    with MolecularDynamics(copper(5), potential, timestep=0.002, beads=beads,
                           comm=comm) as md:  # fmt: skip
        md.velocities(temperature, seed=1)
        options = {"thermostat": "langevin"} if beads > 1 else {}
        md.run(steps, ensemble="nvt", temperature=temperature, seed=2, thermo=10,
               **options)  # fmt: skip
        return md.system, md.thermo


@pytest.mark.parametrize("beads", [1, 2])
def test_a_simulation_of_its_own_on_each_process(monkeypatch, beads):
    """With a communicator of one, each process runs its own temperature,
    as it would alone, while the processes that MPI started would otherwise
    split one."""
    alone = [nvt(t, beads=beads) for t in TEMPERATURES]
    farm = processes.run(
        3, lambda rank: nvt(TEMPERATURES[rank], processes.Alone(), beads), monkeypatch
    )
    for mine, expected in zip(farm, alone):
        if beads == 1:
            same(mine, expected)
        else:
            np.testing.assert_array_equal(
                mine[0]["positions"], expected[0]["positions"]
            )


def test_the_communicator_given_is_that_of_the_processes(monkeypatch):
    """comm= the communicator that MPI started splits as without it."""
    alone = processes.run(2, lambda rank: nvt(300.0), monkeypatch)
    given = processes.run(2, lambda rank: nvt(300.0, processes.world()), monkeypatch)
    for a, b in zip(given, alone):
        same(a, b)


@pytest.mark.parametrize(
    "differ, match",
    [
        ("system", "the system, the time step or the beads"),
        ("velocities", "the velocities"),
        ("run", "the settings of a run"),
        ("controls", "the settings of a run"),
    ],
)
def test_processes_of_one_simulation_refuse_what_differs(monkeypatch, differ, match):
    """A temperature on each process of one dynamics would run one system
    made of the atoms of each, each at its own temperature: every process
    raises instead."""
    from mdapy.simulation.controls import AddForce

    def task(rank):
        system = copper(5 if differ != "system" or rank == 0 else 4)
        with MolecularDynamics(system, EAM.zhou(["Cu"]), timestep=0.002) as md:
            md.velocities(300.0 + 100 * rank * (differ == "velocities"), seed=1)
            temperature = 300.0 + 100 * rank * (differ == "run")
            force = AddForce(np.ones(len(system), dtype=bool),
                             (0.0, 0.0, 0.1 * rank * (differ == "controls")))  # fmt: skip
            md.run(5, ensemble="nvt", temperature=temperature, controls=[force])
            return md.system

    with pytest.raises(InvalidParameter, match=f"differ in {match}: process 1"):
        processes.run(2, task, monkeypatch)


def test_comm_must_be_a_communicator():
    for run in (
        lambda: MolecularDynamics(copper(3), EAM.zhou(["Cu"]), comm="world"),
        lambda: minimize(copper(3), EAM.zhou(["Cu"]), comm=1),
        lambda: neb(*vacancy_hop(), EAM.zhou(["Cu"]), comm=object()),
    ):
        with pytest.raises(InvalidParameter, match="comm must be a communicator"):
            run()


def shaken(seed):
    system = copper(4)
    rng = np.random.default_rng(seed)
    return system.with_columns(
        positions=system["positions"] + rng.normal(0, 0.05, (len(system), 3))
    )


def test_a_minimization_of_its_own_on_each_process(monkeypatch, tmp_path):
    potential = LennardJones({"Cu": (0.4, 2.3)}, 5.0)
    alone = [minimize(shaken(s), potential, max_force=1e-4).energy for s in range(2)]

    def task(rank):
        log = tmp_path / f"min{rank}.log"
        result = minimize(shaken(rank), potential, max_force=1e-4,
                          comm=processes.Alone(), log=log)  # fmt: skip
        return result.energy

    assert processes.run(2, task, monkeypatch) == alone
    # each process its own log
    assert all((tmp_path / f"min{r}.log").read_text() for r in range(2))
    with pytest.raises(InvalidParameter, match="the system or the options"):
        processes.run(2, lambda rank: minimize(shaken(rank), potential), monkeypatch)


def test_a_band_of_its_own_on_each_process(monkeypatch):
    initial, final = vacancy_hop()
    alone = neb(initial, final, EAM.zhou(["Cu"]), images=3, max_force=1e-2)
    farm = processes.run(
        2,
        lambda rank: neb(initial, final, EAM.zhou(["Cu"]), images=3, max_force=1e-2,
                         comm=processes.Alone()),
        monkeypatch,
    )  # fmt: skip
    for result in farm:
        np.testing.assert_array_equal(result.energies, alone.energies)
    with pytest.raises(InvalidParameter, match="the states, the images or the options"):
        processes.run(
            2,
            lambda rank: neb(initial, final, EAM.zhou(["Cu"]), images=3 + rank),
            monkeypatch,
        )


def lennard_jones_in_lammps(comm=None):
    from mdapy.calculators import LAMMPS

    return LAMMPS("pair_style lj/cut 5.0\npair_coeff * * 0.4 2.3", ["Cu"], comm=comm)


def test_lammps_on_the_processes_given(monkeypatch):
    """A LAMMPS without MPI runs on one process, so on a communicator of one,
    not on one of several; a LAMMPS with MPI on several processes computes
    every image of a band on all of them, so neb refuses to share the images
    among them."""
    pytest.importorskip("lammps")
    potential = lennard_jones_in_lammps(processes.Alone())
    assert potential.comm is not None
    expected = lennard_jones_in_lammps().compute(copper(3)).energy
    assert potential.compute(copper(3)).energy == pytest.approx(expected)
    with MolecularDynamics(copper(3), potential, comm=processes.Alone()) as md:
        md.run(2)
    if not potential._mpi:

        def several(rank):
            return lennard_jones_in_lammps(processes.world())

        with pytest.raises(InvalidParameter, match="cannot run on the 2 processes"):
            processes.run(2, several, monkeypatch)
    initial, final = vacancy_hop()
    calculator = lennard_jones_in_lammps()
    calculator._mpi = True  # as a LAMMPS with MPI on every process

    def band(rank):
        return neb(initial, final, calculator, images=2, max_steps=1)

    with pytest.raises(
        InvalidParameter, match="give the calculator comm=MPI.COMM_SELF"
    ):
        processes.run(2, band, monkeypatch)


def test_lammps_passes_on_its_other_errors(monkeypatch):
    """Only the refusal of a communicator by a LAMMPS without MPI falls back
    to a LAMMPS of its own."""
    lammps = pytest.importorskip("lammps")
    from mdapy.calculators import _lammps

    def refusing(*args, **kwargs):
        raise RuntimeError("something else")

    monkeypatch.setattr(lammps, "lammps", refusing)
    with pytest.raises(RuntimeError, match="something else"):
        _lammps._lammps((), processes.Alone())
