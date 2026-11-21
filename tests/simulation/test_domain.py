# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""MDAPY's dynamics split among processes (simulation/_domain.py), the
processes stood in for by threads (processes.py): the bricks hold each atom
once and as ghosts exactly the images within their reach; a split run gives
the run of one process, in every ensemble and with every control, to the
rounding of the forces, which sum in another order; a restart file does not
depend on the number of processes; and what cannot be split is refused.
tests/mpi/ checks the same under mpirun."""

import itertools
import shutil
import sys
import types

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM, ZBL, LennardJones, Overlay, Springs, _gpu
from mdapy.errors import InvalidParameter
from mdapy.simulation import MolecularDynamics, _domain
from mdapy.simulation._controls import _HeatField
from mdapy.simulation.controls import (
    AdaptiveTimestep,
    AddForce,
    Deform,
    ElectronicStopping,
    Hold,
    Indenter,
    Langevin,
    Move,
    Switch,
    Wall,
)
from simulation import processes

from calculators.gpu_stand_in import stand_in


def copper(repeat=7, tilt=0.0, pbc=(True, True, True)):
    crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(repeat,) * 3)
    matrix = crystal.box.matrix.copy()
    matrix[1, 0] += tilt * matrix[0, 0]
    return crystal.with_box(mdapy.Box(matrix, pbc=pbc), scale_positions=True)


# -- the bricks ---------------------------------------------------------------


def bricks_hold_every_atom(comm, h, pbc, reach, seed=7):
    n = 600
    rng = np.random.default_rng(seed)
    lo = np.array([-1.0, 2.0, 0.5])
    x = lo + rng.random((n, 3)) @ h
    ids = np.arange(n, dtype=np.int64)
    bricks = _domain.Bricks(comm, h, lo, pbc, reach)
    s = bricks.fractional(x)
    brick = np.clip(np.floor(s * bricks.counts), 0, bricks.counts - 1).astype(int)
    mine = (brick == bricks.at).all(axis=1)
    fields = {"positions": x[mine], "image": np.zeros((mine.sum(), 3), np.int64),
              "id": ids[mine]}  # fmt: skip
    # every atom moved a little, some across the faces of the bricks and the box
    width = (1.0 / np.linalg.norm(np.linalg.inv(h), axis=0)) / bricks.counts
    step = rng.normal(0, 0.15, (n, 3)) * min(width.min() / 4, 1.0)
    fields["positions"] = fields["positions"] + step[fields["id"]]
    fields = bricks.migrate(fields)
    every = np.concatenate(comm.allgather(fields["id"]))
    assert sorted(every.tolist()) == list(range(n))
    np.testing.assert_allclose(
        fields["positions"] + fields["image"] @ h, (x + step)[fields["id"]], atol=1e-9
    )
    # the ghosts: every image within the reach of the brick
    xg, ig = bricks.borders(fields["positions"], fields["id"])
    local = bricks.local
    lower, upper = bricks.at / bricks.counts, (bricks.at + 1) / bricks.counts
    allx = np.concatenate(comm.allgather(fields["positions"]))
    alli = np.concatenate(comm.allgather(fields["id"]))
    closed = np.asarray(pbc)
    # along a box vector of one brick there are no ghosts, and the end bricks
    # along one that is not periodic reach out of the box
    one = bricks.counts == 1
    low = np.where(one | (~closed & (bricks.at == 0)), -np.inf, lower - bricks.fraction)
    high = np.where(one | (~closed & (bricks.at == bricks.counts - 1)), np.inf,
                    upper + bricks.fraction)  # fmt: skip
    split = [(-1, 0, 1) if p and not o else (0,) for p, o in zip(pbc, one)]
    expect = []
    for shift in itertools.product(*split):
        y = allx + np.array(shift) @ h
        g = bricks.fractional(y)
        near = ((g >= low) & (g < high)).all(axis=1)
        own = np.isin(alli, fields["id"]) & (np.array(shift) == 0).all()
        expect += [
            (int(alli[k]), *np.round(y[k], 9)) for k in np.flatnonzero(near & ~own)
        ]
    got = [(int(i), *np.round(p, 9)) for i, p in zip(ig[local:], xg[local:])]
    assert sorted(got) == sorted(expect)
    # values go to the ghosts of their atoms, and come back from them added
    np.testing.assert_array_equal(bricks.forward(ig[:local].astype(float)), ig)
    copies = np.bincount(np.concatenate(comm.allgather(ig[local:])), minlength=n)
    back = bricks.reverse(np.ones((len(ig), 2)))
    np.testing.assert_array_equal(back[:, 0], 1 + copies[ig[:local]])
    # the ghosts follow their atoms
    nudge = rng.normal(0, 0.01, (n, 3))
    followed = bricks.follow(
        np.concatenate([xg[:local] + nudge[ig[:local]], xg[local:]])
    )
    np.testing.assert_allclose(
        followed[local:], xg[local:] + nudge[ig[local:]], atol=1e-12
    )
    assert (
        bricks.total(local) == n
        and bricks.maximum(comm.Get_rank()) == comm.Get_size() - 1
    )


BOXES = {
    "cubic": (np.diag([20.0, 18.0, 16.0]), (True, True, True), 3.0),
    "triclinic": (np.array([[20.0, 0, 0], [4.0, 18.0, 0], [-3.0, 2.5, 16.0]]),
                  (True, True, True), 2.5),
    "open z": (np.diag([20.0, 18.0, 30.0]), (True, True, False), 3.0),
    "open": (np.diag([24.0, 24.0, 24.0]), (False, False, False), 3.0),
    # bricks thinner than the reach, which pass the ghosts on
    "thin": (np.diag([12.0, 11.0, 10.0]), (True, True, True), 6.5),
    "thin triclinic": (np.array([[12.0, 0, 0], [3.0, 11.0, 0], [-2.0, 1.5, 10.0]]),
                       (True, True, True), 5.5),
    "thin open z": (np.diag([12.0, 11.0, 20.0]), (True, True, False), 6.5),
}  # fmt: skip


@pytest.mark.parametrize("box", list(BOXES))
@pytest.mark.parametrize("size", [1, 2, 3, 4, 6, 8])
def test_the_bricks_hold_every_atom_and_its_ghosts(monkeypatch, box, size):
    h, pbc, reach = BOXES[box]
    processes.run(
        size, lambda rank: bricks_hold_every_atom(processes.world(), h, pbc, reach),
        monkeypatch,
    )  # fmt: skip


def test_the_grid_has_the_least_surface():
    assert _domain.grid(8, np.diag([10.0, 10.0, 10.0]), np.ones(3, bool)) == (2, 2, 2)
    assert _domain.grid(4, np.diag([40.0, 10.0, 10.0]), np.ones(3, bool)) == (4, 1, 1)
    assert _domain.grid(6, np.diag([30.0, 20.0, 10.0]), np.ones(3, bool)) == (3, 2, 1)
    # ties to the first found, as LAMMPS's procs2box takes them
    assert _domain.grid(7, np.diag([10.0, 10.0, 10.0]), np.ones(3, bool)) == (1, 1, 7)


# -- runs ----------------------------------------------------------------------


def cases(device="cpu"):
    """The runs on many processes; device="cuda", the potentials on the
    GPU, which must stand in for it first."""
    gpu = {"device": "cuda", "precision": "double"} if device == "cuda" else {}
    crystal = copper()
    z = crystal["positions"][:, 2]
    bottom, top = z < z.min() + 2.0, z > z.max() - 2.0
    middle = ~(bottom | top)
    eam = EAM.zhou(["Cu"], **gpu)
    lj = LennardJones({"Cu": (0.4, 2.3)}, 5.0, **gpu)
    springs = Springs(crystal["positions"], np.full(len(crystal), 2.0))
    open_z = copper(pbc=(True, True, False))
    # the top layers outside the box, which grows to hold them
    matrix = open_z.box.matrix.copy()
    matrix[2, 2] -= 3.0
    outside = open_z.with_box(mdapy.Box(matrix, pbc=(True, True, False)))
    half = crystal.box.matrix[0, 0] / 2
    out = {
        "nve": (crystal, eam, {}, 4),
        # bricks thinner than the reach of the ghosts
        "small": (copper(3), eam, {"ensemble": "npt", "temperature": 300,
                                   "pressure": 0.0}, 8),
        # thinner across x and y than the reach of the ghosts, one brick each
        "thin": (build_crystal("fcc", 3.615, elements="Cu", repeat=(3, 3, 8)), eam,
                 {"ensemble": "npt", "temperature": 300, "pressure": 0.0}, 2),
        "lj": (copper(5), lj, {}, 8),
        "nvt": (crystal, eam, {"ensemble": "nvt", "temperature": (300, 500)}, 2),
        "npt": (crystal, eam, {"ensemble": "npt", "temperature": 300, "pressure": 1.0}, 2),
        "npt tri": (copper(tilt=0.1), eam, {"ensemble": "npt", "temperature": 300,
                                            "pressure": 1.0, "barostat": "tri"}, 4),
        "nph aniso": (crystal, eam, {"ensemble": "nph", "pressure": 1.0,
                                     "barostat": "aniso"}, 2),
        "langevin": (crystal, eam, {"ensemble": "nvt", "temperature": 400,
                                    "thermostat": "langevin", "seed": 5,
                                    "keep_momentum": True}, 4),
        # compressed along z, across which two bricks split the box
        "nphug": (copper(8), eam, {"ensemble": "nphug", "pressure": 5.0,
                                   "barostat": "z"}, 2),
        "msst": (copper(8), eam, {"ensemble": "msst", "barostat": "z",
                                  "shock_velocity": 60.0}, 2),
        "groups": (crystal, eam, {"controls": [
            Hold(bottom), Move(top, (0.0, 0.0, 0.5), name="top"),
            AddForce(middle, (0.01, 0.0, 0.0)),
            Langevin(middle, 300.0, damping=0.1, seed=9, name="bath")]}, 4),
        "deform": (crystal, eam, {"ensemble": "nvt", "temperature": 300,
                                  "controls": [Deform(xx=1e-3)]}, 2),
        "adaptive": (crystal, eam, {"controls": [AdaptiveTimestep(0.02)]}, 2),
        "wall": (open_z, eam, {"controls": [Wall("z", side="low", velocity=0.5)]}, 2),
        "outside": (outside, eam, {"ensemble": "nvt", "temperature": 300}, 2),
        "indenter": (open_z, eam, {"controls": [Indenter(
            "sphere", center=(half, half, z.max() + 9.5), radius=10.0,
            velocity=(0.0, 0.0, -3.0), name="tip")]}, 2),
        "switch": (crystal, eam, {"ensemble": "nvt", "temperature": 300,
                                  "thermostat": "langevin", "seed": 2,
                                  "controls": [Switch(springs, name="switch")]}, 2),
        "switch to eam": (crystal, lj, {"controls": [Switch(eam, name="switch")]}, 2),
        "hnemd": (crystal, eam, {"ensemble": "nvt", "temperature": 300,
                                 "controls": [_HeatField((0.0, 0.0, 1e-4))]}, 2),
        "hnemd convective": (crystal, eam, {"ensemble": "nvt", "temperature": 300,
                                            "controls": [_HeatField((1e-4, 0, 0), True)]},
                             2),
        # the atoms of the middle of 0.02 eV or more slowed, with a neighbor
        "stopping": (crystal, eam, {"controls": [ElectronicStopping(
            {"Cu": (np.geomspace(0.01, 10.0, 20), 0.5 * np.geomspace(0.1, 3.0, 20))},
            min_energy=0.02, mask=middle)]}, 4),
        # split as its calculators, of the same elements, one thread each
        # computing each pair once among the processes
        "overlay": (crystal, Overlay([eam, ZBL(["Cu"], 1.0, 2.0, **gpu)]), {}, 4),
        "overlay of pairs": (copper(5), Overlay([lj, ZBL(["Cu"], 1.0, 2.0, **gpu)]), {},
                             8),
        "stopping lj": (copper(5), lj, {"controls": [ElectronicStopping(
            {"Cu": (np.geomspace(0.01, 10.0, 20), np.geomspace(0.1, 3.0, 20))},
            min_energy=0.02)]}, 8),
    }  # fmt: skip
    if device == "cpu":
        # computed whole on each process, with ghosts twice the cutoff thick:
        # its calculators of other elements, which the GPU does not add
        out["overlay whole"] = (crystal, Overlay([eam, ZBL(["Ni", "Cu"], 1.0, 2.0)]), {},
                                4)  # fmt: skip
    return out


CASES = cases()


def run(crystal, potential, options, steps=30):
    with MolecularDynamics(crystal, potential, timestep=0.002) as md:
        md.velocities(300, seed=1)
        md.run(steps, thermo=10, **options)
        return md.system, md.thermo


def same(a, b):
    (sa, ta), (sb, tb) = a, b
    x = (sa["positions"] - sb["positions"]) @ np.linalg.inv(sb.box.matrix)
    x[:, sb.box.pbc] -= np.round(x[:, sb.box.pbc])
    assert np.abs(x @ sb.box.matrix).max() < 1e-9
    np.testing.assert_allclose(sa["velocities"], sb["velocities"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(sa.box.matrix, sb.box.matrix, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(sa["image"], sb["image"])
    assert ta.columns == tb.columns and ta.height == tb.height
    for name in ta.columns:
        np.testing.assert_allclose(
            np.stack(ta[name].to_numpy()), np.stack(tb[name].to_numpy()),
            rtol=1e-9, atol=1e-9, err_msg=name,
        )  # fmt: skip


@pytest.mark.parametrize("case", list(CASES))
def test_a_split_run_is_the_run_of_one_process(monkeypatch, case):
    crystal, potential, options, size = CASES[case]
    alone = run(crystal, potential, options)
    for split in processes.run(size, lambda rank: run(crystal, potential, options),
                               monkeypatch):  # fmt: skip
        same(split, alone)


# the GPU takes boxes thicker than twice the cutoff and the skin along each
# periodic direction, and an overlay of the same elements
ON_GPUS = [c for c in CASES if c not in ("small", "thin", "overlay whole")]


@pytest.mark.parametrize("case", ON_GPUS)
def test_a_run_on_gpus_is_the_run_of_one_gpu(monkeypatch, case):
    """Each process holds its atoms and their ghosts on its GPU, stood in for
    on the processor, where the steps run, the ghosts brought each step and
    the sums summed over the processes: the run of one GPU, to the
    rounding."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    crystal, potential, options, size = cases("cuda")[case]
    alone = run(crystal, potential, options)
    for split in processes.run(size, lambda rank: run(crystal, potential, options),
                               monkeypatch):  # fmt: skip
        same(split, alone)


def test_a_run_on_gpus_through_gpu_aware_mpi(monkeypatch):
    """With an MPI that moves the memory of the GPU, what the processes send
    stays on the device (the stand-in's own arrays): the same run."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    monkeypatch.setenv("MDAPY_GPU_AWARE_MPI", "1")
    crystal, potential, options, size = cases("cuda")["nve"]
    alone = run(crystal, potential, options)
    for split in processes.run(size, lambda rank: run(crystal, potential, options),
                               monkeypatch):  # fmt: skip
        same(split, alone)


class _Gathered:
    """A communicator whose processes said values, this one the first."""

    def __init__(self, *others):
        self.others = others

    def allgather(self, value):
        return [value, *self.others]


def test_gpu_aware_mpi_as_the_environment_or_open_mpi_says(monkeypatch):
    from mdapy.simulation import _domain_device

    aware = _domain_device._gpu_aware
    monkeypatch.setenv("MDAPY_GPU_AWARE_MPI", "1")
    assert aware(_Gathered())
    assert not aware(_Gathered(False))  # the same on every process
    monkeypatch.setenv("MDAPY_GPU_AWARE_MPI", " 0 ")
    assert not aware(_Gathered())
    monkeypatch.setenv("MDAPY_GPU_AWARE_MPI", "yes")
    with pytest.raises(InvalidParameter, match="1 or 0, not 'yes'"):
        aware(_Gathered())
    monkeypatch.delenv("MDAPY_GPU_AWARE_MPI")
    monkeypatch.setattr(_domain_device, "_open_mpi_aware", lambda: True)
    assert aware(_Gathered(True))


def test_open_mpi_tells_whether_it_moves_gpu_memory(monkeypatch, tmp_path):
    from mdapy.simulation import _domain_device

    aware = _domain_device._open_mpi_aware
    assert not aware(str(tmp_path / "missing"))  # no maps of the process
    maps = tmp_path / "maps"
    maps.write_text(
        "7f00-7f01 r-xp 00000000 00:00 0 /nowhere/libmpi.so.40.30.6\n"
        "7f01-7f02 r-xp 00000000 00:00 0 /nowhere/libmpi_mpifh.so.40\n"
        "7f02-7f03 rw-p 00000000 00:00 0\n"
    )
    assert not aware(str(maps))  # a library that cannot be loaded
    loaded = []

    def library(answer):
        def load(path):
            loaded.append(path)
            if answer is None:
                return types.SimpleNamespace()  # another MPI: no query
            return types.SimpleNamespace(MPIX_Query_cuda_support=lambda: answer)

        return load

    monkeypatch.setattr(_domain_device.ctypes, "CDLL", library(1))
    assert aware(str(maps))
    assert loaded == ["/nowhere/libmpi.so.40.30.6"]
    monkeypatch.setattr(_domain_device.ctypes, "CDLL", library(0))
    assert not aware(str(maps))
    monkeypatch.setattr(_domain_device.ctypes, "CDLL", library(None))
    assert not aware(str(maps))


def test_memory_of_a_gpu_goes_to_mpi_as_it_is():
    class Device:
        __cuda_array_interface__ = {"shape": (8,), "typestr": "|u1",
                                    "data": (0, False), "version": 3}  # fmt: skip

    given = []

    class Comm:
        def Get_size(self):  # noqa: N802
            return 2

        def Get_rank(self):  # noqa: N802
            return 0

        def Sendrecv(self, sent, dest=None, recvbuf=None, source=None):  # noqa: N802
            given.append((sent, recvbuf))

    bricks = _domain.Bricks(Comm(), np.diag([40.0, 20.0, 20.0]), np.zeros(3),
                            (True, True, True), 2.0)  # fmt: skip
    sent, received = Device(), Device()
    bricks.swap(sent, 1, received, 1)
    assert given[0][0] is sent and given[0][1] is received
    bricks.swap([1.0, 2.0], 1, np.empty(2), 1)
    assert isinstance(given[1][0], np.ndarray)


def potentials(tmp_path):
    """Each of MDAPY's potentials, with a crystal two bricks can split."""
    from mdapy.calculators import (
        ADP,
        MEAM,
        NEP,
        Morse,
        StillingerWeber,
        Tersoff,
        UhlenbeckFord,
    )

    from calculators.meam_files import CASES as MEAMS
    from calculators.meam_files import write_library, write_parameters
    from calculators.nep_files import write_nep
    from calculators.potential_files import write_adp, write_sw, write_tersoff
    from calculators.test_meam import crystal as meam_crystal

    def mixed(structure, a, elements, repeat, p=None):
        system = build_crystal(structure, a, elements=elements[0], repeat=(repeat,) * 3)
        rng = np.random.default_rng(0)
        return system.with_columns(element=list(rng.choice(elements, len(system), p=p)))

    names, lines, structure, a = MEAMS["silicon and carbon, dia, screening beyond rc"]
    meam = MEAM(write_library(tmp_path / "library.meam"), names,
                write_parameters(tmp_path / "x.meam", lines))  # fmt: skip
    repeat = int(np.ceil(2 * (2 * meam.cutoff + 1.0) / a)) + 1
    return {
        "ADP": (ADP(write_adp(tmp_path / "CuNi.adp")), mixed("fcc", 3.6, ("Cu", "Ni"), 7)),
        "MEAM": (meam, meam_crystal(structure, a, meam.elements, repeat=repeat,
                                    jitter=0.05, seed=3)),
        "NEP": (NEP(write_nep(tmp_path / "nep.txt", zbl="universal")),
                mixed("fcc", 3.6, ("Cu", "Ni"), 8)),
        "Tersoff": (Tersoff(write_tersoff(tmp_path / "SiC.tersoff", ("Si", "C"))),
                    mixed("diamond", 5.43, ("Si", "C"), 4, p=[0.8, 0.2])),
        "Stillinger-Weber": (StillingerWeber(write_sw(tmp_path / "SiGe.sw", ("Si", "Ge"))),
                             mixed("diamond", 5.43, ("Si", "Ge"), 4)),
        "Morse": (Morse({("Cu", "Cu"): (0.34, 1.36, 2.87)}, 6.0), copper(5)),
        "Uhlenbeck-Ford": (UhlenbeckFord((0.5, 1.5), 5.0), copper(5)),
    }  # fmt: skip


@pytest.mark.parametrize(
    "name", ["ADP", "MEAM", "NEP", "Tersoff", "Stillinger-Weber", "Morse",
             "Uhlenbeck-Ford"]
)  # fmt: skip
def test_every_potential_of_mdapy_splits(monkeypatch, tmp_path, name):
    potential, system = potentials(tmp_path)[name]
    options = {"ensemble": "npt", "temperature": 600, "pressure": 0.0}
    alone = run(system, potential, options)
    for split in processes.run(2, lambda rank: run(system, potential, options),
                               monkeypatch):  # fmt: skip
        same(split, alone)


@pytest.mark.parametrize("name", ["Tersoff", "Stillinger-Weber"])
@pytest.mark.parametrize("heat", [False, True])
def test_three_body_potentials_on_gpus(monkeypatch, tmp_path, name, heat):
    """On a GPU each the atoms of each process are the centres of the terms,
    with ghosts within the cutoff, and the forces on the ghosts, with their
    virials for the heat flux, go back to their atoms: the run of one GPU."""
    from mdapy.calculators import StillingerWeber, Tersoff

    from calculators.potential_files import write_sw, write_tersoff

    monkeypatch.setattr(_gpu, "cuda", stand_in)
    gpu = {"device": "cuda", "precision": "double"}
    if name == "Tersoff":
        potential = Tersoff(write_tersoff(tmp_path / "SiC.tersoff", ("Si", "C")), **gpu)
    else:
        potential = StillingerWeber(write_sw(tmp_path / "SiGe.sw", ("Si", "Ge")), **gpu)
    system = potentials(tmp_path)[name][1]
    options = {"ensemble": "nvt", "temperature": 600}
    if heat:
        options["controls"] = [_HeatField((2e-4, 0.0, 1e-4))]
    alone = run(system, potential, options)
    for split in processes.run(3, lambda rank: run(system, potential, options),
                               monkeypatch):  # fmt: skip
        same(split, alone)


def test_atoms_that_move_far(monkeypatch):
    """Hot atoms, which move half the skin between two builds of the
    neighbor lists, many times in the run."""
    crystal, potential = copper(), EAM.zhou(["Cu"])

    def hot():
        with MolecularDynamics(crystal, potential, timestep=0.002) as md:
            md.velocities(3000, seed=1)
            md.run(150, thermo=50)
            return md.system, md.thermo

    alone = hot()
    for split in processes.run(2, lambda rank: hot(), monkeypatch):
        same(split, alone)


def test_the_log_names_the_processes(monkeypatch, tmp_path):
    crystal, potential, options, _ = CASES["lj"]

    def task(rank):
        log = tmp_path / f"md{rank}.log"
        with MolecularDynamics(crystal, potential, timestep=0.002, log=log) as md:
            md.run(10, thermo=10)
        return log.exists()

    # the main process alone writes the log
    assert processes.run(3, task, monkeypatch) == [True, False, False]
    assert (
        "MDAPY on the processor, 3 processes of" in (tmp_path / "md0.log").read_text()
    )


def stop(crystal, potential, options, path):
    """The run stopped after 20 of its 30 steps, its restart file written."""
    with MolecularDynamics(crystal, potential, timestep=0.002) as md:
        md.velocities(300, seed=1)
        frames = md.iterate(30, save_every=10, thermo=10, restart=path, restart_every=20,
                            **options)  # fmt: skip
        for frame in frames:
            if frame.info["timestep"] >= 30:
                break
        frames.close()


def finish(path, potential):
    again = MolecularDynamics.load(path, potential)
    assert again.step == 20
    again.resume()
    return again.system, again.thermo


# the box of "outside" grows to hold its atoms in the file
@pytest.mark.parametrize("case", ["langevin", "hnemd convective", "outside"])
def test_restart_files_do_not_depend_on_the_processes(monkeypatch, tmp_path, case):
    crystal, potential, options, _ = CASES[case]
    whole = run(crystal, potential, options)
    path, copy = tmp_path / "md.restart", tmp_path / "copy.restart"
    # written by two processes, read by two and by one
    processes.run(2, lambda rank: stop(crystal, potential, options, path), monkeypatch)
    shutil.copy(path, copy)
    for continued in processes.run(
        2, lambda rank: finish(path, potential), monkeypatch
    ):
        same(continued, whole)
    same(finish(copy, potential), whole)
    # written by one, read by four
    stop(crystal, potential, options, path)
    for continued in processes.run(
        4, lambda rank: finish(path, potential), monkeypatch
    ):
        same(continued, whole)


@pytest.mark.parametrize("case", ["langevin", "hnemd convective", "groups"])
def test_restart_files_on_gpus_do_not_depend_on_the_processes(monkeypatch, tmp_path,
                                                              case):  # fmt: skip
    """Written by two processes on their GPUs, read by two and by one."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    crystal, potential, options, _ = cases("cuda")[case]
    whole = run(crystal, potential, options)
    path, copy = tmp_path / "md.restart", tmp_path / "copy.restart"
    processes.run(2, lambda rank: stop(crystal, potential, options, path), monkeypatch)
    shutil.copy(path, copy)
    for continued in processes.run(
        2, lambda rank: finish(path, potential), monkeypatch
    ):
        same(continued, whole)
    same(finish(copy, potential), whole)


def test_what_the_gpus_refuse(monkeypatch):
    """A box thinner along a vector cut into bricks than the reach of the
    ghosts, a Switch to a potential of a longer reach than the box holds,
    and positions gone wrong, which every process raises together."""
    from mdapy.errors import CalculationFailed

    monkeypatch.setattr(_gpu, "cuda", stand_in)
    eam = EAM.zhou(["Cu"], device="cuda", precision="double")
    refused(monkeypatch, 2, lambda: MolecularDynamics(copper(1), eam, timestep=0.002),
            "fewer processes")  # fmt: skip

    def squeezed():
        with MolecularDynamics(copper(2), eam, timestep=0.002) as md:
            md.run(30, controls=[Deform(zz=-1.5)])

    # the ghosts of EAM within its cutoff and the skin: thinner than twice
    # those along a direction the box is periodic in, which the GPU refuses
    refused(monkeypatch, 2, squeezed, "fewer processes|too thin")

    def switched():
        small = build_crystal("fcc", 3.2, elements="Cu", repeat=(2, 2, 2))
        lj = LennardJones({"Cu": (0.4, 2.3)}, 5.0, device="cuda", precision="double")
        with MolecularDynamics(small, lj, timestep=0.002) as md:
            md.run(10, controls=[Switch(eam)])

    refused(monkeypatch, 2, switched, "fewer processes")

    def lost(rank):
        with MolecularDynamics(copper(), eam, timestep=0.002) as md:
            md.run(2)
            if rank == 0:  # a position of one process alone gone wrong
                atoms = md._MolecularDynamics__engine._Native__atoms
                atoms._atoms.x[0] = np.nan
            with pytest.raises(CalculationFailed, match="not finite"):
                md.run(5)
        return True

    assert processes.run(2, lost, monkeypatch) == [True, True]


def test_velocities_given_on_gpus(monkeypatch):
    """Velocities given to the dynamics go to the device rows of each
    process, and come back in the order of the atoms."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    crystal = copper()
    eam = EAM.zhou(["Cu"], device="cuda", precision="double")
    v = np.random.default_rng(1).normal(0.0, 3.0, (len(crystal), 3))

    def task(rank):
        with MolecularDynamics(crystal.with_columns(velocities=v), eam,
                               timestep=0.002) as md:  # fmt: skip
            md.run(1)
            given = md.system["velocities"]
            forces = (
                md.system["forces"] if "forces" in md.system.table.columns else None
            )
            md.run(3)
            return given, forces, md.system["positions"]

    alone = task(None)
    for split in processes.run(2, task, monkeypatch):
        np.testing.assert_allclose(split[0], alone[0], atol=1e-9)
        np.testing.assert_allclose(split[2], alone[2], atol=1e-9)


def test_one_process_cancelled_stops_them_all(monkeypatch):
    from mdapy.core.threads import CancelToken
    from mdapy.errors import Cancelled

    crystal, potential, _, _ = CASES["lj"]

    def task(rank):
        token = CancelToken()
        if rank == 0:
            token.cancel()
        with MolecularDynamics(crystal, potential, timestep=0.002) as md:
            with pytest.raises(Cancelled):
                md.run(20, thermo=10, cancel=token)
            md.run(5)  # every process goes on together
        return True

    assert processes.run(2, task, monkeypatch) == [True, True]


def test_a_restart_file_that_cannot_be_written(monkeypatch):
    crystal, potential, _, _ = CASES["nve"]

    def task(rank):
        with MolecularDynamics(crystal, potential, timestep=0.002) as md:
            with pytest.raises(InvalidParameter, match="cannot write the restart file"):
                md.save("/nonexistent-folder/md.restart")
            md.run(1)  # every process goes on together
        return True

    assert processes.run(2, task, monkeypatch) == [True, True]


# -- the world and what is refused ------------------------------------------------


def test_the_world(monkeypatch):
    for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE",
                 "PMIX_RANK", "SLURM_STEP_NUM_TASKS"):  # fmt: skip
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(sys.modules, "mpi4py.MPI", raising=False)
    assert _domain.world() is None
    monkeypatch.setenv("PMI_SIZE", "many")
    assert _domain.world() is None
    # srun's tasks without PMIx, whose MPI would not start, are not processes
    # of MPI
    monkeypatch.setenv("SLURM_STEP_NUM_TASKS", "4")
    assert _domain.world() is None
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    monkeypatch.setenv("PMIX_RANK", "0")  # srun --mpi=pmix
    with pytest.raises(InvalidParameter, match="needs mpi4py"):
        _domain.world()
    monkeypatch.delenv("PMIX_RANK")
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")
    monkeypatch.setitem(sys.modules, "mpi4py", None)
    with pytest.raises(InvalidParameter, match="needs mpi4py"):
        _domain.world()
    one = types.SimpleNamespace(Get_size=lambda: 1)
    four = types.SimpleNamespace(Get_size=lambda: 4)
    fake = types.ModuleType("mpi4py")
    fake.MPI = types.SimpleNamespace(COMM_WORLD=one)
    monkeypatch.setitem(sys.modules, "mpi4py", fake)
    assert _domain.world() is None
    fake.MPI.COMM_WORLD = four
    assert _domain.world() is four
    # mpi4py imported, whatever started the processes
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE")
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", fake.MPI)
    assert _domain.world() is four


def refused(monkeypatch, size, task, match):
    def checked(rank):
        with pytest.raises(InvalidParameter, match=match):
            task()

    processes.run(size, checked, monkeypatch)


def test_what_cannot_be_split(monkeypatch):
    crystal = copper()
    sites = Springs(crystal["positions"], 1.0)
    refused(monkeypatch, 2, lambda: MolecularDynamics(crystal, sites, timestep=0.002),
            "not Springs: run it on one process")  # fmt: skip
    # a box cut into bricks along a vector across which it is thinner than
    # the reach of the ghosts, at the start and on the way
    eam = EAM.zhou(["Cu"])
    refused(monkeypatch, 2, lambda: MolecularDynamics(copper(1), eam, timestep=0.002),
            "fewer processes")  # fmt: skip

    def squeezed():
        with MolecularDynamics(copper(2), eam, timestep=0.002) as md:
            md.run(30, controls=[Deform(zz=-1.5)])  # across the bricks

    refused(monkeypatch, 2, squeezed, "fewer processes")

    def switched():
        # 6.4 Å across: thick enough for Lennard-Jones, not for EAM
        small = build_crystal("fcc", 3.2, elements="Cu", repeat=(2, 2, 2))
        lj = LennardJones({"Cu": (0.4, 2.3)}, 5.0)
        with MolecularDynamics(small, lj, timestep=0.002) as md:
            md.run(10, controls=[Switch(eam)])

    refused(monkeypatch, 2, switched, "fewer processes")


def test_an_atom_too_fast(monkeypatch):
    h = np.diag([40.0, 10.0, 10.0])

    def task(rank):
        comm = processes.world()
        bricks = _domain.Bricks(comm, h, np.zeros(3), (True, True, True), 2.0)
        x = np.array([[5.0 + 10.0 * rank, 5.0, 5.0]])
        if rank == 0:
            x[0, 0] += 20.0  # two bricks on
        with pytest.raises(InvalidParameter, match="farther than a brick"):
            bricks.migrate({"positions": x, "image": np.zeros((1, 3), np.int64)})

    processes.run(4, task, monkeypatch)


@pytest.mark.parametrize("name", ["EAM", "Lennard-Jones", "stopping", "overlay"])
def test_processes_with_a_gpu_each(monkeypatch, name):
    """Each process computes the forces of its atoms and ghosts on its GPU,
    stood in for on the processor, the steps on the processor: the run of one
    process on the GPU, to the rounding."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    if name == "EAM":
        crystal, potential = (
            copper(),
            EAM.zhou(["Cu"], device="cuda", precision="double"),
        )
        options = {"ensemble": "npt", "temperature": 300, "pressure": 0.0}
    elif name == "overlay":
        crystal, potential = copper(), Overlay([
            EAM.zhou(["Cu"], device="cuda", precision="double"),
            ZBL(["Cu"], 1.0, 2.0, device="cuda", precision="double")])  # fmt: skip
        options = {"ensemble": "nvt", "temperature": 300}
    elif name == "stopping":  # its neighbors counted on the processor
        crystal, potential = (
            copper(),
            EAM.zhou(["Cu"], device="cuda", precision="double"),
        )
        options = CASES["stopping"][2]
    else:
        crystal = copper(5)
        potential = LennardJones({"Cu": (0.4, 2.3)}, 5.0, device="cuda",
                                 precision="double")  # fmt: skip
        options = {"ensemble": "nvt", "temperature": 300}
    alone = run(crystal, potential, options)
    for split in processes.run(2, lambda rank: run(crystal, potential, options),
                               monkeypatch):  # fmt: skip
        same(split, alone)


@pytest.mark.parametrize("case", ["nve", "npt tri", "groups", "hnemd convective",
                                  "small", "outside", "lj", "switch to eam",
                                  "stopping", "stopping lj", "overlay",
                                  "overlay of pairs"])  # fmt: skip
def test_processes_of_one_thread(monkeypatch, case):
    """On one thread a process computes each pair of its own atoms once, as
    LAMMPS: the run of one process too, to the rounding."""
    from mdapy.core.threads import set_num_threads

    crystal, potential, options, size = CASES[case]
    alone = run(crystal, potential, options)
    set_num_threads(1)
    for split in processes.run(size, lambda rank: run(crystal, potential, options),
                               monkeypatch):  # fmt: skip
        same(split, alone)


def test_each_process_its_gpu(monkeypatch):
    """Under MPI the GPU part takes the device of the rank of the process on
    its node, once, before anything is on a device."""
    chosen = []
    fake = types.SimpleNamespace(
        device_name=lambda: "a GPU", device_count=lambda: 4, use_device=chosen.append
    )
    monkeypatch.setattr(_gpu, "module", lambda: fake)
    for name in ("OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID",
                 "MV2_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"):  # fmt: skip
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_gpu, "_chosen", [])
    assert _gpu.cuda() is fake and chosen == []  # not under MPI
    monkeypatch.setattr(_gpu, "_chosen", [])
    monkeypatch.setenv("MPI_LOCALRANKID", "many")
    monkeypatch.setenv("SLURM_LOCALID", "6")
    _gpu.cuda()
    _gpu.cuda()
    assert chosen == [2]


# -- minimization ----------------------------------------------------------------


def shaken(system, seed=0, scale=0.08):
    rng = np.random.default_rng(seed)
    return system.with_columns(
        positions=system["positions"] + rng.normal(0, scale, (len(system), 3))
    )


def minimizations(tmp_path):
    from mdapy.calculators import MEAM, Tersoff

    from calculators.meam_files import CASES as MEAMS
    from calculators.meam_files import write_library, write_parameters
    from calculators.potential_files import write_tersoff
    from calculators.test_meam import crystal as meam_crystal

    eam = EAM.zhou(["Cu"])
    names, lines, structure, a = MEAMS["silicon and carbon, dia, screening beyond rc"]
    meam = MEAM(write_library(tmp_path / "library.meam"), names,
                write_parameters(tmp_path / "x.meam", lines))  # fmt: skip
    repeat = int(np.ceil(2 * (2 * meam.cutoff + 1.0) / a)) + 1
    silicon = build_crystal("diamond", 5.43, elements="Si", repeat=(4, 4, 4))
    rng = np.random.default_rng(3)
    silicon = silicon.with_columns(element=list(rng.choice(["Si", "C"], len(silicon),
                                                           p=[0.8, 0.2])))  # fmt: skip
    return {
        "eam": (shaken(copper()), eam, {}, 4),
        "eam cell": (shaken(copper(tilt=0.05)), eam, {"cell": "full"}, 2),
        "lj": (shaken(copper(5)), LennardJones({"Cu": (0.4, 2.3)}, 5.0), {}, 8),
        "tersoff": (shaken(silicon, scale=0.05),
                    Tersoff(write_tersoff(tmp_path / "SiC.tersoff", ("Si", "C"))), {}, 4),
        "meam": (meam_crystal(structure, a, meam.elements, repeat=repeat, jitter=0.05,
                              seed=3), meam, {}, 2),
        "small": (shaken(copper(3)), eam, {"cell": "isotropic"}, 8),
        "surface": (shaken(copper(pbc=(True, True, False))), eam, {}, 4),
    }  # fmt: skip


@pytest.mark.parametrize(
    "case", ["eam", "eam cell", "lj", "tersoff", "meam", "small", "surface"]
)
def test_a_minimization_on_many_processes(monkeypatch, tmp_path, case):
    """Every process takes the steps of FIRE over every atom, the forces
    shared among them: the minimization of one process, to the rounding."""
    import io

    from mdapy.simulation import minimize

    system, potential, options, size = minimizations(tmp_path)[case]

    def run(rank):
        log = io.StringIO()
        result = minimize(system, potential, max_force=1e-4, max_steps=300, log=log,
                          **options)  # fmt: skip
        return result, log.getvalue()

    alone, _ = run(None)
    assert alone.steps > 10
    results = processes.run(size, run, monkeypatch)
    for result, log in results:
        assert result.steps == alone.steps and result.converged == alone.converged
        assert result.energy == pytest.approx(alone.energy, rel=1e-11, abs=1e-9)
        np.testing.assert_allclose(result.system["positions"],
                                   alone.system["positions"], atol=1e-7)  # fmt: skip
        np.testing.assert_allclose(result.system.box.matrix, alone.system.box.matrix,
                                   atol=1e-9)  # fmt: skip
    logs = [log for _, log in results]
    assert "energy minimization" in logs[0] and not any(logs[1:])


def test_a_minimization_in_lammps_on_many_processes(monkeypatch):
    """A calculator that does not split, as LAMMPS under MPI, computes on
    every process as on one."""
    from mdapy.simulation import _domain as domain

    class Whole(LennardJones):
        pass

    lj = Whole({"Cu": (0.4, 2.3)}, 5.0)
    monkeypatch.setattr(domain, "_reach", lambda *a, **k: (_ for _ in ()).throw(
        InvalidParameter("no split")))  # fmt: skip
    from mdapy.simulation import minimize

    system = shaken(copper(4))
    alone = minimize(system, lj, max_steps=20)
    for result in processes.run(2, lambda rank: minimize(system, lj, max_steps=20),
                                monkeypatch):  # fmt: skip
        np.testing.assert_allclose(result.system["positions"], alone.system["positions"],
                                   atol=1e-12)  # fmt: skip


def test_a_minimization_in_a_box_too_thin(monkeypatch):
    from mdapy.simulation import minimize

    # 7.2 Å across, cut in two along each vector: thinner than the reach of
    # a cutoff of 8 Å and the skin
    system = shaken(build_crystal("fcc", 3.615, elements="Cu", repeat=(2, 2, 2)))
    lj = LennardJones({"Cu": (0.4, 2.3)}, 8.0)
    with pytest.raises(InvalidParameter, match="fewer processes"):
        processes.run(8, lambda rank: minimize(system, lj, max_steps=5), monkeypatch)


def test_a_minimization_with_a_gpu_each(monkeypatch):
    """Each process computes its atoms and ghosts whole on its GPU, stood in
    for: the minimization of one process on the processor."""
    from mdapy.simulation import minimize

    monkeypatch.setattr(_gpu, "cuda", stand_in)
    system = shaken(copper())
    alone = minimize(system, EAM.zhou(["Cu"]), max_force=1e-4, max_steps=200)
    gpu = EAM.zhou(["Cu"], device="cuda", precision="double")
    for result in processes.run(
        2,
        lambda rank: minimize(system, gpu, max_force=1e-4, max_steps=200),
        monkeypatch,
    ):
        assert result.steps == alone.steps
        np.testing.assert_allclose(result.system["positions"], alone.system["positions"],
                                   atol=1e-7)  # fmt: skip


@pytest.mark.parametrize("size", [2, 4])
def test_adp_on_processes_of_one_thread(monkeypatch, tmp_path, size):
    """ADP on one thread computes each pair once among the processes, its
    sums sent back from the ghosts and its states to them: the run of one
    process, to the rounding, with the pressure."""
    from mdapy.core.threads import set_num_threads

    potential, system = potentials(tmp_path)["ADP"]
    options = {"ensemble": "npt", "temperature": 600, "pressure": 0.0}
    alone = run(system, potential, options)
    set_num_threads(1)
    for split in processes.run(size, lambda rank: run(system, potential, options),
                               monkeypatch):  # fmt: skip
        same(split, alone)
