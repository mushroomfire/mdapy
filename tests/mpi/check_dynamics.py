# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Run under mpirun: MDAPY's dynamics split among the processes gives the run
of one process, in every ensemble and with every control, to the rounding of
the forces, which sum in another order; and its restart files are those of
one process. The first process runs each case again alone and compares.
Prints "ok" on the first process, or raises.

    MDAPY_NUM_THREADS=1 mpirun -np 4 python tests/mpi/check_dynamics.py [case ...]
"""

import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from mpi4py import MPI

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM, LennardJones, Springs, StillingerWeber, Tersoff
from mdapy.simulation import MolecularDynamics, _domain
from mdapy.simulation._controls import _HeatField
from mdapy.simulation.controls import (
    AdaptiveTimestep,
    AddForce,
    Deform,
    Hold,
    Indenter,
    Langevin,
    Move,
    Switch,
    Wall,
)

sys.path.insert(0, str(Path(__file__).parents[1]))
from calculators.potential_files import write_sw, write_tersoff  # noqa: E402

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
alone = _domain.world  # the processes of the run; None for one


def copper(repeat=11, tilt=0.0, pbc=(True, True, True)):
    crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(repeat,) * 3)
    matrix = crystal.box.matrix.copy()
    matrix[1, 0] += tilt * matrix[0, 0]
    return crystal.with_box(mdapy.Box(matrix, pbc=pbc), scale_positions=True)


def cases():
    crystal = copper()
    z = crystal["positions"][:, 2]
    bottom, top = z < z.min() + 2.0, z > z.max() - 2.0
    middle = ~(bottom | top)
    eam = EAM.zhou(["Cu"])
    lj = LennardJones({"Cu": (0.4, 2.3)}, 5.0)
    springs = Springs(crystal["positions"], np.full(len(crystal), 2.0))
    open_z = copper(pbc=(True, True, False))
    length = crystal.box.matrix[0, 0]
    # silicon and germanium, with a potential file of each process's own
    files = Path(tempfile.mkdtemp())
    silicon = build_crystal("diamond", 5.43, elements="Si", repeat=(8, 8, 8))
    kinds = np.random.default_rng(0).choice(["Si", "Ge"], len(silicon))
    sige = silicon.with_columns(element=list(kinds))
    sw = StillingerWeber(write_sw(files / "SiGe.sw", ("Si", "Ge")))
    tersoff = Tersoff(write_tersoff(files / "SiGe.tersoff", ("Si", "Ge")))
    return {
        "nve": (crystal, eam, {}),
        "lj": (crystal, lj, {}),
        "sw": (sige, sw, {"ensemble": "npt", "temperature": 600, "pressure": 0.0}),
        "tersoff": (sige, tersoff, {"ensemble": "npt", "temperature": 600, "pressure": 0.0}),
        "tersoff hnemd": (sige, tersoff, {"ensemble": "nvt", "temperature": 600,
                                          "controls": [_HeatField((1e-4, 0.0, 0.0))]}),
        "nvt": (crystal, eam, {"ensemble": "nvt", "temperature": (300, 500)}),
        "npt": (crystal, eam, {"ensemble": "npt", "temperature": 300, "pressure": 1.0}),
        "npt tri": (
            copper(tilt=0.1),
            eam,
            {"ensemble": "npt", "temperature": 300, "pressure": 1.0, "barostat": "tri"},
        ),
        "nph aniso": (crystal, eam, {"ensemble": "nph", "pressure": 1.0, "barostat": "aniso"}),
        "langevin": (
            crystal,
            eam,
            {"ensemble": "nvt", "temperature": 400, "thermostat": "langevin", "seed": 5,
             "keep_momentum": True},
        ),
        "nphug": (crystal, eam, {"ensemble": "nphug", "pressure": 5.0, "barostat": "z"}),
        "msst": (crystal, eam, {"ensemble": "msst", "barostat": "x", "shock_velocity": 60.0}),
        "groups": (
            crystal,
            eam,
            {"controls": [Hold(bottom), Move(top, (0.0, 0.0, 0.5), name="top"),
                          AddForce(middle, (0.01, 0.0, 0.0)),
                          Langevin(middle, 300.0, damping=0.1, seed=9, name="bath")]},
        ),
        "deform": (crystal, eam, {"ensemble": "nvt", "temperature": 300,
                                  "controls": [Deform(xx=1e-3)]}),
        "adaptive": (crystal, eam, {"controls": [AdaptiveTimestep(0.02)]}),
        "wall": (open_z, eam, {"controls": [Wall("z", side="low", velocity=0.5)]}),
        "indenter": (
            open_z,
            eam,
            {"controls": [Indenter("sphere", center=(length / 2, length / 2, z.max() + 9.5),
                                   radius=10.0, velocity=(0.0, 0.0, -3.0), name="tip")]},
        ),
        "switch": (
            crystal,
            eam,
            {"ensemble": "nvt", "temperature": 300, "thermostat": "langevin", "seed": 2,
             "controls": [Switch(springs, name="switch")]},
        ),
        "hnemd": (crystal, eam, {"ensemble": "nvt", "temperature": 300,
                                 "controls": [_HeatField((0.0, 0.0, 1e-4))]}),
        "hnemd convective": (crystal, eam, {"ensemble": "nvt", "temperature": 300,
                                            "controls": [_HeatField((1e-4, 0.0, 0.0), True)]}),
    }  # fmt: skip


def run(crystal, potential, options, steps=40):
    log = io.StringIO()
    with MolecularDynamics(crystal, potential, timestep=0.002, log=log) as md:
        md.velocities(300, seed=1)
        md.run(steps, thermo=10, **options)
        out = md.system, md.thermo
    split = _domain.world() is not None
    if rank == 0:  # the main process writes the log, which names the processes
        assert (f"{comm.Get_size()} processes" in log.getvalue()) == split
    return out


def compare(a, b, name):
    (sa, ta), (sb, tb) = a, b
    x = sa["positions"] - sb["positions"]
    f = x @ np.linalg.inv(sa.box.matrix)
    f[:, sa.box.pbc] -= np.round(f[:, sa.box.pbc])
    gap = np.abs(f @ sa.box.matrix).max()
    assert gap < 1e-9, f"{name}: positions {gap}"
    assert np.abs(sa["velocities"] - sb["velocities"]).max() < 1e-9, name
    np.testing.assert_allclose(sa.box.matrix, sb.box.matrix, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(sa["image"], sb["image"])
    assert ta.columns == tb.columns and ta.height == tb.height, name
    for column in ta.columns:
        p, q = np.stack(ta[column].to_numpy()), np.stack(tb[column].to_numpy())
        np.testing.assert_allclose(
            p, q, rtol=1e-9, atol=1e-9, err_msg=f"{name} {column}"
        )


def stop(crystal, potential, options, path):
    """The run stopped after 20 of its 40 steps, its restart file written."""
    with MolecularDynamics(crystal, potential, timestep=0.002) as md:
        md.velocities(300, seed=1)
        frames = md.iterate(40, save_every=20, thermo=10, restart=path, restart_every=20,
                            **options)  # fmt: skip
        for frame in frames:
            if frame.info["timestep"] >= 40:
                break
        frames.close()


def finish(path, potential):
    again = MolecularDynamics.load(path, potential)
    again.resume()
    return again.system, again.thermo


names = sys.argv[1:] or list(cases())
everything = cases()
for name in names:
    crystal, potential, options = everything[name]
    split = run(crystal, potential, options)
    if rank == 0:
        _domain.world = lambda: None
        try:
            compare(split, run(crystal, potential, options), name)
        finally:
            _domain.world = alone
    comm.Barrier()
# restart files: of many processes read by many and by one, and of one by
# many; a copy for each, as the run finished from a file writes it again at its
# end
crystal, potential, options = everything["langevin"]
folder = comm.bcast(tempfile.mkdtemp() if rank == 0 else None)
path, copy = os.path.join(folder, "md.restart"), os.path.join(folder, "copy.restart")
whole = run(crystal, potential, options)
stop(crystal, potential, options, path)
if rank == 0:
    shutil.copy(path, copy)
comm.Barrier()
continued = finish(path, potential)
if rank == 0:
    compare(whole, continued, "restart under MPI")
    _domain.world = lambda: None
    try:
        compare(whole, finish(copy, potential), "restart of MPI on one process")
        stop(crystal, potential, options, path)
    finally:
        _domain.world = alone
comm.Barrier()
continued = finish(path, potential)
if rank == 0:
    compare(whole, continued, "restart of one process under MPI")
if rank == 0:
    print("ok", comm.Get_size(), "processes")
