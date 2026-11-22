# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Run under mpirun on a machine with a GPU: each process computes the forces
of its atoms and ghosts on a GPU, the steps on the processor, and gives the
run of one process on the GPU, to the rounding of the forces. The first
process runs each case again alone and compares. Prints "ok" on the first
process, or raises.

    MDAPY_NUM_THREADS=1 mpirun -np 2 python tests/mpi/check_gpu.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from mpi4py import MPI

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM, NEP, LennardJones, StillingerWeber, Tersoff
from mdapy.simulation import MolecularDynamics, _domain
from mdapy.simulation._controls import _HeatField
from mdapy.simulation.controls import Deform, Hold, Langevin, Move

sys.path.insert(0, str(Path(__file__).parents[1]))
from calculators.nep_files import write_nep  # noqa: E402
from calculators.potential_files import write_sw, write_tersoff  # noqa: E402

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world = _domain.world


def copper(repeat=10, pbc=(True, True, True)):
    crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(repeat,) * 3)
    return crystal.with_box(mdapy.Box(crystal.box.matrix, pbc=pbc))


def mixed(elements, p=None, structure=("diamond", 5.43, 8)):
    structure, a, repeat = structure
    system = build_crystal(structure, a, elements=elements[0], repeat=(repeat,) * 3)
    rng = np.random.default_rng(0)
    return system.with_columns(element=list(rng.choice(elements, len(system), p=p)))


crystal = copper()
z = crystal["positions"][:, 2]
bottom, top = z < z.min() + 2.0, z > z.max() - 2.0
eam = EAM.zhou(["Cu"], device="cuda", precision="double")
lj = LennardJones({"Cu": (0.4, 2.3)}, 5.0, device="cuda", precision="double")
files = Path(tempfile.mkdtemp())
sige, sic = mixed(("Si", "Ge")), mixed(("Si", "C"), p=[0.8, 0.2])
sw = StillingerWeber(write_sw(files / "SiGe.sw", ("Si", "Ge")), device="cuda",
                     precision="double")  # fmt: skip
tersoff = Tersoff(write_tersoff(files / "SiC.tersoff", ("Si", "C")), device="cuda",
                  precision="double")  # fmt: skip
nep = NEP(write_nep(files / "nep.txt", zbl="universal"), device="cuda",
          precision="double")  # fmt: skip
cuni = mixed(("Cu", "Ni"), structure=("fcc", 3.6, 12))
heat = _HeatField((2e-4, 0.0, 1e-4))
CASES = {
    "nve": (crystal, eam, {}),
    "npt": (crystal, eam, {"ensemble": "npt", "temperature": 300, "pressure": 1.0}),
    "langevin": (crystal, eam, {"ensemble": "nvt", "temperature": 400,
                                "thermostat": "langevin", "seed": 5,
                                "keep_momentum": True}),
    "groups": (crystal, eam, {"controls": [Hold(bottom), Move(top, (0, 0, 0.5)),
                                           Langevin(~(bottom | top), 300.0, seed=9)]}),
    "deform": (crystal, eam, {"ensemble": "nvt", "temperature": 300,
                              "controls": [Deform(xx=1e-3)]}),
    "lj": (crystal, lj, {"ensemble": "nvt", "temperature": 300}),
    # the atoms of each process the centres, the forces on the ghosts sent back
    "sw": (sige, sw, {"ensemble": "npt", "temperature": 600, "pressure": 0.0}),
    "sw heat": (sige, sw, {"ensemble": "nvt", "temperature": 600, "controls": [heat]}),
    "tersoff": (sic, tersoff, {"ensemble": "npt", "temperature": 600, "pressure": 0.0}),
    "tersoff heat": (sic, tersoff, {"ensemble": "nvt", "temperature": 600,
                                    "controls": [heat]}),
    # and the virials, which the atoms of NEP gather from their neighbors
    "nep": (cuni, nep, {"ensemble": "npt", "temperature": 600, "pressure": 0.0}),
    "nep heat": (cuni, nep, {"ensemble": "nvt", "temperature": 600, "controls": [heat]}),
}  # fmt: skip


def run(crystal, potential, options):
    with MolecularDynamics(crystal, potential, timestep=0.002) as md:
        md.velocities(300, seed=1)
        md.run(40, thermo=10, **options)
        return md.system, md.thermo


def compare(a, b, name):
    (sa, ta), (sb, tb) = a, b
    d = (sa["positions"] - sb["positions"]) @ np.linalg.inv(sb.box.matrix)
    d -= np.round(d)
    assert np.abs(d @ sb.box.matrix).max() < 1e-7, name
    np.testing.assert_allclose(
        sa["velocities"], sb["velocities"], atol=1e-6, err_msg=name
    )
    np.testing.assert_allclose(sa.box.matrix, sb.box.matrix, atol=1e-8, err_msg=name)
    for column in ("temperature", "potential_energy", "pressure", "volume"):
        np.testing.assert_allclose(ta[column], tb[column], rtol=1e-8, atol=1e-8,
                                   err_msg=f"{name} {column}")  # fmt: skip


for name, (system, potential, options) in CASES.items():
    split = run(system, potential, options)
    if rank == 0:
        _domain.world = lambda: None
        try:
            compare(split, run(system, potential, options), name)
        finally:
            _domain.world = world
    comm.Barrier()
if rank == 0:
    print("ok", comm.Get_size(), "processes")
