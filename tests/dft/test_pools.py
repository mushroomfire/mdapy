# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Pools of processes over the pairs of spin and k-point, run by mpirun with
mpi4py (all on one GPU here), against one process: silicon without spin and
iron of PAW with spin, two and three pools, the three of them holding
different numbers of pairs. The energy, the eigenvalues, the forces, the
stress and the magnetization. Run as a script under mpirun, it writes what
rank 0 gets."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mdapy.calculators._dft import scf
from mdapy.calculators._dft.upf import read_upf

from dft import espresso, gpu

BOHR = 0.529177210903


def cases():
    si = read_upf(espresso.pseudo("Si_std.upf"))
    fe = read_upf(espresso.pseudo("pslibrary/Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"))
    h, b = 5.431 / 2 / BOHR, 2.87 / 2 / BOHR
    return {
        "silicon": dict(
            cell=[[0, h, h], [h, 0, h], [h, h, 0]], positions=[[0, 0, 0], [0.26, 0.25, 0.24]], kinds=[0, 0],
            pseudos=[si], cutoff=10.0, kgrid=(2, 2, 2), functional="pbe_pw",
        ),
        "iron": dict(
            cell=[[-b, b, b], [b, -b, b], [b, b, -b]], positions=[[0, 0, 0]], kinds=[0], pseudos=[fe],
            cutoff=15.0, density_cutoff=120.0, kgrid=(2, 2, 2), smearing="gaussian", sigma=0.02, magmoms=[2.5],
        ),
    }  # fmt: skip


def run(case, comm=None):
    c = dict(case)
    r = scf.run(
        np.array(c.pop("cell"), dtype=float), np.array(c.pop("positions"), dtype=float), c.pop("kinds"),
        c.pop("pseudos"), c.pop("cutoff"), energy_tolerance=1e-12, density_tolerance=1e-14, symmetry=False,
        comm=comm, **c,
    )  # fmt: skip
    assert r["converged"]
    keys = (
        "energy",
        "forces",
        "strain_derivative",
        "eigenvalues",
        "magnetization",
        "iterations",
    )
    return {k: np.asarray(r[k]).tolist() for k in keys}


if __name__ == "__main__":  # under mpirun: the results of rank 0
    from mpi4py import MPI

    out = {name: run(case, MPI.COMM_WORLD) for name, case in cases().items()}
    if MPI.COMM_WORLD.Get_rank() == 0:
        Path(sys.argv[1]).write_text(json.dumps(out))
    sys.exit(0)


_dft = gpu.module()
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)


@pytest.mark.parametrize("ranks", [2, 3])
def test_pools_agree_with_one_process(tmp_path, ranks):
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        pytest.skip("no mpirun")
    pytest.importorskip("mpi4py")
    cases()  # skips without the pseudopotentials
    out = tmp_path / "pools.json"
    tests = str(Path(__file__).resolve().parents[1])
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, [tests, os.environ.get("PYTHONPATH")])
        ),
    }
    subprocess.run([mpirun, "-np", str(ranks), "--oversubscribe", sys.executable, __file__, str(out)], env=env,
                   check=True, timeout=1800)  # fmt: skip
    pooled = json.loads(out.read_text())
    for name, case in cases().items():
        one, many = run(case), pooled[name]
        assert many["energy"] == pytest.approx(one["energy"], abs=1e-10), name
        for key, tolerance in [
            ("eigenvalues", 1e-9),
            ("forces", 1e-9),
            ("strain_derivative", 1e-9),
        ]:
            np.testing.assert_allclose(
                many[key], one[key], rtol=0, atol=tolerance, err_msg=f"{name} {key}"
            )
        assert many["magnetization"] == pytest.approx(one["magnetization"], abs=1e-9), (
            name
        )
