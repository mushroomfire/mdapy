# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""A box deformed at constant rates: in MDAPY and in LAMMPS (fix deform 1
with erate or trate, remap x, and the barostat on the lengths left) the same
runs step by step, with the constants of LAMMPS and the masses of the
potential file; the lengths and tilts as the rates give them; the errors."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM, _gpu
from mdapy.errors import InvalidParameter
from mdapy.simulation import Deform, Hold, MolecularDynamics
from mdapy.simulation import _integrator as integrator
from simulation.test_controls import gap, potentials, slab  # noqa: F401

from calculators.gpu_stand_in import stand_in


@pytest.fixture
def lammps_constants(monkeypatch):
    monkeypatch.setattr(integrator, "_KB", 8.617343e-5)
    monkeypatch.setattr(integrator, "_MVV2E", 1.0364269e-4)
    monkeypatch.setattr(integrator, "_FTM2V", 1.0 / 1.0364269e-4)
    monkeypatch.setattr(integrator, "_GPA_TO_EV_PER_A3", 1e4 / 1.6021765e6)
    monkeypatch.setattr(integrator, "_EV_PER_A3_TO_GPA", 1.6021765e6 / 1e4)


CASES = [
    (Deform(xx=0.05), {"ensemble": "nve"}),
    (Deform(xx=0.05, zz=-0.02), {"ensemble": "nvt", "temperature": 300}),
    (Deform(yy=-0.05, kind="true"), {"ensemble": "nvt", "temperature": 300}),
    (Deform(xy=0.1, yz=-0.05), {"ensemble": "nvt", "temperature": 300}),
    (
        Deform(xx=0.05),
        {"ensemble": "npt", "temperature": 300, "pressure": 0.0, "barostat": "aniso"},
    ),
    (
        Deform(xx=0.05),
        {"ensemble": "npt", "temperature": 300, "pressure": 1.0, "barostat": "iso"},
    ),
    (Deform(xx=0.05, yy=0.05), {"ensemble": "nph", "pressure": 0.0}),
]


@pytest.mark.parametrize("deform, options", CASES)
def test_the_same_runs_as_lammps(potentials, lammps_constants, deform, options):  # noqa: F811
    eam, lmp = potentials
    if lmp is None:
        pytest.skip("LAMMPS is not installed")
    system, *_ = slab()
    out = []
    for potential in (lmp, eam):
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.run(
                300,
                thermo=100,
                controls=deform,
                thermostat_damping=0.2,
                barostat_damping=2.0,
                **options,
            )
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    dx, dv = gap(a, b)
    assert dx < 1e-10 and dv < 1e-9
    np.testing.assert_allclose(a.box.matrix, b.box.matrix, rtol=0, atol=1e-10)
    for column in ("temperature", "potential_energy", "volume"):
        np.testing.assert_allclose(ta[column], tb[column], rtol=1e-10)
    np.testing.assert_allclose(ta["pressure"], tb["pressure"], rtol=0, atol=1e-8)


def test_the_box_follows_the_rates(potentials):  # noqa: F811
    eam, _ = potentials
    system, *_ = slab(tilt=0.3)
    h0 = system.box.matrix.copy()
    dt, steps = 0.002, 100
    with MolecularDynamics(system, eam, timestep=dt) as md:
        md.run(steps, controls=Deform(xx=0.05, zz=-0.03, xy=0.1))
        once = md.system.box.matrix
        md.run(steps, controls=Deform(yy=0.02, kind="true"))
        twice = md.system.box.matrix
    t = steps * dt
    np.testing.assert_allclose(once[0, 0], h0[0, 0] * (1 + 0.05 * t), rtol=1e-14)
    np.testing.assert_allclose(once[2, 2], h0[2, 2] * (1 - 0.03 * t), rtol=1e-14)
    np.testing.assert_allclose(once[1, 0], h0[1, 0] + 0.1 * h0[1, 1] * t, atol=1e-13)
    np.testing.assert_allclose(once[2, 0], h0[2, 0], atol=1e-13)  # left
    # the next run starts from the box it finds
    np.testing.assert_allclose(twice[1, 1], once[1, 1] * np.exp(0.02 * t), rtol=1e-13)
    np.testing.assert_allclose(twice[0, 0], once[0, 0], rtol=1e-14)


def test_on_the_gpu(monkeypatch):
    """The deformation on the GPU, its module stood in for on the processor:
    the runs of the processor."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    system = build_crystal("fcc", 3.6, elements="Cu", repeat=(5, 5, 6))
    out = []
    for device in ("cuda", "cpu"):
        with MolecularDynamics(
            system, EAM.zhou(["Cu"], device=device), timestep=0.002
        ) as md:
            md.velocities(300, seed=1)
            md.run(
                40,
                controls=Deform(xx=0.5, zz=-0.2, xy=1.0),
                ensemble="nvt",
                temperature=300,
            )
            md.run(20, controls=Deform(yy=0.5, kind="true"), ensemble="nve")
            out.append(md.system)
    a, b = out
    np.testing.assert_allclose(a.box.matrix, b.box.matrix, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a["positions"], b["positions"], atol=1e-9)
    np.testing.assert_allclose(a["velocities"], b["velocities"], atol=1e-8)


def test_errors(potentials):  # noqa: F811
    eam, lmp = potentials
    system, bottom, *_ = slab()
    with pytest.raises(InvalidParameter, match="not zero"):
        Deform()
    with pytest.raises(InvalidParameter, match="finite"):
        Deform(xx=np.nan)
    with pytest.raises(InvalidParameter, match="kind"):
        Deform(xx=0.1, kind="log")
    assert (
        repr(Deform(xx=0.1, xy=-0.2)) == "Deform(xx=0.1, xy=-0.2, kind='engineering')"
    )
    turned = system.with_box(system.box.matrix[[1, 0, 2]], scale_positions=True)
    for potential in (eam, lmp):
        if potential is None:
            continue
        with MolecularDynamics(system, potential) as md:
            with pytest.raises(InvalidParameter, match="one Deform"):
                md.run(1, controls=[Deform(xx=0.1), Deform(yy=0.1)])
            with pytest.raises(InvalidParameter, match="held or moved"):
                md.run(1, controls=[Deform(xx=0.1), Hold(bottom)])
            with pytest.raises(InvalidParameter, match="engineering rates only"):
                md.run(1, controls=Deform(xy=0.1, kind="true"))
            with pytest.raises(InvalidParameter, match='"iso" or "aniso"'):
                md.run(
                    1,
                    ensemble="nph",
                    pressure=0.0,
                    barostat="tri",
                    controls=Deform(xx=0.1),
                )
            with pytest.raises(InvalidParameter, match='"iso" or "aniso"'):
                md.run(1, ensemble="nph", pressure=0.0, controls=Deform(xy=0.1))
            with pytest.raises(InvalidParameter, match="no length"):
                md.run(
                    1,
                    ensemble="nph",
                    pressure=0.0,
                    controls=Deform(xx=0.1, yy=0.1, zz=0.1),
                )
            md.run(2, controls=Deform(xx=0.1))  # usable after the refusals
        with MolecularDynamics(turned, potential) as md:
            with pytest.raises(InvalidParameter, match="first vector is along x"):
                md.run(1, controls=Deform(xx=0.1))
        sheet = system.with_box(mdapy.Box(system.box.matrix, pbc=(True, True, False)))
        with MolecularDynamics(sheet, potential) as md:
            md.velocities(300, seed=1)
            with pytest.raises(InvalidParameter, match="sheet"):
                md.run(1, ensemble="nph", pressure=0.0, controls=Deform(xx=0.1))
            for deform in (Deform(zz=0.1), Deform(xz=0.1)):
                with pytest.raises(InvalidParameter, match="periodic lengths only"):
                    md.run(1, controls=deform)
            md.run(2, controls=Deform(xx=0.1, xy=0.1))  # in the plane
