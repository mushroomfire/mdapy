# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The multiscale shock technique, fix msst: in MDAPY and in LAMMPS the same
runs step by step, with the constants of LAMMPS; the energy it conserves;
copper that settles on the Rayleigh line and the Hugoniot of a shock; the GPU
stood in; the errors."""

import io

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM, Springs, _gpu
from mdapy.errors import InvalidParameter
from mdapy.simulation import MolecularDynamics
from mdapy.simulation import _integrator as integrator
from mdapy.simulation.controls import Hold
from simulation.test_controls import gap, potentials, slab  # noqa: F401

from calculators.gpu_stand_in import stand_in

RECORDS = (
    "hugoniot",
    "rayleigh",
    "particle_velocity",
    "lagrangian_position",
    "msst_energy",
)


@pytest.fixture
def lammps_constants(monkeypatch):
    monkeypatch.setattr(integrator, "_KB", 8.617343e-5)
    monkeypatch.setattr(integrator, "_MVV2E", 1.0364269e-4)
    monkeypatch.setattr(integrator, "_FTM2V", 1.0 / 1.0364269e-4)
    monkeypatch.setattr(integrator, "_GPA_TO_EV_PER_A3", 1e4 / 1.6021765e6)
    monkeypatch.setattr(integrator, "_EV_PER_A3_TO_GPA", 1.6021765e6 / 1e4)


CASES = [
    {"barostat": "z", "shock_velocity": 60.0},
    {"barostat": "z", "shock_velocity": 60.0, "msst": {"mu": 0.5, "q": 50.0}},
    {"barostat": "x", "shock_velocity": 50.0, "msst": {"tscale": 0.0, "q": 20.0}},
    {
        "barostat": "y",
        "shock_velocity": 55.0,
        "msst": {"tscale": 0.05},
        "hugoniot": (-500.0, 2900.0, 0.5),
    },
]


@pytest.mark.parametrize("options", CASES)
def test_the_same_runs_as_lammps(potentials, lammps_constants, options):  # noqa: F811
    eam, lmp = potentials
    if lmp is None:
        pytest.skip("LAMMPS is not installed")
    system, *_ = slab(tilt=0.2)
    out = []
    for potential in (lmp, eam):
        with MolecularDynamics(system, potential, timestep=0.001) as md:
            md.run(300, ensemble="msst", thermo=50, **options)
            md.run(10, thermo=10)  # constant energy again
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    dx, dv = gap(a, b)
    assert dx < 1e-9 and dv < 1e-8
    np.testing.assert_allclose(a.box.matrix, b.box.matrix, rtol=0, atol=1e-10)
    for column in ("temperature", "potential_energy", "volume"):
        np.testing.assert_allclose(ta[column], tb[column], rtol=1e-10)
    rows = ta["step"].to_numpy() <= 300
    for column in RECORDS:
        np.testing.assert_allclose(
            ta[column].to_numpy()[rows],
            tb[column].to_numpy()[rows],
            rtol=1e-9,
            atol=1e-8,
        )
    assert ta["rayleigh"][-1] is None  # not in the run after it
    k = "xyz".index(options["barostat"])
    if options.get("msst", {}).get("tscale", 0.01) > 0:  # set off by tscale
        assert b.box.matrix[k, k] < system.box.matrix[k, k] - 1e-3


def copper(repeat=4):
    """Copper at 300 K and no pressure."""
    with MolecularDynamics(
        build_crystal("fcc", 3.615, elements="Cu", repeat=repeat),
        EAM.zhou(["Cu"]),
        timestep=0.002,
    ) as md:
        md.velocities(600, seed=1)
        md.run(2000, ensemble="npt", temperature=300, pressure=0.0, barostat="aniso")
        return md.system


def test_the_energy_it_conserves():
    """Without viscosity the total energy and that of the box's variables sum
    to a constant, to the error of the integrator, as the energy of the atoms
    changes by tens of eV."""
    with MolecularDynamics(copper(), EAM.zhou(["Cu"]), timestep=0.001) as md:
        md.run(2000, ensemble="msst", barostat="z", shock_velocity=50.0, thermo=20)
        thermo = md.thermo
    conserved = (thermo["total_energy"] + thermo["msst_energy"]).to_numpy()
    assert np.ptp(conserved) < 0.01  # eV
    assert np.ptp(thermo["total_energy"].to_numpy()) > 20.0
    # the cell moves behind the front: its position falls as Us V / V0
    position = thermo["lagrangian_position"].to_numpy()
    assert (np.diff(position) < 0).all()


def test_copper_settles_behind_a_shock():
    """With a viscosity the compression settles where the Rayleigh line of
    the shock velocity crosses the Hugoniot."""
    with MolecularDynamics(copper(), EAM.zhou(["Cu"]), timestep=0.001) as md:
        md.run(
            8000,
            ensemble="msst",
            barostat="z",
            shock_velocity=55.0,
            msst={"mu": 3.0, "q": 20.0},
            thermo=20,
        )
        rows = md.thermo[len(md.thermo) // 2 :]
    assert abs(rows["rayleigh"].mean()) < 1.0  # GPa
    assert abs(rows["hugoniot"].mean()) < 30.0  # K
    assert rows["particle_velocity"].mean() > 1.0  # Å/ps, compressed
    assert rows["temperature"].mean() > 300.0


def test_on_the_gpu(monkeypatch):
    """msst on the GPU, its module stood in for on the processor: the runs of
    the processor."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    warm = copper(6)
    out = []
    for device in ("cuda", "cpu"):
        with MolecularDynamics(warm, EAM.zhou(["Cu"], device=device)) as md:
            md.run(
                200,
                ensemble="msst",
                barostat="z",
                shock_velocity=50.0,
                msst={"mu": 1.0},
                thermo=50,
            )
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    np.testing.assert_allclose(a["positions"], b["positions"], atol=1e-9)
    np.testing.assert_allclose(a.box.matrix, b.box.matrix, atol=1e-10)
    for column in RECORDS:
        np.testing.assert_allclose(ta[column], tb[column], rtol=1e-8, atol=1e-8)


def test_the_log():
    stream = io.StringIO()
    with MolecularDynamics(copper(3), EAM.zhou(["Cu"]), log=stream) as md:
        md.run(5, ensemble="msst", barostat="z", shock_velocity=50.0)
    assert "shock along z at 50.0 Å/ps, q 10.0, mu 0.0, tscale 0.01" in (
        stream.getvalue()
    )


def test_errors(potentials):  # noqa: F811
    eam, lmp = potentials
    system, *_ = slab()
    everyone = np.ones(len(system), dtype=bool)
    shock = {"barostat": "z", "shock_velocity": 50.0}
    with MolecularDynamics(system, eam) as md:
        for options, match in (
            ({"barostat": "z"}, "needs a shock_velocity"),
            ({"barostat": "z", "shock_velocity": -1.0}, "not negative"),
            ({"barostat": "z", "shock_velocity": True}, "not negative"),
            ({"shock_velocity": 50.0}, "axis of the shock"),
            (shock | {"pressure": 1.0}, "no pressure or temperature"),
            (shock | {"temperature": 300.0}, "no pressure or temperature"),
            (shock | {"controls": Hold(everyone)}, "no controls"),
            (shock | {"msst": {"viscosity": 1.0}}, "dict of q, mu and tscale"),
            (shock | {"msst": [1.0]}, "dict of q, mu and tscale"),
            (shock | {"msst": {"q": 0.0}}, "q positive"),
            (shock | {"msst": {"mu": -1.0}}, "mu not negative"),
            (shock | {"msst": {"tscale": 1.5}}, "tscale from 0 to 1"),
        ):
            with pytest.raises(InvalidParameter, match=match):
                md.run(1, ensemble="msst", **options)
        for options in ({"shock_velocity": 1.0}, {"msst": {}}):
            with pytest.raises(InvalidParameter, match='for the ensemble "msst"'):
                md.run(1, **options)
        md.run(2)  # still runs
    sheet = system.with_box(mdapy.Box(system.box.matrix, pbc=(True, True, False)))
    with MolecularDynamics(sheet, eam) as md:
        with pytest.raises(InvalidParameter, match="periodic in every direction"):
            md.run(1, ensemble="msst", **shock)
    turned = system.with_box(
        system.box.matrix @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]]),
        scale_positions=True,
    )
    engines = [eam] + ([] if lmp is None else [lmp])
    for potential in engines:
        with MolecularDynamics(turned, potential) as md:
            with pytest.raises(InvalidParameter, match="msst needs a box whose"):
                md.run(1, ensemble="msst", **shock)
    springs = Springs(system["positions"], 1.0)
    with MolecularDynamics(system, springs) as md:
        with pytest.raises(InvalidParameter, match="msst needs the virial"):
            md.run(1, ensemble="msst", **shock)
    held = system.with_columns(movable=np.ones((len(system), 3), dtype=bool))
    movable = np.array(held["movable"])
    movable[0, 2] = False
    with MolecularDynamics(held.with_columns(movable=movable), eam) as md:
        with pytest.raises(InvalidParameter, match="atoms held"):
            md.run(1, ensemble="msst", **shock)
