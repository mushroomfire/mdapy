# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The Hugoniostat, fix nphug, and barostats along one axis: in MDAPY and in
LAMMPS the same runs step by step, with the constants of LAMMPS; a crystal
compressed along one axis that settles on its Hugoniot; the GPU stood in;
the errors."""

import io

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM, _gpu
from mdapy.errors import InvalidParameter
from mdapy.simulation import Langevin, MolecularDynamics
from mdapy.simulation import _integrator as integrator
from simulation.test_controls import gap, potentials, slab  # noqa: F401

from calculators.gpu_stand_in import stand_in

RECORDS = ("hugoniot", "shock_velocity", "particle_velocity")


@pytest.fixture
def lammps_constants(monkeypatch):
    monkeypatch.setattr(integrator, "_KB", 8.617343e-5)
    monkeypatch.setattr(integrator, "_MVV2E", 1.0364269e-4)
    monkeypatch.setattr(integrator, "_FTM2V", 1.0 / 1.0364269e-4)
    monkeypatch.setattr(integrator, "_GPA_TO_EV_PER_A3", 1e4 / 1.6021765e6)
    monkeypatch.setattr(integrator, "_EV_PER_A3_TO_GPA", 1.6021765e6 / 1e4)


CASES = [
    {"ensemble": "npt", "temperature": 300, "pressure": 5.0, "barostat": "z"},
    {"ensemble": "nph", "pressure": 5.0, "barostat": "x"},
    {"ensemble": "npt", "temperature": 300, "pressure": (0.0, 3.0), "barostat": "y"},
    {"ensemble": "nphug", "pressure": 10.0},
    {"ensemble": "nphug", "pressure": 10.0, "barostat": "aniso"},
    {"ensemble": "nphug", "pressure": 10.0, "barostat": "tri"},
    {"ensemble": "nphug", "pressure": 20.0, "barostat": "z"},
    {
        "ensemble": "nphug",
        "pressure": 15.0,
        "barostat": "z",
        "hugoniot": (-500.0, 2900.0, 0.5),
    },
]


@pytest.mark.parametrize("options", CASES)
def test_the_same_runs_as_lammps(potentials, lammps_constants, options):  # noqa: F811
    eam, lmp = potentials
    if lmp is None:
        pytest.skip("LAMMPS is not installed")
    system, *_ = slab(tilt=0.2 if options.get("barostat") == "tri" else 0.0)
    out = []
    for potential in (lmp, eam):
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.run(
                300,
                thermo=50,
                thermostat_damping=0.1,
                barostat_damping=1.0,
                **options,
            )
            md.run(10, thermo=10)  # constant energy again
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    dx, dv = gap(a, b)
    assert dx < 1e-9 and dv < 1e-8
    np.testing.assert_allclose(a.box.matrix, b.box.matrix, rtol=0, atol=1e-10)
    for column in ("temperature", "potential_energy", "volume"):
        np.testing.assert_allclose(ta[column], tb[column], rtol=1e-10)
    if options["ensemble"] == "nphug":
        rows = ta["step"].to_numpy() <= 300
        for column in RECORDS:
            np.testing.assert_allclose(
                ta[column].to_numpy()[rows],
                tb[column].to_numpy()[rows],
                rtol=1e-9,
                atol=1e-9,
            )
        assert ta["hugoniot"][-1] is None  # not in the run after it


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


def test_copper_settles_on_its_hugoniot():
    """Compressed to 20 GPa along z, its energy, volume and pressure meet the
    Hugoniot of where it started: the distance from it averages to zero;
    the particle velocity is the shock velocity times the strain."""
    warm = copper()
    with MolecularDynamics(warm, EAM.zhou(["Cu"]), timestep=0.002) as md:
        md.run(0, ensemble="nphug", pressure=20.0, barostat="z")
        first = md.thermo[-1]
        for column in RECORDS:  # the state at the start is the reference
            assert first[column][0] == 0.0
        md.run(
            6000,
            ensemble="nphug",
            pressure=20.0,
            barostat="z",
            thermo=20,
            thermostat_damping=0.1,
            barostat_damping=0.5,
            hugoniot=(
                first["total_energy"][0],
                first["volume"][0],
                -first["stress"][0][2],
            ),
        )
        rows = md.thermo[len(md.thermo) // 2 :]
    assert abs(rows["hugoniot"].mean()) < 10.0
    pressure = -np.stack(rows["stress"].to_numpy())[:, 2]
    assert pressure.mean() == pytest.approx(20.0, abs=0.5)
    assert rows["temperature"].mean() > 330.0  # heated by the compression
    strain = 1.0 - rows["volume"].to_numpy() / first["volume"][0]
    np.testing.assert_allclose(
        rows["particle_velocity"], rows["shock_velocity"] * strain, rtol=1e-12
    )
    assert 40.0 < rows["shock_velocity"].mean() < 60.0  # Å/ps


def test_on_the_gpu(monkeypatch):
    """nphug along z on the GPU, its module stood in for on the processor:
    the runs of the processor."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    warm = copper(5)
    out = []
    for device in ("cuda", "cpu"):
        with MolecularDynamics(warm, EAM.zhou(["Cu"], device=device)) as md:
            md.run(200, ensemble="nphug", pressure=20.0, barostat="z", thermo=50)
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    np.testing.assert_allclose(a["positions"], b["positions"], atol=1e-9)
    np.testing.assert_allclose(a.box.matrix, b.box.matrix, atol=1e-10)
    for column in RECORDS:
        np.testing.assert_allclose(ta[column], tb[column], rtol=1e-8, atol=1e-8)


def test_the_log():
    stream = io.StringIO()
    with MolecularDynamics(copper(3), EAM.zhou(["Cu"]), log=stream) as md:
        md.run(5, ensemble="nphug", pressure=5.0, barostat="z")
    assert "thermostat on the Hugoniot" in stream.getvalue()
    assert "barostat z" in stream.getvalue()


def test_errors(potentials):  # noqa: F811
    eam, lmp = potentials
    system, *_ = slab()
    with MolecularDynamics(system, eam) as md:
        for options, match in (
            ({"pressure": (0.0, 5.0)}, "one pressure"),
            ({"pressure": 5.0, "temperature": 300}, "no temperature"),
            ({"pressure": 5.0, "thermostat": "langevin"}, "nose-hoover"),
            ({"pressure": 5.0, "barostat": "xy"}, "nphug needs a barostat"),
            ({}, "nphug needs a pressure"),
            ({"pressure": 5.0, "hugoniot": (1.0, 2.0)}, "energy, volume, pressure"),
            ({"pressure": 5.0, "hugoniot": (1.0, -2.0, 0.0)}, "positive volume"),
            ({"pressure": 5.0, "hugoniot": (np.nan, 2.0, 0.0)}, "finite"),
            (
                {
                    "pressure": 5.0,
                    "controls": Langevin(np.ones(len(system), bool), 300),
                },
                "Langevin controls",
            ),
        ):
            with pytest.raises(InvalidParameter, match=match):
                md.run(1, ensemble="nphug", **options)
        with pytest.raises(InvalidParameter, match='for the ensemble "nphug"'):
            md.run(1, ensemble="npt", temperature=300, pressure=0.0, hugoniot=(1, 2, 3))
        md.velocities(0.0, seed=1)
        with pytest.raises(InvalidParameter, match="temperature, for the masses"):
            md.run(1, ensemble="nphug", pressure=5.0)
        md.run(2)  # still runs
    sheet = system.with_box(mdapy.Box(system.box.matrix, pbc=(True, True, False)))
    with MolecularDynamics(sheet, eam) as md:
        with pytest.raises(InvalidParameter, match="periodic in every direction"):
            md.run(1, ensemble="nphug", pressure=5.0, barostat="x")
        with pytest.raises(InvalidParameter, match="along z needs a box periodic"):
            md.run(1, ensemble="nph", pressure=5.0, barostat="z")
    turned = system.with_box(
        system.box.matrix @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]]),
        scale_positions=True,
    )
    engines = [eam] + ([] if lmp is None else [lmp])
    for potential in engines:
        with MolecularDynamics(turned, potential) as md:
            with pytest.raises(InvalidParameter, match="along one axis needs a box"):
                md.run(1, ensemble="nph", pressure=5.0, barostat="z")
