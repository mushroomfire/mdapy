# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Molecular dynamics in MDAPY against the same runs in LAMMPS, with the same
EAM potential. With the constants of LAMMPS's metal units and the masses of
the potential file, which LAMMPS takes, the runs agree to rounding; with
MDAPY's constants they agree as these differ (1e-6). Skipped where LAMMPS
is not installed."""

import numpy as np
import pytest

import mdapy
import mdapy.simulation._integrator as integrator
from mdapy.build import build_crystal
from mdapy.calculators import EAM, LAMMPS
from mdapy.simulation import MolecularDynamics

lammps = pytest.importorskip("lammps")

# the masses Zhou's potential file gives Al and Cu
MASSES = {"Al": 26.981539, "Cu": 63.546}


@pytest.fixture(scope="module")
def potentials(tmp_path_factory):
    path = tmp_path_factory.mktemp("eam") / "AlCu.eam.alloy"
    EAM.zhou(["Al", "Cu"]).write(path)
    lmp = LAMMPS(f"pair_style eam/alloy\npair_coeff * * {path} Al Cu", ["Al", "Cu"])
    return EAM(path), lmp


@pytest.fixture
def lammps_constants(monkeypatch):
    """The constants of LAMMPS's metal units in MDAPY's dynamics."""
    monkeypatch.setattr(integrator, "_KB", 8.617343e-5)
    monkeypatch.setattr(integrator, "_MVV2E", 1.0364269e-4)
    monkeypatch.setattr(integrator, "_FTM2V", 1.0 / 1.0364269e-4)
    monkeypatch.setattr(integrator, "_GPA_TO_EV_PER_A3", 1e4 / 1.6021765e6)
    monkeypatch.setattr(integrator, "_EV_PER_A3_TO_GPA", 1.6021765e6 / 1e4)


BOXES = {
    "cube": (np.diag([12.15, 12.15, 12.15]), (True, True, True)),
    "tilted": (
        np.array([[12.2, 0.0, 0.0], [1.3, 11.9, 0.0], [-0.7, 0.9, 12.4]]),
        (True, True, True),
    ),
    # rotated and left-handed: LAMMPS swaps two of its vectors
    "left-handed": (
        np.array([[0.0, 12.2, 0.0], [11.9, 0.4, 0.0], [0.3, -0.5, -12.4]]),
        (True, True, True),
    ),
    "sheet": (
        np.array([[12.15, 0.0, 0.0], [0.8, 12.15, 0.0], [0.0, 0.0, 25.0]]),
        (True, True, False),
    ),
}


def start(potentials, name):
    """Displaced Al with a fifth of Cu in a box, with the velocities of
    300 K from LAMMPS and the masses of the potential file."""
    _, lmp = potentials
    matrix, pbc = BOXES[name]
    system = build_crystal("fcc", 4.05, elements="Al", repeat=(3, 3, 3))
    rng = np.random.default_rng(0)
    shift = [0, 0, 5.0] if name == "sheet" else [0, 0, 0]
    system = system.with_columns(
        element=rng.choice(["Al", "Cu"], len(system), p=[0.8, 0.2]),
        positions=system["positions"] + rng.normal(0, 0.03, (len(system), 3)) + shift,
    )
    if name == "sheet":
        system = system.with_box(mdapy.Box(matrix, pbc=pbc))
    else:
        system = system.with_box(matrix, scale_positions=True)
    with MolecularDynamics(system, lmp, timestep=0.002) as md:
        md.velocities(300, seed=7)
        system = md.system.drop("image")
    return system.with_columns(mass=[MASSES[e] for e in system["element"]])


def both(potentials, system, steps, **options):
    """The system at the end of the same run in LAMMPS and in MDAPY."""
    eam, lmp = potentials
    out = []
    for potential in (lmp, eam):
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.run(steps, thermo=steps, thermostat_damping=0.2, **options)
            out.append((md.system, md.thermo))
    return out


def differences(a, b):
    """The largest differences of the positions (up to periodic images),
    velocities and boxes."""
    d = a.box.to_fractional(a["positions"]) - b.box.to_fractional(b["positions"])
    d -= np.round(d) * a.box.pbc
    return (
        np.abs(d @ a.box.matrix).max(),
        np.abs(a["velocities"] - b["velocities"]).max(),
        np.abs(a.box.matrix - b.box.matrix).max(),
    )


RUNS = [
    ("cube", {"ensemble": "nve"}),
    ("cube", {"ensemble": "nvt", "temperature": (300, 600)}),
    ("cube", {"ensemble": "npt", "temperature": 300, "pressure": 1.0}),
    ("cube", {"ensemble": "npt", "temperature": 300, "pressure": (1.0, 2.0),
              "barostat": "aniso"}),
    ("cube", {"ensemble": "npt", "temperature": 300, "pressure": 0.0,
              "barostat": "tri"}),
    ("cube", {"ensemble": "nph", "pressure": 0.0}),
    ("tilted", {"ensemble": "nvt", "temperature": 300}),
    ("tilted", {"ensemble": "npt", "temperature": 300, "pressure": 0.5}),
    ("tilted", {"ensemble": "npt", "temperature": 300, "pressure": 0.0,
                "barostat": "tri"}),
    ("tilted", {"ensemble": "npt", "temperature": 300, "pressure": 0.0,
                "barostat": "xy"}),
    ("left-handed", {"ensemble": "npt", "temperature": 300, "pressure": 0.0,
                     "barostat": "tri"}),
    ("left-handed", {"ensemble": "npt", "temperature": 300, "pressure": 0.0,
                     "barostat": "xy"}),
    ("sheet", {"ensemble": "npt", "temperature": 300, "pressure": 0.0}),
    ("sheet", {"ensemble": "npt", "temperature": 300, "pressure": 0.0,
               "barostat": "aniso"}),
    ("sheet", {"ensemble": "npt", "temperature": 300, "pressure": 0.0,
               "barostat": "tri"}),
]  # fmt: skip


@pytest.mark.parametrize("box, options", RUNS)
def test_the_same_run_as_lammps(potentials, lammps_constants, box, options):
    system = start(potentials, box)
    (a, thermo_a), (b, thermo_b) = both(
        potentials, system, 300, barostat_damping=2.0, **options
    )
    dx, dv, dbox = differences(a, b)
    assert dx < 1e-10 and dv < 1e-9 and dbox < 1e-10
    for column in ("temperature", "potential_energy", "kinetic_energy", "volume"):
        np.testing.assert_allclose(thermo_a[column], thermo_b[column], rtol=1e-10)
    np.testing.assert_allclose(thermo_a["pressure"], thermo_b["pressure"], atol=1e-8)


@pytest.mark.parametrize(
    "options",
    [
        {"ensemble": "nvt", "temperature": 300},
        {"ensemble": "npt", "temperature": 300, "pressure": 0.0, "barostat": "tri"},
    ],
)
def test_with_the_constants_of_mdapy(potentials, options):
    """MDAPY's constants (CODATA 2018) differ from those of LAMMPS by about
    1e-6: so do the runs, in 300 steps."""
    system = start(potentials, "tilted")
    (a, thermo_a), (b, thermo_b) = both(
        potentials, system, 300, barostat_damping=2.0, **options
    )
    dx, dv, dbox = differences(a, b)
    assert dx < 1e-5 and dv < 1e-4 and dbox < 1e-5
    np.testing.assert_allclose(thermo_a["temperature"], thermo_b["temperature"],
                               rtol=1e-5)  # fmt: skip


def test_langevin_in_both(potentials):
    """Langevin dynamics, whose random forces differ: the same averages,
    and in LAMMPS the same run for the same seed."""
    system = start(potentials, "cube")
    averages = []
    for potential in potentials:
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.run(1000, ensemble="npt", temperature=300, pressure=0.0,
                   thermostat="langevin", seed=11)  # fmt: skip
            md.run(6000, ensemble="npt", temperature=300, pressure=0.0,
                   thermostat="langevin", seed=12, thermo=20)  # fmt: skip
            late = md.thermo.filter(md.thermo["step"] > 1000)
            averages.append(
                (
                    late["temperature"].mean(),
                    late["pressure"].mean(),
                    late["volume"].mean(),
                )  # fmt: skip
            )
    (t_a, p_a, v_a), (t_b, p_b, v_b) = averages
    # 108 atoms: all 3N directions are hot, the temperature counts 3N - 3
    assert t_a == pytest.approx(300 * 108 / 107, rel=0.03)
    assert t_b == pytest.approx(300 * 108 / 107, rel=0.03)
    assert p_a == pytest.approx(0.0, abs=0.2) and p_b == pytest.approx(0.0, abs=0.2)
    assert v_a == pytest.approx(v_b, rel=3e-3)
    _, lmp = potentials
    runs = []
    for _ in range(2):
        with MolecularDynamics(system, lmp, timestep=0.002) as md:
            md.run(50, ensemble="nvt", temperature=300, thermostat="langevin", seed=3)
            runs.append(md.system["positions"])
    np.testing.assert_array_equal(runs[0], runs[1])
    with MolecularDynamics(system, lmp, timestep=0.002) as md:
        md.run(10, ensemble="nvt", temperature=300, thermostat="langevin")
        assert md.step == 10


def test_lammps_does_not_hold_atoms(potentials):
    _, lmp = potentials
    system = start(potentials, "cube")
    held = system.with_columns(movable=np.zeros((len(system), 3), dtype=bool))
    with pytest.raises(mdapy.errors.InvalidParameter, match="movable"):
        MolecularDynamics(held, lmp)
    free = system.with_columns(movable=np.ones((len(system), 3), dtype=bool))
    with MolecularDynamics(free, lmp) as md:
        md.run(1)


@pytest.mark.parametrize("tilt", [0.0, 1.2])
def test_an_open_box_grows_as_in_lammps(potentials, lammps_constants, tilt):
    """Atoms flying off a slab open along z: the box grows to hold them, never
    smaller than the box given, as LAMMPS's boundary m grows it, at every step
    here as LAMMPS builds its lists again at every step; the volume and the
    pressure with it."""
    eam, lmp = potentials
    crystal = build_crystal("fcc", 4.05, elements="Al", repeat=(3, 3, 3))
    matrix = crystal.box.matrix.copy()
    matrix[1, 0] = tilt
    crystal = crystal.with_box(
        mdapy.Box(matrix, pbc=(True, True, False)), scale_positions=True
    )
    z = crystal["positions"][:, 2]
    velocities = np.zeros((len(crystal), 3))
    velocities[z > z.max() - 1.0, 2] = 30.0  # the top layer flies off at 3 km/s
    velocities[z < 1.0, 2] = -20.0  # and the bottom one down
    system = crystal.with_columns(velocities=velocities)
    out = []
    for potential in (lmp, eam):
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            if potential is lmp:
                md.command("neigh_modify every 1 delay 0 check no")
            md.run(200, thermo=50)
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    np.testing.assert_allclose(a.box.matrix, b.box.matrix, rtol=0, atol=1e-8)
    np.testing.assert_allclose(a.box.origin, b.box.origin, rtol=0, atol=1e-8)
    np.testing.assert_allclose(ta["volume"], tb["volume"], rtol=1e-7)
    np.testing.assert_allclose(ta["pressure"], tb["pressure"], rtol=0, atol=1e-5)
    # grown while the atoms flew, back to the box given as they returned
    assert tb["volume"].max() > 1.05 * tb["volume"][0]
    fractional = b.box.to_fractional(b["positions"])[:, 2]
    assert fractional.min() > 0 and fractional.max() < 1
