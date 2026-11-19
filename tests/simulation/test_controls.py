# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Controls of molecular dynamics runs: atoms held, moved and pushed. In
MDAPY and in LAMMPS (fix setforce with velocity set, fix move linear, fix
addforce, the ensemble on the free atoms) the same runs, step by step, with
the constants of LAMMPS and the masses of the potential file; and what
holds whatever the engine: held atoms stay, moved ones go at their velocity,
an added force changes the momentum, a moving wall does work."""

import numpy as np
import pytest

from mdapy.build import build_crystal
from mdapy.calculators import EAM, LAMMPS
from mdapy.errors import InvalidParameter
from mdapy.simulation import AddForce, Hold, MolecularDynamics, Move
from mdapy.simulation import _integrator as integrator

MASSES = {"Al": 26.981539, "Cu": 63.546}


@pytest.fixture(scope="module")
def potentials(tmp_path_factory):
    path = tmp_path_factory.mktemp("eam") / "AlCu.eam.alloy"
    EAM.zhou(["Al", "Cu"]).write(path)
    try:
        import lammps  # noqa: F401
    except ImportError:
        lmp = None
    else:
        lmp = LAMMPS(f"pair_style eam/alloy\npair_coeff * * {path} Al Cu", ["Al", "Cu"])
    return EAM(path), lmp


@pytest.fixture
def lammps_constants(monkeypatch):
    monkeypatch.setattr(integrator, "_KB", 8.617343e-5)
    monkeypatch.setattr(integrator, "_MVV2E", 1.0364269e-4)
    monkeypatch.setattr(integrator, "_FTM2V", 1.0 / 1.0364269e-4)


def slab(tilt=0.0, seed=0):
    """Al with a fifth of Cu, shaken, periodic, with the velocities of 300
    K; its bottom, top and middle layers."""
    system = build_crystal("fcc", 4.05, elements="Al", repeat=(3, 3, 5))
    matrix = system.box.matrix.copy()
    matrix[2, 0] += tilt * matrix[0, 0]
    system = system.with_box(matrix, scale_positions=True)
    rng = np.random.default_rng(seed)
    elements = rng.choice(["Al", "Cu"], len(system), p=[0.8, 0.2])
    system = system.with_columns(
        element=elements,
        mass=[MASSES[e] for e in elements],
        positions=system["positions"] + rng.normal(0, 0.03, (len(system), 3)),
    )
    masses = np.array([MASSES[e] for e in elements])
    # about 300 K: (kT / m)^(1/2) of each component
    sigma = np.sqrt(8.617e-5 * 300.0 / (masses * 1.0364e-4))
    velocities = rng.normal(size=(len(system), 3)) * sigma[:, None]
    velocities -= (masses @ velocities) / masses.sum()
    system = system.with_columns(velocities=velocities)
    z = system.box.to_fractional(system["positions"])[:, 2]
    return system, z < 0.18, z > 0.82, (z > 0.45) & (z < 0.55)


def run(potential, system, steps, controls, **options):
    with MolecularDynamics(system, potential, timestep=0.002) as md:
        md.run(steps, thermo=steps // 2, controls=controls, **options)
        return md.system, md.thermo


def gap(a, b):
    d = a.box.to_fractional(a["positions"]) - b.box.to_fractional(b["positions"])
    d -= np.round(d)
    return np.abs(d @ a.box.matrix).max(), np.abs(
        a["velocities"] - b["velocities"]
    ).max()


CASES = [
    (0.0, "xyz", {"ensemble": "nve"}),
    (0.0, "xyz", {"ensemble": "nvt", "temperature": (300, 500)}),
    (0.0, "z", {"ensemble": "nvt", "temperature": 300}),
    (0.0, "xy", {"ensemble": "nve"}),
    (0.3, "xyz", {"ensemble": "nvt", "temperature": 300}),
]


@pytest.mark.parametrize("tilt, directions, options", CASES)
def test_the_same_runs_as_lammps(
    potentials, lammps_constants, tilt, directions, options
):
    eam, lmp = potentials
    if lmp is None:
        pytest.skip("LAMMPS is not installed")
    system, bottom, top, middle = slab(tilt)
    controls = [
        Hold(bottom, directions=directions),
        Move(top, (0.4, -0.2, 0.0), name="top"),
        AddForce(middle, (0.0, 0.05, 0.02)),
    ]
    (a, ta), (b, tb) = (
        run(p, system, 300, controls, thermostat_damping=0.2, **options)
        for p in (lmp, eam)
    )
    dx, dv = gap(a, b)
    assert dx < 1e-9 and dv < 1e-8
    for column in ("temperature", "potential_energy", "kinetic_energy"):
        np.testing.assert_allclose(ta[column], tb[column], rtol=1e-9)
    np.testing.assert_allclose(
        np.stack(ta["top_force"]), np.stack(tb["top_force"]), rtol=0, atol=1e-8
    )


def test_what_the_controls_do(potentials):
    eam, _ = potentials
    system, bottom, top, middle = slab()
    velocity = np.array([0.4, -0.2, 0.1])
    with MolecularDynamics(system, eam, timestep=0.002) as md:
        md.run(
            200,
            thermo=50,
            ensemble="nvt",
            temperature=300,
            controls=[Hold(bottom), Move(top, velocity, name="top")],
        )
        a, thermo = md.system, md.thermo
        # the next run is free again
        md.run(10, thermo=10)
        after = md.thermo
    unwrapped = a["positions"] + a["image"] @ a.box.matrix
    np.testing.assert_allclose(
        unwrapped[bottom], system["positions"][bottom], atol=1e-12
    )
    assert np.abs(a["velocities"][bottom]).max() == 0.0
    np.testing.assert_allclose(
        unwrapped[top], system["positions"][top] + velocity * 0.4, atol=1e-10
    )
    np.testing.assert_allclose(
        a["velocities"][top], np.tile(velocity, (top.sum(), 1)), atol=1e-13
    )
    # the temperature of the atoms that move freely, without the moved ones
    free = ~(bottom | top)
    masses = system["mass"]
    twice = (masses[free, None] * a["velocities"][free] ** 2).sum() * integrator._MVV2E
    expected = twice / (3 * free.sum() * integrator._KB)
    assert thermo["temperature"][-1] == pytest.approx(expected, rel=1e-12)
    assert thermo["top_force"].null_count() == 0
    assert after["top_force"][-1] is None


def test_an_added_force_changes_the_momentum(potentials):
    """Between the atoms the forces cancel: the momentum grows by the force
    added to each atom times their number and the time."""
    eam, _ = potentials
    system, _, _, middle = slab()
    force = np.array([0.02, -0.01, 0.03])
    p0 = system["mass"] @ system["velocities"]
    with MolecularDynamics(system, eam, timestep=0.002) as md:
        md.run(100, thermo=100, controls=AddForce(middle, force))
        a = md.system
    p = a["mass"] @ a["velocities"]
    expected = p0 + middle.sum() * force * 0.2 / integrator._MVV2E
    np.testing.assert_allclose(p, expected, rtol=1e-9, atol=1e-9)


def test_a_moving_wall_does_work(potentials):
    """At constant energy the energy of the atoms grows by the work of the
    wall that moves them: minus the force they exert on it times its
    velocity, over the time."""
    eam, _ = potentials
    system, bottom, top, _ = slab()
    velocity = np.array([0.5, 0.0, 0.0])
    with MolecularDynamics(system, eam, timestep=0.001) as md:
        md.run(400, thermo=1, controls=[Hold(bottom), Move(top, velocity, name="top")])
        thermo = md.thermo
    force = np.stack(thermo["top_force"])
    power = -(force @ velocity)
    work = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * 0.001)])
    energy = thermo["total_energy"].to_numpy()
    assert np.abs(energy - energy[0] - work).max() < 2e-3 * np.abs(work).max() + 1e-4


def test_errors(potentials):
    eam, lmp = potentials
    system, bottom, top, _ = slab()
    n = len(system)
    with MolecularDynamics(system, eam) as md:
        with pytest.raises(
            InvalidParameter,
            match="Hold, Move, AddForce, Langevin, Deform, AdaptiveTimestep, Switch or Wall",
        ):
            md.run(1, controls=["hold"])
        with pytest.raises(InvalidParameter, match="values, not one"):
            md.run(1, controls=Hold(np.ones(3, dtype=bool)))
        with pytest.raises(InvalidParameter, match="held or moved by another"):
            md.run(1, controls=[Move(top, (1, 0, 0)), Hold(top)])
        with pytest.raises(InvalidParameter, match="held or moved by another"):
            md.run(1, controls=[Hold(top), Move(top, (1, 0, 0))])
        with pytest.raises(InvalidParameter, match="named"):
            md.run(
                1,
                controls=[
                    Move(top, (1, 0, 0), name="w"),
                    Move(bottom, (0, 0, 0), name="w"),
                ],
            )
        with pytest.raises(InvalidParameter, match="barostat"):
            md.run(
                1, ensemble="npt", temperature=300, pressure=0.0, controls=Hold(bottom)
            )
    with pytest.raises(InvalidParameter, match="mask"):
        Hold(np.zeros((n, 3), dtype=bool))
    with pytest.raises(InvalidParameter, match="directions"):
        Hold(bottom, directions="xx")
    with pytest.raises(InvalidParameter, match="velocity"):
        Move(top, (1, 0))
    with pytest.raises(InvalidParameter, match="force"):
        AddForce(top, (np.nan, 0, 0))
    with pytest.raises(InvalidParameter, match="name"):
        Move(top, (1, 0, 0), name="a b")
    assert "Hold(" in repr(Hold(bottom)) and "Move(" in repr(Move(top, (1, 0, 0)))
    assert "AddForce(" in repr(AddForce(top, (1, 0, 0)))
    if lmp is not None:
        tilted, bottom, _, _ = slab(0.3)
        rotated = tilted.with_box(
            tilted.box.matrix[[1, 0, 2]] * [1, 1, 1], scale_positions=True
        )
        with MolecularDynamics(rotated, lmp) as md:
            with pytest.raises(InvalidParameter, match="aligned|first vector"):
                md.run(1, controls=Hold(bottom, directions="z"))


def test_lammps_details(potentials, lammps_constants):
    """A run with forces added alone, an empty group, and the limits."""
    eam, lmp = potentials
    if lmp is None:
        pytest.skip("LAMMPS is not installed")
    system, bottom, top, middle = slab()
    controls = [
        AddForce(middle, (0.0, 0.05, 0.02)),
        Hold(np.zeros(len(system), dtype=bool)),
    ]
    (a, ta), (b, tb) = (run(p, system, 100, controls) for p in (lmp, eam))
    dx, dv = gap(a, b)
    assert dx < 1e-10 and dv < 1e-9
    np.testing.assert_allclose(ta["temperature"], tb["temperature"], rtol=1e-10)
    with MolecularDynamics(system, lmp) as md:
        with pytest.raises(InvalidParameter, match="at most 24"):
            md.run(1, controls=[AddForce(middle, (0, 0, 0))] * 25)
        with pytest.raises(InvalidParameter, match="barostat"):
            md.run(
                1, ensemble="npt", temperature=300, pressure=0.0, controls=Hold(bottom)
            )
        md.run(2, controls=Hold(bottom))  # usable after the refusals


def test_moved_atoms_under_langevin(potentials):
    eam, _ = potentials
    system, bottom, top, _ = slab()
    velocity = np.array([0.2, 0.0, -0.1])
    with MolecularDynamics(system, eam, timestep=0.002) as md:
        md.run(
            50,
            ensemble="nvt",
            temperature=300,
            thermostat="langevin",
            seed=1,
            controls=[Move(top, velocity), Hold(bottom, directions="z")],
        )
        a = md.system
    np.testing.assert_allclose(
        a["velocities"][top], np.tile(velocity, (top.sum(), 1)), atol=1e-13
    )
    assert np.abs(a["velocities"][bottom, 2]).max() == 0.0
    assert np.abs(a["velocities"][bottom, :2]).max() > 0.0


def test_on_the_gpu(potentials, monkeypatch):
    """The controls of the dynamics on the GPU, its module stood in for on
    the processor: the runs of the processor, and the heat of the baths."""
    from mdapy.calculators import _gpu
    from mdapy.simulation import Langevin

    from calculators.gpu_stand_in import stand_in

    monkeypatch.setattr(_gpu, "cuda", stand_in)
    system = build_crystal("fcc", 3.6, elements="Cu", repeat=(4, 4, 6))
    rng = np.random.default_rng(0)
    system = system.with_columns(
        positions=system["positions"] + rng.normal(0, 0.03, (len(system), 3))
    )
    z = system.box.to_fractional(system["positions"])[:, 2]
    bottom, top, middle = z < 0.2, z > 0.8, (z > 0.45) & (z < 0.55)
    controls = [
        Hold(bottom, directions="xz"),
        Move(top, (0.4, -0.2, 0.1), name="top"),
        AddForce(middle, (0.0, 0.05, 0.02)),
    ]
    out = []
    for device in ("cuda", "cpu"):
        with MolecularDynamics(
            system, EAM.zhou(["Cu"], device=device), timestep=0.002
        ) as md:
            md.velocities(300, seed=1)
            md.run(40, ensemble="nvt", temperature=300, thermo=20, controls=controls)
            md.run(5, thermo=5)  # free again
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    np.testing.assert_allclose(a["positions"], b["positions"], atol=1e-9)
    np.testing.assert_allclose(a["velocities"], b["velocities"], atol=1e-8)
    np.testing.assert_allclose(ta["temperature"], tb["temperature"], rtol=1e-9)
    np.testing.assert_allclose(
        np.stack(ta["top_force"].drop_nulls()),
        np.stack(tb["top_force"].drop_nulls()),
        atol=1e-9,
    )
    hot = Langevin(bottom, 600.0, damping=0.1, seed=3, name="hot")
    cold = Langevin(top, 100.0, damping=0.1, name="cold")
    with MolecularDynamics(
        system, EAM.zhou(["Cu"], device="cuda"), timestep=0.002
    ) as md:
        md.velocities(300, seed=1)
        md.run(1000, thermo=100, controls=[hot, cold, Hold(middle, directions="z")])
        thermo = md.thermo
    energy = thermo["total_energy"].to_numpy()
    heat = thermo["hot_energy"].to_numpy() + thermo["cold_energy"].to_numpy()
    assert np.abs(energy - energy[0] - heat).max() < 0.03
    with MolecularDynamics(system, EAM.zhou(["Cu"], device="cuda")) as md:
        many = [Move(z == z, (0, 0, 0), name="all")] + [
            Langevin(np.zeros(len(system), dtype=bool), 300.0) for _ in range(31)
        ]
        with pytest.raises(InvalidParameter, match="at most 31"):
            md.run(1, controls=many)


def baths(system, bottom, top):
    from mdapy.simulation import Langevin

    return [
        Langevin(bottom, 600.0, damping=0.1, seed=3, name="hot"),
        Langevin(top, 100.0, damping=0.1, name="cold"),  # an unpredictable seed
    ]


@pytest.mark.parametrize("engine", ["mdapy", "lammps"])
def test_heat_baths_balance_the_energy(potentials, engine):
    """A hot and a cold bath at the two ends: heat flows in at one and out
    at the other, and the energy of the atoms changes by the heat the two
    gave, to the error of the integrator."""
    eam, lmp = potentials
    potential = eam if engine == "mdapy" else lmp
    if potential is None:
        pytest.skip("LAMMPS is not installed")
    system, bottom, top, _ = slab()
    with MolecularDynamics(system, potential, timestep=0.002) as md:
        md.run(4000, thermo=200, controls=baths(system, bottom, top))
        thermo = md.thermo
    energy = thermo["total_energy"].to_numpy()
    heat = thermo["hot_energy"].to_numpy() + thermo["cold_energy"].to_numpy()
    assert thermo["hot_energy"][-1] > 20.0 and thermo["cold_energy"][-1] < -20.0
    assert np.abs(energy - energy[0] - heat).max() < 0.03


def test_a_bath_holds_its_temperature(potentials):
    from mdapy.simulation import Langevin

    eam, _ = potentials
    system, bottom, _, _ = slab()
    every = np.ones(len(system), dtype=bool)
    with MolecularDynamics(system, eam, timestep=0.002) as md:
        md.run(1000, controls=Langevin(every, (300.0, 500.0), damping=0.1, seed=1))
        md.run(4000, thermo=20, controls=Langevin(every, 500.0, damping=0.1, seed=2))
        assert md.thermo["temperature"][-200:].mean() == pytest.approx(500.0, rel=0.05)
        # a seed repeats, and the held directions keep still
        first = md.system
    for _ in range(2):
        with MolecularDynamics(system, eam, timestep=0.002) as md:
            md.run(
                50,
                controls=[
                    Langevin(every, 500.0, seed=9),
                    Hold(bottom, directions="z"),
                ],
            )
            again = md.system
        if _ == 0:
            reference = again
    np.testing.assert_array_equal(again["velocities"], reference["velocities"])
    assert np.abs(again["velocities"][bottom, 2]).max() == 0.0
    assert not np.array_equal(first["velocities"], again["velocities"])


def test_bath_errors(potentials):
    from mdapy.simulation import Langevin

    eam, _ = potentials
    system, bottom, top, _ = slab()
    with MolecularDynamics(system, eam) as md:
        with pytest.raises(InvalidParameter, match="take the place of the thermostat"):
            md.run(1, ensemble="nvt", temperature=300, controls=Langevin(top, 300.0))
        with pytest.raises(InvalidParameter, match="two Langevin baths"):
            md.run(1, controls=[Langevin(top, 300.0), Langevin(top, 400.0)])
        with pytest.raises(InvalidParameter, match="moved by a control is in"):
            md.run(1, controls=[Langevin(top, 300.0), Move(top, (1, 0, 0))])
    with pytest.raises(InvalidParameter, match="temperature"):
        Langevin(top, -1.0)
    with pytest.raises(InvalidParameter, match="temperature"):
        Langevin(top, (1.0, 2.0, 3.0))
    with pytest.raises(InvalidParameter, match="damping"):
        Langevin(top, 300.0, damping=0.0)
    with pytest.raises(InvalidParameter, match="seed"):
        Langevin(top, 300.0, seed=0)
    assert "300.0 K" in repr(Langevin(top, 300.0))
    assert "300.0 to 400.0 K" in repr(Langevin(top, (300.0, 400.0)))
