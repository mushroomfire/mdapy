# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Path-integral molecular dynamics: the normal modes of the free ring
polymer; the free ring polymer kept exactly and the energy of RPMD kept;
atoms tied by springs, harmonic oscillators, at the energies of the finite
ring polymer in closed form, quantum far from the classical; restarts;
the beads and the centroids; on the GPU stood in; the errors."""

import math

import numpy as np
import pytest

import mdapy
from mdapy import units
from mdapy.calculators import LennardJones, Springs, _gpu
from mdapy.errors import InvalidParameter
from mdapy.simulation import MolecularDynamics, _domain, _pimd
from mdapy.simulation.controls import Hold

from calculators.gpu_stand_in import stand_in

KB, MVV2E = units.BOLTZMANN, units.AMU_A2_PER_PS2_TO_EV
HBAR = units.PLANCK / (2 * math.pi)
NVT = {"ensemble": "nvt", "thermostat": "langevin"}


def hydrogen(n=64, quantum=0.2, seed=0):
    """Hydrogen atoms each tied to its own site by a spring of hbar w =
    quantum eV, apart from each other."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 20, (n, 3))
    system = mdapy.System(x, mdapy.Box(20.0), elements="H")
    w = quantum / HBAR
    return system, Springs(x, np.full(n, 1.008 * w * w * MVV2E)), w


def finite(beads, w, temperature, n):
    """The mean potential energy of n oscillators of 3 dimensions in a ring
    polymer of the beads, and the kinetic energy of its estimators."""
    beta = 1.0 / (KB * temperature)
    wp = beads / (beta * HBAR)
    wk = 2 * wp * np.sin(np.pi * np.arange(beads) / beads)
    return 3 * n / (2 * beta) * float(np.sum(w**2 / (w**2 + wk**2)))


def test_the_normal_modes():
    """C is orthogonal and turns the springs of the ring, a Laplacian, into
    the frequencies 2 sin(pi k / P) squared."""
    for p in (1, 2, 3, 4, 7, 8):
        c, frequencies = _pimd.normal_modes(p)
        np.testing.assert_allclose(c @ c.T, np.eye(p), atol=1e-14)
        ring = (
            2 * np.eye(p)
            - np.roll(np.eye(p), 1, axis=1)
            - np.roll(np.eye(p), -1, axis=1)
        )
        if p == 1:
            ring = np.zeros((1, 1))
        elif p == 2:
            ring = np.array([[2.0, -2.0], [-2.0, 2.0]])
        np.testing.assert_allclose(c @ ring @ c.T, np.diag(frequencies**2), atol=1e-12)


def test_the_free_ring_polymer_and_rpmd_keep_their_energy():
    """Without a potential the free ring polymer turns exactly; with the
    springs to the sites RPMD keeps the energy of the ring polymer to an
    error that falls as the square of the time step, symplectic."""
    system, springs, _ = hydrogen(16)
    free = LennardJones({"H": (0.0, 1.0)}, 3.0)

    def spread(potential, dt):
        with MolecularDynamics(system, potential, timestep=dt, beads=6) as md:
            md.velocities(300, seed=1)
            md.run(int(round(0.1 / dt)), ensemble="nve", temperature=300,
                   thermo=int(round(0.01 / dt)))  # fmt: skip
            energy = md.thermo["ring_polymer_energy"].to_numpy()
        return np.ptp(energy) / abs(energy).max()

    assert spread(free, 0.0005) < 1e-12
    coarse, fine = spread(springs, 0.0005), spread(springs, 0.00025)
    assert fine < 2e-3 and 3.0 < coarse / fine < 5.0


@pytest.mark.parametrize("beads", [4, 8])
def test_oscillators_at_the_energies_of_the_ring_polymer(beads):
    """At 300 K and hbar w = 0.2 eV, the potential energy and both
    estimators of the kinetic energy, averaged, are those of the finite
    ring polymer in closed form, far above the classical 3 n k T / 2."""
    n, temperature = 64, 300.0
    system, springs, w = hydrogen(n)
    with MolecularDynamics(system, springs, timestep=0.0005, beads=beads) as md:
        md.velocities(temperature, seed=1)
        md.run(1000, temperature=temperature, thermostat_damping=0.05, seed=3, **NVT)
        md.run(6000, temperature=temperature, thermostat_damping=0.05, seed=4,
               thermo=5, **NVT)  # fmt: skip
        th = md.thermo.filter(md.thermo["step"] > 1000)
    expected = finite(beads, w, temperature, n)
    assert expected > 2.5 * 1.5 * n * KB * temperature
    for column in ("potential_energy", "kinetic_energy", "primitive_kinetic_energy"):
        assert th[column].mean() == pytest.approx(expected, rel=0.03), column
    assert th["temperature"].mean() == pytest.approx(temperature, rel=0.03)


def test_beads_and_centroids():
    system, springs, _ = hydrogen(8)
    with MolecularDynamics(system, springs, timestep=0.0005, beads=4) as md:
        md.velocities(300, seed=1)
        assert md.thermo.height == 0
        md.run(20, temperature=300, seed=2, **NVT)
        beads = md.beads
        assert len(beads) == 4 and [b.info["bead"] for b in beads] == [0, 1, 2, 3]
        centroid = md.system
        unwrapped = np.mean(
            [b["positions"] + b["image"] @ b.box.matrix for b in beads], 0
        )
        wrapped, _ = centroid.box.wrap(unwrapped)
        np.testing.assert_allclose(centroid["positions"], wrapped, atol=1e-12)
        np.testing.assert_allclose(
            centroid["velocities"],
            np.mean([b["velocities"] for b in beads], 0),
            atol=1e-12,
        )
        frames = list(md.iterate(10, temperature=300, seed=2, save_every=5, **NVT))
        assert len(frames) == 3 and "bead" not in frames[-1].info
    with MolecularDynamics(system, springs, timestep=0.0005) as md:
        assert len(md.beads) == 1  # classical: the atoms alone


def test_velocities_are_those_of_the_temperature():
    system, springs, _ = hydrogen(32)
    with MolecularDynamics(system, springs, timestep=0.0005, beads=8) as md:
        md.velocities(250, seed=3)
        md.run(0, temperature=250, **NVT)
        assert md.thermo["temperature"][0] == pytest.approx(250, rel=1e-12)
        centroid = md.system["velocities"]
        assert np.abs(1.008 * centroid.sum(axis=0)).max() < 1e-10
        md.velocities(0, seed=3)
        assert np.abs(md.beads[2]["velocities"]).max() == 0.0


def test_a_run_finished_from_its_restart(tmp_path):
    system, springs, _ = hydrogen(16)
    options = {"temperature": 300, "seed": 5, "thermo": 10, **NVT}
    with MolecularDynamics(system, springs, timestep=0.0005, beads=4) as md:
        md.velocities(300, seed=1)
        md.run(60, **options)
        whole = md.system, md.thermo
    path = tmp_path / "pimd.restart"
    with MolecularDynamics(system, springs, timestep=0.0005, beads=4) as md:
        md.velocities(300, seed=1)
        frames = md.iterate(
            60, save_every=30, restart=path, restart_every=30, **options
        )
        for frame in frames:
            if frame.info["timestep"] >= 60:
                break
        frames.close()
    again = MolecularDynamics.load(path, springs)
    assert again.step == 30 and len(again.beads) == 4
    again.resume()
    np.testing.assert_allclose(
        again.system["positions"], whole[0]["positions"], atol=1e-12
    )
    np.testing.assert_allclose(
        again.system["velocities"], whole[0]["velocities"], atol=1e-12
    )
    for column in ("potential_energy", "kinetic_energy", "pressure"):
        np.testing.assert_allclose(
            again.thermo[column].to_numpy(), whole[1][column].to_numpy(), rtol=1e-12,
            err_msg=column,
        )  # fmt: skip


def test_on_the_gpu_stood_in(monkeypatch):
    """The forces of each bead on the GPU, the steps on the processor: the
    run of the processor."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    from mdapy.build import build_crystal
    from mdapy.calculators import EAM

    crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(5, 5, 5))
    import io

    out, stream = [], io.StringIO()
    for device in ("cpu", "cuda"):
        eam = EAM.zhou(["Cu"], device=device, precision="double")
        log = stream if device == "cuda" else None
        with MolecularDynamics(crystal, eam, timestep=0.002, beads=3, log=log) as md:
            md.velocities(50, seed=1)
            md.run(30, temperature=50, seed=2, thermo=10, **NVT)
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    assert "path integrals of 3 beads; the forces on the GPU" in stream.getvalue()
    np.testing.assert_allclose(a["velocities"], b["velocities"], atol=1e-10)
    np.testing.assert_allclose(ta["pressure"], tb["pressure"], rtol=1e-10)


def test_a_mass_column_and_the_log():
    """The masses of a mass column, deuterium for hydrogen; the log names
    the beads."""
    import io

    system, springs, _ = hydrogen(8)
    heavy = system.with_columns(mass=np.full(8, 2.014))
    stream = io.StringIO()
    with MolecularDynamics(heavy, springs, timestep=0.0005, beads=4, log=stream) as md:
        md.velocities(300, seed=1)
        md.run(10, temperature=300, seed=2, **NVT)
    assert "path integrals of 4 beads; the processor" in stream.getvalue()
    assert md_closed_mass(heavy) == pytest.approx(2.014)
    # the same draws of the same temperature: deuterium slower by sqrt(m_H / m_D)
    speeds = []
    for atoms in (system, heavy):
        with MolecularDynamics(atoms, springs, beads=4) as md:
            md.velocities(300, seed=1)
            speeds.append(np.stack([b["velocities"] for b in md.beads]))
    np.testing.assert_allclose(
        speeds[1], speeds[0] * np.sqrt(1.008 / 2.014), rtol=1e-12
    )


def md_closed_mass(system):
    with MolecularDynamics(system, Springs(system["positions"], np.ones(len(system))),
                           beads=2) as md:  # fmt: skip
        return float(md.beads[1]["mass"][0])


def test_atoms_that_overlap():
    from mdapy.errors import CalculationFailed

    system = mdapy.System([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]], mdapy.Box(20.0),
                          elements="H")  # fmt: skip
    lj = LennardJones({"H": (0.01, 1.0)}, 3.0)
    with pytest.raises(CalculationFailed, match="not finite"):
        with MolecularDynamics(system, lj, beads=2) as md:
            md.run(1, temperature=300, **NVT)


def test_errors(monkeypatch):
    system, springs, _ = hydrogen(8)
    for beads in (0, -2, 1.5, True):
        with pytest.raises(InvalidParameter, match="beads must be"):
            MolecularDynamics(system, springs, beads=beads)
    held = system.with_columns(movable=np.zeros((8, 3), dtype=bool))
    with pytest.raises(InvalidParameter, match="no movable column"):
        MolecularDynamics(held, springs, beads=2)
    untyped = mdapy.System(system["positions"], system.box)
    with pytest.raises(Exception, match="masses"):
        MolecularDynamics(untyped, springs, beads=2)
    with pytest.raises(InvalidParameter, match="MDAPY calculator"):
        _pimd._RingPolymer(system, "springs", 0.001, 2)

    class EnergyOnly(Springs):
        implemented = frozenset({"energy"})

    with pytest.raises(InvalidParameter, match="does not compute energies and forces"):
        MolecularDynamics(system, EnergyOnly(system["positions"], np.ones(8)), beads=2)
    with MolecularDynamics(system, springs, beads=2) as md:
        for options, message in (
            ({"ensemble": "npt", "temperature": 300, "pressure": 0.0}, '"nvt" or "nve"'),
            ({"ensemble": "nvt", "temperature": 300}, "PILE_L"),
            ({"ensemble": "nve"}, "needs a temperature"),
            ({"ensemble": "nve", "temperature": (300, 400)}, "one positive temperature"),
            ({"controls": [Hold(np.ones(8, dtype=bool))], "temperature": 300},
             "no controls"),
        ):  # fmt: skip
            with pytest.raises(InvalidParameter, match=message):
                md.run(1, **options)
    monkeypatch.setattr(_domain, "world", lambda: object())
    with pytest.raises(InvalidParameter, match="one process"):
        MolecularDynamics(system, springs, beads=2)
