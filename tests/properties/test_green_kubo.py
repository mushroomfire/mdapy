# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The Green-Kubo conductivity: the correlation of a flux against its sums
by brute force, a process of known correlation, and LAMMPS's own fix
ave/correlate on the same run of Lennard-Jones argon. The LAMMPS parts are
skipped where LAMMPS is not installed."""

import dataclasses

import numpy as np
import pytest

from mdapy import units
from mdapy.build import build_crystal
from mdapy.errors import InvalidParameter
from mdapy.properties._green_kubo import _W_PER_M_K, _running
from mdapy.properties.transport import green_kubo, green_kubo_from_flux


def test_the_correlation_by_brute_force():
    flux = np.random.default_rng(0).normal(size=(500, 3))
    lags = 40
    found = _running(flux, lags)
    correlation = np.array(
        [
            [
                [np.mean(flux[: len(flux) - k, a] * flux[k:, b]) for b in range(3)]
                for a in range(3)
            ]
            for k in range(lags)
        ]
    )
    correlation = 0.5 * (correlation + correlation.transpose(0, 2, 1))
    expected = np.zeros_like(correlation)
    for k in range(1, lags):
        expected[k] = np.trapezoid(correlation[: k + 1], axis=0)
    np.testing.assert_allclose(found, expected, rtol=1e-10, atol=1e-12)


def test_a_flux_of_known_correlation():
    """An Ornstein-Uhlenbeck process of variance s² and time tau: the
    integral of its correlation is s² tau, independent along each axis."""
    rng = np.random.default_rng(1)
    count, interval, tau, s = 800000, 0.01, 0.2, 2.0
    decay = np.exp(-interval / tau)
    flux = np.empty((count, 3))
    flux[0] = rng.normal(0, s, 3)
    noise = rng.normal(0, s * np.sqrt(1 - decay**2), (count, 3))
    for k in range(1, count):
        flux[k] = decay * flux[k - 1] + noise[k]
    volume, temperature = 1000.0, 300.0
    result = green_kubo_from_flux(
        flux,
        interval=interval,
        volume=volume,
        temperature=temperature,
        correlation=2.0,
        blocks=10,
    )
    expected = s**2 * tau / (units.BOLTZMANN * temperature**2 * volume) * _W_PER_M_K
    diagonal = np.diag(result.kappa)
    assert diagonal.mean() == pytest.approx(expected, rel=0.05)
    # each axis within its error, which the blocks estimate
    error = np.diag(result.errors).mean()
    assert (np.abs(diagonal - expected) < 4 * error).all()
    assert 0.01 * expected < error < 0.1 * expected
    assert np.abs(result.kappa - np.diag(diagonal)).max() < 4 * error
    assert result.times[-1] == pytest.approx(2.0)
    assert result.running.shape == (len(result.times), 3, 3)
    np.testing.assert_array_equal(result.flux, flux)
    assert repr(result).startswith("GreenKubo(")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.kappa = None


def test_errors():
    flux = np.zeros((100, 3))
    good = dict(interval=0.01, volume=100.0, temperature=300.0, correlation=0.1)
    with pytest.raises(InvalidParameter, match="flux"):
        green_kubo_from_flux(flux[:, :2], **good)
    with pytest.raises(InvalidParameter, match="flux"):
        green_kubo_from_flux(np.full((100, 3), np.nan), **good)
    for name in good:
        with pytest.raises(InvalidParameter, match=name):
            green_kubo_from_flux(flux, **{**good, name: 0.0})
    for blocks in (1, 2.5, True):
        with pytest.raises(InvalidParameter, match="blocks"):
            green_kubo_from_flux(flux, **good, blocks=blocks)
    with pytest.raises(InvalidParameter, match="shorter than the correlation"):
        green_kubo_from_flux(flux, **{**good, "correlation": 0.5})


ARGON = "pair_style lj/cut 8.5\npair_coeff * * 0.0104 3.405"


def test_against_fix_ave_correlate_of_lammps(tmp_path):
    """The same run of solid argon: LAMMPS correlates the heat flux and
    integrates it as its example of Green-Kubo does; the flux it samples
    gives the same conductivity."""
    pytest.importorskip("lammps")
    from mdapy.calculators import LAMMPS
    from mdapy.simulation import MolecularDynamics

    potential = LAMMPS(ARGON, elements=["Ar"])
    system = build_crystal("fcc", 5.3, elements="Ar", repeat=3)
    every, lags, temperature, timestep = 2, 50, 40.0, 0.002
    path = tmp_path / "flux.txt"
    with MolecularDynamics(system, potential, timestep=timestep) as md:
        md.velocities(2 * temperature, seed=3)
        md.run(1000, ensemble="nvt", temperature=temperature)
        md.command(
            "reset_timestep 0\n"
            "compute ke all ke/atom\n"
            "compute pe all pe/atom\n"
            "compute stress all centroid/stress/atom NULL virial\n"
            "compute flux all heat/flux ke pe stress\n"
            f"fix samples all ave/time {every} 1 {every} c_flux[1] c_flux[2] c_flux[3] "
            f'file {path} format " %.17g"\n'
            f"fix JJ all ave/correlate {every} {lags} {every * lags} "
            "c_flux[1] c_flux[2] c_flux[3] type auto ave running\n"
            f"variable k11 equal trap(f_JJ[3])\n"
            f"variable k22 equal trap(f_JJ[4])\n"
            f"variable k33 equal trap(f_JJ[5])"
        )
        md.run(4000, ensemble="nve", thermo=every * lags)
        traps = [md.lammps.extract_variable(f"k{x}{x}") for x in (1, 2, 3)]
        volume = system.box.volume
    flux = np.loadtxt(path, comments="#")[:, 1:]
    found = green_kubo_from_flux(
        flux,
        interval=every * timestep,
        volume=volume,
        temperature=temperature,
        correlation=(lags - 1) * every * timestep,
    )
    scale = every * timestep / (units.BOLTZMANN * temperature**2 * volume) * _W_PER_M_K
    np.testing.assert_allclose(
        np.diag(found.kappa), np.array(traps) * scale, rtol=1e-12
    )


def test_solid_argon():
    """The whole run, at constant temperature then constant energy: a
    conductivity of the order of that of argon, the same along x, y and z
    within its errors."""
    pytest.importorskip("lammps")
    from mdapy.calculators import LAMMPS

    potential = LAMMPS(ARGON, elements=["Ar"])
    system = build_crystal("fcc", 5.3, elements="Ar", repeat=4)
    result = green_kubo(
        system,
        potential,
        40.0,
        timestep=0.002,
        equilibration=2000,
        production=20000,
        every=2,
        correlation=4.0,
        blocks=4,
    )
    diagonal = np.diag(result.kappa)
    assert (diagonal > 0.1).all() and (diagonal < 10).all()
    assert result.temperature == pytest.approx(40.0, rel=0.1)
    assert result.volume == pytest.approx(system.box.volume)
    spread = np.abs(diagonal - diagonal.mean()).max()
    assert spread < 4 * np.diag(result.errors).max() + 0.1 * diagonal.mean()


def test_errors_of_the_run(tmp_path):
    pytest.importorskip("lammps")
    from mdapy.calculators import LAMMPS

    potential = LAMMPS(ARGON, elements=["Ar"])
    system = build_crystal("fcc", 5.3, elements="Ar", repeat=3)
    with pytest.raises(InvalidParameter, match="temperature"):
        green_kubo(system, potential, 0.0)
    with pytest.raises(InvalidParameter, match="positive"):
        green_kubo(system, potential, 40.0, production=0)
    open_box = system.with_box(system.box.matrix).with_box(
        type(system.box)(system.box.matrix, pbc=(True, True, False))
    )
    with pytest.raises(InvalidParameter, match="periodic"):
        green_kubo(open_box, potential, 40.0)
    # Stillinger-Weber, of which LAMMPS has no centroid stress of each atom
    path = tmp_path / "Si.sw"
    path.write_text(
        "Si Si Si 2.1683 2.0951 1.80 21.0 1.20 -0.333333333333 7.049556277 "
        "0.6022245584 4.0 0.0 0.0\n"
    )
    silicon = LAMMPS(f"pair_style sw\npair_coeff * * {path} Si", elements=["Si"])
    crystal = build_crystal("diamond", 5.431, elements="Si", repeat=2)
    with pytest.raises(InvalidParameter, match="centroid stress"):
        green_kubo(crystal, silicon, 300.0, equilibration=10, production=20, every=2)


def shaken(structure, a, elements, repeat, seed=0):
    """A crystal with its atoms shaken and moving, in a tilted box in the
    form of LAMMPS."""
    import mdapy

    crystal = build_crystal(structure, a, elements=elements[0], repeat=repeat)
    matrix = crystal.box.matrix.copy()
    matrix[1, 0], matrix[2, 0], matrix[2, 1] = 0.4, -0.3, 0.2
    crystal = crystal.with_box(mdapy.Box(matrix), scale_positions=True)
    rng = np.random.default_rng(seed)
    n = len(crystal)
    return crystal.with_columns(
        positions=crystal["positions"] + rng.normal(0, 0.05, (n, 3)),
        velocities=rng.normal(0, 3.0, (n, 3)),
        element=list(rng.choice(elements, n)),
    )


@pytest.mark.parametrize("kind", ["eam", "lj"])
def test_the_flux_of_lammps(kind, tmp_path, monkeypatch):
    """LAMMPS's compute heat/flux with the centroid stress of each atom, with
    the kinetic energies of LAMMPS's constant of metal units: MDAPY's heat flux
    of a pair potential; of EAM, that of its pair forces split in halves,
    which LAMMPS's centroid stress of EAM is, and not MDAPY's of Fan et al.
    (2015)."""
    pytest.importorskip("lammps")
    import mdapy
    from mdapy.calculators import EAM, LAMMPS, LennardJones
    from mdapy.properties._green_kubo import _flux
    from mdapy.simulation import MolecularDynamics

    from calculators.eam_files import write_setfl

    if kind == "eam":
        setfl = write_setfl(tmp_path / "CuNi.eam.alloy")
        ours = EAM(setfl)
        theirs = LAMMPS(
            f"pair_style eam/alloy\npair_coeff * * {setfl} Cu Ni", ["Cu", "Ni"]
        )
        system = shaken("fcc", 3.6, ["Cu", "Ni"], 3)
    else:
        ours = LennardJones({"Ar": (0.0104, 3.405)}, 8.5)
        theirs = LAMMPS(ARGON, elements=["Ar"])
        system = shaken("fcc", 5.3, ["Ar"], 4)
    system = system.with_columns(
        mass=mdapy.elements.atomic_masses(list(system["element"]))
    )
    with MolecularDynamics(system, theirs) as md:
        md.command(
            "compute ke all ke/atom\n"
            "compute pe all pe/atom\n"
            "compute stress all centroid/stress/atom NULL virial\n"
            "compute flux all heat/flux ke pe stress\n"
            "run 0"
        )
        expected = np.array(md.lammps.numpy.extract_compute("flux", 0, 1)[:3])
    monkeypatch.setattr(units, "AMU_A2_PER_PS2_TO_EV", 1.0364269e-4)
    found = _flux(system, ours)
    if kind == "eam":
        halves = ours.compute(system, ("atomic_energies", "atomic_virials"))
        v = system["velocities"]
        kinetic = 0.5 * system["mass"] * (v * v).sum(axis=1) * 1.0364269e-4
        split = (halves.atomic_energies + kinetic) @ v + np.einsum(
            "iab,ib->a", halves.atomic_virials, v
        )
        np.testing.assert_allclose(
            split, expected, rtol=1e-11, atol=1e-11 * np.abs(expected).max()
        )
        assert np.abs(found - expected).max() > 1e-3 * np.abs(expected).max()
        return
    np.testing.assert_allclose(
        found, expected, rtol=1e-11, atol=1e-11 * np.abs(expected).max()
    )


@pytest.mark.parametrize("kind", ["nep", "lj", "sw", "tersoff", "eam"])
def test_the_flux_is_the_change_of_the_energy_moment(kind, tmp_path):
    """The heat flux of a cluster times its volume, sum_i E_i v_i + sum_i W_i
    v_i, is the time derivative of sum_i r_i E_i along the dynamics, by
    central differences: the virial of each atom of NEP, Stillinger-Weber and
    Tersoff from the energies of the others (Fan et al. 2015), the terms of
    these two whole with their centres, and of a pair potential split in
    halves."""
    import mdapy
    from mdapy.calculators import NEP, LennardJones, StillingerWeber, Tersoff
    from mdapy.calculators._base import HEAT_FLUX_TERMS
    from mdapy.properties._green_kubo import _flux
    from mdapy.properties._phonons import _masses

    from calculators.nep_files import write_nep
    from calculators.potential_files import write_sw, write_tersoff

    if kind == "nep":
        potential, step = NEP(write_nep(tmp_path / "nep.txt")), 1e-5
    elif kind == "eam":
        from mdapy.calculators import EAM

        potential, step = EAM.zhou(["Cu", "Ni"]), 1e-6
    elif kind == "lj":
        potential, step = LennardJones({"Cu": (0.4, 2.3), "Ni": (0.5, 2.2)}, 5.0), 1e-6
    elif kind == "sw":
        potential, step = StillingerWeber(write_sw(tmp_path / "a.sw")), 1e-6
    else:
        path = write_tersoff(tmp_path / "a.tersoff", ("Si", "C"))
        potential, step = Tersoff(path), 1e-6
    if kind in ("sw", "tersoff"):
        elements = ["Si", "Ge"] if kind == "sw" else ["Si", "C"]
        crystal = shaken("diamond", 5.3, elements, 2, seed=1)
    else:
        crystal = shaken("fcc", 3.6, ["Cu", "Ni"], 3, seed=1)
    cluster = crystal.with_box(mdapy.Box(np.eye(3) * 60.0, pbc=False)).with_columns(
        positions=crystal["positions"] + 20.0
    )
    m = _masses(cluster)
    mvv2e = units.AMU_A2_PER_PS2_TO_EV
    x, v = cluster["positions"], cluster["velocities"]
    forces = potential.compute(cluster, ("forces",)).forces
    a = forces / m[:, None] / mvv2e
    moments = []
    for sign in (1.0, -1.0):
        moved = cluster.with_columns(
            positions=x + sign * step * v + 0.5 * step**2 * a,
            velocities=v + sign * step * a,
        )
        energies = potential.compute(
            moved, ("atomic_energies", HEAT_FLUX_TERMS)
        ).atomic_energies
        vs = moved["velocities"]
        energies = energies + 0.5 * m * (vs * vs).sum(axis=1) * mvv2e
        moments.append(energies @ moved["positions"])
    expected = (moments[0] - moments[1]) / (2 * step)
    found = _flux(cluster, potential)
    np.testing.assert_allclose(
        found, expected, rtol=0, atol=1e-6 * np.abs(expected).max()
    )


def test_solid_argon_in_mdapy():
    """The whole run in MDAPY's own dynamics with its Lennard-Jones: a
    conductivity of the order of that of argon."""
    from mdapy.calculators import LennardJones

    potential = LennardJones({"Ar": (0.0104, 3.405)}, 8.5)
    system = build_crystal("fcc", 5.3, elements="Ar", repeat=4)
    result = green_kubo(
        system,
        potential,
        40.0,
        timestep=0.002,
        equilibration=1000,
        production=16000,
        every=4,
        correlation=4.0,
        blocks=4,
    )
    mean = np.trace(result.kappa) / 3
    assert 0.2 < mean < 3.0
    assert result.temperature == pytest.approx(40.0, rel=0.1)
    assert result.volume == pytest.approx(system.box.volume)
    assert len(result.flux) == 4000


def test_errors_in_mdapy(tmp_path):
    from mdapy.calculators import ADP

    from calculators.potential_files import write_adp

    # ADP: the virials of its many-body terms split among their atoms
    crystal = build_crystal("fcc", 3.6, elements="Cu", repeat=2)
    with pytest.raises(InvalidParameter, match="centroid stress"):
        green_kubo(crystal, ADP(write_adp(tmp_path / "a.adp")), 300.0)
    with pytest.raises(InvalidParameter, match="calculator"):
        green_kubo(crystal, "Si.sw", 300.0)
