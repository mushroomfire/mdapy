# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Homogeneous nonequilibrium molecular dynamics: the work of the driving
field is the change of the energy, the momentum is kept, a turned box drives
the turned flux, and solid argon conducts heat along the field."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import ADP, LAMMPS, LennardJones, StillingerWeber, _gpu
from mdapy.core.threads import CancelToken
from mdapy.errors import Cancelled, InvalidParameter
from mdapy.lattice_dynamics import hnemd
from mdapy.simulation import Deform, MolecularDynamics
from mdapy.simulation._controls import _HeatField

from calculators.gpu_stand_in import stand_in
from calculators.potential_files import write_adp, write_sw

ARGON = LennardJones({"Ar": (0.0104, 3.405)}, 8.5, shift=True)


def warm(seed=1, repeat=3):
    """Solid argon with the velocities of about 40 K."""
    crystal = build_crystal("fcc", 5.28, elements="Ar", repeat=repeat)
    with MolecularDynamics(crystal, ARGON, timestep=0.004) as md:
        md.velocities(80.0, seed=seed)
        md.run(500, ensemble="nvt", temperature=40.0)
        return md.system


def driven(system, field, steps=400, timestep=0.001):
    with MolecularDynamics(system, ARGON, timestep=timestep) as md:
        md.run(steps, thermo=1, controls=_HeatField(field))
        return md.thermo, md.system


def test_the_work_of_the_field_is_the_change_of_the_energy():
    """At constant energy the field does the work F . sum_i W_i v_i on the
    atoms, the pushes less their mean doing none more as the momentum is
    zero: the total energy changes by its integral over time."""
    field = np.array([0.02, -0.01, 0.005])
    thermo, end = driven(warm(), field)
    energy = thermo["total_energy"].to_numpy()
    power = np.stack(thermo["heat_flux"].to_numpy()) @ field
    work = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * 0.001)])
    assert abs(energy[-1] - energy[0]) > 20 * abs(
        (energy - energy[0] - work)[-1]
    )  # the field changed the energy, and by its work
    np.testing.assert_allclose(energy - energy[0], work, rtol=0, atol=2e-4)
    masses = mdapy.elements.atomic_masses(["Ar"])[0]
    assert np.abs(masses * end["velocities"].sum(axis=0)).max() < 1e-9


def test_a_turned_box_drives_the_turned_flux():
    system = warm()
    angle = 0.6
    turn = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    turned = system.with_box(
        mdapy.Box(system.box.matrix @ turn.T, origin=system.box.origin @ turn.T)
    ).with_columns(
        positions=system["positions"] @ turn.T,
        velocities=system["velocities"] @ turn.T,
    )
    field = np.array([0.01, 0.0, 0.0])
    a, _ = driven(system, field, steps=100)
    b, _ = driven(turned, field @ turn.T, steps=100)
    np.testing.assert_allclose(
        np.stack(b["heat_flux"]), np.stack(a["heat_flux"]) @ turn.T, atol=1e-8
    )


def test_solid_argon_conducts_along_the_field():
    result = hnemd(
        build_crystal("fcc", 5.28, elements="Ar", repeat=4),
        ARGON,
        40.0,
        field=(0.0, 0.003, 0.0),
        timestep=0.004,
        equilibration=2000,
        production=20000,
        every=10,
    )
    assert 0.2 < result.along < 1.5
    assert result.kappa[1] == result.along
    assert abs(result.kappa[0]) < 0.5 and abs(result.kappa[2]) < 0.5
    assert result.temperature == pytest.approx(40.0, rel=0.05)
    assert result.running.shape == (2000, 3) and len(result.times) == 2000
    assert result.times[-1] == pytest.approx(80.0)
    assert (result.errors > 0).all()
    assert repr(result).startswith("HNEMD(")
    assert not result.running.flags.writeable


def test_progress_and_cancel():
    seen = []
    system = build_crystal("fcc", 5.28, elements="Ar", repeat=3)
    options = dict(equilibration=20, production=100, every=10, blocks=2)
    hnemd(
        system,
        ARGON,
        40.0,
        field=(3e-3, 0, 0),
        progress=lambda d, t: seen.append((d, t)),
        **options,
    )
    assert seen[-1] == (120, 120) and [d for d, _ in seen] == sorted(d for d, _ in seen)
    token = CancelToken()
    with pytest.raises(Cancelled):
        hnemd(
            system,
            ARGON,
            40.0,
            field=(3e-3, 0, 0),
            cancel=token,
            progress=lambda d, t: token.cancel(),
            **options,
        )


def test_errors(tmp_path, monkeypatch):
    system = build_crystal("fcc", 5.28, elements="Ar", repeat=3)
    for wrong, match in (
        (dict(field=(0.0, 0.0, 0.0)), "not be zero"),
        (dict(field=(1.0, 0.0)), "three finite"),
        (dict(production=0), "positive"),
        (dict(blocks=1), "blocks"),
        (dict(production=20, every=10, blocks=5), "a sample each"),
    ):
        with pytest.raises(InvalidParameter, match=match):
            hnemd(system, ARGON, 40.0, **({"field": (3e-3, 0, 0)} | wrong))
    with pytest.raises(InvalidParameter, match="temperature"):
        hnemd(system, ARGON, 0.0, field=(3e-3, 0, 0))
    open_box = system.with_box(mdapy.Box(system.box.matrix, pbc=(True, True, False)))
    with pytest.raises(InvalidParameter, match="periodic"):
        hnemd(open_box, ARGON, 40.0, field=(3e-3, 0, 0))
    with pytest.raises(InvalidParameter, match="other than LAMMPS"):
        hnemd(system, "Ar.lj", 40.0, field=(3e-3, 0, 0))
    # ADP: the virials of its many-body terms split among their atoms
    copper = build_crystal("fcc", 3.6, elements="Cu", repeat=3)
    with pytest.raises(InvalidParameter, match="centroid stress"):
        hnemd(
            copper,
            ADP(write_adp(tmp_path / "a.adp")),
            300.0,
            field=(3e-3, 0, 0),
            equilibration=10,
            production=100,
            every=10,
        )
    with MolecularDynamics(system, ARGON) as md:
        with pytest.raises(InvalidParameter, match="one heat field"):
            md.run(1, controls=[_HeatField((1e-3, 0, 0)), _HeatField((0, 1e-3, 0))])
        with pytest.raises(InvalidParameter, match="not supported"):
            md.run(1, controls=[_HeatField((1e-3, 0, 0)), Deform(xx=0.1)])
    try:
        import lammps  # noqa: F401
    except ImportError:
        pass
    else:
        lmp = LAMMPS(
            "pair_style lj/cut 8.5\npair_coeff * * 0.0104 3.405\npair_modify shift yes",
            ["Ar"],
        )
        with pytest.raises(InvalidParameter, match="other than LAMMPS"):
            hnemd(system, lmp, 40.0, field=(3e-3, 0, 0))
        with MolecularDynamics(system, lmp) as md:
            with pytest.raises(InvalidParameter, match="MDAPY's own dynamics"):
                md.run(1, controls=_HeatField((1e-3, 0, 0)))


def test_on_the_gpu(monkeypatch):
    """The field on the GPU, its module stood in for on the processor: the
    driven run and the heat flux of the processor."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    system = warm(repeat=4)
    field = np.array([0.01, -0.005, 0.002])
    out = []
    for device in ("cuda", "cpu"):
        potential = LennardJones(
            {"Ar": (0.0104, 3.405)}, 8.5, shift=True, device=device
        )
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.run(100, thermo=10, controls=_HeatField(field))
            out.append((md.system, md.thermo))
    (a, ta), (b, tb) = out
    np.testing.assert_allclose(a["velocities"], b["velocities"], atol=1e-9)
    np.testing.assert_allclose(
        np.stack(ta["heat_flux"]), np.stack(tb["heat_flux"]), rtol=1e-9, atol=1e-12
    )


def test_stillinger_weber_on_the_processor_and_the_gpu(tmp_path, monkeypatch):
    """Silicon of Stillinger-Weber driven by the field, the energies and
    virials of the heat flux computed at each step: on the processor and on
    the GPU, its module stood in for, alike, and at constant energy the field
    does the work of its flux."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    path = write_sw(tmp_path / "Si.sw", ("Si",))
    crystal = build_crystal("diamond", 5.43, elements="Si", repeat=3)
    field = np.array([0.05, -0.025, 0.0125])
    out = []
    for device in ("cuda", "cpu"):
        potential = StillingerWeber(path, device=device)
        with MolecularDynamics(crystal, potential, timestep=0.0005) as md:
            md.velocities(600.0, seed=1)
            md.run(600, thermo=1, controls=_HeatField(field))
            out.append(md.thermo)
    on_gpu, here = out
    np.testing.assert_allclose(
        np.stack(on_gpu["heat_flux"]), np.stack(here["heat_flux"]), rtol=1e-9, atol=1e-9
    )
    energy = here["total_energy"].to_numpy()
    power = np.stack(here["heat_flux"].to_numpy()) @ field
    work = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * 0.0005)])
    residual = np.abs(energy - energy[0] - work).max()
    assert abs(energy[-1] - energy[0]) > 20 * residual


def hot_argon(seed=1):
    """Argon as a dense fluid at 150 K, its crystal expanded."""
    crystal = build_crystal("fcc", 5.6, elements="Ar", repeat=4)
    with MolecularDynamics(crystal, ARGON, timestep=0.002) as md:
        md.velocities(300.0, seed=seed)
        md.run(1000, ensemble="nvt", temperature=150.0)
        return md.system


def test_the_convective_field_does_the_work_of_the_whole_flux(monkeypatch):
    """With the energies each atom carries, each atom also pushed by the field
    times its energy less their mean: at constant energy the field does the
    work F . (sum_i E_i v_i + sum_i W_i v_i), the momentum stays zero, and on
    the GPU, its module stood in for, the run is that of the processor."""
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    system = hot_argon()
    field = np.array([0.02, -0.01, 0.005])
    out = []
    for device in ("cuda", "cpu"):
        potential = LennardJones(
            {"Ar": (0.0104, 3.405)}, 8.5, shift=True, device=device
        )
        with MolecularDynamics(system, potential, timestep=0.002) as md:
            md.run(1000, thermo=1, controls=_HeatField(field, convective=True))
            out.append((md.thermo, md.system))
    (on_gpu, _), (here, end) = out
    np.testing.assert_allclose(
        np.stack(on_gpu["heat_flux"]), np.stack(here["heat_flux"]), rtol=1e-9, atol=1e-9
    )
    energy = here["total_energy"].to_numpy()
    power = np.stack(here["heat_flux"].to_numpy()) @ field
    work = np.concatenate([[0.0], np.cumsum(0.5 * (power[1:] + power[:-1]) * 0.002)])
    residual = np.abs(energy - energy[0] - work).max()
    assert abs(energy[-1] - energy[0]) > 1000 * residual
    masses = mdapy.elements.atomic_masses(["Ar"])[0]
    assert np.abs(masses * end["velocities"].sum(axis=0)).max() < 1e-9
    # the flux with the energies carried differs from its potential part
    with MolecularDynamics(system, ARGON, timestep=0.002) as md:
        md.run(1000, thermo=1, controls=_HeatField(field))
        potential_part = np.stack(md.thermo["heat_flux"].to_numpy())
    assert np.abs(potential_part - np.stack(here["heat_flux"])).max() > 1.0


def test_the_conductivity_with_the_energies_carried(tmp_path):
    result = hnemd(
        hot_argon(),
        ARGON,
        150.0,
        field=(5e-3, 0, 0),
        equilibration=100,
        production=500,
        every=5,
        convective=True,
    )
    assert np.isfinite(result.kappa).all() and result.flux.shape == (100, 3)
    two = hot_argon().with_columns(element=["Ar", "Kr"] * 128)
    with pytest.raises(InvalidParameter, match="one element and one mass"):
        hnemd(two, ARGON, 150.0, field=(5e-3, 0, 0), convective=True)
    heavy = hot_argon().with_columns(mass=np.linspace(39.0, 41.0, 256))
    with pytest.raises(InvalidParameter, match="one element and one mass"):
        hnemd(heavy, ARGON, 150.0, field=(5e-3, 0, 0), convective=True)
