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
from mdapy.calculators import LAMMPS, LennardJones, StillingerWeber, _gpu
from mdapy.core.threads import CancelToken
from mdapy.errors import Cancelled, InvalidParameter
from mdapy.lattice_dynamics import hnemd
from mdapy.simulation import Deform, MolecularDynamics
from mdapy.simulation._controls import _HeatField

from calculators.gpu_stand_in import stand_in

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
    path = tmp_path / "Si.sw"
    path.write_text(
        "Si Si Si 2.1683 2.0951 1.80 21.0 1.20 -0.333333333333 7.049556277 "
        "0.6022245584 4.0 0.0 0.0\n"
    )
    silicon = build_crystal("diamond", 5.431, elements="Si", repeat=3)
    with pytest.raises(InvalidParameter, match="centroid stress"):
        hnemd(
            silicon,
            StillingerWeber(path),
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
    monkeypatch.setattr(_gpu, "cuda", stand_in)
    gpu = LennardJones({"Ar": (0.0104, 3.405)}, 8.5, device="cuda")
    with pytest.raises(InvalidParameter, match="on the processor"):
        thick = build_crystal("fcc", 5.28, elements="Ar", repeat=4)
        hnemd(
            thick,
            gpu,
            40.0,
            field=(3e-3, 0, 0),
            equilibration=10,
            production=100,
            every=10,
        )
