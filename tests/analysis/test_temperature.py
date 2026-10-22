# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest
from lattices import crystal

import mdapy
from mdapy import units
from mdapy.analysis import AtomicTemperature
from mdapy.errors import InvalidParameter, MissingProperty


def thermal(temperature=300.0, repeat=(10, 10, 10), seed=0):
    system = crystal("fcc", 3.6, repeat, element="Cu")
    mass = mdapy.elements.atomic_masses(["Cu"])[0]
    sigma = np.sqrt(units.BOLTZMANN * temperature / (mass * units.AMU_A2_PER_PS2_TO_EV))
    rng = np.random.default_rng(seed)
    return system.with_columns(
        velocities=rng.normal(scale=sigma, size=(len(system), 3))
    )


def reference(system, cutoff):
    nl = mdapy.NeighborList.build(system, cutoff)
    mass = (
        system.table["mass"].to_numpy()
        if "mass" in system.table.columns
        else (mdapy.elements.atomic_masses(system.table["element"].to_list()))
    )
    v = system.table["velocities"].to_numpy()
    result = []
    for i in range(len(system)):
        group = [i] + [j for j in nl.j[nl.offsets[i] : nl.offsets[i + 1]] if j != i]
        m, u = mass[group], v[group]
        center = (m[:, None] * u).sum(0) / m.sum()
        kinetic = (
            0.5 * (m * ((u - center) ** 2).sum(1)).sum() * units.AMU_A2_PER_PS2_TO_EV
        )
        result.append(2 * kinetic / (3 * (len(group) - 1) * units.BOLTZMANN))
    return np.array(result)


def test_maxwell_boltzmann_velocities():
    temperature = AtomicTemperature(3.0).compute(thermal()).temperature
    assert temperature.mean() == pytest.approx(300.0, rel=0.02)


def test_matches_a_reference_and_ignores_drift():
    system = thermal(repeat=(3, 3, 3), seed=1)
    result = AtomicTemperature(3.0).compute(system).temperature
    np.testing.assert_allclose(result, reference(system, 3.0), rtol=1e-12)
    moving = system.with_columns(
        velocities=system.table["velocities"].to_numpy() + [5.0, -3.0, 1.0]
    )
    np.testing.assert_allclose(
        AtomicTemperature(3.0).compute(moving).temperature, result, rtol=1e-9
    )
    rng = np.random.default_rng(2)
    heavy = system.with_columns(mass=rng.uniform(10, 100, len(system)))
    np.testing.assert_allclose(
        AtomicTemperature(4.0).compute(heavy).temperature,
        reference(heavy, 4.0),
        rtol=1e-12,
    )
    # a given neighbor list, rather than the grid, sums in another order
    listed = AtomicTemperature(4.0).compute(
        heavy, neighbors=mdapy.NeighborList.build(heavy, 4.5)
    )
    np.testing.assert_allclose(listed.temperature, reference(heavy, 4.0), rtol=1e-12)


def test_columns_isolated_atoms_and_errors():
    system = thermal(repeat=(3, 3, 3))
    result = AtomicTemperature(3.0).compute(system)
    assert (
        system.with_columns(result).table["temperature"].to_list()
        == result.temperature.tolist()
    )
    lonely = mdapy.System(
        [[0.0, 0, 0], [5.0, 5, 5]], mdapy.Box(20.0, pbc=False), elements="Ar"
    )
    lonely = lonely.with_columns(velocities=np.ones((2, 3)))
    assert np.isnan(AtomicTemperature(2.0).compute(lonely).temperature).all()
    assert repr(AtomicTemperature(2.0)) == "AtomicTemperature(cutoff=2.0)"
    with pytest.raises(MissingProperty, match="velocities"):
        AtomicTemperature(3.0).compute(crystal("fcc", 3.6))
    bare = mdapy.System(system.positions, system.box).with_columns(
        velocities=system.table["velocities"].to_numpy()
    )
    with pytest.raises(MissingProperty, match="mass or an element"):
        AtomicTemperature(3.0).compute(bare)
    with pytest.raises(InvalidParameter, match="cutoff must be positive"):
        AtomicTemperature(0.0)
