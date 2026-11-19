"""Thermal velocities of a system: the temperature exactly, no total
momentum, those of MolecularDynamics.velocities in MDAPY, a part of the
atoms, held directions, and the errors."""

import numpy as np
import pytest

from mdapy import units
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.simulation import MolecularDynamics, thermal_velocities


def alloy():
    system = build_crystal("fcc", 3.6, elements="Cu", repeat=(4, 4, 4))
    kinds = np.random.default_rng(0).choice(["Cu", "Ni"], len(system))
    return system.with_columns(element=list(kinds))


def temperature(system, v, dof):
    masses = np.where(
        system.table["element"].cast(str).to_numpy() == "Cu", 63.546, 58.6934
    )
    twice = np.einsum("i,ij,ij->", masses, v, v) * units.AMU_A2_PER_PS2_TO_EV
    return twice / (dof * units.BOLTZMANN), masses


def test_those_of_the_dynamics():
    system = alloy()
    v = thermal_velocities(system, 600.0, seed=7)
    t, masses = temperature(system, v, 3 * len(system) - 3)
    assert t == pytest.approx(600.0, rel=1e-12)
    np.testing.assert_allclose(masses @ v, 0.0, atol=1e-10)
    with MolecularDynamics(system, EAM.zhou(["Cu", "Ni"])) as md:
        md.velocities(600.0, seed=7)
        np.testing.assert_allclose(md.system["velocities"], v, rtol=1e-12, atol=1e-13)
    # a run starts from them
    with MolecularDynamics(
        system.with_columns(velocities=v), EAM.zhou(["Cu", "Ni"])
    ) as md:
        assert md.thermo.is_empty()
        md.run(0)
        assert md.thermo["temperature"][0] == pytest.approx(600.0, rel=1e-12)


def test_a_part_and_held_directions():
    system = alloy()
    flyer = system["positions"][:, 2] > 7.0
    v = thermal_velocities(system, 300.0, seed=2, mask=flyer)
    assert (v[~flyer] == 0).all()
    t, masses = temperature(system, v, 3 * int(flyer.sum()) - 3)
    assert t == pytest.approx(300.0, rel=1e-12)
    np.testing.assert_allclose(masses[flyer] @ v[flyer], 0.0, atol=1e-10)
    movable = np.ones((len(system), 3), dtype=bool)
    movable[:, 2] = False
    held = system.with_columns(movable=movable)
    v = thermal_velocities(held, 300.0, seed=2)
    assert (v[:, 2] == 0).all()
    t, _ = temperature(held, v, int(movable.sum()))
    assert t == pytest.approx(300.0, rel=1e-12)
    with MolecularDynamics(held, EAM.zhou(["Cu", "Ni"])) as md:
        md.velocities(300.0, seed=2)
        np.testing.assert_allclose(md.system["velocities"], v, rtol=1e-12, atol=1e-13)
    # nothing to move, or no temperature
    assert (thermal_velocities(system, 0.0, seed=1) == 0).all()
    one = np.zeros(len(system), dtype=bool)
    one[0] = True
    assert (thermal_velocities(system, 300.0, seed=1, mask=one) == 0).all()
    none = np.zeros(len(system), dtype=bool)
    assert (thermal_velocities(system, 300.0, seed=1, mask=none) == 0).all()
    every = thermal_velocities(
        system.with_columns(movable=np.ones((len(system), 3), dtype=bool)),
        300.0,
        seed=1,
    )
    np.testing.assert_array_equal(every, thermal_velocities(system, 300.0, seed=1))
    masses = system.with_columns(mass=np.full(len(system), 60.0)).drop("element")
    assert thermal_velocities(masses, 300.0, seed=1).shape == (len(system), 3)


def test_errors():
    system = alloy()
    with pytest.raises(InvalidParameter, match="temperature"):
        thermal_velocities(system, -1.0, seed=1)
    with pytest.raises(InvalidParameter, match="seed"):
        thermal_velocities(system, 300.0, seed=0)
    with pytest.raises(InvalidParameter, match="seed"):
        thermal_velocities(system, 300.0, seed=True)
    with pytest.raises(InvalidParameter, match="mask"):
        thermal_velocities(system, 300.0, seed=1, mask=np.ones(3, dtype=bool))
    with pytest.raises(MissingProperty, match="masses"):
        thermal_velocities(system.drop("element"), 300.0, seed=1)
