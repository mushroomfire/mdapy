# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest

import mdapy
from mdapy import units
from mdapy.calculators import EAM, elastic_constants, minimize
from mdapy.calculators._elastic import _strained
from mdapy.errors import InvalidParameter

from .eam_files import alloy, write_setfl
from .test_testing import Harmonic


@pytest.fixture(scope="module")
def potential(tmp_path_factory):
    return EAM(write_setfl(tmp_path_factory.mktemp("eam") / "CuNi.eam.alloy"))


@pytest.fixture(scope="module")
def copper(potential):
    """fcc Cu of the synthetic potential at zero stress."""
    system = alloy(seed=0, repeat=3, jitter=0.0, a=3.29)
    system = system.with_columns(element=["Cu"] * len(system))
    result = minimize(
        system, potential, cell="full", max_force=1e-8, max_stress=1e-7, max_steps=10000
    )
    assert result.converged
    return result.system


def curvature(potential, system, strain):
    """d2E/dx2 / V in GPa for the strain x * strain, by central differences."""
    h = 1e-4
    energy = [
        potential.compute(_strained(system, s * h * strain)).energy for s in (-1, 0, 1)
    ]
    return (
        (energy[0] - 2 * energy[1] + energy[2])
        / h**2
        / system.box.volume
        * units.EV_PER_A3_TO_GPA
    )


def test_cubic_constants_match_the_energy(potential, copper):
    result = elastic_constants(copper, potential)
    c = result.constants
    np.testing.assert_allclose(c, c.T, atol=1e-6 * c.max())
    np.testing.assert_allclose(np.diag(c)[:3], c[0, 0], rtol=1e-9)
    np.testing.assert_allclose(np.diag(c)[3:], c[3, 3], rtol=1e-9)
    np.testing.assert_allclose([c[0, 1], c[0, 2], c[1, 2]], c[0, 1], rtol=1e-9)
    assert np.abs(c[:3, 3:]).max() < 1e-6 * c.max()
    # the same constants from the curvature of the energy
    uniaxial = np.diag([1.0, 0, 0])
    shear = np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0]])  # engineering shear yz
    assert curvature(potential, copper, uniaxial) == pytest.approx(c[0, 0], rel=1e-4)
    assert curvature(potential, copper, shear) == pytest.approx(c[3, 3], rel=1e-4)
    bulk = curvature(potential, copper, np.eye(3)) / 9
    assert result.bulk_modulus("voigt") == pytest.approx(bulk, rel=1e-4)
    assert result.bulk_modulus("reuss") == pytest.approx(
        result.bulk_modulus("voigt"), rel=1e-9
    )
    voigt, reuss, hill = (result.shear_modulus(m) for m in ("voigt", "reuss", "hill"))
    assert voigt >= hill >= reuss and hill == pytest.approx(0.5 * (voigt + reuss))
    assert not result.constants.flags.writeable


def test_relaxation_softens_an_alloy(potential):
    system = alloy(seed=3, repeat=2, jitter=0.0, a=3.3)
    relaxed = minimize(
        system, potential, cell="full", max_force=1e-7, max_stress=1e-6, max_steps=20000
    )
    assert relaxed.converged
    soft = elastic_constants(relaxed.system, potential).constants
    stiff = elastic_constants(relaxed.system, potential, relax=False).constants
    assert (np.diag(soft) <= np.diag(stiff) + 1e-6).all()
    assert (np.diag(stiff) - np.diag(soft)).max() > 1e-3  # the atoms do move


def test_errors(potential, copper):
    with pytest.raises(InvalidParameter, match="periodic in every direction"):
        elastic_constants(
            copper.with_box(mdapy.Box(copper.box.matrix, pbc=(True, True, False))),
            potential,
        )
    with pytest.raises(InvalidParameter, match="strain must be positive"):
        elastic_constants(copper, potential, strain=0.0)

    class NoVirial(Harmonic):
        implemented = frozenset({"energy", "forces"})

    with pytest.raises(
        InvalidParameter, match=r"does not compute \['energy', 'forces', 'virial'\]"
    ):
        elastic_constants(copper, NoVirial())
    with pytest.raises(InvalidParameter, match=r"does not compute \['virial'\]"):
        elastic_constants(copper, NoVirial(), relax=False)
    rattled = copper.with_positions(
        copper.positions
        + np.random.default_rng(1).normal(scale=0.1, size=copper.positions.shape)
    )
    with pytest.raises(InvalidParameter, match="did not converge in 2 steps"):
        elastic_constants(rattled, potential, max_steps=2)
    with pytest.raises(InvalidParameter, match="average must be"):
        elastic_constants(copper, potential, relax=False).bulk_modulus("mean")
