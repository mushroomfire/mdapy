# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import dataclasses

import numpy as np
import pytest

from mdapy import units
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics import (
    elastic_constants,
    primitive_cell,
    quasi_harmonic,
    thermal_elastic_constants,
)
from mdapy.lattice_dynamics._thermal_elastic import (
    _birch,
    _deformed,
    _hessian,
    _stencil,
)
from mdapy.simulation import minimize

from calculators.eam_files import write_setfl

OPTIONS = {"supercell": 4, "mesh": 8}


@pytest.fixture(scope="module")
def copper(tmp_path_factory):
    potential = EAM(write_setfl(tmp_path_factory.mktemp("eam") / "CuNi.eam.alloy"))
    relaxed = minimize(
        build_crystal("fcc", 3.29, elements="Cu"),
        potential,
        cell="full",
        max_force=1e-10,
        max_stress=1e-8,
        max_steps=10000,
    ).system
    qha = quasi_harmonic(
        primitive_cell(relaxed), potential, [0.0, 100.0, 300.0], **OPTIONS
    )
    return potential, relaxed, qha


@pytest.fixture(scope="module")
def constants(copper):
    potential, _, qha = copper
    return thermal_elastic_constants(qha, potential, **OPTIONS)


def test_cubic_and_consistent_with_the_bulk_modulus(copper, constants):
    _, _, qha = copper
    assert constants.method == "quasi-harmonic"
    assert (
        constants.isothermal.shape == (3, 6, 6)
        and not constants.adiabatic.flags.writeable
    )
    for c in constants.isothermal:
        np.testing.assert_allclose(c, c.T, atol=1e-6 * c.max())
        np.testing.assert_allclose(np.diag(c)[:3], c[0, 0], rtol=1e-6)
        np.testing.assert_allclose(np.diag(c)[3:], c[3, 3], rtol=1e-6)
        assert np.abs(c[3:, :3]).max() < 1e-6 * c.max()
    # (C11 + 2 C12) / 3 is the curvature of the free energy against volume
    bulk = (constants.isothermal[:, 0, 0] + 2 * constants.isothermal[:, 0, 1]) / 3
    np.testing.assert_allclose(bulk, qha.bulk_modulus, rtol=2e-2)
    # the crystal softens as it heats
    assert np.all(np.diff(constants.isothermal[:, 3, 3]) < 0)


def test_adiabatic_constants(copper, constants):
    _, _, qha = copper
    np.testing.assert_array_equal(constants.adiabatic[0], constants.isothermal[0])
    # shears are the same; the bulk modulus grows by C_P / C_V
    np.testing.assert_allclose(
        constants.adiabatic[:, 3:, 3:],
        constants.isothermal[:, 3:, 3:],
        rtol=1e-9,
        atol=1e-12,
    )
    for k in (1, 2):
        t, v = qha.temperatures[k], qha.volume[k]
        bulk = (constants.isothermal[k, 0, 0] + 2 * constants.isothermal[k, 0, 1]) / 3
        alpha = qha.thermal_expansion[k]
        cp = qha.heat_capacity[k]
        cv = cp - t * v * qha.bulk_modulus[k] * units.GPA_TO_EV_PER_A3 * alpha**2
        adiabatic = (
            constants.adiabatic[k, 0, 0] + 2 * constants.adiabatic[k, 0, 1]
        ) / 3
        # with the bulk modulus of the elastic constants in the coupling
        expected = (
            bulk
            + t
            * v
            * (bulk * units.GPA_TO_EV_PER_A3) ** 2
            * alpha**2
            / cv
            * units.EV_PER_A3_TO_GPA
        )
        assert adiabatic == pytest.approx(expected, rel=1e-6)


def test_quasi_static(copper):
    potential, relaxed, qha = copper
    quasi_static = thermal_elastic_constants(qha, potential, method="quasi-static")
    assert quasi_static.method == "quasi-static"
    # softer as the crystal expands, as the static energy alone gives it
    assert np.all(np.diff(quasi_static.isothermal[:, 0, 0]) < 0)
    # a grid that starts just below the volumes of the temperatures: the
    # four cells nearest them
    short = quasi_harmonic(
        primitive_cell(relaxed),
        potential,
        [0.0, 100.0],
        volumes=np.linspace(0.995, 1.06, 6),
        **OPTIONS,
    )
    again = thermal_elastic_constants(short, potential, method="quasi-static")
    np.testing.assert_allclose(
        again.isothermal,
        quasi_static.isothermal[:2],
        atol=2e-3 * quasi_static.isothermal.max(),
    )
    # the second derivatives of the energy at zero stress are the elastic
    # constants of the stress
    h = 5e-3
    values = {
        np.round(v / h).astype(int).tobytes(): np.array(
            [potential.compute(_deformed(relaxed, v)).energy]
        )
        for v in _stencil(h)
    }
    curvature = (
        _hessian(values, h)[..., 0] / relaxed.box.volume * units.EV_PER_A3_TO_GPA
    )
    np.testing.assert_allclose(
        curvature,
        elastic_constants(relaxed, potential).constants,
        atol=2e-3 * curvature.max(),
    )


def test_stress_strain_coefficients_under_pressure():
    c = np.zeros((6, 6))
    c[:3, :3] = 100.0
    c[np.arange(6), np.arange(6)] = [200, 200, 200, 50, 50, 50]
    b = _birch(c, 2.0)
    assert b[0, 0] == 198 and b[0, 1] == 102 and b[3, 3] == 48
    np.testing.assert_array_equal(_birch(c, 0.0), c)


def test_invalid_arguments(copper):
    potential, _, qha = copper
    with pytest.raises(InvalidParameter, match="method must be"):
        thermal_elastic_constants(qha, potential, method="static")
    with pytest.raises(InvalidParameter, match="strain must be positive"):
        thermal_elastic_constants(qha, potential, strain=0)
    with pytest.raises(InvalidParameter, match="no cells"):
        thermal_elastic_constants(dataclasses.replace(qha, cells=()), potential)


def test_strained_cells_that_do_not_relax(copper):
    potential, _, _ = copper
    # hcp: its atoms move inside a strained cell
    hcp = minimize(
        build_crystal("hcp", 2.33, elements="Cu"),
        potential,
        cell="full",
        max_force=1e-10,
        max_stress=1e-8,
        max_steps=20000,
    ).system
    qha = quasi_harmonic(hcp, potential, [100.0], **OPTIONS)
    with pytest.raises(InvalidParameter, match="did not relax"):
        thermal_elastic_constants(qha, potential, method="quasi-static", max_steps=1)
