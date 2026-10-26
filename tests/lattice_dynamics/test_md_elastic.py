# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Elastic constants from molecular dynamics in LAMMPS. Skipped where LAMMPS
is not installed."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM, LAMMPS
from mdapy.errors import InvalidParameter
from mdapy.lattice_dynamics import elastic_constants, md_elastic_constants
from mdapy.simulation import minimize

from calculators.eam_files import write_setfl

lammps = pytest.importorskip("lammps")


@pytest.fixture(scope="module")
def copper(tmp_path_factory):
    setfl = write_setfl(tmp_path_factory.mktemp("eam") / "CuNi.eam.alloy")
    relaxed = minimize(
        build_crystal("fcc", 3.29, elements="Cu", repeat=(3, 3, 3)),
        EAM(setfl),
        cell="full",
        max_force=1e-10,
        max_stress=1e-8,
        max_steps=10000,
    ).system
    potential = LAMMPS(
        f"pair_style eam/alloy\npair_coeff * * {setfl} Cu Ni", ["Cu", "Ni"]
    )
    return EAM(setfl), potential, relaxed


def test_cold_crystal_has_the_static_constants(copper):
    # at 1 K the crystal hardly moves: the constants of the energy at rest,
    # but for the strain of 1 %
    eam, potential, relaxed = copper
    static = elastic_constants(relaxed, eam).constants
    result = md_elastic_constants(
        relaxed, potential, 1.0, equilibration=500, production=1000, every=10
    )
    np.testing.assert_allclose(result.constants, static, atol=3e-3 * static.max())
    assert result.errors.shape == (6, 6) and result.errors.max() < 0.5
    assert not result.errors.flags.writeable
    assert result.bulk_modulus() == pytest.approx(
        (static[0, 0] + 2 * static[0, 1]) / 3, rel=3e-3
    )


def test_hot_crystal_is_softer(copper):
    eam, potential, relaxed = copper
    cold = md_elastic_constants(
        relaxed, potential, 1.0, equilibration=300, production=600
    )
    hot = md_elastic_constants(
        relaxed, potential, 600.0, equilibration=1000, production=2000, barostat="aniso"
    )
    assert hot.constants[3, 3] < cold.constants[3, 3]
    assert (hot.errors > cold.errors).all()


@pytest.mark.parametrize(
    "options, message",
    [
        ({"temperature": 0.0}, "temperature must be positive"),
        ({"strain": 0.0}, "strain must be positive"),
        ({"production": 0}, "must be positive"),
        ({"blocks": 1}, "blocks must be at least 2"),
        ({"production": 20, "every": 10}, "a sample each"),
    ],
)
def test_invalid_arguments(copper, options, message):
    _, potential, relaxed = copper
    arguments = {"temperature": 300.0} | options
    temperature = arguments.pop("temperature")
    with pytest.raises(InvalidParameter, match=message):
        md_elastic_constants(relaxed, potential, temperature, **arguments)
    cluster = relaxed.with_box(mdapy.Box(30.0, pbc=False))
    with pytest.raises(InvalidParameter, match="periodic"):
        md_elastic_constants(cluster, potential, 300.0)
