# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Elastic constants from molecular dynamics, in MDAPY and in LAMMPS where it
is installed; the Born term of MDAPY against that of LAMMPS."""

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import EAM, LAMMPS, LennardJones
from mdapy.errors import InvalidParameter
from mdapy.properties._md_elastic import _born
from mdapy.properties.elastic import elastic_constants, md_elastic_constants
from mdapy.simulation import MolecularDynamics, minimize

from calculators.eam_files import write_setfl


@pytest.fixture(scope="module")
def setfl(tmp_path_factory):
    return write_setfl(tmp_path_factory.mktemp("eam") / "CuNi.eam.alloy")


def lammps_eam(setfl):
    pytest.importorskip("lammps")
    return LAMMPS(f"pair_style eam/alloy\npair_coeff * * {setfl} Cu Ni", ["Cu", "Ni"])


@pytest.fixture(scope="module", params=["mdapy", "lammps"])
def copper(request, setfl):
    """MDAPY's EAM, the calculator of the dynamics, MDAPY's own or LAMMPS,
    and the relaxed crystal."""
    eam = EAM(setfl)
    relaxed = minimize(
        build_crystal("fcc", 3.29, elements="Cu", repeat=(3, 3, 3)),
        eam,
        cell="full",
        max_force=1e-10,
        max_stress=1e-8,
        max_steps=10000,
    ).system
    potential = eam if request.param == "mdapy" else lammps_eam(setfl)
    return eam, potential, relaxed


def test_the_born_term_of_lammps(setfl):
    """MDAPY's Born term of a shaken crystal in a tilted box, the 21 values of
    LAMMPS's compute born/matrix numdiff for the same atoms."""
    potential = lammps_eam(setfl)
    crystal = build_crystal("fcc", 3.6, elements="Cu", repeat=(4, 4, 4))
    matrix = crystal.box.matrix.copy()
    matrix[1, 0], matrix[2, 0], matrix[2, 1] = 0.8, -0.5, 0.3  # the form of LAMMPS
    crystal = crystal.with_box(mdapy.Box(matrix), scale_positions=True)
    rng = np.random.default_rng(0)
    kinds = rng.choice(["Cu", "Ni"], len(crystal))
    shaken = crystal.with_columns(
        positions=crystal["positions"] + rng.normal(0, 0.1, (len(crystal), 3)),
        element=list(kinds),
    )
    ours = _born(shaken, EAM(setfl))
    with MolecularDynamics(shaken, potential) as md:
        md.command("compute mdapy_v all pressure NULL virial")
        md.command("compute mdapy_b all born/matrix numdiff 1e-5 mdapy_v")
        md.command("run 0")
        theirs = np.array(md.lammps.numpy.extract_compute("mdapy_b", 0, 1)[:21])
    np.testing.assert_allclose(
        ours, theirs, rtol=1e-7, atol=1e-7 * np.abs(theirs).max()
    )


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


def test_symmetric_and_reproducible(copper):
    """Each entry off the diagonal the mean of its two estimates; the same
    seed the same constants, another seed others."""
    _, potential, relaxed = copper
    options = {"equilibration": 200, "production": 400, "every": 10}
    first = md_elastic_constants(relaxed, potential, 300.0, **options)
    again = md_elastic_constants(relaxed, potential, 300.0, **options)
    other = md_elastic_constants(relaxed, potential, 300.0, seed=2, **options)
    np.testing.assert_array_equal(first.constants, first.constants.T)
    np.testing.assert_array_equal(first.errors, first.errors.T)
    np.testing.assert_array_equal(first.constants, again.constants)
    assert not np.array_equal(first.constants, other.constants)


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


FLUCTUATION = {"method": "fluctuation", "equilibration": 500, "every": 10}


def test_fluctuations_of_a_cold_crystal_give_the_static_constants(copper):
    """At 1 K the Born term of the crystal at rest, which in fcc relaxes no
    atoms: the static constants, more closely than the strains of 1 %."""
    eam, potential, relaxed = copper
    static = elastic_constants(relaxed, eam).constants
    result = md_elastic_constants(
        relaxed, potential, 1.0, production=2000, **FLUCTUATION
    )
    np.testing.assert_allclose(result.constants, static, atol=5e-4 * static.max())
    np.testing.assert_array_equal(result.constants, result.constants.T)
    assert result.errors.shape == (6, 6) and result.errors.max() < 0.05


def test_fluctuations_and_strains_agree(copper):
    """The two methods of the same hot crystal within their errors."""
    _, potential, relaxed = copper
    hot = {"timestep": 0.002, "equilibration": 1000}
    strained = md_elastic_constants(relaxed, potential, 300.0, production=4000, **hot)
    fluctuating = md_elastic_constants(
        relaxed, potential, 300.0, production=40000, method="fluctuation", **hot
    )
    for a, b in [(0, 0), (1, 1), (2, 2), (0, 1), (3, 3), (5, 5)]:
        spread = np.hypot(strained.errors[a, b], fluctuating.errors[a, b])
        assert abs(strained.constants[a, b] - fluctuating.constants[a, b]) < 4 * spread


@pytest.mark.parametrize("engine", ["mdapy", "lammps"])
def test_fluctuations_of_an_ideal_gas(engine):
    """Atoms without forces: no Born term and no fluctuations of the virial,
    the kinetic term alone, and under its pressure P the stress-strain
    coefficients B11 = B12 = P and B44 = 0."""
    gas = mdapy.System(
        np.random.default_rng(0).random((64, 3)) * 20.0,
        mdapy.Box(20.0),
        elements=["Ar"] * 64,
    )
    if engine == "mdapy":
        potential = LennardJones({"Ar": (0.0, 3.4)}, 3.0)
    else:
        pytest.importorskip("lammps")
        potential = LAMMPS("pair_style zero 3.0\npair_coeff * *", ["Ar"])
    pressure = 0.05  # GPa
    result = md_elastic_constants(
        gas,
        potential,
        300.0,
        pressure=pressure,
        barostat="iso",
        production=20000,
        **FLUCTUATION,
    )
    c, e = result.constants, result.errors
    for a, b in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
        assert c[a, b] == pytest.approx(pressure, abs=4 * e[a, b])
        assert e[a, b] < 0.02 * pressure
    np.testing.assert_allclose(np.diag(c)[3:], 0.0, atol=5 * np.diag(e)[3:].max())


def test_the_method_must_be_known(copper):
    _, potential, relaxed = copper
    with pytest.raises(InvalidParameter, match="method must be"):
        md_elastic_constants(relaxed, potential, 300.0, method="green-kubo")
    with pytest.raises(InvalidParameter, match="calculator"):
        md_elastic_constants(relaxed, "Cu.eam.alloy", 300.0)
