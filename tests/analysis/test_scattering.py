# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The total scattering of X-rays and neutrons: the sum over the wave vectors
of the box in NumPy, the number functions when every element weighs the
same, and the transform of an analytic structure factor."""

import numpy as np
import pytest

from mdapy.analysis import (
    PowderDiffraction,
    RadialDistributionFunction,
    StructureFactor,
    TotalScattering,
    TotalScatteringResult,
)
from mdapy.build import build_crystal, random_alloy
from mdapy.core.box import Box
from mdapy.core.system import System
from mdapy.elements import neutron_scattering_lengths, xray_form_factors
from mdapy.errors import InvalidParameter, MissingProperty


def alloy(repeat=3, seed=1):
    crystal = build_crystal("fcc", 3.6, elements="Cu", repeat=(repeat,) * 3)
    system = random_alloy(crystal, {"Cu": 0.6, "Ni": 0.4}, seed=seed)
    shake = np.random.default_rng(seed).normal(0, 0.15, (len(system), 3))
    return system.with_columns(positions=system["positions"] + shake)


def brute_force(system, q_max, bins, weight):
    """S(q) of the Faber-Ziman definition from the sum over the wave vectors
    of the box, each atom weighted by weight(element, q at the bin center)."""
    x = system["positions"]
    reciprocal = 2 * np.pi * np.linalg.inv(system.box.matrix).T
    n_max = (q_max / np.linalg.norm(reciprocal, axis=1)).astype(int) + 1
    hkl = np.stack(
        np.meshgrid(*(np.arange(-n, n + 1) for n in n_max), indexing="ij"), -1
    ).reshape(-1, 3)
    vectors = hkl @ reciprocal
    size = np.linalg.norm(vectors, axis=1)
    edges = np.linspace(0, q_max, bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    keep = (size > 0) & (size <= q_max)
    vectors, size = vectors[keep], size[keep]
    which = np.clip(np.searchsorted(edges, size, side="left") - 1, 0, bins - 1)
    symbols = system["element"]
    elements = sorted(set(symbols.tolist()))
    fractions = {e: np.mean(symbols == e) for e in elements}
    s = np.full(bins, np.nan)
    for k in range(bins):
        here = which == k
        if not here.any():
            continue
        w = {e: weight(e, centers[k]) for e in elements}
        amplitude = sum(
            w[e] * np.exp(1j * vectors[here] @ x[symbols == e].T).sum(axis=1)
            for e in elements
        )
        mean = sum(fractions[e] * w[e] for e in elements)
        square = sum(fractions[e] * w[e] ** 2 for e in elements)
        intensity = np.mean(np.abs(amplitude) ** 2) / len(system)
        s[k] = 1 + (intensity - square) / mean**2
    return s


@pytest.mark.parametrize("radiation", ["neutron", "xray"])
def test_the_sum_over_the_wave_vectors(radiation):
    system = alloy()

    def weight(element, q):
        if radiation == "neutron":
            return neutron_scattering_lengths([element])[0]
        return xray_form_factors([element], q)[0]

    result = TotalScattering(radiation, 5.0, 25, r_max=5.0).compute(system)
    reference = brute_force(system, 5.0, 25, weight)
    np.testing.assert_array_equal(np.isnan(result.s), np.isnan(reference))
    finite = np.isfinite(reference)
    np.testing.assert_allclose(result.s[finite], reference[finite], rtol=0, atol=1e-10)
    assert result.radiation == radiation


def test_equal_weights_give_the_number_functions():
    system = alloy(repeat=4)
    same = TotalScattering(
        "neutron", 8.0, 80, r_max=6.0, r_bins=120, weights={"Ni": 7.718}
    ).compute(system)
    assert same.weights == {"Cu": 7.718, "Ni": 7.718}
    number = StructureFactor(8.0, 80).compute(system)
    np.testing.assert_allclose(same.s, number.s, rtol=0, atol=1e-12, equal_nan=True)
    rdf = RadialDistributionFunction(6.0, 120).compute(system)
    np.testing.assert_allclose(same.g, rdf.g, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        same.pdf, 4 * np.pi * same.density * rdf.r * (rdf.g - 1), rtol=0, atol=1e-12
    )
    assert same.density == pytest.approx(len(system) / system.box.volume)


def test_the_parts_add_up():
    system = alloy(repeat=4)
    for radiation in ("xray", "neutron"):
        result = TotalScattering(radiation, 8.0, 80, method="debye").compute(system)
        assert set(result.partial) == {("Ni", "Ni"), ("Ni", "Cu"), ("Cu", "Cu")}
        np.testing.assert_allclose(
            sum(result.partial.values()), result.s - 1, atol=1e-12
        )
        x = {e: np.mean(system["element"] == e) for e in ("Cu", "Ni")}
        mean = sum(x[e] * np.asarray(result.weights[e]) for e in x)
        np.testing.assert_allclose(result.f, (result.s - 1) * mean**2, atol=1e-9)
    # X-rays weigh by the form factors at each q, and at q = 0 in real space
    xray = TotalScattering("xray", 8.0, 80).compute(system)
    np.testing.assert_allclose(xray.weights["Cu"], xray_form_factors(["Cu"], xray.q)[0])


def test_heavy_water_weights():
    """Deuterium for hydrogen changes the sign of its weight."""
    ice = build_crystal("fcc", 4.0, elements="O", repeat=(3, 3, 3))
    x = ice["positions"]
    hydrogens = np.concatenate([x + [0.96, 0, 0], x + [-0.24, 0.93, 0]])
    from mdapy.transform import combine

    first = ice.with_columns(positions=hydrogens[: len(x)], element=["H"] * len(x))
    second = ice.with_columns(positions=hydrogens[len(x) :], element=["H"] * len(x))
    water = combine(combine(ice, first), second)
    light = TotalScattering("neutron", 6.0, 30, r_max=4.0).compute(water)
    heavy = TotalScattering(
        "neutron", 6.0, 30, r_max=4.0, weights={"H": 6.6681}
    ).compute(water)
    assert light.weights["H"] < 0 < heavy.weights["H"]
    assert not np.allclose(light.g, heavy.g)


def test_the_transform_of_an_analytic_structure_factor():
    """S(q) - 1 = A sin(q r0)/(q r0): one shell at r0, whose transform peaks
    there, the ripples of the cut damped by the function of Lorch."""
    q = np.linspace(0.05, 30.0, 600)
    r0, a = 2.5, 0.8
    shell = TotalScatteringResult(
        q=q, s=1 + a * np.sin(q * r0) / (q * r0), f=q * 0, partial={},
        r=np.linspace(0.5, 5.0, 901), g=q * 0, pdf=q * 0, weights={},
        radiation="neutron", density=0.1,
    )  # fmt: skip
    plain = shell.transform()
    damped = shell.transform(lorch=True)
    r = shell.r
    assert r[np.argmax(plain)] == pytest.approx(r0, abs=0.01)
    assert r[np.argmax(damped)] == pytest.approx(r0, abs=0.01)
    away = np.abs(r - r0) > 0.6
    assert np.abs(damped[away]).max() < 0.3 * np.abs(plain[away]).max()
    # the area of the peak: (2/π) ∫ (a/r0) sin(q r0) sin(q r) dq up to Q is
    # (a/r0) sin(Q(r - r0)) / (π (r - r0)) near r0, whose integral over
    # |r - r0| < h is (a/r0)(2/π) Si(Q h)
    dr = r[1] - r[0]
    near = np.abs(r - r0) < 0.5
    t = np.linspace(1e-9, q[-1] * 0.5, 200001)
    si = np.trapezoid(np.sin(t) / t, t)
    assert plain[near].sum() * dr == pytest.approx(a / r0 * 2 / np.pi * si, rel=0.01)
    part = shell.transform(np.array([r0]), q_range=(1.0, 10.0))
    assert part.shape == (1,) and part[0] < plain[np.argmax(plain)]
    with pytest.raises(InvalidParameter, match="q_range"):
        shell.transform(q_range=(5.0, 2.0))
    with pytest.raises(InvalidParameter, match="q_range"):
        shell.transform(q_range=(0.0, 40.0))


def test_errors():
    system = alloy()
    with pytest.raises(InvalidParameter, match="radiation"):
        TotalScattering("electron", 5.0)
    with pytest.raises(InvalidParameter, match="75.40"):
        TotalScattering("xray", 80.0)
    with pytest.raises(InvalidParameter, match="method"):
        TotalScattering("xray", 5.0, method="fast")
    with pytest.raises(MissingProperty, match="element"):
        bare = System(table=system.table.drop("element"), box=system.box)
        TotalScattering("xray", 5.0).compute(bare)
    with pytest.raises(InvalidParameter, match="mean weight"):
        TotalScattering("neutron", 5.0, weights={"Cu": 0.0, "Ni": 0.0}).compute(system)
    technetium = system.with_columns(element=["Tc"] * len(system))
    with pytest.raises(InvalidParameter, match="Tc"):
        TotalScattering("neutron", 5.0).compute(technetium)
    flat = system.with_box(Box(system.box.matrix, pbc=(True, True, False)))
    with pytest.raises(InvalidParameter):
        TotalScattering("xray", 5.0).compute(flat)


def test_the_reflections_of_copper():
    """In fcc copper every allowed reflection has all atoms in phase: its
    intensity per atom is m N f(q)² Lp(θ), m the multiplicity."""
    a, n, wavelength = 3.615, 4, 1.5406
    copper = build_crystal("fcc", a, elements="Cu", repeat=(n, n, n))
    result = PowderDiffraction(wavelength, (30, 100), 1400).compute(copper)
    for hkl, multiplicity in (((1, 1, 1), 8), ((2, 0, 0), 6), ((2, 2, 0), 12),
                              ((3, 1, 1), 24), ((2, 2, 2), 8)):  # fmt: skip
        q = 2 * np.pi * np.sqrt(sum(h * h for h in hkl)) / a
        theta = np.arcsin(q * wavelength / (4 * np.pi))
        lp = (1 + np.cos(2 * theta) ** 2) / (np.sin(theta) ** 2 * np.cos(theta))
        expected = (
            multiplicity * len(copper) * xray_form_factors(["Cu"], q)[0] ** 2 * lp
        )
        k = np.searchsorted(np.linspace(30, 100, 1401), np.degrees(2 * theta)) - 1
        assert result.intensity[k] == pytest.approx(expected, rel=1e-10)
    # nothing between the reflections of a perfect crystal
    assert np.sort(result.intensity)[-6] < 1e-12 * result.intensity.max()


@pytest.mark.parametrize("radiation", ["xray", "neutron"])
def test_every_wave_vector_at_its_angle(radiation):
    system = alloy()
    wavelength, low, high, bins = 1.2, 20.0, 110.0, 300
    result = PowderDiffraction(wavelength, (low, high), bins, radiation=radiation,
                               debye_waller=0.4).compute(system)  # fmt: skip
    # every vector of the box in NumPy, each sign
    reciprocal = 2 * np.pi * np.linalg.inv(system.box.matrix).T
    q_max = 4 * np.pi * np.sin(np.radians(high) / 2) / wavelength
    n_max = (q_max / np.linalg.norm(reciprocal, axis=1)).astype(int) + 1
    hkl = np.stack(
        np.meshgrid(*(np.arange(-m, m + 1) for m in n_max), indexing="ij"), -1
    ).reshape(-1, 3)
    vectors = hkl @ reciprocal
    size = np.linalg.norm(vectors, axis=1)
    keep = (size > 0) & (size <= q_max)
    vectors, size = vectors[keep], size[keep]
    symbols = system["element"]
    amplitude = 0
    for e in ("Cu", "Ni"):
        if radiation == "xray":
            w = xray_form_factors([e], size)[0]
        else:
            w = neutron_scattering_lengths([e])[0]
        amplitude = amplitude + w * np.exp(
            1j * vectors @ system["positions"][symbols == e].T
        ).sum(axis=1)
    theta = np.arcsin(size * wavelength / (4 * np.pi))
    lp = 1 / (np.sin(theta) ** 2 * np.cos(theta))
    if radiation == "xray":
        lp = lp * (1 + np.cos(2 * theta) ** 2)
    each = (
        np.abs(amplitude) ** 2
        / len(system)
        * lp
        * np.exp(-0.8 * (size / (4 * np.pi)) ** 2)
    )
    edges = np.linspace(low, high, bins + 1)
    which = np.searchsorted(edges, np.degrees(2 * theta), side="right") - 1
    inside = (which >= 0) & (which < bins)
    expected = np.bincount(which[inside], each[inside], bins)
    np.testing.assert_allclose(
        result.intensity, expected, rtol=1e-10, atol=1e-9 * expected.max()
    )
    assert result.radiation == radiation and result.wavelength == wavelength
    np.testing.assert_allclose(result.two_theta, 0.5 * (edges[1:] + edges[:-1]))


def test_broadening_keeps_the_intensity():
    copper = build_crystal("fcc", 3.615, elements="Cu", repeat=(3, 3, 3))
    sharp = PowderDiffraction(two_theta=(20, 120), bins=2000).compute(copper)
    wide = PowderDiffraction(two_theta=(20, 120), bins=2000, broadening=0.5).compute(
        copper
    )
    step = sharp.two_theta[1] - sharp.two_theta[0]
    assert wide.intensity.sum() * step == pytest.approx(
        sharp.intensity.sum() * step, rel=1e-6
    )
    # each peak a Gaussian of the width asked for
    k = np.argmax(wide.intensity)
    half = wide.two_theta[wide.intensity > wide.intensity[k] / 2]
    assert half[-1] - half[0] == pytest.approx(0.5, abs=2 * step)
    assert not np.any(
        PowderDiffraction(lorentz_polarization=False).compute(copper).intensity < 0
    )


def test_powder_errors():
    system = alloy()
    for bad in (
        {"wavelength": 0.0},
        {"two_theta": (90, 30)},
        {"two_theta": (10, 190)},
        {"bins": 0},
        {"radiation": "electron"},
        {"debye_waller": -1.0},
        {"broadening": 0.0},
        {"wavelength": 0.1, "two_theta": (10, 170)},  # beyond the form factors
    ):
        with pytest.raises(InvalidParameter):
            PowderDiffraction(**bad)
    with pytest.raises(MissingProperty, match="element"):
        PowderDiffraction().compute(
            System(table=system.table.drop("element"), box=system.box)
        )
    flat = system.with_box(Box(system.box.matrix, pbc=(True, True, False)))
    with pytest.raises(InvalidParameter, match="periodic"):
        PowderDiffraction().compute(flat)
