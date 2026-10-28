# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.analysis import SpatialCorrelation
from mdapy.errors import InvalidParameter

MATRIX = [[40.0, 0, 0], [5.0, 36.0, 0], [-3.0, 2.0, 42.0]]


def scattered(pbc=True, n=6000, seed=0):
    """Atoms at random in a triclinic box, some just below its lower faces,
    with two related properties and velocities; along directions that are
    not periodic, all inside the box."""
    rng = np.random.default_rng(seed)
    box = mdapy.Box(MATRIX, origin=[1.0, -2.0, 0.5], pbc=pbc)
    fractional = rng.random((n, 3))
    fractional[:20] -= 0.002
    for e in range(3):
        if not box.pbc[e]:
            fractional[:, e] = np.clip(fractional[:, e], 0.001, 0.999)
    a = rng.normal(size=n)
    return mdapy.System(box.to_cartesian(fractional), box).with_columns(
        a=a, b=a + 0.5 * rng.normal(size=n), velocities=rng.normal(size=(n, 3))
    )


@pytest.mark.parametrize(
    "pbc, window",
    [
        (True, True),
        ((True, False, True), True),
        ((True, False, True), False),
        (False, True),
    ],
)
def test_as_ovito(pbc, window):
    """OVITO's spatial correlation function: the grid, the real and
    reciprocal space, the direct summation and the moments."""
    pytest.importorskip("ovito")
    from ovito.data import DataCollection
    from ovito.modifiers import SpatialCorrelationFunctionModifier
    from ovito.pipeline import Pipeline, StaticSource

    system = scattered(pbc)
    data = DataCollection()
    cell = np.zeros((3, 4))
    cell[:, :3] = system.box.matrix.T
    cell[:, 3] = system.box.origin
    data.create_cell(cell, pbc=tuple(bool(p) for p in system.box.pbc))
    particles = data.create_particles(count=len(system))
    particles.create_property("Position", data=system["positions"])
    particles.create_property("A", data=system["a"])
    particles.create_property("B", data=system["b"])
    pipeline = Pipeline(source=StaticSource(data=data))
    pipeline.modifiers.append(
        SpatialCorrelationFunctionModifier(
            property1="A",
            property2="B",
            grid_spacing=1.5,
            apply_window=window,
            direct_summation=True,
            neighbor_cutoff=6.0,
            neighbor_bins=40,
        )
    )
    theirs = pipeline.compute()
    ours = SpatialCorrelation(
        "a", "b", grid_spacing=1.5, window=window, direct=True, cutoff=6.0, bins=40
    ).compute(system)
    for name, x, y in [
        ("correlation-real-space", ours.r, ours.correlation),
        ("correlation-real-space-rdf", ours.r, ours.rdf),
        ("correlation-reciprocal-space", ours.q, ours.reciprocal),
        ("correlation-neighbor", ours.neighbor_r, ours.neighbor_correlation),
        ("correlation-neighbor-rdf", ours.neighbor_r, ours.neighbor_rdf),
    ]:
        table = theirs.tables[name]
        np.testing.assert_allclose(x, table.xy()[:, 0], rtol=1e-12)
        np.testing.assert_allclose(y, np.asarray(table.y), rtol=1e-9, atol=1e-12)
    assert len(ours.r) == 11
    attributes = theirs.attributes
    np.testing.assert_allclose(
        [
            *ours.means,
            *ours.mean_squares,
            ours.mean_product,
        ],
        [
            attributes[f"CorrelationFunction.{key}"]
            for key in ("mean1", "mean2", "variance1", "variance2", "covariance")
        ],
        rtol=1e-12,
    )


def test_autocorrelation_and_symmetry():
    """Without a second property, the autocorrelation; the two properties
    swapped give the same functions."""
    system = scattered()
    auto = SpatialCorrelation("a").compute(system)
    same = SpatialCorrelation("a", "a").compute(system)
    np.testing.assert_array_equal(auto.correlation, same.correlation)
    ab = SpatialCorrelation("a", "b", direct=True).compute(system)
    ba = SpatialCorrelation("b", "a", direct=True).compute(system)
    np.testing.assert_allclose(ab.correlation, ba.correlation, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(ab.reciprocal, ba.reciprocal, rtol=1e-10)
    np.testing.assert_allclose(
        ab.neighbor_correlation, ba.neighbor_correlation, rtol=1e-12
    )
    np.testing.assert_allclose(ab.means, ba.means[::-1])
    assert ab.neighbor_r is not None and auto.neighbor_r is None
    assert repr(SpatialCorrelation("a")) == "SpatialCorrelation('a', 'a')"


def test_constant_properties_give_the_pair_distribution():
    """Properties of one value each are the density times them: the
    correlation is their product times the radial distribution function."""
    system = scattered().with_columns(two=2.0, three=3.0)
    result = SpatialCorrelation("two", "three", direct=True).compute(system)
    np.testing.assert_allclose(result.correlation, 6 * result.rdf, rtol=1e-10)
    np.testing.assert_allclose(
        result.neighbor_correlation, 6 * result.neighbor_rdf, rtol=1e-12
    )
    # an ideal gas: g(r) about 1 beyond the grid cells
    assert np.abs(result.rdf[2:] - 1).max() < 0.1
    np.testing.assert_allclose(result.mean_product, 6.0)


def test_components_expressions_and_missing_values():
    """A component or an expression as a property; atoms of a NaN value are
    left out of the grid, as if they were 0, and make the moments NaN."""
    system = scattered()
    x = SpatialCorrelation("velocities.x").compute(system)
    named = SpatialCorrelation("vx").compute(
        system.with_columns(vx=system["velocities"][:, 0])
    )
    np.testing.assert_array_equal(x.correlation, named.correlation)
    doubled = SpatialCorrelation("2 * a").compute(system)
    np.testing.assert_allclose(
        doubled.correlation, 4 * SpatialCorrelation("a").compute(system).correlation
    )
    values = np.asarray(system["a"]).copy()
    values[:100] = np.nan
    missing = SpatialCorrelation("a").compute(system.with_columns(a=values))
    values[:100] = 0
    zero = SpatialCorrelation("a").compute(system.with_columns(a=values))
    np.testing.assert_array_equal(missing.correlation, zero.correlation)
    assert np.isnan(missing.means[0])


def test_errors():
    system = scattered()
    with pytest.raises(InvalidParameter, match="one value per atom"):
        SpatialCorrelation("velocities").compute(system)
    with pytest.raises(InvalidParameter, match="too large"):
        SpatialCorrelation("a", grid_spacing=20.0).compute(system)
    with pytest.raises(InvalidParameter, match="no atoms"):
        SpatialCorrelation("a").compute(
            system.filter(np.zeros(len(system), dtype=bool))
        )
    for name, value in [
        ("grid_spacing", 0),
        ("cutoff", -1.0),
        ("bins", 0),
        ("bins", 2.5),
        ("bins", True),
    ]:
        with pytest.raises(InvalidParameter, match=name):
            SpatialCorrelation("a", **{name: value})
