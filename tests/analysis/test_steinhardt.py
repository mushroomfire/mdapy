# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest
from numpy.polynomial import legendre

import mdapy
from mdapy.analysis import SteinhardtParameters
from mdapy.errors import InvalidParameter

from lattices import crystal, icosahedron


def unit_vectors(system, k):
    nl = mdapy.NeighborList.build(system, num_neighbors=k)
    v = nl.vectors.reshape(len(system), k, 3)
    return v / np.linalg.norm(v, axis=2, keepdims=True), nl.j.reshape(len(system), k)


def reference(system, degree, k, average=False):
    """q_l from the addition theorem: (4 pi / (2l + 1)) sum_m q_lm(a) q_lm(b)*
    is the mean of P_l over the bonds of a and b."""
    u, j = unit_vectors(system, k)

    def overlap(a, b):
        return legendre.legval(u[a] @ u[b].T, [0] * degree + [1]).mean()

    q = []
    for i in range(len(system)):
        members = [i, *j[i]] if average else [i]
        total = sum(overlap(a, b) for a in members for b in members)
        q.append(np.sqrt(total) / len(members))
    return np.array(q)


@pytest.mark.parametrize(
    "kind, a, k, q4, q6",
    [
        ("fcc", 3.6, 12, 0.190941, 0.574524),
        ("hcp", 2.5, 12, 0.097222, 0.484762),
        ("bcc", 2.87, 8, 0.509175, 0.628539),
        ("bcc", 2.87, 14, 0.036369, 0.510688),
    ],
)
def test_perfect_crystals(kind, a, k, q4, q6):
    """Values of the literature, to their six digits."""
    result = SteinhardtParameters(num_neighbors=k).compute(crystal(kind, a))
    np.testing.assert_allclose(result[4], q4, atol=1e-6)
    np.testing.assert_allclose(result[6], q6, atol=1e-6)
    averaged = SteinhardtParameters(num_neighbors=k, average=True).compute(
        crystal(kind, a)
    )
    np.testing.assert_allclose(averaged.q, result.q, rtol=1e-12)  # all atoms alike


def test_icosahedron_and_cutoff():
    center = SteinhardtParameters((6,), num_neighbors=12).compute(icosahedron())[6][0]
    assert center == pytest.approx(np.sqrt(11) / 5, abs=1e-12)  # 0.663325
    fcc = crystal("fcc", 3.6)
    by_cutoff = SteinhardtParameters(cutoff=3.0).compute(fcc)
    np.testing.assert_allclose(
        by_cutoff.q, SteinhardtParameters().compute(fcc).q, rtol=1e-12
    )


@pytest.mark.parametrize("average", [False, True])
def test_matches_the_addition_theorem(average):
    system = crystal("fcc", 3.6, jitter=0.2, seed=1)
    degrees = tuple(range(13))
    result = SteinhardtParameters(degrees, average=average).compute(system)
    np.testing.assert_allclose(result[0], 1.0, rtol=1e-12)
    for degree in (1, 2, 4, 6, 8, 12):
        np.testing.assert_allclose(
            result[degree],
            reference(system, degree, 12, average),
            rtol=1e-10,
            atol=1e-12,
        )


def test_rotation_threads_and_columns():
    system = crystal("bcc", 2.87, jitter=0.1, seed=2)
    q, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(3, 3)))
    turned = mdapy.System(
        system.positions @ q.T, mdapy.Box(system.box.matrix @ q.T), elements="Cu"
    )
    parameters = SteinhardtParameters((4, 6, 8), num_neighbors=8, average=True)
    result = parameters.compute(system)
    np.testing.assert_allclose(parameters.compute(turned).q, result.q, rtol=1e-10)
    with mdapy.num_threads(3):
        np.testing.assert_array_equal(parameters.compute(system).q, result.q)
    nl = mdapy.NeighborList.build(system, 4.0)
    np.testing.assert_array_equal(parameters.compute(system, neighbors=nl).q, result.q)
    table = system.with_columns(result).table
    assert {"q4_average", "q6_average", "q8_average"} <= set(table.columns)
    assert list(SteinhardtParameters((6,)).compute(system).columns()) == ["q6"]
    assert repr(parameters) == (
        "SteinhardtParameters(degrees=(4, 6, 8), num_neighbors=8, average=True)"
    )
    assert "cutoff=3.0" in repr(SteinhardtParameters(cutoff=3.0))
    with pytest.raises(InvalidParameter, match="degree 2 was not computed"):
        result[2]
    with pytest.raises(InvalidParameter, match="fewer than 8"):
        parameters.compute(system, neighbors=nl.within(2.0))


def test_atoms_without_neighbors():
    pair = mdapy.System(
        [[0.0, 0, 0], [5.0, 0, 0]], mdapy.Box(20.0, pbc=False), elements="Ar"
    )
    result = SteinhardtParameters((0, 6), cutoff=2.0).compute(pair)
    np.testing.assert_array_equal(result.q, 0.0)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"degrees": ()}, "degrees must be from 0 to 20"),
        ({"degrees": (21,)}, "degrees must be from 0 to 20"),
        ({"num_neighbors": 4, "cutoff": 3.0}, "not both"),
        ({"cutoff": 0.0}, "cutoff must be positive"),
        ({"num_neighbors": 0}, "num_neighbors must be at least 1"),
    ],
)
def test_invalid_options(options, message):
    with pytest.raises(InvalidParameter, match=message):
        SteinhardtParameters(**options)
