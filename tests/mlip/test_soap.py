# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.mlip import SOAP, farthest_point_sampling, structure_descriptors
from mdapy.transform import replicate

TRICLINIC = np.array([[11.0, 0, 0], [2.0, 10.0, 0], [-1.5, 1.0, 12.0]])


def alloy(pbc=True, seed=0, n=90):
    """Atoms of Cu, Ni and Al at random in a triclinic box, no two closer
    than 1.6 Å."""
    rng = np.random.default_rng(seed)
    box = mdapy.Box(TRICLINIC, origin=[1.0, -2.0, 0.5], pbc=pbc)
    points = []
    while len(points) < n:
        p = box.to_cartesian(rng.random((1, 3)))[0]
        steps = box.to_fractional(p - np.array(points).reshape(-1, 3) + box.origin)
        if pbc:
            steps -= np.round(steps)
        if not len(points) or np.linalg.norm(steps @ box.matrix, axis=1).min() >= 1.6:
            points.append(p)
    elements = rng.choice(["Cu", "Ni", "Al"], n, p=[0.5, 0.3, 0.2])
    return mdapy.System(np.array(points), box, elements=list(elements))


@pytest.mark.parametrize("pbc", [True, False])
@pytest.mark.parametrize(
    "r_cut, n_max, l_max, sigma, crossover",
    [
        (4.0, 4, 3, 0.5, True),
        (5.0, 8, 6, 0.5, True),
        (5.0, 5, 10, 1.0, False),
        (3.0, 2, 0, 0.3, True),
    ],
)
def test_as_dscribe(pbc, r_cut, n_max, l_max, sigma, crossover):
    """DScribe's SOAP with its Gaussian radial functions, of every atom;
    species absent from the system take their places in the layout."""
    pytest.importorskip("dscribe")
    from dscribe.descriptors import SOAP as Reference

    from mdapy.interop import to_ase

    system = alloy(pbc)
    species = ["Al", "Cu", "Ni", "Fe"]
    ours = SOAP(species, r_cut, n_max, l_max, sigma, crossover=crossover)
    theirs = Reference(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        periodic=pbc,
        compression={"mode": "off" if crossover else "crossover"},
    ).create(to_ase(system))
    found = ours.descriptors(system)
    assert found.shape == theirs.shape == (len(system), ours.dimension)
    np.testing.assert_allclose(found, theirs, rtol=0, atol=1e-9 * np.abs(theirs).max())


def test_invariance():
    """The same descriptors for a turned, moved or reordered system, and for
    each copy of a replicated one."""
    system = alloy()
    soap = SOAP(["Cu", "Ni", "Al"], 4.5, 6, 6)
    found = soap.descriptors(system)
    q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    turned = system.with_box(
        mdapy.Box(system.box.matrix @ q.T, origin=system.box.origin @ q.T + 3.0)
    ).with_columns(positions=system["positions"] @ q.T + 3.0)
    np.testing.assert_allclose(soap.descriptors(turned), found, rtol=1e-10, atol=1e-12)
    order = np.random.default_rng(2).permutation(len(system))
    shuffled = system.with_table(system.table[order])
    np.testing.assert_allclose(
        soap.descriptors(shuffled), found[order], rtol=1e-12, atol=1e-14
    )
    copies = soap.descriptors(replicate(system, (2, 1, 1)))
    np.testing.assert_allclose(copies[: len(system)], found, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(copies[len(system) :], found, rtol=1e-12, atol=1e-14)


def test_crystals_and_frames():
    """Atoms of one site of a perfect crystal share one descriptor, which
    tells fcc and bcc apart; frames of whole structures, chosen far apart."""
    fcc = build_crystal("fcc", 3.6, elements="Cu", repeat=(3, 3, 3))
    bcc = build_crystal("bcc", 2.87, elements="Cu", repeat=(4, 4, 4))
    soap = SOAP(["Cu"], 5.0, 6, 6)
    a, b = soap.descriptors(fcc), soap.descriptors(bcc)
    np.testing.assert_allclose(
        a, np.broadcast_to(a[0], a.shape), rtol=0, atol=1e-12 * a.max()
    )
    assert np.linalg.norm(a[0] - b[0]) > 1e-3 * np.linalg.norm(a[0])
    frames = [fcc, bcc, fcc, bcc.with_columns(positions=bcc["positions"] + 0.01)]
    points = structure_descriptors(frames, soap)
    assert points.shape == (4, soap.dimension)
    assert sorted(farthest_point_sampling(points, 2).tolist()) in ([0, 1], [0, 3])


def test_neighbors_threads_and_repr():
    system = alloy(seed=4)
    soap = SOAP(["Al", "Cu", "Ni"], 4.0, 4, 4)
    found = soap.descriptors(system)
    longer = mdapy.NeighborList.build(system, soap.cutoff + 1.0)
    np.testing.assert_allclose(
        soap.descriptors(system, neighbors=longer), found, rtol=1e-12, atol=1e-14
    )
    with mdapy.num_threads(1):
        np.testing.assert_array_equal(soap.descriptors(system), found)
    assert soap.cutoff == pytest.approx(4.0 + 0.5 * np.sqrt(-2 * np.log(1e-3)))
    assert soap.dimension == 3 * 4 * (3 * 4 + 1) // 2 * 5
    assert SOAP("Cu", crossover=False).species == ("Cu",)
    assert (
        repr(soap) == "SOAP(('Al', 'Ni', 'Cu'), r_cut=4.0, n_max=4, l_max=4, sigma=0.5)"
    )


def test_errors():
    for kwargs, match in [
        ({"species": []}, "at least one"),
        ({"species": ["Xx"]}, "unknown element"),
        ({"species": ["Cu", "Cu"]}, "twice"),
        ({"species": ["Cu"], "r_cut": 1.0}, "r_cut"),
        ({"species": ["Cu"], "n_max": 0}, "n_max"),
        ({"species": ["Cu"], "n_max": 2.5}, "n_max"),
        ({"species": ["Cu"], "l_max": 21}, "l_max"),
        ({"species": ["Cu"], "l_max": True}, "l_max"),
        ({"species": ["Cu"], "sigma": 0}, "sigma"),
        ({"species": ["Cu"], "r_cut": 1.5, "n_max": 40}, "linearly dependent"),
    ]:
        with pytest.raises(InvalidParameter, match=match):
            SOAP(**kwargs)
    with pytest.raises(InvalidParameter, match="not among the species"):
        SOAP(["Cu"]).descriptors(alloy())
    with pytest.raises(MissingProperty, match="element"):
        SOAP(["Cu"]).descriptors(mdapy.System(np.zeros((1, 3)), mdapy.Box(10.0)))
